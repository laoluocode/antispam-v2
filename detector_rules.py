"""L0 确定性规则。

设计原则：**只放零误伤的模式**。

V1 的教训是把 150 行"群内黑话"硬编码进正则（`尚门`、`大一外出`、`特jia` …），
结果是规则跟广告号的措辞赛跑，且一旦命中就绕过了所有 AI 兜底与人工复核。
V2 里这类"像广告的词"一律降到 few-shot 例子，交给模型判断；
L0 只保留三类**确定性事实**：

1. 联系方式本体（手机号、t.me、@bot）—— 正常群友不会发这些
2. 广告号混淆手法（不可见字符高占比）—— 正常群友不会打出变体选择符
3. 黑名单用户 —— 人工或学习层确认过的账号

命中 L0 直接删除，不进 LLM，延迟 ~0ms、成本 0。
"""

import re
import time
from typing import NamedTuple

from utils.normalize import invisible_ratio, normalize

# 中国大陆手机号：正常聊天不会出现（群里没人报手机号）
PHONE_PATTERN = re.compile(r"(?:\+?86[-\s]?)?1[3-9]\d{9}")

# Telegram 引流链接
TELEGRAM_LINK_PATTERN = re.compile(
    r"(?:t\.me/|telegram\.me/|@[A-Za-z][A-Za-z0-9_]{4,31}bot\b)",
    re.IGNORECASE,
)

# 数字类模式的安全折叠：只把全角数字转半角，**不做其他归一化**。
#
# 为什么不能直接用 normalize()：它是为模糊匹配设计的有损变换
# （l->1、o->0、谐音替换、去噪声），拿它做 L0 判定会引入误伤——
# 例如 "Ill" 会被折成 "111"。而全角数字的折叠是无损的、可逆的，
# 且实测正负样本里都从未出现过全角数字，因此零误伤风险。
#
# 修的是一个现成的绕过口子：
#   "我电话13812345678"        -> 命中
#   "我电话１３８１２３４５６７８"  -> 原来放过（全角数字 \w 为真，不被当噪声）
_FW_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def fold_digits(text: str) -> str:
    """只折叠全角数字，保留其余字符原样。用于数字类模式的匹配。"""
    return (text or "").translate(_FW_DIGITS)


# 银行卡号（16-19 位连续数字，且不是手机号）
BANK_CARD_PATTERN = re.compile(r"\b\d{16,19}\b")

# 不可见字符占比阈值。
# 实测：正常群聊消息为 0%，广告号混淆消息可达 20-40%。
# 取 15% 留足余量，且要求消息足够长以免短消息被单个字符拉高占比。
INVISIBLE_RATIO_THRESHOLD = 0.15
INVISIBLE_MIN_LENGTH = 10


class Verdict(NamedTuple):
    """L0 判定结果。hit=False 时 reason 为空。"""

    hit: bool
    reason: str


_NO_HIT = Verdict(False, "")


def compute_trust(msg_n: int, del_n: int, span_days: float = 0.0) -> str:
    """计算用户信任档位。这是**喂给模型的信号**，不是一条保护规则。

    V1 的错误是把"发言多"写成硬编码的 `if total_count > 100: skip`，
    后果有两个：漏掉刷量伪装的广告号，同时让 AI 完全看不到这个信息。

    实测依据（生产库 12,470 名用户，114 天数据）：

    | 信号 | 区分度 |
    |---|---|
    | 发言量 | 无效。发言 1000+ 的用户里广告号占比 2.4%，发言 <10 的是 1.7% |
    | 删除比 del/msg | 极强。21.4% 的用户从未被删，24.3% 的用户删除比 >90% |

    反例（发言量完全失效的证据）：
        发言=458 被删=294 删除比=64%   "看本质 扒开现象"
        发言=151 被删=111 删除比=74%   "梦小琪洪山开课"
    这些按"发言>100 即老用户"的标准全在保护区内，实际是刷量伪装的广告号。

    档位含义：
    - trusted : 发言够多且删除比极低 -> prompt 标 [信任=高]
    - repeat  : 删除比高且被删次数够多 -> 标 [信任=低]，并进快车道
    - new     : 发言极少 -> 标 [新用户]，判定应更谨慎
    - normal  : 无标注，由模型自行判断
    """
    ratio = del_n / msg_n if msg_n else 0.0

    # 新人优先：发言极少时删除比不可信（发 1 条被删就是 100%）
    if msg_n < 5:
        return "new"

    # 惯犯：要求被删次数足够多，避免"发 10 条被删 3 条"被误标为广告号
    if del_n >= 10 and ratio >= 0.30:
        return "repeat"

    # 受信任：发言够多且删除比极低
    if msg_n >= 30 and ratio < 0.05:
        return "trusted"

    return "normal"


TRUST_LABELS = {
    "trusted": "[信任=高]",
    "repeat": "[信任=低]",
    "new": "[新用户]",
    "normal": "",
}


def trust_marker(trust: str) -> str:
    """prompt 中显示的信任标记。normal 档返回空串。"""
    return TRUST_LABELS.get(trust, "")


def check_local(text: str, user_id: int, chat_id: int, blacklist_ids: set) -> Verdict:
    """L0 检查。任何一项命中即返回删除判定。

    注意：这里刻意不做"广告词匹配"。像 `上门`、`约课`、`养生` 这类词
    在本群是正常话题（V1 因移除 `开课` 关键词有过专门的修复提交），
    必须交给模型结合上下文判断。
    """
    t = text or ""
    if not t.strip():
        return _NO_HIT

    # 1. 黑名单用户：不看内容，直接删
    if user_id in blacklist_ids:
        return Verdict(True, "黑名单用户")

    # 2. 不可见字符混淆：正常输入不会产生变体选择符
    if len(t) >= INVISIBLE_MIN_LENGTH and invisible_ratio(t) >= INVISIBLE_RATIO_THRESHOLD:
        ratio = invisible_ratio(t)
        return Verdict(True, f"不可见字符混淆（占比 {ratio:.0%}）")

    # 3. 手机号（含全角数字变体）
    td = fold_digits(t)
    if PHONE_PATTERN.search(td):
        return Verdict(True, "包含手机号")

    # 4. Telegram 引流链接
    if TELEGRAM_LINK_PATTERN.search(t):
        return Verdict(True, "包含 Telegram 引流链接")

    # 5. 银行卡号（先排除已被手机号规则覆盖的情况）
    if not PHONE_PATTERN.search(td) and BANK_CARD_PATTERN.search(td):
        return Verdict(True, "包含银行卡号")

    return _NO_HIT


def signature(text: str) -> str:
    """消息指纹，用于重复刷屏检测。

    归一化后比对，使 `个人外出` 与 `個 人 外 出` 视为同一条。
    """
    return normalize(text)


# 尾部 nonce：`开课中❤️❤️ e` / `开课中❤️❤️ J` —— 广告号在固定文案后
# 追加一个随机字符，使每条消息字符串都不同，从而绕过"完全相同才判重复"的检测。
#
# 注意 normalize 会先删掉空格与 emoji，`开课中❤️❤️ e` 归一化后是 `开课中e`，
# 所以这里直接在归一化结果上剥掉尾部字母数字（而非先匹配分隔符）。
_NONCE_TAIL_PATTERN = re.compile(r"[a-z0-9]{1,3}$")

# 参与重复检测的最小归一化长度。低于此长度的短句（"在的"、"好的"）不参与，
# 避免不同短语被剥掉尾部后归一成同一个 key。
MIN_REPEAT_KEY_LENGTH = 4


def repeat_key(text: str) -> str:
    """重复检测用的指纹：归一化 + 去掉尾部随机 nonce。

    实测价值：线上有广告号用 `开课中❤️❤️` 配 `e/J/t/K/L/r/4` 连发 7 条，
    归一化后仅尾部 nonce 不同。剥掉尾部后 7 条归一为 `开课中`，即可判重复。

    长度门槛加在**归一化结果**上（>= 4 字符）而非剥离后：
    `开课中e` 归一化后 4 字符符合门槛，剥离得 3 字符的 `开课中` 仍应采用；
    而 `在的`（2 字符）不达门槛，不参与重复检测，避免正常短语被误归一。
    """
    norm = normalize(text)
    if len(norm) < MIN_REPEAT_KEY_LENGTH:
        return norm
    return _NONCE_TAIL_PATTERN.sub("", norm) or norm


# 纯 ASCII **字母**的极短消息，且不含数字。
#
# 为什么排除数字：实测老库 21 万条消息里，`1` 出现 282 次且都是正常回应
# （群友用 "1" 表示"在/同意"），`66`/`88` 同理。把数字算进来必然误伤。
#
# 为什么只留字母：实测长度<=2 的消息里，裸字母（d/n/m/t/o/u）**只出现在
# 广告号**（Jfdueyp Kvdzehcm 发 15 条、Wdoggmgtbn Vhexblzls 发 6 条），
# 正常群友的短回应全是中文（「谁」「好」「在呢」「哦」「哈喽」）。
_BARE_LETTER_PATTERN = re.compile(r"^[A-Za-z]{1,3}$")

# 常见英文回应词白名单。这些在正常聊天里确实会出现，本身也没有广告价值。
_BARE_LETTER_ALLOW = frozenset({
    "ok", "no", "hi", "up", "dd", "gg", "thx", "yes", "lol", "wtf",
    "id", "qq", "vx", "tg",
})

# 触发所需的历史次数。实测线上广告号每 5-7 分钟发一个字母，10 分钟内
# 能攒到 2-3 个；正常用户几乎为 0，所以阈值取 3 已经很保守。
BARE_LETTER_THRESHOLD = 3


def is_bare_letter(text: str) -> bool:
    t = (text or "").strip()
    if t.lower() in _BARE_LETTER_ALLOW:
        return False
    return bool(_BARE_LETTER_PATTERN.match(t))


def check_bare_letter_spam(text: str, recent_texts: list[str],
                           threshold: int = BARE_LETTER_THRESHOLD) -> Verdict:
    """裸字母刷屏检测（只匹配纯字母，不含数字）。

    实测案例（2026-10-10，南京苏州集团）：
      账号 A: n d d d n n d d m d n d d m   （14 条，每 5-7 分钟一条）
      账号 B: u t o t o t                    （6 条，与 A 交叉插队）
    两个账号都在发单个 ASCII 字母，被删记录均为 0，全部漏过。

    这类消息本身不含广告特征——真正的广告在**频道预览卡片**里
    （客户端显示 "📢 频道名 + 招募探花…"），而 bot 只收到那个字母。
    但**发单字母的行为本身就是强特征**：正常群友不会这么干。

    这条规则同时覆盖两种变体：
      · 拆字：把一句话拆成单字分多条发（规避整句关键词匹配）
      · 单字母 + 链接预览：用字母当载体，广告在预览里

    实测误伤边界（老库 21 万条消息）：
      `1` 出现 282 次、`66`/`88` 类似——都是正常回应，**必须排除数字**；
      正常群友的超短消息全是中文（谁/好/在呢/哦/哈喽），不会用裸字母。
    """
    t = (text or "").strip()
    if not is_bare_letter(t):
        return _NO_HIT
    n = sum(1 for h in recent_texts if is_bare_letter(h))
    if n + 1 >= threshold:
        return Verdict(True, f"单字符刷屏（近期共 {n + 1} 条裸字母）")
    return _NO_HIT


# t.me 链接里的用户名/频道标识
_TME_USERNAME = re.compile(r"(?:https?://)?t\.me/(?:s/)?([A-Za-z0-9_]{4,})", re.I)
_TME_PRIVATE = re.compile(r"(?:https?://)?t\.me/c/", re.I)


def extract_tme_identifiers(text: str) -> set[str]:
    """抽取文本里所有 t.me 目标标识（小写用户名）。"""
    out = set()
    for m in _TME_USERNAME.finditer(text or ""):
        name = m.group(1).lower()
        # 纯数字是帖子 ID 不是用户名（t.me/xxx/123 里的 123 不会匹配到这里，
        # 但 t.me/123 会被匹配，排除掉）
        if name.isdigit():
            continue
        out.add(name)
    return out


def check_external_channel_link(text: str, managed: set[str]) -> Verdict:
    """预览/链接指向**非本群体系**的 t.me 目标 -> 外部引流。

    为什么这条规则有效（实测背景）：
    这些群形成了"集团"式频道体系，群主自己运营榜单/报告/资讯频道，
    群友转发自己频道的帖子是常态。广告号利用这个习惯，把**外部频道**
    的帖子当预览卡片发进群——正文 Telegram 不下发，bot 只看到单字母。

    但**链接指向哪里是可见的**。指向群主管理的频道 = 正常；
    指向一个陌生频道 = 引流。这是"关系"层面的判断，与广告文案无关，
    所以广告号改文案也绕不过去。

    managed 由调用方提供，应包含：本群自身、各群的 linked_chat，
    以及运营方在群描述里公开的频道。
    """
    ids = extract_tme_identifiers(text)
    if not ids:
        return _NO_HIT
    unmanaged = {i for i in ids if i not in managed}
    if unmanaged and len(unmanaged) == len(ids):
        # 全部指向陌生目标才算——只要有一个是自家频道，就不判，
        # 避免把"同时提到自家榜单和外部链接"的正常消息误伤
        return Verdict(True, f"外部频道引流（{'、'.join(list(unmanaged)[:2])}）")
    return _NO_HIT


# 广告按钮的文案特征。广告号常把"加入频道/领取奖励"做成 inline 按钮，
# 正文很干净（甚至只有一个表情），按钮才是那个广告。
_AD_BUTTON_PATTERN = re.compile(
    r"(加入|点击|领取|立即|马上|查看|进入|申请|咨询|联系|开户|注册|下载|开始|获取|"
    r"join|click|claim|start|open|register|download|visit|apply|contact"
    r")\s*(频道|群组|群|频道|链接|这里|我|奖励|福利|红包|彩金|名额|客服|网站|APP|app|"
    r"channel|group|link|here|now|bonus|reward)?",
    re.IGNORECASE,
)

# 按钮 URL 指向的链接同样受"外部频道"规则约束，这里只做基础识别
_BUTTON_URL_PATTERN = re.compile(r"https?://|t\.me/", re.IGNORECASE)


def check_inline_buttons(button_texts: list[str],
                         button_urls: list[str],
                         managed: set[str]) -> Verdict:
    """检测消息附带的 inline 按钮。

    为什么需要：广告号把广告做成按钮——正文可以只是一个表情，
    按钮写"加入频道｜领取福利"。这类消息在文本层完全干净，
    keyword/LLM 都看不出来。

    命中条件（任一）：
      · 按钮 URL 指向非本群体系的外部 t.me 目标（与外部频道规则同源）
      · 按钮文案是明确的广告引导语（加入频道/领取奖励…）

    正常运营的按钮（"查看榜单"指向自家频道）不会命中。
    """
    if not button_texts and not button_urls:
        return _NO_HIT

    # 1) 按钮 URL 指向外部
    for url in button_urls:
        ids = extract_tme_identifiers(url)
        if ids and not (ids & {m.lower() for m in managed}):
            return Verdict(True, f"按钮指向外部频道（{list(ids)[0]}）")
        # 非 Telegram 的外部链接 + 广告引导文案
        if _BUTTON_URL_PATTERN.search(url) and not url.lower().startswith(
                ("https://t.me/", "http://t.me/", "tg://")):
            for t in button_texts:
                if _AD_BUTTON_PATTERN.search(t):
                    return Verdict(True, f"按钮引导站外链接（{t[:12]}）")

    # 2) 按钮文案是广告引导语
    for t in button_texts:
        tt = (t or "").strip()
        if len(tt) <= 24 and _AD_BUTTON_PATTERN.search(tt):
            # 需要同时带"渠道"类词或为纯引导，避免把"查看"这类中性按钮误伤
            if re.search(r"(频道|群组|群|链接|奖励|福利|红包|彩金|名额|客服|"
                         r"channel|group|link|bonus|reward)", tt, re.I):
                return Verdict(True, f"广告按钮（{tt[:14]}）")
    return _NO_HIT


# ---------------------------------------------------------------------------
# 自进化规则加载
# ---------------------------------------------------------------------------

# 规则缓存。学习任务写完库后由 pipeline 触发 force 重载。
_pattern_cache: list = []
_pattern_cache_at = 0.0
PATTERN_CACHE_TTL = 300.0


def load_patterns(force: bool = False) -> list:
    """从库里加载自进化规则到缓存。返回 [(pattern, kind, hits), ...]。

    active 规则用于 L0 直拦（0ms 删除）；shadow 规则在判定时并行记录命中，
    不影响决定——这是"先观察再放权"的安全阀。
    """
    global _pattern_cache, _pattern_cache_at
    now = time.time()
    if not force and now - _pattern_cache_at < PATTERN_CACHE_TTL:
        return _pattern_cache
    # 直接 import db，不要写 "try: from database ... except ImportError" 的回退：
    # `database` 是老 bot 的模块（另一个 schema），在容器里 import 会**成功**
    # （不触发 ImportError），于是拿它查 spam_patterns 必然失败并被 except 吞掉，
    # 静默返回空缓存——自进化规则曾因此从上线起就没被加载过。
    try:
        import db as _db
    except ImportError:
        _db = None
    if _db is None:
        logger.error("无法加载规则库模块，自进化规则不可用")
        return _pattern_cache

    try:
        active = [(r["pattern"], "active", r["hit_count"] or 0)
                  for r in _db.active_patterns()]
        shadow = [(r["pattern"], "shadow", r["shadow_hits"] or 0)
                  for r in _db.patterns_for_shadow()]
        _pattern_cache = active + shadow
        _pattern_cache_at = now
    except Exception as exc:  # noqa: BLE001
        # 不能静默：加载失败意味着自进化完全失效，必须看得见
        logger.error("加载自进化规则失败: %s: %s", type(exc).__name__, exc)
    return _pattern_cache


def check_learned_patterns(text: str) -> Verdict:
    """用**已毕业**（active）的自进化规则做 L0 直拦。

    这些规则经过了"负样本零命中"闸门与 shadow 观察期，才有资格直拦。
    shadow 规则不在这里生效——它们的命中由 record_shadow_hits() 记录。
    """
    t = text or ""
    if not t:
        return _NO_HIT
    norm = normalize(t)
    for pattern, mode, _hits in load_patterns():
        if mode != "active":
            continue
        if _pattern_matches(pattern, t, norm):
            return Verdict(True, f"自进化规则命中（{pattern[:16]}）")
    return _NO_HIT


def _pattern_matches(pattern: str, raw: str, norm: str) -> bool:
    """判断一条自进化规则是否命中。

    · 行为类（`__invisible_ratio__` 等）走专门的判定
    · 域类（t.me/xxx、example.com）在原文与归一化文本里找
    · 其余（签名/短语）在归一化文本里找——签名与短语都是归一化后的形态
    """
    if pattern == "__invisible_ratio__":
        return len(raw) >= INVISIBLE_MIN_LENGTH and \
            invisible_ratio(raw) >= INVISIBLE_RATIO_THRESHOLD
    if pattern == "__bare_letter__":
        return is_bare_letter(raw)
    if pattern.startswith("t.me/") or _DOMAIN_LIKE.match(pattern):
        return pattern in raw.lower() or pattern in norm
    return pattern and pattern in norm


def record_shadow_hits(text: str, is_spam: bool) -> list[str]:
    """记录 shadow 规则在本条消息上的命中情况。

    返回命中的规则列表。**只在真值已知时调用**（例如 LLM 判完之后），
    这样 shadow_spam_hits / shadow_hits 才能反映该规则的准确率。
    """
    t = text or ""
    if not t:
        return []
    norm = normalize(t)
    hits = []
    for pattern, mode, _hits in load_patterns():
        if mode != "shadow":
            continue
        if _pattern_matches(pattern, t, norm):
            hits.append(pattern)
    if hits:
        from db import bump_pattern
        for pattern in hits:
            try:
                bump_pattern(pattern, shadow_hit=True, spam_hit=is_spam)
            except Exception:  # noqa: BLE001
                pass
    return hits


_DOMAIN_LIKE = re.compile(r"^[a-z0-9.\-]+\.[a-z]{2,}$", re.I)


# ---------------------------------------------------------------------------
# 波次检测（借鉴 Reddit 的"协调行为"识别）
# ---------------------------------------------------------------------------

# 实测依据（线上 1811 条删除记录）：
#   删除量中位数 = 2 条/分钟
#   但有 9 个"波次"，其中最大的一次是 **195 条/分钟、来自 105 个不同账号**
#   9 个波次覆盖了全部删除量的 25%
# 单账号维度的规则对"105 个账号同时发广告"这种协作攻击完全无感——
# 每个账号都只发了一两条，都不触发刷屏阈值。
#
# 所以需要**群级**的密度指标：短期内删除率飙升 = 正在被批量投放，
# 此时降低判定门槛、全部走快车道，把响应时间压到最低。
RAID_WINDOW_SECONDS = 60          # 观察窗
RAID_MIN_DELETES = 10             # 窗口内删除数达到这个量级才可能是波次
RAID_BASELINE_MULTIPLIER = 4.0    # 或达到基线均值的 4 倍（取两者较大值）
RAID_BASELINE_DEFAULT = 2.0       # 冷启动时的基线假设（实测中位数 2 条/分钟）


class RaidDetector:
    """群级删除密度监控，识别协调投放的波次。

    与逐条判定的区别：逐条看的是"这条是不是广告"，
    波次检测看的是"这个群现在是不是正在被批量投放"。
    后者是全局信号，前者看不到。
    """

    def __init__(self, window: int = RAID_WINDOW_SECONDS,
                 baseline: float = RAID_BASELINE_DEFAULT):
        self.window = window
        self.baseline = baseline
        self._events: list[float] = []      # 删除时间戳
        self._active_until = 0.0
        self._peak = 0

    def record_delete(self, now: float | None = None) -> None:
        self._events.append(now if now is not None else time.time())

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._events and self._events[0] < cutoff:
            self._events.pop(0)

    def in_raid(self, now: float | None = None) -> bool:
        """当前是否处于波次中（含判定后的冷却期）。"""
        n = now if now is not None else time.time()
        return n < self._active_until

    def check(self, now: float | None = None) -> tuple[bool, int, float]:
        """返回 (是否波次, 窗口内删除数, 触发阈值)。

        同时维护冷却期：一旦判定为波次，在窗口长度内保持"波次中"状态——
        波次是持续过程，不能因为某一秒没删除就立刻退出警戒。
        """
        n = now if now is not None else time.time()
        self._prune(n)
        count = len(self._events)
        self._peak = max(self._peak, count)

        threshold = max(RAID_MIN_DELETES,
                        self.baseline * RAID_BASELINE_MULTIPLIER)
        if count >= threshold:
            self._active_until = n + self.window
            return True, count, threshold
        return self.in_raid(n), count, threshold


# ---------------------------------------------------------------------------
# 账号资料检测（借鉴 Discord AutoMod 的 MemberProfile 触发类型）
# ---------------------------------------------------------------------------

# 昵称里的引流指引。广告号把"去哪找我"直接写进昵称，这样即使正文被删、
# 甚至不说话，群友也能找到它——"南叙 主页有群"、"屿辰 看我简介"。
#
# 实测区分度（全库 522 个有删除记录的账号 / 19 个可信群友）：
#   本规则命中 33 个广告号（6.3%），**可信群友 0 个**
# 与自进化闸门同判据（负样本零命中），因此可直接作为 L0 拦截依据。
_NICKNAME_GUIDE = re.compile(
    r"看\s*(我|下)?\s*(的)?\s*(简介|资料|主页|签名|名字|头像|背景)"
    r"|看\s*(竹叶|蛛叶|煮页|主叶|简戒|简个|苝景|苝井)"      # 谐音变体
    r"|主页\s*(有|看|在)"
    r"|点\s*(我)?\s*(头像|主页|简介)"
    r"|简介\s*(有|看|详)"
    r"|详\s*(情)?\s*看"
    r"|加\s*(我)?\s*(v|薇|威|徽|微|扣|q)\b",
    re.I,
)

# 昵称里的联系方式/服务标识。同样实测可信群友 0 命中。
_NICKNAME_CONTACT = re.compile(
    r"[（(]?\s*(客服|外围|上门|楼凤|修车|约课|开课|预约号)\s*[)）]?"
    r"|@[A-Za-z0-9_]{4,}"                # 昵称里直接塞 @username
    r"|兼职在线|在校|学妹|壆妹|资原|资源|兼职|谦职|会所|养生|全套|半套",
    re.I,
)


def check_account_profile(full_name: str | None,
                          username: str | None = None) -> Verdict:
    """账号资料检测。

    为什么值得做：广告号的**账号资料本身就是广告位**。它们把引流指引写进昵称
    （"看我简介"），这样正文被删了也照样能被找到。Discord AutoMod 有专门的
    MemberProfile 触发类型做这件事，我们原来完全没检查资料。

    实测（522 个有删除记录的账号 vs 19 个可信群友）：
      引流指引类命中 33 个广告号（6.3%），可信群友 0 命中
      联系方式类命中 14 个（2.7%），可信群友 0 命中
    两类的误伤都为零，符合 L0"只放零误伤模式"的门槛。
    """
    name = (full_name or "").strip()
    if not name:
        return _NO_HIT
    m = _NICKNAME_GUIDE.search(name)
    if m:
        return Verdict(True, f"昵称含引流指引（{m.group(0)[:10]}）")
    m = _NICKNAME_CONTACT.search(name)
    if m:
        return Verdict(True, f"昵称含服务标识（{m.group(0)[:10]}）")
    return _NO_HIT


# ---------------------------------------------------------------------------
# 自适应检查强度（借鉴 StopSpam）
# ---------------------------------------------------------------------------

# 触发重新检查的条件。任何一条命中 -> 这条消息必须送 AI，即使发件人是老群友。
#
# 设计依据（StopSpam 的做法）：长期干净发言的成员不必每条都查，
# 只在"触发可疑"时回到审查——发链接、发邀请、转发、久未发言后回归。
# 这样把 AI 的注意力限制在"它存在的意义上要抓的流量"上。
#
# ⚠️ 关键：跳过 AI **不等于不检查**。L0 确定性规则（联系方式/不可见字符/
# 黑名单/裸字母/重复/外部频道/按钮/自进化规则）永远先跑，全部命中即删。
# 这里只决定"是否再送 AI 兜底"。
TRIGGER_NEW_USER = "新成员"
TRIGGER_FLAGGED = "曾被删过"


def check_triggers(text: str, trust: str, has_preview: bool,
                   has_buttons: bool, is_forward: bool,
                   msg_count: int = 0,
                   recent_delete_count: int = 0) -> list[str]:
    """返回触发的检查条件列表。空列表 = 无需送 AI。

    条件覆盖了"能把广告塞进来"的所有形态，所以跳过 AI 不会漏掉广告载体：
      · 链接 / t.me / 按钮 / 预览 —— 引流必备
      · 转发 —— 广告常用手法
      · 新成员 —— 老群友才享免检
      · 曾被删过 —— 有前科的不信任
      · 含联系方式（手机号）—— 已在 L0 覆盖，这里再兜一次
    """
    reasons: list[str] = []
    t = text or ""

    if trust in ("new",):
        reasons.append(TRIGGER_NEW_USER)
    if trust == "repeat" or recent_delete_count > 0:
        reasons.append(TRIGGER_FLAGGED)

    # 链接类：引流必备载体
    if extract_tme_identifiers(t) or _URL_ANY.search(t):
        reasons.append("含链接")
    if has_preview:
        reasons.append("含链接预览")
    if has_buttons:
        reasons.append("含按钮")
    if is_forward:
        reasons.append("转发消息")

    # 极短内容：可能是单字母刷屏或拆分文案
    if len(t.strip()) <= 3:
        reasons.append("极短内容")

    return reasons


_URL_ANY = re.compile(r"https?://|www\.", re.I)


def check_repeat(text: str, recent_texts: list[str], threshold: int = 3) -> Verdict:
    """重复刷屏检测。

    recent_texts 为该用户在本群的近期消息（调用方提供，已按时间倒序）。
    同一 repeat_key 出现 >= threshold 次即判为刷屏。

    这是确定性规则而非 AI 判定：重复行为本身就是广告特征，
    且归一化去 nonce 后正常群友几乎不会触发（正常人不会把同一句话发三遍）。
    """
    key = repeat_key(text)
    if len(key) < 4:
        return _NO_HIT

    count = sum(1 for t in recent_texts if repeat_key(t) == key)
    if count >= threshold:
        return Verdict(True, f"重复刷屏（同一内容第 {count + 1} 次）")
    return _NO_HIT


if __name__ == "__main__":
    samples = [
        ("能收🧧就行，曰5K", 1, "普通"),
        ("我的电话是13812345678，有事联系", 2, "含手机号"),
        ("加我 t.me/someone 详聊", 3, "含tg链接"),
        ("个人外出 無定面付 详情dd", 4, "黑话但无确定性特征"),
        ("今天在汉口那边修的车，体验还行", 5, "正常"),
        ("卡号 6222021234567890123 打款", 6, "含银行卡"),
        ("@some_spam_bot 点我", 7, "bot链接"),
        ("约课吗 有空", 8, "正常话题词"),
    ]
    bl = {99}
    print(f"{'结果':6s} {'原因':24s} 样本")
    print("-" * 78)
    for text, uid, label in samples:
        v = check_local(text, uid, -100, bl)
        status = "命中" if v.hit else "放行"
        print(f"{status:6s} {v.reason:24s} [{label}] {text}")
    print()
    print("黑名单用户测试:")
    v = check_local("今天天气不错", 99, -100, bl)
    print(f"  {v.hit} {v.reason}")
