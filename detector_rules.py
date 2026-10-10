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
from typing import NamedTuple

from utils.normalize import invisible_ratio, normalize

# 中国大陆手机号：正常聊天不会出现（群里没人报手机号）
PHONE_PATTERN = re.compile(r"(?:\+?86[-\s]?)?1[3-9]\d{9}")

# Telegram 引流链接
TELEGRAM_LINK_PATTERN = re.compile(
    r"(?:t\.me/|telegram\.me/|@[A-Za-z][A-Za-z0-9_]{4,31}bot\b)",
    re.IGNORECASE,
)

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

    # 3. 手机号
    if PHONE_PATTERN.search(t):
        return Verdict(True, "包含手机号")

    # 4. Telegram 引流链接
    if TELEGRAM_LINK_PATTERN.search(t):
        return Verdict(True, "包含 Telegram 引流链接")

    # 5. 银行卡号（先排除已被手机号规则覆盖的情况）
    if not PHONE_PATTERN.search(t) and BANK_CARD_PATTERN.search(t):
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
