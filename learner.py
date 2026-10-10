"""规则自进化。

## 目标定位

**不是省钱**——实测 LLM 判定单次 0.0005 元，600 次调用才 0.3 元，钱不是瓶颈。
规则的价值是**速度**：规则 ~0ms 删除，LLM 要等 15-60 秒。
所以自进化的目标是"让广告在出现的那一秒消失"，而不是压缩 API 开销。

## 为什么 V1 的机制失败了

V1 是「命中 3 次 + 准确率 100% → 永久直拦」，结果 224 条规则毕业，里面
全是 `肾虚`、`火力全开`、`投放` 这类通用词，把规则变成了静默秒删机器。

根因：**准确率门槛会被基率误导**。一个规则命中 100 次里 99 次是广告，
看着 99% 准确，但那 1 次误删是真人，而且撤销不了。

## 本模块的闸门：负样本零命中

> 候选规则必须在**全部「正常消息」语料上零命中**，才允许升级为直拦。

这比准确率门槛硬得多，而且是**可证明安全**的：只要历史上没有任何一条正常
消息命中过它，它就（在已见数据上）不可能误删。我们正好有真实语料——
线上未被删除的消息就是天然的负样本集。

## 候选从**对比**里挖，不是从广告里挖

只从广告文本挖短语，会挖出所有群内高频词（"修车"、"开课"、"武汉"）——
它们在广告里出现多，在正常聊天里出现更多。
所以每个候选都要算**区分度**：广告中出现次数 vs 正常语料中命中次数。

## 三级毕业

| 级别 | 条件 | 路由 |
|---|---|---|
| A 直拦 | 负样本零命中 + 正样本 >= MIN_SPAM_HITS | L0 秒删（0ms，不花 AI） |
| B 送 AI 加权 | 负样本命中率 < 1% | 进 prompt 标高可疑，AI 仍可推翻 |
| C 弱信号 | 其他 | 只进 few-shot 例子 |

B 级是 V1 缺失的一层：让规则影响判定但不做最终决定。

## 新规则先跑 shadow

升级前先在 shadow 模式观察：LLM 判定时并行记录命中情况但**不影响决定**。
观察期满且负样本仍零命中才升 A 级。
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import db
from detector_rules import is_bare_letter, extract_tme_identifiers
from utils.normalize import invisible_ratio, normalize

logger = logging.getLogger(__name__)

# ---- 闸门与门槛 ----

# 负样本必须零命中。这是唯一允许自动毕业的条件。
MAX_HAM_HITS = 0

# 成为候选所需的最少广告命中次数，按类型区分。
# 太低会把一次性内容固化成规则（V1 的 3 次门槛就是这么崩的）。
# 域名/URL 复用率天然比文案高（一个引流域名会被多个号反复贴），门槛可低些。
MIN_SPAM_HITS = 5
MIN_SPAM_HITS_BY_KIND = {
    "signature": 5,    # 同一文案反复出现，复用率最高的信号
    "phrase": 5,
    "domain": 3,       # 一个引流域名会被多个账号反复贴
    "behavior": 5,
}

# shadow 观察期：至少命中这么多次才考虑毕业
MIN_SHADOW_HITS = 10

# 每天最多毕业多少条规则，防止规则爆炸
MAX_PROMOTIONS_PER_DAY = 3

# 语料取材窗口与上限。
#
# 教训（来自 @BayesSpamSniperBot 作者的上线故障复盘）：他的机器人上线后
# 误删严重，根因是训练集里垃圾广告远多于正常消息——分类器学成了"倾向判广告"。
# 我们的闸门依赖正常消息语料，所以正常语料被**静默截断**会导致闸门失效，
# 这是同一个失效模式。
#
# 实测：武汉档案馆群的正常消息已达 1,476 条（总量 2,985），逼近旧上限 3000。
# 因此上限调到 20000（远超实际量级），并在被截断时**明确告警**而不是静默截断。
CORPUS_DAYS = 7
CORPUS_LIMIT = 20000

# 负样本相对正样本的最低倍数。低于它说明正常语料偏少，
# 闸门的可靠性下降，此时跳过本轮学习并告警。
MIN_HAM_PER_SPAM = 0.5

# 候选短语长度范围（归一化后）
MIN_PHRASE_LEN = 4
MAX_PHRASE_LEN = 24

# 通用词黑名单：这些词在正常聊天里几乎必然出现，作为短语候选没有区分度。
# 命中即丢弃——它们正是 V1 把那 224 条规则搞砸的原因。
GENERIC_WORDS = frozenset({
    "武汉", "南京", "苏州", "哈尔滨", "洪山", "江岸", "武昌", "汉阳", "光谷",
    "开课", "修车", "上门", "约课", "全套", "半套", "养生", "外围", "服务",
    "老师", "学生", "学妹", "小妹", "兼职", "项目", "资源", "投资", "收益",
    "一天", "一天赚", "日入", "日赚", "红包", "频道", "群组", "联系", "私聊",
    "有空", "在线", "新人", "少妇", "萝莉", "御姐", "妹妹", "小姐姐",
})

# URL / 域名候选
_DOMAIN_PATTERN = re.compile(r"https?://([a-z0-9.\-]+\.[a-z]{2,})", re.I)

# 谐音归一化后仍保留的"暗语"片段：包含这些字的片段更可能是广告术语
_AD_HINT_CHARS = ("约", "课", "门", "单", "押", "付", "跑", "洗", "资", "米",
                  "包", "夜", "聊", "课", "外", "出", "接")


@dataclass
class Candidate:
    """一个候选规则。"""

    pattern: str
    kind: str                    # signature | domain | phrase | behavior
    spam_hits: int = 0
    ham_hits: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        total = self.spam_hits + self.ham_hits
        return self.spam_hits / total if total else 0.0

    def min_hits(self) -> int:
        return MIN_SPAM_HITS_BY_KIND.get(self.kind, MIN_SPAM_HITS)

    def passes_gate(self) -> bool:
        """负样本零命中闸门 + 达到该类型的最低命中量。

        负样本零命中是自动毕业的**唯一**条件；命中量门槛只是防止把
        一次性内容固化成规则。
        """
        return self.ham_hits <= MAX_HAM_HITS and self.spam_hits >= self.min_hits()


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------

def load_corpus(chat_id: int, days: int = CORPUS_DAYS,
                limit: int = CORPUS_LIMIT) -> tuple[list[str], list[str]]:
    """取本群的正负样本语料。返回 (spam_texts, ham_texts)。

    正样本 = spam_records 里 action='delete' 的消息正文（LLM 或规则判定的）
    负样本 = messages 里**未被删除**的消息

    注意负样本是"我们当前认为正常"的消息，其中可能混有漏网广告——
    这会让闸门偏严（好事），不会让误伤发生。

    ⚠️ 两个必须防的失效模式：
      1. **静默截断**：语料被 LIMIT 切掉一半，闸门就只覆盖了一半的正常消息，
         零命中结论不可靠。这里在触及上限时明确告警。
      2. **正常语料饥饿**：某些群广告占比极高（实测有群 385 删 / 26 留），
         正常样本太少会让闸门的统计意义下降。调用方用 MIN_HAM_PER_SPAM 判断。
    """
    spam_rows = db.query("""
        SELECT text_snapshot FROM spam_records
        WHERE chat_id=? AND action='delete' AND text_snapshot IS NOT NULL
          AND LENGTH(TRIM(text_snapshot)) >= 2
          AND created_at >= datetime('now', ?)
        ORDER BY id DESC LIMIT ?
    """, (chat_id, f"-{days} days", limit))

    ham_rows = db.query("""
        SELECT m.text FROM messages m
        WHERE m.chat_id=? AND m.text IS NOT NULL AND LENGTH(TRIM(m.text)) >= 2
          AND m.timestamp >= datetime('now', ?)
          AND NOT EXISTS (
              SELECT 1 FROM spam_records s
              WHERE s.chat_id = m.chat_id AND s.message_id = m.message_id
                AND s.action = 'delete')
        ORDER BY m.id DESC LIMIT ?
    """, (chat_id, f"-{days} days", limit))

    if len(spam_rows) >= limit or len(ham_rows) >= limit:
        logger.warning(
            "语料触及上限 chat=%s spam=%d ham=%d limit=%d —— "
            "闸门只覆盖了部分正常消息，结论可能不可靠",
            chat_id, len(spam_rows), len(ham_rows), limit)

    return ([r["text_snapshot"] for r in spam_rows],
            [r["text"] for r in ham_rows])


# ---------------------------------------------------------------------------
# 候选挖掘：四类
# ---------------------------------------------------------------------------

def mine_signatures(spam: list[str], ham: list[str]) -> list[Candidate]:
    """精确签名：同一条内容被反复发送。

    实测案例：`来洗mi 挣一万` 被同一个号连发多次；`充电宝投放，日入过w的新项目`
    被不同号复用。这类内容复用率极高，且正常群友不会把同一句话发 5 遍。

    用归一化后的完整文本做 key，因此谐音/空格变体也会归到一起。
    """
    spam_keys = Counter(normalize(t) for t in spam if t)
    ham_keys = {normalize(t) for t in ham if t}
    out = []
    for key, n in spam_keys.items():
        if len(key) < MIN_PHRASE_LEN:
            continue
        cand = Candidate(pattern=key, kind="signature", spam_hits=n,
                         ham_hits=1 if key in ham_keys else 0)
        if cand.spam_hits >= MIN_SPAM_HITS:
            out.append(cand)
    return out


def mine_domains(spam: list[str], ham: list[str]) -> list[Candidate]:
    """域名 / Telegram 目标：指向固定引流目的地。

    零误伤性最强的候选之一——正常群友分享的链接（自家榜单、视频站）
    与广告的引流域名天然分离。
    """
    spam_dom = Counter()
    ham_dom = set()
    for t in spam:
        for d in _DOMAIN_PATTERN.findall(t or ""):
            # t.me 交给 extract_tme_identifiers 处理成 t.me/<用户名>，
            # 否则会产生裸 't.me' 候选——它在正常语料里必然命中，白占闸门。
            if d.lower() not in ("t.me", "telegram.me"):
                spam_dom[d.lower()] += 1
        for u in extract_tme_identifiers(t or ""):
            spam_dom[f"t.me/{u}"] += 1
    for t in ham:
        for d in _DOMAIN_PATTERN.findall(t or ""):
            if d.lower() not in ("t.me", "telegram.me"):
                ham_dom.add(d.lower())
        for u in extract_tme_identifiers(t or ""):
            ham_dom.add(f"t.me/{u}")

    out = []
    for dom, n in spam_dom.items():
        cand = Candidate(pattern=dom, kind="domain", spam_hits=n,
                         ham_hits=1 if dom in ham_dom else 0)
        if cand.spam_hits >= 3:      # 域名复用比短语更常见，门槛可低些
            out.append(cand)
    return out


def mine_phrases(spam: list[str], ham: list[str]) -> list[Candidate]:
    """谐音暗语短语：归一化后的高频片段，且必须在正常语料零命中。

    这是四类里最容易出错的一类，所以：
      · 归一化后再切短语（谐音变体自动归并）
      · 排除通用词（GENERIC_WORDS）——它们正是 V1 搞砸的原因
      · 要求片段含"暗语特征字"，降低挖出普通词组的机会
      · 负样本零命中由闸门统一把关
    """
    def grams(text: str, lo: int, hi: int):
        norm = normalize(text)
        for n in range(lo, min(hi, len(norm)) + 1):
            for i in range(len(norm) - n + 1):
                yield norm[i:i + n]

    ham_gram_set = set()
    for t in ham:
        ham_gram_set.update(grams(t, MIN_PHRASE_LEN, MAX_PHRASE_LEN))

    spam_grams = Counter()
    for t in spam:
        # 同一文本内的重复片段只算一次，避免单条长文刷高计数
        spam_grams.update(set(grams(t, MIN_PHRASE_LEN, MAX_PHRASE_LEN)))

    survived = []
    for gram, n in spam_grams.items():
        if n < MIN_SPAM_HITS:
            continue
        if any(w in gram for w in GENERIC_WORDS):
            continue
        if not any(c in gram for c in _AD_HINT_CHARS):
            continue
        survived.append((gram, n))

    # 子串剪枝：`洗m1挣`、`来洗m1挣`、`来洗m1挣一` 都是同一条文案的子串，
    # 全部保留会让规则数爆炸且毫无额外覆盖。只留极大片段——
    # 即不被任何**命中次数不少于自己**的更长片段包含。
    #
    # 注意不能跨命中数剪枝：命中 10 次的短片段可能比命中 5 次的长片段更有价值。
    survived.sort(key=len, reverse=True)
    out = []
    for gram, n in survived:
        if any(gram in longer and m >= n for longer, m in survived
               if longer != gram and len(longer) > len(gram)):
            continue
        out.append(Candidate(
            pattern=gram, kind="phrase", spam_hits=n,
            ham_hits=1 if gram in ham_gram_set else 0))
    return out


def mine_behaviors(spam: list[str], ham: list[str]) -> list[Candidate]:
    """行为类特征。这些是"消息形态"而非内容，广告号换文案也绕不过。

    目前只产出可命中的确定性形态：
      · 不可见字符占比高（客户端看不见，人眼无从察觉）
      · 裸字母（正常群友的短回应是中文，不是裸 ASCII 字母）
    """
    out = []
    inv_spam = sum(1 for t in spam
                   if len(t) >= 10 and invisible_ratio(t) >= 0.15)
    inv_ham = sum(1 for t in ham
                  if len(t) >= 10 and invisible_ratio(t) >= 0.15)
    if inv_spam >= MIN_SPAM_HITS:
        out.append(Candidate(pattern="__invisible_ratio__", kind="behavior",
                             spam_hits=inv_spam, ham_hits=inv_ham))

    bare_spam = sum(1 for t in spam if is_bare_letter(t))
    bare_ham = sum(1 for t in ham if is_bare_letter(t))
    if bare_spam >= MIN_SPAM_HITS:
        out.append(Candidate(pattern="__bare_letter__", kind="behavior",
                             spam_hits=bare_spam, ham_hits=bare_ham))
    return out


def mine_all(spam: list[str], ham: list[str]) -> list[Candidate]:
    """四类候选全跑，按价值排序（零误伤优先、命中量次之）。"""
    cands = (mine_signatures(spam, ham)
             + mine_domains(spam, ham)
             + mine_behaviors(spam, ham)
             + mine_phrases(spam, ham))
    # 去重：同一 pattern 只保留命中量最大的一条
    best: dict[str, Candidate] = {}
    for c in cands:
        cur = best.get(c.pattern)
        if cur is None or c.spam_hits > cur.spam_hits:
            best[c.pattern] = c

    # 跨类型子串剪枝。
    # 各类型内部的剪枝不够：`无押小额口苓红包洗资` 会作为 signature 存活，
    # 而 phrase 类里它的子串 `无押小额口苓红` 因为"有更长且命中不少于自己的
    # 片段"被剪掉——两边一合并，短的反而活了下来，长的也在，形成冗余。
    # 这里跨类型再剪一次：短片段若被某条命中量不少于自己的更长片段包含，
    # 就丢弃。域名类不参与（t.me/wdgx 是独立标识，不该被长文本包含）。
    items = sorted(best.values(),
                   key=lambda c: (len(c.pattern), -c.spam_hits), reverse=True)
    kept: list[Candidate] = []
    for c in items:
        if c.kind != "domain":
            subsumed = any(
                k.kind != "domain"
                and len(k.pattern) > len(c.pattern)
                and c.pattern in k.pattern
                and k.spam_hits >= c.spam_hits
                for k in items
            )
            if subsumed:
                continue
        kept.append(c)

    return sorted(kept, key=lambda c: (c.ham_hits, -c.spam_hits))


# ---------------------------------------------------------------------------
# 生命周期：shadow -> active
# ---------------------------------------------------------------------------

def refresh_candidates(chat_id: int) -> dict:
    """挖一轮候选并写入 shadow 模式。

    已存在同 pattern 的不重复写（spam_patterns 有 UNIQUE 约束）。
    """
    spam, ham = load_corpus(chat_id)
    if not spam or not ham:
        return {"skipped": True, "reason": "语料不足",
                "spam": len(spam), "ham": len(ham)}

    # 正常语料饥饿保护。
    # 借鉴 @BayesSpamSniperBot 上线故障的教训：他的训练集里广告远多于正常消息，
    # 分类器学成"倾向判广告"，上线后大规模误删。
    # 我们的闸门完全依赖正常语料——正常样本太少时"零命中"就失去了统计意义，
    # 所以宁可不学，也不能在证据不足时放权。
    ratio = len(ham) / max(1, len(spam))
    if ratio < MIN_HAM_PER_SPAM:
        logger.warning(
            "正常语料饥饿 chat=%s spam=%d ham=%d 比值=%.2f < %.2f —— "
            "闸门统计意义不足，本轮跳过学习",
            chat_id, len(spam), len(ham), ratio, MIN_HAM_PER_SPAM)
        return {"skipped": True, "reason": "正常语料不足（闸门不可靠）",
                "spam": len(spam), "ham": len(ham), "ratio": round(ratio, 2)}

    cands = mine_all(spam, ham)
    passed = [c for c in cands if c.passes_gate()]

    added = 0
    for c in passed:
        try:
            db.add_pattern(chat_id, c.pattern, mode="shadow")
            added += 1
        except Exception:  # noqa: BLE001
            pass

    db.execute("""INSERT INTO chat_settings (chat_id, key, value)
                  VALUES (?,?,?) ON CONFLICT(chat_id,key)
                  DO UPDATE SET value=excluded.value""",
               (chat_id, "learn_last_run", str(int(time.time()))))

    logger.info("自进化 chat=%s 语料 spam=%d ham=%d 候选=%d 过闸=%d 新增=%d",
                chat_id, len(spam), len(ham), len(cands), len(passed), added)
    return {"candidates": len(cands), "passed": len(passed), "added": added,
            "spam": len(spam), "ham": len(ham)}


def eligible_for_promotion(chat_id: int) -> list:
    """找出可以毕业（shadow -> active）的规则。

    条件（全部满足）：
      · shadow 命中次数 >= MIN_SHADOW_HITS
      · **shadow 期间未被任何正常消息命中**（shadow_spam_hits == shadow_hits）
    第二条是闸门在运行期的延续——观察期内只要有一条正常消息命中，
    说明该规则会误伤，不予毕业。
    """
    rows = db.query("""
        SELECT * FROM spam_patterns
        WHERE mode='shadow' AND (chat_id=? OR chat_id IS NULL)
          AND shadow_hits >= ?
          AND shadow_hits = shadow_spam_hits
        ORDER BY shadow_hits DESC
    """, (chat_id, MIN_SHADOW_HITS))
    return list(rows)


def run_lifecycle(chat_id: int, promote_limit: int = MAX_PROMOTIONS_PER_DAY) -> dict:
    """一轮完整生命周期：挖候选 -> 毕业达标规则。"""
    stats = refresh_candidates(chat_id)

    promoted = []
    for row in eligible_for_promotion(chat_id)[:promote_limit]:
        db.promote_pattern(row["id"])
        promoted.append(row["pattern"])
        logger.warning("规则毕业(直拦) chat=%s pattern=%r shadow命中=%d",
                       chat_id, row["pattern"], row["shadow_hits"])
    stats["promoted"] = promoted
    return stats


def validation_report(chat_id: int, limit: int = 20) -> dict:
    """给管理员看的验证报告：哪些候选可以毕业、哪些被闸门拦住。"""
    spam, ham = load_corpus(chat_id)
    cands = mine_all(spam, ham)
    passed = [c for c in cands if c.passes_gate()]
    blocked = [c for c in cands
               if not c.passes_gate() and c.ham_hits > 0]
    return {
        "spam": len(spam), "ham": len(ham),
        "total": len(cands), "passed": passed[:limit],
        "blocked": blocked[:limit],
        "promotable": eligible_for_promotion(chat_id),
    }
