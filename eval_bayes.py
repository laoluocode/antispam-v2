"""离线实验：朴素贝叶斯分类器在我们自己的语料上表现如何。

动机：@BayesSpamSniperBot 的作者用朴素贝叶斯做 TG 反广告，思路来自
保罗·格雷厄姆 2002 年的《A Plan for Spam》。文章核心论点是
"关键字匹配容易被绕过，而贝叶斯能从整体用词分布判断"。

我们已经在用 LLM 做语义判断 + 自进化规则做确定性拦截。
这里要回答的问题是：**再补一个本地贝叶斯分类器，有没有增量价值？**

它潜在的价值有两块：
  1. 作为信号进 prompt（提升 LLM 判断力）
  2. 高置信时直接定案（省掉一次 LLM 调用）
两者都要求它在我们的语料上有真实区分度。

实验设计：从线上库导出语料，做词袋 + 拉普拉斯平滑的朴素贝叶斯，
用留出法（时间切分）评估，并打印最有区分度的词——那些词最能说明
"它是学到了广告特征，还是学到了本群方言"。
"""

import asyncio
import math
import os
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, "/root/ribaobot/antispambot")

from utils.normalize import normalize  # noqa: E402

DB = "/app/data/antispam.db"
CHAT = int(os.environ.get("CHAT_ID", "-1003874534750"))


def load_corpus(days: int = 7):
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    spam = [r["text_snapshot"] for r in c.execute("""
        SELECT text_snapshot FROM spam_records
        WHERE chat_id=? AND action='delete' AND text_snapshot IS NOT NULL
          AND LENGTH(TRIM(text_snapshot)) >= 2
          AND created_at >= datetime('now', ?)
        ORDER BY id ASC
    """, (CHAT, f"-{days} days"))]
    ham = [r["text"] for r in c.execute("""
        SELECT m.text FROM messages m
        WHERE m.chat_id=? AND m.text IS NOT NULL AND LENGTH(TRIM(m.text)) >= 2
          AND m.timestamp >= datetime('now', ?)
          AND NOT EXISTS (SELECT 1 FROM spam_records s
                          WHERE s.chat_id=m.chat_id AND s.message_id=m.message_id
                            AND s.action='delete')
        ORDER BY m.id ASC
    """, (CHAT, f"-{days} days"))]
    c.close()
    return spam, ham


# 中文字符级 2-gram + 英文/数字词。中文没有空格，按字符 n-gram 切分是
# 文本分类的常规做法（也正是归一化后能吃掉谐音变体的原因）。
def tokenize(text: str) -> list[str]:
    norm = normalize(text)
    tokens = list(norm)                       # 单字
    tokens += [norm[i:i + 2] for i in range(len(norm) - 1)]   # 2-gram
    return tokens


class NaiveBayes:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.spam_counts = Counter()
        self.ham_counts = Counter()
        self.spam_total = 0
        self.ham_total = 0
        self.n_spam = 0
        self.n_ham = 0
        self.vocab = set()

    def train(self, spam: list[str], ham: list[str]):
        for t in spam:
            toks = tokenize(t)
            self.spam_counts.update(toks)
            self.spam_total += len(toks)
            self.n_spam += 1
        for t in ham:
            toks = tokenize(t)
            self.ham_counts.update(toks)
            self.ham_total += len(toks)
            self.n_ham += 1
        self.vocab = set(self.spam_counts) | set(self.ham_counts)

    def log_ratio(self, text: str) -> float:
        """返回 log(P(spam|text)/P(ham|text))，正数偏向广告。"""
        toks = tokenize(text)
        if not toks:
            return 0.0
        prior = math.log((self.n_spam + 1) / (self.n_ham + 1))
        v = len(self.vocab) + 1
        s = prior
        for tok in toks:
            ps = (self.spam_counts.get(tok, 0) + self.alpha) / (self.spam_total + self.alpha * v)
            ph = (self.ham_counts.get(tok, 0) + self.alpha) / (self.ham_total + self.alpha * v)
            s += math.log(ps / ph)
        return s / len(toks)                 # 按长度归一，避免长文本得分虚高

    def top_discriminative(self, n: int = 25):
        """最有区分度的 token——用来判断它学到的到底是广告特征还是群方言。"""
        v = len(self.vocab) + 1
        scored = []
        for tok in self.vocab:
            if len(tok) < 2:
                continue
            ps = (self.spam_counts.get(tok, 0) + self.alpha) / (self.spam_total + self.alpha * v)
            ph = (self.ham_counts.get(tok, 0) + self.alpha) / (self.ham_total + self.alpha * v)
            scored.append((math.log(ps / ph), tok,
                           self.spam_counts.get(tok, 0), self.ham_counts.get(tok, 0)))
        scored.sort(reverse=True)
        return scored[:n], scored[-n:]


def evaluate(nb: NaiveBayes, spam: list[str], ham: list[str], thr: float):
    tp = sum(1 for t in spam if nb.log_ratio(t) >= thr)
    fp = sum(1 for t in ham if nb.log_ratio(t) >= thr)
    fn = len(spam) - tp
    tn = len(ham) - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


async def main():
    spam, ham = load_corpus(7)
    print("=" * 78)
    print("  语料（群 %s，近 7 天）" % CHAT)
    print("=" * 78)
    print("  spam = %d 条，ham = %d 条" % (len(spam), len(ham)))
    if len(spam) < 20 or len(ham) < 20:
        print("  语料不足，无法评估")
        return

    # 时间切分：前 70% 训练，后 30% 测试。
    # 不能用随机切分——同一波广告的重复文案会同时落进训练和测试集，
    # 那样准确率会虚高（数据泄漏）。
    s_cut, h_cut = int(len(spam) * 0.7), int(len(ham) * 0.7)
    tr_s, te_s = spam[:s_cut], spam[s_cut:]
    tr_h, te_h = ham[:h_cut], ham[h_cut:]
    print("  时间切分: 训练 spam=%d ham=%d / 测试 spam=%d ham=%d"
          % (len(tr_s), len(tr_h), len(te_s), len(te_h)))

    nb = NaiveBayes()
    nb.train(tr_s, tr_h)

    print()
    print("=" * 78)
    print("  阈值扫描（测试集）")
    print("=" * 78)
    print("  %8s %8s %8s %8s %8s" % ("阈值", "精确率", "召回率", "F1", "误伤/漏"))
    best = None
    for thr in (-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3):
        m = evaluate(nb, te_s, te_h, thr)
        print("  %8.2f %7.1f%% %7.1f%% %8.3f  fp=%d fn=%d"
              % (thr, m["precision"] * 100, m["recall"] * 100, m["f1"],
                 m["fp"], m["fn"]))
        if best is None or m["f1"] > best[1]["f1"]:
            best = (thr, m)
    print("  最佳阈值 %.2f -> F1=%.3f" % (best[0], best[1]["f1"]))

    print()
    print("=" * 78)
    print("  最有区分度的 token（判断它学到的是广告特征还是群方言）")
    print("=" * 78)
    top, bottom = nb.top_discriminative(20)
    print("  --- 最偏广告 ---")
    for score, tok, sc, hc in top:
        print("    %+7.2f  %-8r spam=%-4d ham=%-4d" % (score, tok, sc, hc))
    print("  --- 最偏正常 ---")
    for score, tok, sc, hc in bottom:
        print("    %+7.2f  %-8r spam=%-4d ham=%-4d" % (score, tok, sc, hc))

    print()
    print("=" * 78)
    print("  与现有方案的对比")
    print("=" * 78)
    print("  当前线上（南京苏州集团近 6 小时）: 规则承担率 46%，其余走 LLM")
    print("  贝叶斯最佳 F1 = %.3f" % best[1]["f1"])
    print("  （LLM 的历史离线评估: 阈值 0.50 时 P=89.3%% R=95.5%% F1=0.923）")


asyncio.run(main())
