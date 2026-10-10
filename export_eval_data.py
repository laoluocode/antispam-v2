"""从生产库导出评测样本。

⚠️ **安全约定：绝不直接查询生产库。**

2026-10-10 的教训：直接在生产库上跑 `GROUP BY user_id` + 关联子查询，
在 `journal_mode=DELETE`（非 WAL）下长时间持有读事务，导致正在运行的 bot
出现 `database is locked` 写入失败。事后 `PRAGMA quick_check` 确认数据完整，
但那是运气，不是设计。

因此本脚本先用 SQLite backup API 把库**复制到临时文件**，随后所有重查询
都打在副本上。backup API 对源库只持有极短的读锁。

样本必须**自带用户的信任信号**，否则评测无法验证"老用户保护"是否起作用。
V1 的评测（model_eval_log）没有这个信息，所以它测不出误伤。

信任信号的三个分量：

1. **删除比** `del_n / msg_n` —— 主信号。
   实测分布两极分化：21.4% 的用户从未被删，24.3% 的用户删除比 >90%。
   而"发言量"完全无区分度：发言 1000+ 的用户里广告号占比 2.4%，
   发言 <10 的用户里是 1.7%，几乎一样。

2. **发言量** `msg_n` —— 次要信号，仅作辅助（<5 条视为新人）。

3. **活跃跨度** `span_days` —— 刷量的号往往跨度短而密集。

**分层采样**：按发言量分 5 档各取等量样本。否则 88% 的用户都是 <10 条的，
样本会全部集中在低活跃用户上，测不出信任信号的价值。
"""

import argparse
import json
import os
import random
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_rules import compute_trust

DB_DEFAULT = "/root/ribaobot/jianghu_daily.db"

# 发言量分层（对应实测分布：<10 占 88.2%，100+ 仅 1.6%）
STRATA = [
    ("inactive", 0, 10),
    ("low", 10, 30),
    ("mid", 30, 100),
    ("active", 100, 1000),
    ("veteran", 1000, None),
]


def _in_stratum(msg_n: int, lo: int, hi) -> bool:
    return msg_n >= lo and (hi is None or msg_n < hi)


def _stratum_of(msg_n: int) -> str:
    for name, lo, hi in STRATA:
        if _in_stratum(msg_n, lo, hi):
            return name
    return "unknown"


def snapshot(db_path: str) -> str:
    """用 SQLite backup API 复制数据库到临时文件。

    对源库只持有极短的读锁，且不产生长时间运行的读事务——
    这正是直接查询生产库时踩的坑。
    """
    fd, tmp = tempfile.mkstemp(prefix="antispam_eval_", suffix=".db")
    os.close(fd)
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    dst = sqlite3.connect(tmp)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return tmp


def _user_stats(conn: sqlite3.Connection) -> dict:
    """一次性算出所有用户的信任信号。只在副本上执行。

    活跃跨度用 SQL 里的 julianday 直接算 —— 早先的实现在 Python 里
    对每个用户单独发一次 julianday 查询，12,470 个用户就是 12,470 次往返，
    在单核机器上要跑好几分钟。
    """
    stats = {}
    rows = conn.execute("""
        SELECT m.user_id,
               COUNT(*)                                      AS msg_n,
               julianday(MAX(m.timestamp)) - julianday(MIN(m.timestamp)) AS span_days,
               (SELECT COUNT(DISTINCT s.message_id) FROM spam_records s
                 WHERE s.user_id = m.user_id AND s.action = 'delete')     AS del_n
        FROM messages m
        GROUP BY m.user_id
    """).fetchall()
    for uid, msg_n, span, del_n in rows:
        span = round(span or 0.0, 1)
        stats[uid] = {
            "msg_n": msg_n,
            "del_n": del_n or 0,
            "span_days": span,
            "trust": compute_trust(msg_n, del_n or 0, span),
        }
    return stats


def export(db_path: str, days: int, per_stratum: int, out_path: str,
           users_per_stratum: int = 120, keep_snapshot: bool = False) -> dict:
    print("  正在复制数据库副本（避免占用生产库）…")
    snap = snapshot(db_path)
    try:
        return _export_from(snap, days, per_stratum, out_path, users_per_stratum)
    finally:
        if keep_snapshot:
            print(f"  副本保留于 {snap}")
        else:
            try:
                os.unlink(snap)
            except OSError:
                pass


def _export_from(db_path: str, days: int, per_stratum: int, out_path: str,
                 users_per_stratum: int) -> dict:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    stats = {"per_stratum": {}, "strata_missing": []}

    user_stats = _user_stats(conn)

    # 把"已删除消息"集合读进内存做过滤。
    # 实测（副本上，21 万条消息 / 13.8 万条删除记录）：
    #   spam 用 SQL EXISTS 子查询      24.62s   <- 相关子查询逐行探测，极慢
    #   ham  用 SQL NOT EXISTS        4.05s
    #   读进内存 + Python 集合过滤      0.11s   <- 快 224 倍
    # 集合只有 13.8 万项，内存代价可忽略。
    deleted = set(c.execute(
        "SELECT chat_id, message_id FROM spam_records WHERE action = 'delete'"
    ).fetchall())
    stats["deleted_set_size"] = len(deleted)

    # 按档位分组用户并对人多的档位抽样 —— 限制 IN 列表长度，
    # 否则 SQLite 会退化成全表扫描（messages 表没有 user_id 索引）。
    strata_ids = {}
    for name, lo, hi in STRATA:
        ids = [uid for uid, us in user_stats.items() if _in_stratum(us["msg_n"], lo, hi)]
        if len(ids) > users_per_stratum:
            ids = random.Random(42).sample(ids, users_per_stratum)
        strata_ids[name] = ids
        stats["per_stratum"][f"{name}/users"] = len(ids)

    # 一次取出所有候选消息（不含删除判断），再在内存里分堆。
    # 多取一些以便分堆后两类都够数。
    want = per_stratum * 4

    rows = []
    for name, _lo, _hi in STRATA:
        ids = strata_ids[name]
        if not ids:
            stats["per_stratum"][f"{name}/spam"] = 0
            stats["per_stratum"][f"{name}/ham"] = 0
            stats["strata_missing"].append(name)
            continue
        ph = ",".join("?" * len(ids))
        got = c.execute(f"""
            SELECT m.text, m.message_id, m.chat_id, m.user_id, m.timestamp
            FROM messages m
            WHERE m.timestamp >= datetime('now', ?)
              AND m.text IS NOT NULL AND LENGTH(TRIM(m.text)) >= 2
              AND m.user_id IN ({ph})
            ORDER BY m.timestamp DESC
            LIMIT ?
        """, (f"-{days} days", *ids, want)).fetchall()

        spam_n = ham_n = 0
        for text, mid, cid, uid, ts in got:
            is_spam = (cid, mid) in deleted
            if is_spam and spam_n >= per_stratum:
                continue
            if not is_spam and ham_n >= per_stratum:
                continue
            if is_spam:
                spam_n += 1
            else:
                ham_n += 1
            rows.append((text, mid, cid, uid, ts, "spam" if is_spam else "ham"))

        stats["per_stratum"][f"{name}/spam"] = spam_n
        stats["per_stratum"][f"{name}/ham"] = ham_n
        if spam_n == 0 or ham_n == 0:
            stats["strata_missing"].append(
                f"{name}(spam={spam_n},ham={ham_n})"
            )

    seen, out = set(), []
    for text, mid, cid, uid, ts, label in rows:
        key = (cid, mid)
        if key in seen:
            continue
        seen.add(key)
        us = user_stats.get(uid, {})
        out.append({
            "text": text,
            "label": label,
            "source": "deleted" if label == "spam" else "not_deleted",
            "msg_id": mid,
            "chat_id": cid,
            "user_id": uid,
            "ts": str(ts)[:19],
            "stratum": _stratum_of(us.get("msg_n", 0)),
            "msg_n": us.get("msg_n", 0),
            "del_n": us.get("del_n", 0),
            "span_days": us.get("span_days", 0.0),
            "trust": us.get("trust", "normal"),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats["written"] = len(out)
    stats["spam"] = sum(1 for r in out if r["label"] == "spam")
    stats["ham"] = sum(1 for r in out if r["label"] == "ham")
    stats["trust_dist"] = {}
    for r in out:
        stats["trust_dist"][r["trust"]] = stats["trust_dist"].get(r["trust"], 0) + 1
    conn.close()
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--days", type=int, default=60,
                    help="回溯天数。默认 60 天以覆盖高活跃用户")
    ap.add_argument("--per-stratum", type=int, default=100, help="每档每类的样本数")
    ap.add_argument("--users-per-stratum", type=int, default=120,
                    help="每档最多参与抽样的用户数。限制 IN 列表长度，"
                         "否则 SQLite 会退化成全表扫描（messages 表无 user_id 索引）")
    ap.add_argument("--out", default="eval_samples.jsonl")
    ap.add_argument("--keep-snapshot", action="store_true",
                    help="保留数据库副本（调试用）")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"数据库不存在: {args.db}", file=sys.stderr)
        sys.exit(1)

    st = export(args.db, args.days, args.per_stratum, args.out,
                args.users_per_stratum, args.keep_snapshot)
    print(f"导出到 {args.out}")
    print(f"  合计 {st['written']} 条（广告 {st['spam']} / 正常 {st['ham']}）")
    print()
    print("  分层采样结果:")
    for k in sorted(st["per_stratum"]):
        flag = "  <- 无样本" if st["per_stratum"][k] == 0 else ""
        print(f"    {k:18s} {st['per_stratum'][k]:5d}{flag}")
    if st["strata_missing"]:
        print(f"  缺失档位: {st['strata_missing']}")
    print()
    print("  信任档位分布:")
    for k, v in sorted(st["trust_dist"].items(), key=lambda x: -x[1]):
        print(f"    {k:10s} {v:5d} 条 ({v/st['written']:.1%})")
