"""从生产库导出评测样本。

只读，不修改任何数据。

样本来源：
- 正例：spam_records 中被删除的消息，join messages 取回正文。
  注意 V1 的实时层在入库**之前**就删除了消息，因此这类消息在 messages 表里查不到，
  只有走批量路径删除的才有正文。这正是 V1 缺少 text_snapshot 字段的后果。
- 负例：messages 表中未被删除的消息（近似认为是正常聊天）。
  这是弱标签——V1 漏删的广告也会混进来。因此负例的准确率需要人工抽检。

输出 JSONL，每行 {"text", "label", "source", "msg_id", "chat_id", "ts"}。
"""

import argparse
import json
import sqlite3
import sys

DB_DEFAULT = "/root/ribaobot/jianghu_daily.db"


def export(db_path: str, days: int, limit: int, out_path: str) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c = conn.cursor()

    stats = {}

    # --- 正例：被删除且能取回正文的消息 ---
    positives = c.execute(
        """
        SELECT m.text, s.created_at, s.message_id, s.chat_id, s.reason
        FROM spam_records s
        JOIN messages m
          ON m.message_id = s.message_id AND m.chat_id = s.chat_id
        WHERE s.action = 'delete'
          AND s.created_at >= datetime('now', ?)
          AND m.text IS NOT NULL AND LENGTH(TRIM(m.text)) >= 2
        ORDER BY s.created_at DESC
        LIMIT ?
        """,
        (f"-{days} days", limit),
    ).fetchall()
    stats["positives_available"] = len(positives)

    # --- 负例：未被删除的消息 ---
    negatives = c.execute(
        """
        SELECT text, timestamp, message_id, chat_id, ''
        FROM messages
        WHERE timestamp >= datetime('now', ?)
          AND text IS NOT NULL AND LENGTH(TRIM(text)) >= 2
          AND message_id NOT IN (
              SELECT message_id FROM spam_records WHERE action = 'delete'
          )
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (f"-{days} days", limit),
    ).fetchall()
    stats["negatives_available"] = len(negatives)

    # --- 正面移除记录（管理员撤销/解禁），是高质量反例 ---
    revived = c.execute(
        """
        SELECT m.text, s.created_at, s.message_id, s.chat_id, s.reason
        FROM spam_records s
        JOIN messages m
          ON m.message_id = s.message_id AND m.chat_id = s.chat_id
        WHERE s.action IN ('unmute', 'mute_ignore')
          AND m.text IS NOT NULL
        LIMIT 100
        """,
    ).fetchall()
    stats["revived"] = len(revived)

    rows = []
    for text, ts, mid, cid, reason in positives:
        rows.append({
            "text": text, "label": "spam", "source": "deleted",
            "msg_id": mid, "chat_id": cid, "ts": str(ts)[:19], "reason": reason,
        })
    for text, ts, mid, cid, _ in negatives:
        rows.append({
            "text": text, "label": "ham", "source": "not_deleted",
            "msg_id": mid, "chat_id": cid, "ts": str(ts)[:19],
        })
    for text, ts, mid, cid, reason in revived:
        rows.append({
            "text": text, "label": "ham", "source": "admin_revived",
            "msg_id": mid, "chat_id": cid, "ts": str(ts)[:19], "reason": reason,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats["written"] = len(rows)
    stats["spam"] = sum(1 for r in rows if r["label"] == "spam")
    stats["ham"] = sum(1 for r in rows if r["label"] == "ham")
    conn.close()
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--limit", type=int, default=400, help="每类上限")
    ap.add_argument("--out", default="eval_samples.jsonl")
    args = ap.parse_args()

    if args.db.startswith("/root/") and not __import__("os").path.exists(args.db):
        print(f"数据库不存在: {args.db}", file=sys.stderr)
        sys.exit(1)

    st = export(args.db, args.days, args.limit, args.out)
    print(f"导出到 {args.out}")
    print(f"  正例(广告)   : {st['spam']} 条（库中可用 {st['positives_available']}）")
    print(f"  负例(正常)   : {st['ham']} 条（库中可用 {st['negatives_available']}）")
    print(f"  其中管理员撤销: {st['revived']} 条（高质量反例）")
    print(f"  合计         : {st['written']} 条")
