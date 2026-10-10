"""影子期判定分析。

影子模式下新 bot 只记录不删除，所以这段时间的价值全在于**把判定差异变成
可复核的规则更新**。本脚本把两个 bot 的判定逐条对照，按价值排序输出。

为什么需要它：few-shot 样本的正常写入路径是"举报确认"和"误删恢复"，
而影子期不删消息就点不了误删恢复——若不主动分析，跑一周也学不到东西。
自动学习又会拿模型自己的输出训练自己、放大错误，所以这里的定位是
**产出人工可复核的候选清单**，由人确认后才写进 few-shot。

用法：
    python analyze_shadow.py --days 3
    python analyze_shadow.py --days 7 --export-csv 分歧.csv
    python analyze_shadow.py --add-positive "文本"     # 人工确认后写入
    python analyze_shadow.py --add-negative "文本"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db

NEW_DB_DEFAULT = "data/antispam.db"
OLD_DB_DEFAULT = "/root/ribaobot/jianghu_daily.db"


def _attach_old(conn: sqlite3.Connection, path: str) -> bool:
    """挂载老库（只读）。路径不存在时降级为"只报告新 bot 判定"。"""
    if not path or not os.path.exists(path):
        return False
    try:
        conn.execute("ATTACH DATABASE ? AS old", (f"file:{path}?mode=ro",))
        return True
    except sqlite3.Error as exc:
        print(f"  ⚠️ 无法挂载老库 {path}: {exc}", file=sys.stderr)
        return False


def collect(conn: sqlite3.Connection, days: int, has_old: bool) -> dict:
    """把新 bot 的判定与老 bot 的删除决定按 (chat_id, message_id) 对齐。"""
    cutoff = f"-{days} days"

    new_rows = conn.execute("""
        SELECT chat_id, message_id, user_id, full_name, text_snapshot,
               confidence, action, created_at
        FROM spam_records
        WHERE created_at >= datetime('now', ?)
        ORDER BY created_at DESC
    """, (cutoff,)).fetchall()

    if not has_old:
        return {"new": new_rows, "old": {}, "has_old": False}

    old_rows = conn.execute("""
        SELECT chat_id, message_id, reason, created_at
        FROM old.spam_records
        WHERE action = 'delete' AND created_at >= datetime('now', ?)
    """, (cutoff,)).fetchall()
    old = {(r[0], r[1]): {"reason": r[2], "at": r[3]} for r in old_rows}
    return {"new": new_rows, "old": old, "has_old": True}


def analyze(data: dict, cut: float) -> dict:
    """分四类对照。判定用置信度阈值，与新 bot 实际使用的一致。"""
    new_rows, old = data["new"], data["old"]

    new_verdict = {}
    for r in new_rows:
        cid, mid, uid, name, text, conf, action, at = r
        # 同一 (chat, msg) 可能同时有 observe 和 delete 记录，取置信度最高的
        prev = new_verdict.get((cid, mid))
        c = conf if conf is not None else 0.0
        if prev is None or c > prev["conf"]:
            new_verdict[(cid, mid)] = {
                "user_id": uid, "name": name, "text": text or "",
                "conf": c, "action": action, "at": at,
            }

    both, only_new, only_old = [], [], []
    for key, v in new_verdict.items():
        is_spam = v["conf"] >= cut
        in_old = key in old
        if is_spam and in_old:
            both.append({**v, "chat_id": key[0], "message_id": key[1],
                         "old_reason": old[key]["reason"]})
        elif is_spam and not in_old:
            only_new.append({**v, "chat_id": key[0], "message_id": key[1]})
        elif not is_spam and in_old:
            only_old.append({**v, "chat_id": key[0], "message_id": key[1],
                             "old_reason": old[key]["reason"]})

    # 老 bot 删了、新 bot 连记录都没有（没被送判定）= 潜在漏判
    unseen_by_new = [k for k in old if k not in new_verdict]

    return {
        "both": both, "only_new": only_new, "only_old": only_old,
        "unseen_by_new": unseen_by_new,
        "has_old": data["has_old"],
        "new_total": len(new_verdict),
        "old_total": len(old),
    }


def _why(text: str) -> str:
    """粗粒度归因，帮人快速扫读（不是判定依据，只是分类标签）。"""
    import re
    from utils.normalize import invisible_ratio
    tags = []
    if len(text) >= 10 and invisible_ratio(text) >= 0.15:
        tags.append("不可见字符混淆")
    if re.search(r"(?:t\.me/|@\w+bot)", text, re.I):
        tags.append("含链接")
    if re.search(r"(?:\+?86[-\s]?)?1[3-9]\d{9}", text):
        tags.append("含手机号")
    if re.search(r"(无定|無定|面付|缅付|上门|尚门|可约|苛约|接单|约课|开课)",
                 text):
        tags.append("招揽词")
    if re.search(r"(跑分|洗资|洗钱|代收|代付|usdt|U|日入|日赚|一天\s*\d)",
                 text, re.I):
        tags.append("金融")
    if re.search(r"(壮阳|催情|延时|阳痿|肾虚|药|印度代购)", text):
        tags.append("药品")
    if re.search(r"(机场|VPN|梯子|节点|G不限时|流量)", text, re.I):
        tags.append("网络服务")
    return "、".join(tags) if tags else "—"


def report(res: dict, cut: float, days: int, limit: int) -> None:
    print("=" * 78)
    print(f"  影子期判定分析（近 {days} 天，阈值 {cut}）")
    print("=" * 78)

    if not res["has_old"]:
        # 新群没有老 bot 做对照，也就没有免费的真值标签。
        # 这时分析器的价值是"把模型的判断摊开给人看"，让人补标注。
        spam = res["only_new"] + res["both"]
        print("  ⚠️ 本库没有可对照的老 bot（新群场景），只能人工标注")
        print()
        print(f"  新 bot 判为广告：{len(spam)} 条")
        print()
        print("  处理方式：扫下面这份清单，")
        print("    · 判对了 → 无需操作（它已经是正例）")
        print("    · 判错了 → 用 --add-negative \"文本\" 写成反例")
        print("    · 漏了的 → 用 --add-positive \"文本\" 写成正例")
        print()
        print("  按置信度分档（低档更可能判错，优先看）:")
        for lo, hi, label in ((0.0, 0.6, "0.50-0.59 最可疑"),
                              (0.6, 0.8, "0.60-0.79 需确认"),
                              (0.8, 0.9, "0.80-0.89 较可信"),
                              (0.9, 1.01, "0.90-1.00 高可信")):
            items = [v for v in spam if lo <= v["conf"] < hi]
            if not items:
                continue
            print(f"\n  ▸ {label}  共 {len(items)} 条")
            for v in items[:limit]:
                print(f"      [{v['conf']:.2f}] {_why(v['text']):20s} {v['text'][:44]!r}")
            if len(items) > limit:
                print(f"      …另有 {len(items) - limit} 条")
        return

    both, only_new, only_old = res["both"], res["only_new"], res["only_old"]
    n_new_spam = len(both) + len(only_new)
    n_old = res["old_total"]

    print(f"  新 bot 判为广告 : {n_new_spam} 条")
    print(f"  老 bot 删除     : {n_old} 条")
    print(f"  两边一致        : {len(both)} 条")
    print(f"  仅新 bot 判删   : {len(only_new)} 条  <- 老 bot 可能漏删，值得看")
    print(f"  仅老 bot 删了   : {len(only_old)} 条  <- 新 bot 可能漏判，重点")
    print(f"  老 bot 删了但新 bot 未判定: {len(res['unseen_by_new'])} 条")

    if n_old:
        agree = len(both) / n_old
        print(f"\n  相对老 bot 的召回率: {agree:.1%}"
              f"（一致 {len(both)} / 老 bot 删除 {n_old}）")

    # --- 置信度分布：用于校准阈值 ---
    print("\n  新 bot 判为广告的置信度分布:")
    buckets = Counter()
    for v in both + only_new:
        c = v["conf"]
        b = ("0.95-1.00" if c >= 0.95 else "0.85-0.94" if c >= 0.85
             else "0.70-0.84" if c >= 0.70 else "0.50-0.69")
        buckets[b] += 1
    for b in ("0.95-1.00", "0.85-0.94", "0.70-0.84", "0.50-0.69"):
        n = buckets.get(b, 0)
        bar = "█" * min(40, n)
        print(f"    {b}  {n:5d}  {bar}")

    # --- 重点一：仅老 bot 删了（新 bot 漏判）---
    print("\n" + "-" * 78)
    print("  【重点】仅老 bot 删了 —— 新 bot 可能漏判，这些决定规则更新方向")
    print("-" * 78)
    if not only_old:
        print("    （无）新 bot 覆盖了老 bot 的全部删除")
    else:
        by_tag = defaultdict(list)
        for v in only_old:
            by_tag[_why(v["text"])].append(v)
        for tag, items in sorted(by_tag.items(), key=lambda x: -len(x[1])):
            print(f"\n    ▸ {tag}  共 {len(items)} 条")
            for v in items[:8]:
                print(f"      [{v['conf']:.2f}] {v['text'][:52]!r}")
                if v.get("old_reason"):
                    print(f"             老 bot 理由: {v['old_reason'][:56]}")
            if len(items) > 8:
                print(f"      …另有 {len(items) - 8} 条")

    # --- 重点二：仅新 bot 判删 ---
    print("\n" + "-" * 78)
    print("  【重点】仅新 bot 判删 —— 老 bot 漏删的，验证新引擎的价值")
    print("-" * 78)
    if not only_new:
        print("    （无）")
    else:
        high = [v for v in only_new if v["conf"] >= 0.85]
        mid = [v for v in only_new if v["conf"] < 0.85]
        for label, items in (("高置信 ≥0.85", high), ("中间带 0.50-0.84", mid)):
            print(f"\n    ▸ {label}  共 {len(items)} 条")
            for v in items[:10]:
                print(f"      [{v['conf']:.2f}] {_why(v['text']):20s} {v['text'][:44]!r}")
            if len(items) > 10:
                print(f"      …另有 {len(items) - 10} 条")

    # --- 招揽词命中的对照：规则更新最直接的输入 ---
    print("\n" + "-" * 78)
    print("  按特征归因（老 bot 删了 vs 新 bot 删了）")
    print("-" * 78)
    old_tags = Counter(_why(v["text"]) for v in both + only_old)
    new_tags = Counter(_why(v["text"]) for v in both + only_new)
    all_tags = set(old_tags) | set(new_tags)
    print(f"    {'特征':24s} {'老bot':>7} {'新bot':>7}")
    for t in sorted(all_tags, key=lambda x: -(old_tags[x] + new_tags[x]))[:14]:
        print(f"    {t:24s} {old_tags[t]:>7} {new_tags[t]:>7}")


def export_csv(res: dict, path: str, cut: float) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["类别", "chat_id", "message_id", "置信度", "信任标记",
                    "昵称", "内容", "特征", "老bot理由"])
        groups = ([("两边一致", res["both"]), ("仅新bot判删", res["only_new"]),
                   ("仅老bot删了", res["only_old"])]
                  if res["has_old"] else
                  [("新bot判删", res["only_new"] + res["both"])])
        for label, items in groups:
            for v in items:
                w.writerow([label, v["chat_id"], v["message_id"],
                            f"{v['conf']:.2f}", "", v["name"], v["text"],
                            _why(v["text"]), v.get("old_reason", "")])
    print(f"\n  已导出 {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-db", default=NEW_DB_DEFAULT)
    ap.add_argument("--old-db", default=OLD_DB_DEFAULT,
                    help="对照用的老 bot 库。只在你自己的群（新老并存）时才有意义；"
                         "新群没有对照，留空即可")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--chat-id", type=int, default=0, help="只看某个群（0=全部）")
    ap.add_argument("--cut", type=float, default=None,
                    help="置信度阈值，默认取群配置")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--export-csv", default="")
    ap.add_argument("--add-positive", default="", help="人工确认后写入正例")
    ap.add_argument("--add-negative", default="", help="人工确认后写入反例")
    args = ap.parse_args()

    if not os.path.exists(args.new_db):
        print(f"新 bot 数据库不存在: {args.new_db}", file=sys.stderr)
        sys.exit(1)

    db.init_db(args.new_db)

    # 人工写入路径
    if args.add_positive or args.add_negative:
        chats = ([c["chat_id"] for c in db.list_active_chats()]
                 if not args.chat_id else [args.chat_id])
        for cid in chats:
            if args.add_positive:
                db.add_fewshot(cid, "positive", args.add_positive, "manual")
            if args.add_negative:
                db.add_fewshot(cid, "negative", args.add_negative, "manual")
        print(f"已写入 {len(chats)} 个群。当前样本数：")
        for cid in chats:
            print(f"  chat={cid} 正例={len(db.fewshots(cid, 'positive', 999))} "
                  f"反例={len(db.fewshots(cid, 'negative', 999))}")
        return

    cut = args.cut
    if cut is None:
        chats = db.list_active_chats()
        cut = float(chats[0]["confidence_cut"]) if chats else 0.5

    conn = sqlite3.connect(args.new_db)
    has_old = _attach_old(conn, args.old_db)
    data = collect(conn, args.days, has_old)
    conn.close()

    res = analyze(data, cut)
    report(res, cut, args.days, args.limit)
    if args.export_csv:
        export_csv(res, args.export_csv, cut)


if __name__ == "__main__":
    main()
