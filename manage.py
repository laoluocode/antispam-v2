"""群管理 CLI。

手动加群用（Phase 4 之前不做自助接入）。直接改数据库，不需要重启即可生效
（pipeline 每轮读一次 chats 表）。

用法:
    python manage.py add -1001234567890 --title "某某群" --nature general
    python manage.py add -1001234567890 --title "某某群" --nature adult_service --strict
    python manage.py list
    python manage.py set -1001234567890 --confidence-cut 0.85 --shadow-mode 0
    python manage.py set -1001234567890 --strictness standard
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
from prompts import NATURE_PRESETS

# 严格档位 -> 阈值。
# 阈值是成本/召回/误删之间唯一的调节旋钮，且改它**不动 prompt、不废缓存**。
#
# 命名注意：这里的 "standard" 就是实测最优档（F1 0.923），数值上是三者中最低的。
# 不用 "loose" 这种词是因为它听上去像"不严谨"，而实测它精确率反而最好——
# 所谓"精确率低"是评估集把 V1 漏删的广告标成了正常，属于标签噪声。
# 权衡看的是 F1 而非单看精确率。
STRICTNESS_CUT = {
    "strict": 0.85,     # 只删铁证   召回 81.2% / 精确 92.3% / F1 0.864
    "balanced": 0.70,   # 偏保守     召回 87.5% / 精确 91.6% / F1 0.895
    "standard": 0.50,   # 实测最优   召回 95.5% / 精确 89.3% / F1 0.923  <- 默认
    "aggressive": 0.30, # 多删       召回 98.0% / 精确 86.9% / F1 0.921
}


def cmd_add(args) -> None:
    exists = db.get_chat(args.chat_id)
    if exists:
        print(f"群 {args.chat_id} 已存在，用 manage.py set 修改")
        return
    # 新群默认影子模式关闭：影子期不删消息，广告会在群里公开堆几天，
    # 而新群没有老 bot 做对照、拿不到真值标签，这几天的"学习"价值很低。
    shadow = 1 if args.shadow else 0
    db.register_chat(args.chat_id, args.title, args.nature, shadow_mode=bool(shadow))
    cut = STRICTNESS_CUT[args.strictness]
    db.set_chat_field(args.chat_id, "strictness", args.strictness)
    db.set_chat_field(args.chat_id, "confidence_cut", cut)
    print(f"已添加群 {args.chat_id}（{args.title}）")
    print(f"  性质      : {args.nature}（{NATURE_PRESETS.get(args.nature, {}).get('nature', '?')}）")
    print(f"  严格度    : {args.strictness} -> 阈值 {cut}")
    print(f"  影子模式  : {'开（只记录不删除）' if shadow else '关（真删）'}")
    print(f"  默认预算  : 300000 token/天")
    print()
    print("  注意：机器人必须是群管理员且有「删除消息」权限，否则判定无法执行。")


def cmd_set(args) -> None:
    if not db.get_chat(args.chat_id):
        print(f"群 {args.chat_id} 不存在，先 add")
        return
    changed = []
    if args.title:
        db.set_chat_field(args.chat_id, "title", args.title)
        changed.append(f"title={args.title}")
    if args.nature:
        db.set_chat_field(args.chat_id, "chat_nature", args.nature)
        changed.append(f"nature={args.nature}")
    if args.strictness:
        cut = STRICTNESS_CUT[args.strictness]
        db.set_chat_field(args.chat_id, "strictness", args.strictness)
        db.set_chat_field(args.chat_id, "confidence_cut", cut)
        changed.append(f"strictness={args.strictness} 阈值={cut}")
    if args.confidence_cut is not None:
        db.set_chat_field(args.chat_id, "confidence_cut", args.confidence_cut)
        changed.append(f"confidence_cut={args.confidence_cut}")
    if args.shadow_mode is not None:
        db.set_chat_field(args.chat_id, "shadow_mode", args.shadow_mode)
        changed.append(f"shadow_mode={args.shadow_mode}")
    if args.token_cap is not None:
        db.set_chat_field(args.chat_id, "daily_token_cap", args.token_cap)
        changed.append(f"daily_token_cap={args.token_cap}")
    if args.fastlane is not None:
        db.set_chat_field(args.chat_id, "fastlane_seconds", args.fastlane)
        changed.append(f"fastlane_seconds={args.fastlane}")
    if args.batch is not None:
        db.set_chat_field(args.chat_id, "batch_seconds", args.batch)
        changed.append(f"batch_seconds={args.batch}")
    print("已更新 " + ", ".join(changed) if changed else "无改动")


def cmd_list(args) -> None:
    rows = db.list_active_chats()
    if not rows:
        print("尚无注册的群")
        return
    print(f"{'chat_id':>16} {'影子':>5} {'严格度':>9} {'阈值':>5} {'性质':>14}  标题")
    print("-" * 88)
    for r in rows:
        print(f"{r['chat_id']:>16} {'开' if r['shadow_mode'] else '关':>5} "
              f"{r['strictness']:>9} {r['confidence_cut']:>5.2f} "
              f"{r['chat_nature']:>14}  {r['title'] or ''}")
    print()
    print("  阈值含义：越高越保守（少删）。改阈值不动 prompt，也就不会废掉缓存。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("DB_NAME", "data/antispam.db"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="添加群")
    a.add_argument("chat_id", type=int)
    a.add_argument("--title", default="")
    a.add_argument("--nature", default="general", choices=list(NATURE_PRESETS))
    a.add_argument("--strictness", default="standard", choices=list(STRICTNESS_CUT),
                   help="默认 standard=阈值0.50（离线实测 F1 最优）")
    a.add_argument("--shadow", action="store_true",
                   help="开启影子模式（只记录不删除）。新群不建议开")
    a.set_defaults(func=cmd_add)

    b = sub.add_parser("set", help="修改群配置")
    b.add_argument("chat_id", type=int)
    b.add_argument("--title")
    b.add_argument("--nature", choices=list(NATURE_PRESETS))
    b.add_argument("--strictness", choices=list(STRICTNESS_CUT))
    b.add_argument("--confidence-cut", type=float, dest="confidence_cut")
    b.add_argument("--shadow-mode", type=int, choices=[0, 1], dest="shadow_mode")
    b.add_argument("--token-cap", type=int, dest="token_cap")
    b.add_argument("--fastlane", type=int, help="L1 快车道攒批窗口（秒）")
    b.add_argument("--batch", type=int, help="L2 常规攒批窗口（秒）")
    b.set_defaults(func=cmd_set)

    c = sub.add_parser("list", help="列出所有群")
    c.set_defaults(func=cmd_list)

    args = ap.parse_args()
    d = os.path.dirname(os.path.abspath(args.db))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    db.init_db(args.db)
    args.func(args)


if __name__ == "__main__":
    main()
