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


def cmd_words(args) -> None:
    """管理每群的词汇表。

    为什么需要：每个群有自己的惯例词。某群的"导航"是目录机器人的固定指令，
    判定器把它当广告会持续误删，而每次改 prompt 都要重新部署 + 废一次缓存。
    做成每群一张表，群里发现误判当场加词即可。

    改词表会更新 prompt 前缀 -> 该群缓存失效一次（约 0.004 元），
    所以尽量攒几个词一起加，而不是逐词频繁改。
    """
    if not db.get_chat(args.chat_id):
        print(f"群 {args.chat_id} 不存在")
        return
    kind = "blocked" if args.blocked else "normal"

    if args.remove:
        left = db.remove_word(args.chat_id, args.remove, kind)
        print(f"已从{'禁词' if kind == 'blocked' else '正常词'}表移除「{args.remove}」")
    elif args.add:
        left = db.add_word(args.chat_id, args.add, kind)
        label = "禁词" if kind == "blocked" else "正常词"
        print(f"已把「{args.add}」加入{label}表")
    else:
        left = None

    nw = db.normal_words(args.chat_id)
    bw = db.blocked_words(args.chat_id)
    print()
    print(f"群 {args.chat_id} 当前词汇表:")
    print(f"  正常词（单独出现不判广告）: {'、'.join(nw) if nw else '（空）'}")
    print(f"  禁词  （出现即高置信度）  : {'、'.join(bw) if bw else '（空）'}")
    if left is None:
        print()
        print("用法：")
        print("  manage.py words -100xxx --add 导航")
        print("  manage.py words -100xxx --add 尾课 --blocked")
        print("  manage.py words -100xxx --remove 导航")
        print("  manage.py words -100xxx          # 只看当前词表")


def cmd_cost(args) -> None:
    """成本报告：观察真实日消费，用于定预算。

    注意历史数据的计价口径：早先的实现把所有调用都按高峰价估算，
    而空闲时段（含全部周末）实际价格减半，所以**早期记录约偏高两倍**。
    现在计价按当前时段自动判断。
    """
    days = db.usage_daily_total(args.days)
    if not days:
        print("暂无用量记录。")
        return
    print(f"近 {args.days} 天成本（元）:")
    print(f"  {'日期':<12} {'调用':>7} {'缓存命中率':>10} {'成本':>10}")
    print("  " + "-" * 44)
    for d in days:
        h, m = d["hit_tokens"] or 0, d["miss_tokens"] or 0
        rate = h / (h + m) if (h + m) else 0
        print(f"  {d['usage_date']:<12} {d['calls']:>7} {rate:>9.0%} "
              f"{d['cost_yuan']:>9.4f}")
    total = sum(d["cost_yuan"] or 0 for d in days)
    print("  " + "-" * 44)
    print(f"  {'合计':<12} {'':>7} {'':>10} {total:>9.4f}")
    if days:
        print(f"  日均: {total / len(days):.4f} 元/天")

    print()
    print("  今日各群明细:")
    today = days[0]["usage_date"] if days else ""
    print(f"  {'chat_id':>17} {'调用':>6} {'命中率':>7} {'成本':>10}")
    for r in db.usage_by_day(args.days):
        if r["usage_date"] != today:
            continue
        h, m = r["hit_tokens"] or 0, r["miss_tokens"] or 0
        rate = h / (h + m) if (h + m) else 0
        title = db.get_chat(r["chat_id"])
        name = (title["title"] if title else "") or ""
        print(f"  {r['chat_id']:>17} {r['calls']:>6} {rate:>6.0%} "
              f"{r['cost_yuan']:>9.4f}  {str(name)[:18]}")


def cmd_gbl(args) -> None:
    """全局黑名单：跨群生效的账号黑名单。

    为什么需要：同一批广告号会同时打多个群（实测：南京苏州集团与武汉集团
    同期出现同一套广告）。某个群确认过的号，其他群当场就拦截，
    不必各自重新踩一遍——群越多，每个群越安全。
    本群删除次数达到阈值的账号会自动进入此表。
    """
    if args.remove:
        ok = db.global_blacklist_remove(args.remove)
        print("已移出全局黑名单" if ok else "不在全局黑名单中")
    if args.add:
        db.global_blacklist_add(args.add, None, "管理员手工加入", 0)
        print(f"已加入全局黑名单: {args.add}")

    rows = db.global_blacklist_list(200)
    if not rows:
        print("全局黑名单为空。")
    else:
        print(f"全局黑名单（{len(rows)} 人，对所有群生效）:")
        print(f"  {'user_id':>14} {'命中群数':>7}  昵称 / 原因")
        print("  " + "-" * 72)
        for r in rows:
            print(f"  {r['user_id']:>14} {r['hits']:>7}  "
                  f"{str(r['full_name'] or '')[:16]:16s} {str(r['reason'] or '')[:34]}")
    print()
    print("用法：manage.py gbl [--add <user_id>] [--remove <user_id>]")


def cmd_links(args) -> None:
    """查看/设置本群体系的频道标识。

    用途：判断一条 t.me 链接是"自家频道"还是"外部引流"。
    这些群的运营方把自家榜单/报告/资讯频道写在群描述里，程序会定期自动抓取；
    本命令用于查看抓取结果，或在描述没写全时手工补充。
    """
    if args.add:
        cur = db.managed_links(args.chat_id)
        cur.add(args.add.lstrip("@").lower())
        db.set_chat_field(args.chat_id, "managed_links", ",".join(sorted(cur)))
        print(f"已加入 {args.add}")
    if args.remove:
        cur = db.managed_links(args.chat_id)
        cur.discard(args.remove.lstrip("@").lower())
        db.set_chat_field(args.chat_id, "managed_links", ",".join(sorted(cur)))
        print(f"已移除 {args.remove}")

    row = db.get_chat(args.chat_id)
    if not row:
        print(f"群 {args.chat_id} 不存在")
        return
    print(f"群 {args.chat_id}（{row['title']}）")
    print(f"  群自己的 username : {row['username'] or '(未设置)'}")
    print(f"  原始 managed_links: {row['managed_links'] or '(空)'}")
    print(f"  生效的自家频道集合:")
    for n in sorted(db.managed_links(args.chat_id)):
        print(f"    @{n}")
    print()
    print("  指向上述之外的目标的 t.me 链接会被判为外部引流并删除。")
    print("  用法：manage.py links -100xxx [--add 频道名] [--remove 频道名]")


def cmd_previews(args) -> None:
    """列出近期链接预览指纹及引用它们的账号数。

    用途：广告号用"短文本 + 频道预览"引流，正文拿不到，但
    **多个账号指向同一预览目标**这个协同特征是可见的。
    账号数 >= 3 的指纹会被自动判为协同引流并删除。
    """
    rows = db.list_previews(args.chat_id, args.hours, 30)
    if not rows:
        print(f"近 {args.hours} 小时无预览记录。")
        print("若群里有此类广告却没记录，说明消息里没有可提取的 URL —— ")
        print("查看日志中的「低信息量消息」行确认 bot 实际收到了什么。")
        return
    print(f"近 {args.hours} 小时预览指纹（按引用账号数排序）:")
    print(f"  {'账号数':>5}  {'最后出现':<20} URL")
    print("  " + "-" * 78)
    for r in rows:
        flag = "  <- 会自动判删" if r["senders"] >= 3 else ""
        print(f"  {r['senders']:>5}  {str(r['last_seen'])[:19]:<20} "
              f"{str(r['url'] or '')[:44]}{flag}")


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
        nw = db.normal_words(r["chat_id"])
        bw = db.blocked_words(r["chat_id"])
        if nw or bw:
            print(f"{'':>16} 词表: 正常词[{'、'.join(nw) or '空'}] "
                  f"禁词[{'、'.join(bw) or '空'}]")
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

    h = sub.add_parser("cost", help="成本报告（观察真实日消费）")
    h.add_argument("--days", type=int, default=7)
    h.set_defaults(func=cmd_cost)

    g = sub.add_parser("gbl", help="全局黑名单（跨群生效）")
    g.add_argument("--add", type=int, help="加入一个 user_id")
    g.add_argument("--remove", type=int, help="移出一个 user_id")
    g.set_defaults(func=cmd_gbl)

    f = sub.add_parser("links", help="查看/设置本群体系的频道标识")
    f.add_argument("chat_id", type=int)
    f.add_argument("--add", help="加入一个频道 username")
    f.add_argument("--remove", help="移除一个频道 username")
    f.set_defaults(func=cmd_links)

    e = sub.add_parser("previews", help="查看链接预览指纹（排查协同引流）")
    e.add_argument("chat_id", type=int)
    e.add_argument("--hours", type=int, default=24)
    e.set_defaults(func=cmd_previews)

    d = sub.add_parser("words", help="管理群词汇表（正常词/禁词）")
    d.add_argument("chat_id", type=int)
    d.add_argument("--add", help="加入一个词")
    d.add_argument("--remove", help="移除一个词")
    d.add_argument("--blocked", action="store_true",
                   help="操作禁词表（默认操作正常词表）")
    d.set_defaults(func=cmd_words)

    args = ap.parse_args()
    d = os.path.dirname(os.path.abspath(args.db))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    db.init_db(args.db)
    args.func(args)


if __name__ == "__main__":
    main()
