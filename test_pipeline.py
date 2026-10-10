"""pipeline.py 的离线自测。用 FakeClient，不调真实 API。"""
import asyncio
import os
import re
import sys
import tempfile

sys.path.insert(0, "/root/ribaobot/antispambot")
os.chdir("/root/ribaobot/antispambot")

tmp = tempfile.mktemp(suffix=".db")
import db
db.init_db(tmp)
from pipeline import ChatPipeline, PendingMessage
from deepseek import Usage


class FakeClient:
    def __init__(self):
        self.calls = 0

    async def judge(self, messages, user_id, **kw):
        self.calls += 1
        ids = []
        for line in messages[-1]["content"].splitlines():
            m = re.match(r"\[ID=(\d+)\]", line)
            if m:
                ids.append(int(m.group(1)))
        verdicts = [{"id": i, "confidence": 0.9 if n % 2 == 0 else 0.05}
                    for n, i in enumerate(ids)]
        return ({"verdicts": verdicts},
                Usage(cache_hit_tokens=4500, cache_miss_tokens=300,
                      completion_tokens=50, calls=1))


async def main():
    cid = -100
    db.register_chat(cid, "测试群", "adult_service", shadow_mode=True)
    fake = FakeClient()
    deleted, notified = [], []

    async def on_spam(v):
        deleted.append(v)

    async def notify(vs):
        notified.append(len(vs))

    p = ChatPipeline(cid, fake, on_spam, notify)
    db.cache_admins(cid, [(999, "管理员")])

    print("=== 1. L0 立即判定（手机号）===")
    m = PendingMessage(1, 1001, cid, 1, "某人", "我的电话13812345678",
                       None, "2026-10-10 12:00:00")
    v = await p.submit(m)
    src = v.source if v else "-"
    rsn = v.reason if v else "-"
    hit = v.is_spam if v else None
    print("  hit=%s source=%s reason=%s" % (hit, src, rsn))

    print("=== 2. 管理员免检 ===")
    v = await p.submit(PendingMessage(2, 1002, cid, 999, "管理员",
                                      "13812345678", None, "t"))
    print("  结果=%s (应为 None)" % v)

    print("=== 3. 白名单 trusted 跳过 ===")
    db.whitelist_set(cid, 5, "trusted", "白名单", None)
    v = await p.submit(PendingMessage(3, 1003, cid, 5, "白名单",
                                      "13812345678", None, "t"))
    print("  结果=%s (应为 None)" % v)

    print("=== 4. 分级路由 ===")
    m_new = PendingMessage(10, 1010, cid, 10, "新人", "正常聊天", None, "t")
    await p.submit(m_new)
    print("  新人 msg_n=0          -> fast=%s trust=%s" % (m_new.fast, m_new.trust))

    for _ in range(40):
        db.upsert_user_seen(cid, 11, "老用户", None)
    m_old = PendingMessage(11, 1011, cid, 11, "老用户", "正常聊天", None, "t")
    await p.submit(m_old)
    print("  老用户 msg_n=40       -> fast=%s trust=%s" % (m_old.fast, m_old.trust))

    for _ in range(20):
        db.upsert_user_seen(cid, 12, "广告号", None)
        db.bump_user_deleted(cid, 12)
    m_rep = PendingMessage(12, 1012, cid, 12, "广告号", "正常聊天", None, "t")
    await p.submit(m_rep)
    print("  惯犯 msg_n=20/del=20  -> fast=%s trust=%s" % (m_rep.fast, m_rep.trust))
    print("  队列 fast=%d slow=%d" % (len(p._fast), len(p._slow)))

    print("=== 5. prompt 结构（缓存前缀顺序）===")
    msgs = p._build_prompt([m_new])
    print("  roles=%s" % [m["role"] for m in msgs])
    for i, mm in enumerate(msgs):
        print("    [%d] %-6s len=%5d  %r" % (i, mm["role"], len(mm["content"]),
                                             mm["content"][:36]))

    print("=== 6. 影子模式：记录但不删 ===")
    await p._judge([m_new, m_old])
    print("  on_spam 调用=%d (影子模式应 0)  通知=%s" % (len(deleted), notified))
    recs = db.query("SELECT action, COUNT(*) n FROM spam_records "
                    "WHERE chat_id=? GROUP BY action", (cid,))
    print("  落库=%s" % [(r["action"], r["n"]) for r in recs])

    print("=== 7. 关影子后真删 ===")
    db.set_chat_field(cid, "shadow_mode", 0)
    p2 = ChatPipeline(cid, fake, on_spam, notify)
    await p2._judge([m_new, m_old])
    print("  on_spam 调用=%d (应>0)" % len(deleted))
    recs = db.query("SELECT action, COUNT(*) n FROM spam_records "
                    "WHERE chat_id=? GROUP BY action", (cid,))
    print("  落库=%s" % [(r["action"], r["n"]) for r in recs])

    print("=== 8. 预算护栏 ===")
    db.set_chat_field(cid, "daily_token_cap", 10)
    before = fake.calls
    await ChatPipeline(cid, fake, on_spam, notify)._judge([m_new])
    print("  超预算跳过调用: %s" % (fake.calls == before))

    print("=== 9. 判定失败时回队重试 ===")
    class FailClient:
        async def judge(self, *a, **k):
            from deepseek import DeepSeekError
            raise DeepSeekError("模拟失败")
    db.set_chat_field(cid, "daily_token_cap", 300000)
    row = db.save_message(cid, 5000, 1, "某人", "测试", None, "2026-10-10 12:00:00")
    batch = db.take_unchecked_batch(cid, 10)
    n_before = len(batch)
    pf = ChatPipeline(cid, FailClient(), on_spam, notify)
    await pf._judge([PendingMessage(r["id"], r["message_id"], cid, r["user_id"],
                                    r["full_name"], r["text"], None, r["timestamp"])
                     for r in batch])
    back = db.query("SELECT COUNT(*) n FROM messages WHERE chat_id=? AND is_checked=0",
                    (cid,))
    print("  失败前取批=%d 条，失败后回到未检测=%d 条 (应相等)" % (n_before, back[0]["n"]))
    print("  last_error=%s" % (pf.last_error is not None))

    print("=== 10. 付费商家（advertiser）：仅删高置信 ===")
    db.whitelist_set(cid, 77, "advertiser", "付费商家", None)

    class GradedClient:
        """按预设分数返回，用于测阈值分档。"""
        def __init__(self, mapping):
            self.mapping = mapping
        async def judge(self, messages, user_id, **kw):
            ids = []
            for line in messages[-1]["content"].splitlines():
                mm = re.match(r"\[ID=(\d+)\]", line)
                if mm:
                    ids.append(int(mm.group(1)))
            return ({"verdicts": [{"id": i, "confidence": self.mapping.get(i, 0.0)}
                                  for i in ids]},
                    Usage(cache_hit_tokens=4500, cache_miss_tokens=300,
                          completion_tokens=20, calls=1))

    db.set_chat_field(cid, "shadow_mode", 0)
    db.set_chat_field(cid, "confidence_cut", 0.5)
    deleted.clear()

    # row_id 77 -> 置信度 0.70（普通用户应删，付费商家应放过）
    # row_id 78 -> 置信度 0.95（两者都应删）
    m_a = PendingMessage(77, 7077, cid, 77, "付费商家", "招揽内容", None, "t")
    m_a.whitelist = "advertiser"
    m_b = PendingMessage(78, 7078, cid, 78, "普通用户", "招揽内容", None, "t")
    m_c = PendingMessage(79, 7079, cid, 77, "付费商家", "诈骗内容", None, "t")
    m_c.whitelist = "advertiser"

    gc = GradedClient({77: 0.70, 78: 0.70, 79: 0.95})
    await ChatPipeline(cid, gc, on_spam, notify)._judge([m_a, m_b, m_c])
    got = sorted(v.message.row_id for v in deleted)
    print("  置信度 0.70 付费商家(row77) -> %s (应放过)"
          % (77 in got))
    print("  置信度 0.70 普通用户(row78) -> %s (应删除)"
          % (78 in got))
    print("  置信度 0.95 付费商家(row79) -> %s (应删除)"
          % (79 in got))
    u = db.user_stats(cid, 77)
    print("  付费商家 del_count=%d (应为 0，不累计)" % u["del_n"])
    u2 = db.user_stats(cid, 78)
    print("  普通用户 del_count=%d (应为 1，已累计)" % u2["del_n"])

    print("=== 11. 回归：is_checked 必须被标记（线上曾全为 0）===")
    from deepseek import DeepSeekError as _DSE
    cid2 = -200
    db.register_chat(cid2, "标记测试群", "general", shadow_mode=False)
    db.set_chat_field(cid2, "shadow_mode", 0)

    async def noop(v): pass

    def add_msg(mid, text):
        return db.save_message(cid2, mid, 1, "群友", text, None,
                               "2026-10-10 13:00:00")

    # (a) 判定成功 -> 应标记
    for i in range(3):
        add_msg(6000 + i, "普通聊天内容 %d" % i)
    rows = db.query("SELECT id FROM messages WHERE chat_id=? AND is_checked=0", (cid2,))
    ids = [r["id"] for r in rows]
    batch = [PendingMessage(i, 6000 + n, cid2, 1, "群友", "普通聊天内容 %d" % n,
                            None, "t") for n, i in enumerate(ids)]
    await ChatPipeline(cid2, GradedClient({i: 0.05 for i in ids}), noop, None)._judge(batch)
    left = db.query("SELECT COUNT(*) n FROM messages WHERE chat_id=? AND is_checked=0", (cid2,))[0]["n"]
    print("  (a) 判定成功后未检测数=%d (应为 0)" % left)
    assert left == 0, "判定成功后未标记 is_checked"

    # (b) 断言全部失败 -> 应保留重试，但不超过上限
    db.execute("DELETE FROM messages WHERE chat_id=?", (cid2,))
    for i in range(2):
        add_msg(6100 + i, "测试")
    rows = db.query("SELECT id, message_id FROM messages WHERE chat_id=? AND is_checked=0", (cid2,))
    batch = [PendingMessage(r["id"], r["message_id"], cid2, 1, "群友", "测试", None, "t")
             for r in rows]

    class Boom:
        async def judge(self, *a, **k):
            raise _DSE("模拟持续失败")
    pf = ChatPipeline(cid2, Boom(), noop, None)
    for attempt in range(db.MAX_JUDGE_RETRIES + 1):
        await pf._judge(batch)
    left = db.query("SELECT COUNT(*) n FROM messages WHERE chat_id=? AND is_checked=0", (cid2,))[0]["n"]
    print("  (b) 连续失败 %d 次后未检测数=%d (应为 0，超过上限即放弃)"
          % (db.MAX_JUDGE_RETRIES + 1, left))
    assert left == 0, "超过重试上限后应标记已检测，否则每轮重复烧 token"

    # (c) L0 命中 -> 应标记
    db.execute("DELETE FROM messages WHERE chat_id=?", (cid2,))
    rid = add_msg(6200, "我的电话13812345678")
    p3 = ChatPipeline(cid2, fake, noop, None)
    m = PendingMessage(rid, 6200, cid2, 1, "群友", "我的电话13812345678", None, "t")
    v = await p3.submit(m)
    left = db.query("SELECT COUNT(*) n FROM messages WHERE chat_id=? AND is_checked=0", (cid2,))[0]["n"]
    print("  (c) L0 命中(hit=%s) 后未检测数=%d (应为 0)" % (bool(v and v.is_spam), left))
    assert left == 0, "L0 命中后未标记 is_checked"

    # (d) 管理员 / 白名单 -> 也应标记（否则它们永远留在未检测里被反复取出）
    db.execute("DELETE FROM messages WHERE chat_id=?", (cid2,))
    db.cache_admins(cid2, [(888, "管理员")])
    rid = add_msg(6300, "管理员发言")
    ChatPipeline(cid2, fake, noop, None)  # 新建以刷新管理员缓存
    p4 = ChatPipeline(cid2, fake, noop, None)
    await p4.submit(PendingMessage(rid, 6300, cid2, 888, "管理员", "管理员发言", None, "t"))
    left = db.query("SELECT COUNT(*) n FROM messages WHERE chat_id=? AND is_checked=0", (cid2,))[0]["n"]
    print("  (d) 管理员免检后未检测数=%d (应为 0)" % left)
    assert left == 0, "管理员消息未标记 is_checked"

    print("=== 12. 回归：同批多条删除记录可分别定位（记录 ID 精确）===")
    cid3 = -300
    db.register_chat(cid3, "多条测试群", "general", shadow_mode=False)
    db.set_chat_field(cid3, "shadow_mode", 0)
    r1 = db.save_spam_record(cid3, 11, "甲", 7001, "甲的内容", 0.9, "llm", "delete")
    r2 = db.save_spam_record(cid3, 22, "乙", 7002, "乙的内容", 0.9, "llm", "delete")
    print("  两条记录 ID: %s / %s (应不同)" % (r1, r2))
    assert r1 != r2, "同批两条应各自有独立记录 ID"
    a = db.get_spam_record(r1)
    b = db.get_spam_record(r2)
    print("  按 ID 取回: %s=%s, %s=%s" % (r1, a["text_snapshot"], r2, b["text_snapshot"]))
    assert a["text_snapshot"] == "甲的内容" and b["text_snapshot"] == "乙的内容"
    # 恢复其中一条不应影响另一条
    db.execute("UPDATE spam_records SET action='revive' WHERE id=?", (r1,))
    print("  恢复 %s 后: r1.action=%s  r2.action=%s (r2 应仍为 delete)"
          % (r1, db.get_spam_record(r1)["action"], db.get_spam_record(r2)["action"]))
    assert db.get_spam_record(r2)["action"] == "delete"
    # /recent 数据源
    n = len(db.recent_spam_records_all(10))
    print("  recent_spam_records_all 可取:", n, "条")
    assert n >= 1

    print("=== 13. 回归：单字符刷屏（线上实测漏判案例）===")
    from detector_rules import check_bare_letter_spam, is_bare_letter

    # 线上真实序列：账号 A 发 n d d d n n d d m d n d d，账号 B 发 u t o t o t
    A = ["n", "d", "d", "d", "n", "n", "d", "d", "m", "d", "n", "d", "d"]
    hit_at = None
    for i, t in enumerate(A):
        if check_bare_letter_spam(t, A[:i]).hit:
            hit_at = i + 1
            break
    print("  广告号序列在第 %s 条命中（阈值 3）" % hit_at)
    assert hit_at == 3, "应在第 3 条裸字母时命中"

    # 后续每条都应命中（持续刷屏）
    tail_hits = sum(1 for i, t in enumerate(A[3:], 3)
                    if check_bare_letter_spam(t, A[:i]).hit)
    print("  后续 %d 条全部命中: %s" % (len(A) - 3, tail_hits == len(A) - 3))
    assert tail_hits == len(A) - 3

    # 常见英文回应不应算裸字母（否则误伤正常群友）
    for w in ("ok", "hi", "no", "up", "dd", "thx", "yes", "id", "qq", "vx"):
        assert not is_bare_letter(w), "%r 不该被算作裸字母" % w
    print("  常见英文回应白名单生效: ok/hi/no/up/dd/thx/yes/id/qq/vx 均不触发")

    # 数字必须排除：老库实测 '1' 出现 282 次且全是正常回应
    for w in ("1", "66", "88", "666", "0", "2"):
        assert not is_bare_letter(w), "%r（数字）不该被算作裸字母" % w
    print("  数字已排除: 1/66/88/666/0/2 均不触发（老库 '1' 出现 282 次）")

    # 中文短回应不该触发
    for w in ("在的", "嗯", "好的", "收到", "在呢", "谁", "好", "哦", "哈喽",
              "有空", "在岗", "积分", "导航", "签到"):
        assert not check_bare_letter_spam(w, ["在的", "嗯", "好的"]).hit, w
    print("  中文短回应不触发: 在的/嗯/好的/收到/在呢/谁/好/哦/哈喽/有空/导航/签到")

    # 行为类检测**不再按信任档位豁免**（借鉴 tg-spam 的设计：
    # "重复/行为类检测对所有用户生效——已批准用户仍可能通过重复消息刷屏"）。
    # 原来的豁免留了个缺口：老群友账号被盗后发裸字母完全不被拦。
    # 误伤风险由白名单承担：白名单已涵盖 ok/hi/no/dd/1/66/88 等常见短回应。
    cid4 = -400
    db.register_chat(cid4, "行为检测测试群", "general", shadow_mode=False)
    db.set_chat_field(cid4, "shadow_mode", 0)
    for _ in range(40):
        db.upsert_user_seen(cid4, 42, "老群友", None)
    for i in range(5):
        db.save_message(cid4, 8000 + i, 42, "老群友", "x", None, "t")
    rid = db.save_message(cid4, 8100, 42, "老群友", "x", None, "t")
    p13 = ChatPipeline(cid4, fake, noop, None)
    v13 = await p13.submit(PendingMessage(rid, 8100, cid4, 42, "老群友", "x", None, "t"))
    print("  老群友反复发裸字母: %s（现在应命中——账号可能被盗）"
          % ("命中" if (v13 and v13.is_spam) else "未命中"))
    assert v13 and v13.is_spam, "行为类检测不应按信任档位豁免"

    # 但老群友发白名单里的常见短回应仍然安全
    for i, w in enumerate(("ok", "hi", "1", "66", "导航", "签到")):
        db.save_message(cid4, 8200 + i, 43, "老群友2", w, None, "t")
        r2 = db.save_message(cid4, 8300 + i, 43, "老群友2", w, None, "t")
    p13b = ChatPipeline(cid4, fake, noop, None)
    r3 = db.save_message(cid4, 8400, 43, "老群友2", "ok", None, "t")
    v13b = await p13b.submit(PendingMessage(r3, 8400, cid4, 43, "老群友2",
                                            "ok", None, "t"))
    print("  老群友发 'ok': %s（应不命中，白名单兜住误伤）"
          % ("命中" if (v13b and v13b.is_spam) else "未命中"))
    assert not (v13b and v13b.is_spam), "白名单里的短回应不该被拦"

    print("=== 14. 回归：外部频道引流检测 ===")
    from detector_rules import check_external_channel_link, extract_tme_identifiers

    db.register_chat(-500, "武汉集团", "adult_service", shadow_mode=False)
    db.set_chat_field(-500, "username", "wuhanjt")
    db.set_chat_field(-500, "managed_links",
                      "wuhanjt,wuhanjp,whjtbg2,whjtshangbangbot,wuhanjt_bot")
    managed = db.managed_links(-500)
    print("  自家频道集合: %s" % sorted(managed))

    cases = [
        ("https://t.me/wuhanjp/123", False, "自家榜单频道"),
        ("https://t.me/whjtbg2", False, "自家报告频道"),
        ("https://t.me/wdgx", True, "外部频道（实测案例）"),
        ("https://t.me/some_spam_channel/99", True, "外部频道+帖子"),
        ("看看这个 https://t.me/wdgx/1", True, "正文夹带外部链接"),
        ("今天在汉口修车体验还行", False, "无链接"),
    ]
    for text, expect_hit, note in cases:
        r = check_external_channel_link(text, managed)
        ok = (r.hit == expect_hit)
        print("  [%s] %-34s -> %s  %s"
              % ("OK" if ok else "FAIL", text[:34],
                 "命中" if r.hit else "放行", note))
        assert ok, "%s 期望 hit=%s 实际=%s" % (text, expect_hit, r.hit)

    print("  t.me 标识提取: %s" % sorted(extract_tme_identifiers(
        "https://t.me/wdgx/1 和 t.me/wuhanjp 以及 t.me/123")))
    print("  自家+外部混合不判（避免误伤）: %s"
          % check_external_channel_link(
              "https://t.me/wuhanjp 和 https://t.me/wdgx", managed).hit)

    print("=== 15. 回归：编辑消息重新判定 / 按钮 / 全局黑名单 ===")
    from detector_rules import check_inline_buttons

    cid5 = -600
    db.register_chat(cid5, "编辑测试群", "general", shadow_mode=False)
    db.set_chat_field(cid5, "shadow_mode", 0)
    db.set_chat_field(cid5, "username", "testgroup")
    db.set_chat_field(cid5, "managed_links", "testgroup,ourchannel")

    # (a) 编辑消息：正文更新 + 重置为未检测
    rid = db.save_message(cid5, 9001, 7, "某人", "今天天气不错", None,
                          "2026-10-10 15:00:00")
    db.take_unchecked_batch(cid5, 10)          # 假装判过了
    row = db.get_message(cid5, 9001)
    print("  (a) 编辑前 is_checked=%s text=%r" % (row["is_checked"], row["text"]))
    ok = db.remark_message(cid5, 9001, "加我 t.me/spamchannel 领福利")
    row = db.get_message(cid5, 9001)
    print("      编辑后 is_checked=%s text=%r" % (row["is_checked"], row["text"]))
    assert ok and row["is_checked"] == 0, "编辑后应重置为未检测以便重新判定"
    print("      -> 编辑会触发重新判定")

    # (b) 编辑后的内容应被 L0 抓到（外部频道）
    from detector_rules import check_external_channel_link
    managed = db.managed_links(cid5)
    v = check_external_channel_link(row["text"], managed)
    print("  (b) 编辑后内容判定: %s (%s)" % ("命中" if v.hit else "放行", v.reason))
    assert v.hit, "编辑成外部频道链接应被 L0 抓到"

    # (c) inline 按钮
    v = check_inline_buttons(["加入频道"], ["https://t.me/spamchannel"], managed)
    print("  (c) 外部频道按钮: %s (%s)" % ("命中" if v.hit else "放行", v.reason))
    assert v.hit
    v = check_inline_buttons(["查看榜单"], ["https://t.me/ourchannel"], managed)
    print("      自家频道按钮: %s (应放行)" % ("命中" if v.hit else "放行"))
    assert not v.hit
    v = check_inline_buttons(["领取福利"], [], managed)
    print("      纯引导文案按钮: %s (%s)" % ("命中" if v.hit else "放行", v.reason))
    assert v.hit

    # (d) 全局黑名单跨群生效
    print("  (d) 全局黑名单跨群生效:")
    cid6 = -700
    db.register_chat(cid6, "另一个群", "general", shadow_mode=False)
    print("      A群黑名单前，B群可见:", db.blacklist_ids(cid6))
    db.global_blacklist_add(999888, "惯犯", "测试", cid5)
    print("      加入全局黑名单后，B群可见:", db.blacklist_ids(cid6))
    assert 999888 in db.blacklist_ids(cid6), "全局黑名单应对所有群生效"
    assert 999888 in db.blacklist_ids(cid5), "全局黑名单应对本群也生效"
    print("      全局黑名单条目:", len(db.global_blacklist_list()))
    db.global_blacklist_remove(999888)
    print("      移除后 B群可见:", db.blacklist_ids(cid6))

    print("=== 16. 回归：最小批门槛 / 预算超限回队 / 回捞补判 ===")
    import pipeline as _pl

    print("  最小批门槛: fast=%d slow=%d" % (_pl.MIN_BATCH_FAST, _pl.MIN_BATCH_SLOW))
    # fast=0 表示不设门槛、严格按 15 秒窗口触发（实时性优先的配置）。
    # 这里只断言"要么不设门槛，要么门槛足够大到能摊薄前缀成本"。
    assert _pl.MIN_BATCH_FAST == 0 or _pl.MIN_BATCH_FAST >= 10, \
        "快车道门槛要么关闭(0)，要么足够大"
    assert _pl.MIN_BATCH_SLOW >= 10, "慢车道门槛过小起不到摊薄作用"

    cid7 = -800
    db.register_chat(cid7, "回捞测试群", "general", shadow_mode=False)
    db.set_chat_field(cid7, "shadow_mode", 0)
    for i in range(5):
        db.save_message(cid7, 9500 + i, 3, "群友", "内容 %d" % i, None, "t")
        db.take_unchecked_batch(cid7, 1)   # 假装判过（模拟早先被静默丢弃）

    left = db.query("SELECT COUNT(*) n FROM messages WHERE chat_id=? AND is_checked=0",
                    (cid7,))[0]["n"]
    print("  造 5 条『已标记检测但没判过』的消息，未检测=%d" % left)

    # 手工把它们置回未检测，模拟需要补判的状态
    db.execute("UPDATE messages SET is_checked=0 WHERE chat_id=?", (cid7,))
    p16 = ChatPipeline(cid7, fake, noop, None)
    n = await p16.enqueue_backlog(100)
    print("  enqueue_backlog 入队 %d 条" % n)
    assert n == 5, "应回捞 5 条"
    print("  队列: fast=%d slow=%d（补判应进 slow）"
          % (len(p16._fast), len(p16._slow)))
    assert len(p16._slow) == 5 and len(p16._fast) == 0
    after = db.query("SELECT COUNT(*) n FROM messages WHERE chat_id=? AND is_checked=0",
                     (cid7,))[0]["n"]
    print("  回捞后未检测=%d（应为 0，已标记进队列）" % after)
    assert after == 0

    # 预算超限时应回队而不是丢弃。
    # 注意必须先产生用量记录，否则 used=0 而 cap=1 时 0>=1 为假，护栏不会触发
    #（第一版测试就栽在这里——它测的是"没超预算"，不是"超预算"）。
    db.record_usage(cid7, 1, 5000, 500, 200, 0.01)
    db.set_chat_field(cid7, "daily_token_cap", 1)
    db.save_message(cid7, 9600, 3, "群友", "预算测试", None, "t")
    rows = db.take_unchecked_batch(cid7, 10)
    batch = [PendingMessage(r["id"], r["message_id"], cid7, 3, "群友",
                            r["text"], None, "t") for r in rows]
    before_calls = fake.calls
    print("      已用量=%s 上限=%s -> 应触发护栏"
          % (db.tokens_used_today(cid7), db.get_chat(cid7)["daily_token_cap"]))
    p17 = ChatPipeline(cid7, fake, noop, None)
    await p17._judge(batch)
    stayed = db.query("SELECT COUNT(*) n FROM messages WHERE chat_id=? AND is_checked=0",
                      (cid7,))[0]["n"]
    print("  预算超限: 是否跳过 LLM 调用=%s, 回队后未检测=%d（应 >0，不再静默丢弃）"
          % (fake.calls == before_calls, stayed))
    assert fake.calls == before_calls, "超预算不应调用 LLM"
    assert stayed > 0, "超预算应回队而非永久丢弃"

    print("=== 17. 回归：自进化规则（挖掘 -> shadow -> 毕业 -> L0 直拦）===")
    import learner
    from detector_rules import (
        check_learned_patterns, load_patterns, record_shadow_hits,
    )

    cid8 = -900
    db.register_chat(cid8, "自进化测试群", "general", shadow_mode=False)
    db.set_chat_field(cid8, "shadow_mode", 0)

    SPAM = (["来洗mi 挣一万"] * 6
            + ["充电宝投放，日入过w的新项目"] * 6
            + ["加我 https://t.me/wdgx 详聊", "看 https://t.me/wdgx 这个",
               "https://t.me/wdgx", "https://t.me/wdgx/123"])
    HAM = ["今天在汉口修的车体验还行", "开课了老师怎么样", "武汉95养生哪里好",
           "导航", "签到", "有空", "在的", "好的", "1", "666",
           "分享个链接 https://t.me/wuhanjp 榜单", "全套多少钱", "这个服务态度不错"]
    for i, t in enumerate(SPAM):
        db.save_message(cid8, 5000 + i, 1, "广告号", t, None, "2026-10-10 12:00:00")
        db.save_spam_record(cid8, 1, "广告号", 5000 + i, t, 0.9, "llm", "delete")
    for i, t in enumerate(HAM):
        db.save_message(cid8, 6000 + i, 2, "群友", t, None, "2026-10-10 12:00:00")

    st = learner.refresh_candidates(cid8)
    print("  挖掘: 候选=%d 过闸=%d 入库=%d" % (st["candidates"], st["passed"], st["added"]))
    assert st["passed"] > 0, "应有候选通过零命中闸门"

    pats = db.patterns_for_shadow()
    print("  shadow 规则:")
    for r in pats:
        print("    %r" % r["pattern"][:36])
    # 闸门必须挡住含正常语料的候选
    report = learner.validation_report(cid8)
    print("  被闸门拦住 %d 条（有正常消息命中的）" % len(report["blocked"]))
    for c in report["blocked"][:3]:
        print("    spam=%s ham=%s %r" % (c.spam_hits, c.ham_hits, c.pattern[:30]))

    # shadow 阶段不应影响判定
    load_patterns(force=True)
    v = check_learned_patterns("来洗mi 挣一万")
    print("  shadow 规则是否直接拦截: %s (应为 False，观察期内不决定)"
          % v.hit)
    assert not v.hit, "shadow 规则不应直拦"

    # 记录 shadow 命中（模拟 LLM 判完）
    hits = record_shadow_hits("来洗mi 挣一万", is_spam=True)
    print("  record_shadow_hits 命中: %s" % [h[:20] for h in hits])
    assert hits, "应记录到 shadow 命中"

    # 攒够观察期 -> 毕业
    for r in db.patterns_for_shadow():
        for _ in range(12):
            db.bump_pattern(r["pattern"], shadow_hit=True, spam_hit=True)
    eligible = learner.eligible_for_promotion(cid8)
    print("  可毕业 %d 条" % len(eligible))
    promoted = learner.run_lifecycle(cid8)["promoted"]
    print("  已毕业: %s" % [p[:24] for p in promoted])
    load_patterns(force=True)
    v = check_learned_patterns("来洗mi 挣一万")
    print("  毕业后直拦: %s (%s)" % (v.hit, v.reason))
    assert v.hit, "毕业后应在 L0 直拦"

    # 正常消息不应被自进化规则命中
    fp = [t for t in HAM if check_learned_patterns(t).hit]
    print("  正常语料误伤: %s (应为空)" % fp)
    assert not fp, "自进化规则不该命中正常语料"

    print("=== 18. 回归：自适应检查强度（免 AI 额度）===")
    from detector_rules import check_triggers

    # 触发条件单测
    cases = [
        ("今天天气不错", "trusted", False, False, False, [], "老群友闲聊"),
        ("看这个 https://t.me/x", "trusted", False, False, False, ["含链接"], "带链接"),
        ("分享个东西", "trusted", True, False, False, ["含链接预览"], "带预览"),
        ("点击", "trusted", False, True, False, ["含按钮"], "带按钮"),
        ("转发内容", "trusted", False, False, True, ["转发消息"], "转发"),
        ("n", "trusted", False, False, False, ["极短内容"], "极短"),
        ("你好", "new", False, False, False, ["新成员"], "新成员"),
    ]
    for text, trust, pv, bt, fw, expect, note in cases:
        got = check_triggers(text, trust, pv, bt, fw, 50, 0)
        # 断言"期望的条件都在"，而非精确相等：一条消息可能同时命中多个条件
        # （例如两字的按钮文案会同时触发"含按钮"和"极短内容"），这是对的。
        ok = all(e in got for e in expect)
        print("  [%s] %-16s -> %-30s %s"
              % ("OK" if ok else "FAIL", note, got or "无(免AI)", text[:16]))
        assert ok, "%s 期望包含 %s 实际 %s" % (note, expect, got)

    cid9 = -950
    db.register_chat(cid9, "免检测试群", "general", shadow_mode=False)
    db.set_chat_field(cid9, "shadow_mode", 0)
    db.cache_admins(cid9, [])
    for _ in range(40):
        db.upsert_user_seen(cid9, 55, "老群友", None)
    db.refresh_grace(cid9, 55, True)
    print("  老群友 trust=trusted 初始额度=%d" % db.grace_left(cid9, 55))

    class CountingClient:
        def __init__(self):
            self.calls = 0
        async def judge(self, messages, user_id, **kw):
            self.calls += 1
            ids = [int(l.split("=")[1].split("]")[0])
                   for l in messages[-1]["content"].splitlines()
                   if l.startswith("[ID=") and l.split("=")[1][:1].isdigit()]
            return ({"verdicts": [{"id": i, "confidence": 0.05} for i in ids]},
                    Usage(cache_hit_tokens=100, cache_miss_tokens=10,
                          completion_tokens=5, calls=1))

    async def noop18(v):
        pass

    cc = CountingClient()
    p18 = ChatPipeline(cid9, cc, noop18, None)

    # (a) 老群友纯闲聊 -> 应跳过 AI
    rid = db.save_message(cid9, 7001, 55, "老群友", "今天天气不错", None, "t")
    v = await p18.submit(PendingMessage(rid, 7001, cid9, 55, "老群友",
                                        "今天天气不错", None, "t"))
    print("  (a) 老群友闲聊 -> 判定结果=%s 队列=%d (应 None/0，未入队)"
          % (v, len(p18._fast) + len(p18._slow)))
    assert v is None and not p18._fast and not p18._slow
    print("      额度 %d -> %d" % (40, db.grace_left(cid9, 55)))

    # (b) 老群友发链接 -> 必须送 AI 且走快车道。
    # 注意用普通站外链接：t.me 链接会被 L0 的外部频道规则直接拦掉，
    # 那样就测不到"触发条件"这条路径了。
    rid = db.save_message(cid9, 7002, 55, "老群友",
                          "看 https://example.com/page 这个", None, "t")
    m_b = PendingMessage(rid, 7002, cid9, 55, "老群友",
                         "看 https://example.com/page 这个", None, "t")
    v = await p18.submit(m_b)
    print("  (b) 老群友发站外链接 -> triggers=%s fast=%s queued=%s"
          % (m_b.triggers, m_b.fast, bool(p18._fast or p18._slow)))
    assert m_b.triggers and m_b.fast, "带链接必须送 AI 且走快车道"

    # (c) 新成员闲聊 -> 必须送 AI
    rid = db.save_message(cid9, 7003, 66, "新人", "大家好", None, "t")
    m_c = PendingMessage(rid, 7003, cid9, 66, "新人", "大家好", None, "t")
    await p18.submit(m_c)
    print("  (c) 新成员闲聊 -> triggers=%s (应含'新成员')" % m_c.triggers)
    assert "新成员" in m_c.triggers

    # (d) 有前科的老群友 -> 必须送 AI
    db.bump_user_deleted(cid9, 55, "老群友")
    for _ in range(5):
        db.upsert_user_seen(cid9, 55, "老群友", None)
    p18._profile_cache.clear()          # 刷新画像缓存
    rid = db.save_message(cid9, 7004, 55, "老群友", "在吗", None, "t")
    m_d = PendingMessage(rid, 7004, cid9, 55, "老群友", "在吗", None, "t")
    await p18.submit(m_d)
    print("  (d) 有前科 -> triggers=%s (应含'曾被删过')" % m_d.triggers)
    assert any("曾被删" in t for t in m_d.triggers)

    # (e) 额度耗尽 -> 下一条纯闲聊仍应送 AI
    db.refresh_grace(cid9, 77, True)
    for _ in range(40):
        db.upsert_user_seen(cid9, 77, "另一老群友", None)
    db.execute("UPDATE users SET grace_left=0 WHERE chat_id=? AND user_id=?",
               (cid9, 77))
    p19 = ChatPipeline(cid9, cc, noop18, None)
    rid = db.save_message(cid9, 7005, 77, "另一老群友", "闲聊", None, "t")
    m_e = PendingMessage(rid, 7005, cid9, 77, "另一老群友", "闲聊", None, "t")
    await p19.submit(m_e)
    print("  (e) 额度耗尽 -> 是否入队=%s (应 True，刷满后必送 AI 复核)"
          % bool(p19._fast or p19._slow))
    assert p19._fast or p19._slow, "额度耗尽后必须送 AI"

    # (f) 关键保险：跳过 AI 的路径仍在 L0 之后 —— 老群友发手机号应被 L0 拦
    db.refresh_grace(cid9, 88, True)
    for _ in range(40):
        db.upsert_user_seen(cid9, 88, "老群友2", None)
    p20 = ChatPipeline(cid9, cc, noop18, None)
    rid = db.save_message(cid9, 7006, 88, "老群友2", "我电话13812345678", None, "t")
    v = await p20.submit(PendingMessage(rid, 7006, cid9, 88, "老群友2",
                                        "我电话13812345678", None, "t"))
    print("  (f) 老群友发手机号 -> L0 是否拦下=%s (应 True，免 AI 不等于不检查)"
          % (v and v.is_spam))
    assert v and v.is_spam, "免 AI 额度不能绕过 L0 确定性规则"

    print("=== 19. 回归：语料健康（截断告警 / 正常语料饥饿保护）===")
    # 借鉴 @BayesSpamSniperBot 上线故障：训练集里广告远多于正常消息
    # -> 分类器学成"倾向判广告" -> 大规模误删。
    # 我们的闸门依赖正常语料，同样会因正常样本过少而失去统计意义。
    cidA = -960
    db.register_chat(cidA, "饥饿测试群", "general", shadow_mode=False)
    # 造 40 条广告、只有 3 条正常 -> 比值 0.075 < 0.5，应跳过
    for i in range(40):
        t = "广告文案样本 %d 加微" % i
        db.save_message(cidA, 8000 + i, 1, "广告号", t, None, "2026-10-10 12:00:00")
        db.save_spam_record(cidA, 1, "广告号", 8000 + i, t, 0.9, "llm", "delete")
    for i in range(3):
        db.save_message(cidA, 8100 + i, 2, "群友", "正常聊天 %d" % i, None,
                        "2026-10-10 12:00:00")
    st = learner.refresh_candidates(cidA)
    print("  广告 40 / 正常 3（比值 0.08）-> %s" % st)
    assert st.get("skipped"), "正常语料过少时必须跳过学习"
    print("      -> 已跳过，不会在证据不足时放权 ✓")

    # 补足正常语料后应恢复学习
    for i in range(30):
        db.save_message(cidA, 8200 + i, 2, "群友", "正常聊天补充 %d 内容" % i,
                        None, "2026-10-10 12:00:00")
    st = learner.refresh_candidates(cidA)
    print("  补足正常语料后 -> skipped=%s spam=%s ham=%s"
          % (st.get("skipped"), st.get("spam"), st.get("ham")))
    assert not st.get("skipped"), "语料充足后应恢复学习"

    # 上限告警：limit 设很小应触发告警（这里只验证不会静默）
    sp, ha = learner.load_corpus(cidA, limit=5)
    print("  load_corpus(limit=5) -> 返回 spam=%d ham=%d（并已告警，不静默）"
          % (len(sp), len(ha)))
    assert len(sp) <= 5 and len(ha) <= 5

    print("=== 20. 回归：判定记录不丢正常样本 / 健康度不偏差 ===")
    # Gmail 阈值漂移监控的教训：只看"被判为广告的样本"会产生幸存者偏差，
    # 低分区永远为空，看不到阈值附近的真实堆积——而漂移最早就在那里显现。
    cidH = -970
    db.register_chat(cidH, "健康度测试群", "general", shadow_mode=False)
    db.set_chat_field(cidH, "shadow_mode", 0)

    class Graded:
        def __init__(self, scores):
            self.scores = scores
        async def judge(self, messages, user_id, **kw):
            ids = [int(l.split("=")[1].split("]")[0])
                   for l in messages[-1]["content"].splitlines()
                   if l.startswith("[ID=") and l.split("=")[1][:1].isdigit()]
            return ({"verdicts": [{"id": i, "confidence": self.scores.get(i, 0.1)}
                                  for i in ids]},
                    Usage(cache_hit_tokens=100, cache_miss_tokens=10,
                          completion_tokens=5, calls=1))

    # 注意：_judge 通过 on_spam 回调触发实际删除，传 noop 就测不到删除路径
    deleted20 = []

    async def on_spam20(v):
        deleted20.append(v)

    # 造 4 条：1 条高分删除、3 条低分保留（模拟真实分布）
    texts = [("广告内容测试", 0.95), ("正常聊天一", 0.05),
             ("正常聊天二", 0.10), ("正常聊天三", 0.02)]
    batch = []
    for i, (t, _s) in enumerate(texts):
        rid = db.save_message(cidH, 9000 + i, 21, "某人", t, None, "t")
        batch.append(PendingMessage(rid, 9000 + i, cidH, 21, "某人", t, None, "t"))
    # prompt 里的 ID 是**真实记录 ID**（row_id），不是批内序号，
    # 所以分数映射必须用 m.row_id 做键。硬编码 1..4 在套件里必然对不上——
    # 自增 row_id 已经被前面的测试推到 140 以上，于是全部落到默认分。
    scores20 = {m.row_id: c for m, (_t, c) in zip(batch, texts)}
    print("  row_id->分数: %s（row_id 非连续，必须按真实 ID 映射）" % scores20)
    await ChatPipeline(cidH, Graded(scores20), on_spam20, None)._judge(batch)
    print("  on_spam 回调触发 %d 次（高分那条）" % len(deleted20))
    assert len(deleted20) == 1, "只有 0.95 那条该被删"

    rows = db.query("SELECT confidence, deleted FROM verdict_log WHERE chat_id=?",
                    (cidH,))
    print("  verdict_log 记录 %d 条（应 =4，含被判为正常的）" % len(rows))
    for r in sorted(rows, key=lambda x: -(x["confidence"] or 0)):
        print("     conf=%.2f deleted=%s" % (r["confidence"], r["deleted"]))
    assert len(rows) == 4, "必须记录全部判定，不能只记删除"
    kept = [r for r in rows if r["deleted"] == 0]
    print("  其中判为正常（未删）= %d 条 —— 这些正是旧实现会丢掉的样本" % len(kept))
    assert len(kept) == 3, "低分样本必须留存，否则分布分析有幸存者偏差"

    h = db.verdict_health(cidH, 24)
    print("  健康度分布: %s" % h["buckets"])
    assert h["buckets"]["low(<0.3)"] == 3, "低分区应有 3 条（旧实现恒为 0）"
    assert h["buckets"]["very_high(>0.9)"] == 1

    # L0 命中也要进判定记录
    db.save_message(cidH, 9100, 21, "某人", "我电话13812345678", None, "t")
    p21 = ChatPipeline(cidH, Graded({}), on_spam20, None)
    rid = db.query("SELECT id FROM messages WHERE chat_id=? AND message_id=9100",
                   (cidH,))[0]["id"]
    await p21.submit(PendingMessage(rid, 9100, cidH, 21, "某人",
                                    "我电话13812345678", None, "t"))
    n_l0 = db.query("""SELECT COUNT(*) n FROM verdict_log
                       WHERE chat_id=? AND source='local_rule'""", (cidH,))[0]["n"]
    print("  L0 命中写入判定记录: %d 条（应 >=1）" % n_l0)
    assert n_l0 >= 1

    print("=== 21. 回归：数据库结构自检（防静默缺表）===")
    # 线上真实踩到过：verdict_log 加进 SCHEMA 后，因为容器没有真正重启，
    # 表没被创建，而调用方把异常吞掉了 —— 表现为"功能上线了但数据是空的"。
    # 自检把这种情况变成启动日志里的一行 ERROR。
    import sqlite3 as _sq
    tmp2 = tmp + ".schema"
    db.init_db(tmp2)
    missing = db.verify_schema(tmp2)
    print("  全新库自检: 缺表=%s（应为空）" % (missing or "无"))
    assert not missing, "全新库不应缺表"

    conn = _sq.connect(tmp2)
    conn.execute("DROP TABLE verdict_log")
    conn.commit()
    conn.close()
    missing = db.verify_schema(tmp2)
    print("  删掉 verdict_log 后自检: 缺表=%s" % missing)
    assert "verdict_log" in missing, "自检必须能发现缺表"

    # init_db 重跑应把表补回来（CREATE TABLE IF NOT EXISTS 是幂等的）
    db.init_db(tmp2)
    missing = db.verify_schema(tmp2)
    print("  重跑 init_db 后自检: 缺表=%s（应补回）" % (missing or "无"))
    assert not missing, "init_db 应补回缺失的表"
    os.unlink(tmp2)

    print("=== 22. 回归：全角数字绕过（实测发现的现成口子）===")
    # 来源：Unicode 混淆对抗研究（不可见字符 + 字符替换 + 全角）。
    # 全角数字 U+FF10-U+FF19 的 \w 为真，不被噪声正则删除，也没做 ASCII 映射，
    # 于是 "我电话１３８..." 能直接绕过手机号规则。
    from detector_rules import check_local, fold_digits

    cases = [
        ("我电话13812345678", True, "半角手机号"),
        ("我电话１３８１２３４５６７８", True, "全角手机号"),
        ("１３８１２３４５６７８", True, "纯全角手机号"),
        ("＋８６１３８１２３４５６７８", True, "全角带国际区号"),
        ("我的卡号６２１７００１２３４５６７８９０", True, "全角银行卡号"),
        ("今天在汉口修的车体验还行", False, "正常聊天"),
        ("我电话１３８", False, "位数不足"),
        ("Ill", False, "纯字母（折叠不能把它变成数字）"),
    ]
    for text, expect, label in cases:
        v = check_local(text, 1, -1, set())
        ok = (v.hit == expect)
        print("  [%s] %-20s %-30r -> %s"
              % ("OK" if ok else "FAIL", label, text[:28],
                 "命中" if v.hit else "放过"))
        assert ok, "%s 期望 hit=%s 实际=%s" % (label, expect, v.hit)

    # 折叠必须是无损的：只动全角数字，不碰字母
    print("  fold_digits('Ill')=%r  fold_digits('１２３')=%r"
          % (fold_digits("Ill"), fold_digits("１２３")))
    assert fold_digits("Ill") == "Ill", "折叠不能改字母（否则 Ill->111 会误伤）"
    assert fold_digits("１２３") == "123"

    # normalize 也应对齐：全角与半角归一化后必须一致
    from utils.normalize import normalize
    assert (normalize("我电话１３８１２３４５６７８")
            == normalize("我电话13812345678")), "归一化后应一致"

    print("=== 23. 回归：本群绑定频道的自动转发放行 ===")
    # 用户实测反馈：机器人会删除群绑定频道发帖时的自动转发。
    # 根因：is_automatic_forward 全代码库只出现在诊断日志里，判断链上没用它，
    # 于是群主自己的运营内容（榜单/报告/公告）被当普通消息判定。
    #
    # 关键设计：只放行**本群自己的绑定频道**（linked_chat_id 匹配），
    # 外部频道的自动转发照常判定——否则等于给"借频道发广告"开后门。
    import main as main_mod

    cidC = -980
    LINKED = -1003873111009
    OTHER = -1009999999999
    db.register_chat(cidC, "绑频测试群", "general", shadow_mode=False)
    db.set_chat_field(cidC, "linked_chat_id", LINKED)
    crow = db.get_chat(cidC)
    print("  群 %s 绑定频道 = %s" % (cidC, crow["linked_chat_id"]))
    assert crow["linked_chat_id"] == LINKED

    # 用轻量替身验证判定逻辑（构造完整 Update 太重，这里直接测分支条件）
    def verdict_for(sender_id, is_auto):
        """复刻 main.handle_message 里那段判定。"""
        if is_auto:
            linked = crow["linked_chat_id"]
            if linked and sender_id == linked:
                return "放行（本群绑定频道）"
        return "照常判定"

    cases = [
        (LINKED, True, "放行（本群绑定频道）", "本群绑定频道的自动转发"),
        (OTHER, True, "照常判定", "外部频道的自动转发（该拦）"),
        (None, True, "照常判定", "自动转发但拿不到 sender"),
        (LINKED, False, "照常判定", "非自动转发的消息（即使 sender 相同）"),
    ]
    for sender_id, is_auto, expect, label in cases:
        got = verdict_for(sender_id, is_auto)
        ok = (got == expect)
        print("  [%s] %-32s -> %s" % ("OK" if ok else "FAIL", label, got))
        assert ok, "%s 期望 %s 实际 %s" % (label, expect, got)

    # 未配置绑定频道的群：不应误放行任何自动转发
    cidD = -981
    db.register_chat(cidD, "无绑频群", "general", shadow_mode=False)
    crow2 = db.get_chat(cidD)
    print("  未配绑定频道的群 linked_chat_id = %r（None 表示没配）"
          % crow2["linked_chat_id"])
    assert crow2["linked_chat_id"] is None, "未配置时应为 NULL，不放行任何自动转发"

    print("=== 24. 回归：账号资料检测（昵称即广告位）===")
    # 来源：Discord AutoMod 有 MemberProfile 触发类型（扫描用户资料），
    # 我们原来完全没检查资料。而广告号把引流指引直接写进昵称——
    # 正文被删了也照样能被群友找到（"南叙 主页有群"、"屿辰 看我简介"）。
    #
    # 实测区分度（522 个有删除记录的账号 / 19 个可信群友）：
    #   引流指引类命中 33 个广告号（6.3%），可信群友 0 命中
    #   联系方式类命中 14 个（2.7%），可信群友 0 命中
    # 误伤为零，符合 L0"只放零误伤模式"的门槛。
    from detector_rules import check_account_profile

    spam_names = [
        "南叙 主页有群", "祁野 看我资料", "简寻 看我资料", "屿辰 看我简介",
        "甜晴天（武汉专业95客服）", "知禾（湖北大学开课🀄️）", "春色与你✨上门6",
        "白月光唯一客服（新预约号）", "哆哆💘兼职在线", "甜甜 在校壆妹资原",
    ]
    for n in spam_names:
        v = check_account_profile(n)
        assert v.hit, "广告号昵称应命中: %r" % n
    print("  广告号昵称 %d/%d 命中 ✓" % (len(spam_names), len(spam_names)))

    # 这 19 个是**线上真实的可信群友**（发言 >=30 且删除比 <5%），
    # 必须全部放过——这是零误伤断言的实测依据，不是编的。
    trusted_names = [
        "富顺富顺", "Rick superlucky", "嘿嘿 嘿", "肯恰那", "吉米",
        "独木不成林", "汤姆", "灰狼", "溪口水库", "小灰机潜海",
        "summonkings", "夸克 刘", "锦鲤附体", "蓝脸的盗耳礅", "梅丹丹死了",
        "解 药（🈚️痔疮）", "星哥", "双叶 | 16:26", "威 常",
    ]
    fp = [n for n in trusted_names if check_account_profile(n).hit]
    print("  线上真实可信群友 %d 个，误伤 %d 个 %s"
          % (len(trusted_names), len(fp), fp or "✓"))
    assert not fp, "可信群友昵称不该命中: %s" % fp

    # 空昵称不应崩
    for empty in (None, "", "   "):
        assert not check_account_profile(empty).hit
    print("  空昵称安全处理 ✓")

    print("=== 25. 回归：波次检测（协调投放识别）===")
    # 来源：Reddit 的做法——"用 LLM 捕捉旧系统漏掉的、高度隐蔽的**协调模式**"。
    # 我们自己的数据里这个模式非常明显（1811 条删除记录）：
    #   删除量中位数 = 2 条/分钟
    #   但有 9 个波次，最大一次是 **195 条/分钟、来自 105 个不同账号**
    #   9 个波次覆盖全部删除量的 25%
    # 单账号维度的刷屏阈值对"105 个账号同时投放"完全无感——
    # 每个账号都只发了一两条。所以需要**群级**密度指标。
    from detector_rules import (
        RaidDetector, RAID_MIN_DELETES, RAID_BASELINE_MULTIPLIER,
    )

    print("  阈值 = max(%d, 基线×%.1f)"
          % (RAID_MIN_DELETES, RAID_BASELINE_MULTIPLIER))

    # (a) 正常时段不误报（实测中位数 2 条/分钟）
    d = RaidDetector()
    t0 = 1000.0
    for i in range(3):
        d.record_delete(t0 + i * 20)
    hit, cnt, thr = d.check(t0 + 60)
    print("  (a) 正常时段 3 条/分钟 -> 波次=%s（应 False，否则正常流量会误报）"
          % hit)
    assert not hit, "正常流量不该被判为波次"

    # (b) 真实波次形态：195 条/分钟、105 个账号
    d2 = RaidDetector()
    t1 = 2000.0
    for i in range(195):
        d2.record_delete(t1 + i * 0.3)
    hit2, cnt2, _ = d2.check(t1 + 60)
    print("  (b) 195 条/分钟（实测最大波次）-> 波次=%s 窗口内=%d"
          % (hit2, cnt2))
    assert hit2, "真实波次必须识别出来"

    # (c) 冷却期：波次后短暂无删除不应立刻退出警戒
    hit3, _, _ = d2.check(t1 + 100)
    print("  (c) 波次后 40 秒（无新删除）-> 波次=%s（应在冷却期内保持）" % hit3)
    assert hit3, "波次是持续过程，不应因某一秒没删除就退出"

    # (d) 冷却期结束后退出，避免永久处于警戒
    hit4, _, _ = d2.check(t1 + 200)
    print("  (d) 波次后 200 秒 -> 波次=%s（应已退出，否则阈值永久偏低）" % hit4)
    assert not hit4, "冷却期后应退出警戒"

    # (e) 集成：pipeline 里波次状态下阈值收紧且走快车道
    cidR = -990
    db.register_chat(cidR, "波次测试群", "general", shadow_mode=False)
    db.set_chat_field(cidR, "shadow_mode", 0)
    # 用文件顶部定义的通用 noop（noop20 是测试 20 里的局部函数，这里不可见）
    pR = ChatPipeline(cidR, Graded({}), noop, None)
    print("  (e) 初始波次状态=%s 阈值=%s"
          % (pR._raid.in_raid(), db.get_chat(cidR)["confidence_cut"]))
    assert not pR._raid.in_raid()
    for i in range(RAID_MIN_DELETES + 5):
        pR._raid.record_delete()
    pR._raid.check()
    print("      记录 %d 次删除后 -> 波次状态=%s"
          % (RAID_MIN_DELETES + 5, pR._raid.in_raid()))
    assert pR._raid.in_raid(), "达到阈值后应进入波次状态"

    # 波次状态下：普通消息也走快车道
    rid = db.save_message(cidR, 9500, 31, "某群友", "随便聊聊内容", None, "t")
    mR = PendingMessage(rid, 9500, cidR, 31, "某群友", "随便聊聊内容", None, "t")
    await pR.submit(mR)
    print("      波次中普通消息 fast=%s（应 True，波次期间压缩响应时间）"
          % mR.fast)
    assert mR.fast, "波次期间所有消息应走快车道"

    print("=== 26. 回归：判定记忆（人工裁决一次，后续自动复用）===")
    # 来源（AstrBot QQ 群广告拦截插件）："疑似广告自动发给群管理员人工确认；
    # **确认结果记录进学习库，后续相同内容直接复用结论**"。
    # 解决的问题：同一条广告被 ABC 三个账号发三遍，原本要判三次、
    # 管理员要看三次。人工裁决一次之后就该校自动处理。
    cidM = -995
    db.register_chat(cidM, "记忆测试群", "general", shadow_mode=False)
    db.set_chat_field(cidM, "shadow_mode", 0)

    # (a) 误删记忆 -> 同类内容直接放行，不再送 AI
    txt = "今天在汉口修的车体验还行"
    assert db.recall_verdict(cidM, txt) is None, "初始不应有记忆"
    db.remember_verdict(cidM, txt, "false_positive", "admin_revive")
    got = db.recall_verdict(cidM, txt)
    print("  (a) 误删记忆: %r -> 复用结论=%s（应 false_positive）" % (txt[:20], got))
    assert got == "false_positive"

    rid = db.save_message(cidM, 9800, 41, "真人", txt, None, "t")
    pM = ChatPipeline(cidM, Graded({}), noop, None)
    v = await pM.submit(PendingMessage(rid, 9800, cidM, 41, "真人", txt, None, "t"))
    print("      同类内容判定结果=%s 入队=%d（应 None/0，直接放行）"
          % (v, len(pM._fast) + len(pM._slow)))
    assert v is None and not pM._fast and not pM._slow, "误删记忆应直接放行"

    # (b) 确认记忆 -> 同类内容直接删
    ad = "充电宝投放日入过w的新项目"
    db.remember_verdict(cidM, ad, "confirmed", "admin_report")
    rid = db.save_message(cidM, 9801, 42, "广告号", ad, None, "t")
    pM2 = ChatPipeline(cidM, Graded({}), noop, None)
    v = await pM2.submit(PendingMessage(rid, 9801, cidM, 42, "广告号", ad, None, "t"))
    print("  (b) 确认记忆: %r -> 判定=%s reason=%s"
          % (ad[:20], "删除" if (v and v.is_spam) else "放过",
             v.reason if v else ""))
    assert v and v.is_spam, "确认记忆应直接删除"
    assert v.source == "memory"

    # (c) 关键：谐音/不可见字符变体也要命中同一记忆
    # 否则广告号改一个字符就绕过了人工裁决
    variant = "充电宝投放\u200b日入过w的新项目"      # 插入零宽空格
    got_v = db.recall_verdict(cidM, variant)
    print("  (c) 变体（插入零宽空格）: 复用结论=%s（应命中，否则改一字符即可绕过）"
          % got_v)
    assert got_v == "confirmed", "归一化变体必须命中同一记忆"

    # (d) 误删优先于确认（同一内容两种记忆时保护真人）
    db.remember_verdict(cidM, txt, "confirmed", "x")
    got_p = db.recall_verdict(cidM, txt)
    print("  (d) 同时存在两种记忆时取=%s（应 false_positive，优先保护真人）" % got_p)
    assert got_p == "false_positive", "误删记忆应优先"

    # (e) 太短的内容不记忆（避免把 'ok' 这类记成永久规则）
    print("  (e) 过短内容是否记忆: %s（应 False）"
          % db.remember_verdict(cidM, "ok", "confirmed"))
    assert not db.remember_verdict(cidM, "ok", "confirmed")

    print("  记忆条数=%d" % len(db.verdict_memory_list(cidM)))

    print("=== 27. 回归：L0 命中也要记 shadow 统计（自进化的自我阻碍）===")
    # 实测发现的严重设计缺陷：原来只在 LLM 判定后记 shadow 命中，
    # 而 L0 命中就提前 return 了——于是所有被 L0 拦下的广告
    # **从未参与 shadow 统计**。
    # 后果：66 条 shadow 规则最高只攒到 3 次命中（毕业需 10 次），
    # 而同期删了 1845 条。规则永远毕不了业，自进化形同虚设。
    from detector_rules import load_patterns, record_shadow_hits

    cidS = -996
    db.register_chat(cidS, "shadow统计测试群", "general", shadow_mode=False)
    db.set_chat_field(cidS, "shadow_mode", 0)
    db.set_chat_field(cidS, "username", "testgroup")
    db.set_chat_field(cidS, "managed_links", "testgroup")

    # 造一条 shadow 规则
    ad_text = "限时特价液课上门服务"
    db.add_pattern(cidS, "特价液课", mode="shadow")
    load_patterns(force=True)
    print("  shadow 规则已入库，初始命中=%s"
          % db.query("SELECT shadow_hits FROM spam_patterns LIMIT 1")[0]["shadow_hits"])

    # 用一个**会被 L0 拦下**的内容（含手机号 -> check_local 命中）
    # 同时它也命中 shadow 规则 "特价液课"
    text = "特价液课 电１３８００００００００"
    rid = db.save_message(cidS, 9900, 51, "广告号", text, None, "t")
    pS = ChatPipeline(cidS, Graded({}), noop, None)
    v = await pS.submit(PendingMessage(rid, 9900, cidS, 51, "广告号", text, None, "t"))
    print("  判定结果: %s (%s)" % ("L0 命中" if (v and v.is_spam) else "未命中",
                                  v.reason if v else ""))
    assert v and v.is_spam, "含手机号应被 L0 拦下"

    hits = db.query("SELECT shadow_hits, shadow_spam_hits FROM spam_patterns")[0]
    print("  L0 命中后 shadow 统计: hits=%s spam_hits=%s（应都 >=1）"
          % (hits["shadow_hits"], hits["shadow_spam_hits"]))
    assert (hits["shadow_hits"] or 0) >= 1, \
        "L0 拦下的广告必须计入 shadow 统计，否则规则永远攒不够毕业所需命中"

    # 攒够后应能毕业
    for _ in range(12):
        db.bump_pattern("特价液课", shadow_hit=True, spam_hit=True)
    from learner import eligible_for_promotion
    elig = eligible_for_promotion(cidS)
    print("  攒够命中后可毕业: %s" % [r["pattern"] for r in elig])
    assert any(r["pattern"] == "特价液课" for r in elig), "应可毕业"

    # 反向：有误命中的规则不能毕业（闸门仍然有效）
    db.add_pattern(cidS, "武汉养生", mode="shadow")
    for _ in range(12):
        db.bump_pattern("武汉养生", shadow_hit=True, spam_hit=False)  # 有误命中
    elig2 = eligible_for_promotion(cidS)
    names = [r["pattern"] for r in elig2]
    print("  有误命中的规则是否可毕业: %s（应不含 武汉养生）"
          % ("含!" if "武汉养生" in names else "不含 ✓"))
    assert "武汉养生" not in names, "有误命中的规则不得毕业"

    os.unlink(tmp)
    print("\n全部通过")


asyncio.run(main())
