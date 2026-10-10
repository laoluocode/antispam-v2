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

    os.unlink(tmp)
    print("\n全部通过")


asyncio.run(main())
