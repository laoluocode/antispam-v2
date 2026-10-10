"""检测流水线：攒批、分级路由、判定执行。

三层响应速度，按用户风险分流（这是"露头秒"的关键，判定器本身快慢无关紧要）：

| 层 | 响应 | 成本 | 覆盖 |
|---|---|---|---|
| L0 确定性规则 | ~0ms | 0 | 不可见字符混淆、手机号、t.me、黑名单、重复刷屏 |
| L1 快车道 | ~15s | 低 | 新人 / 惯犯 / 高风险用户的消息 |
| L2 常规 | ~90s | 极低 | 其余流量 |

**为什么按用户风险分流而不是全部实时**：真正会秒发秒撤的是垃圾号，而垃圾号的
特征就是"新"或"惯犯"。老用户偶尔发广告不会盯着屏幕秒撤。
所以 L1 只覆盖高风险流量，却抓住了绝大多数需要秒删的消息。

**成本为什么可以接受**：稳定前缀命中缓存后，小请求的边际成本几乎为零。
实测一次调用 3,214 miss + 42,621 hit token；单条实时只是在未命中部分加了
约 40 token。所以"每条实时"约 7 元/月，"按 50 条攒批"约 4 元/月——差距很小，
不是此前设想的 50 倍。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import db
from deepseek import DeepSeekClient, DeepSeekError, estimate_cost_yuan
from detector_rules import (
    INVISIBLE_MIN_LENGTH,
    INVISIBLE_RATIO_THRESHOLD,
    check_bare_letter_spam,
    check_external_channel_link,
    check_inline_buttons,
    check_local,
    check_repeat,
    compute_trust,
    trust_marker,
)
from link_preview import has_link_preview, preview_fingerprint
from prompts import BATCH_INSTRUCTION, GLOBAL_PROMPT, build_fewshots, build_group_profile
from utils.normalize import invisible_ratio

logger = logging.getLogger(__name__)

# L1 快车道触发条件：这些用户的消息优先送判定
FASTLANE_TRUST_LEVELS = {"new", "repeat"}

# 重复检测看多少条近期消息
REPEAT_WINDOW = 8
REPEAT_THRESHOLD = 3

# 付费商家（advertiser 档）判定为删除所需的最低置信度。
# 他们付了钱，招揽性质的广告应放过；但诈骗/黑产/外部产品推销照样要删，
# 否则白名单就变成了投毒通道。取 0.85 与严格档的"铁证"线对齐。
ADVERTISER_CUT = 0.85

# 数据库查询的短 TTL 缓存。避免每条消息都打一次库。
PROFILE_TTL = 30.0
ADMIN_TTL = 300.0

# 最小批大小门槛：0 表示不设门槛，到时间窗就发。
#
# 背景：实测发现每次调用要重发约 1,685 token 的稳定前缀，而平均批只有 2 条，
# 导致 81% 的 token 花在固定开销上。设门槛能把单条摊薄成本降约 10 倍。
#
# 但**实时性优先时可关掉**：门槛让快车道从"最多等 15 秒"变成"最多等攒够 20 条"，
# 在低流量群可能等很久。当前按"实时性优先"配置：快车道 0（15 秒必发），
# 慢车道保留小额门槛避免过度调用。
MIN_BATCH_FAST = 0    # 0 = 不设门槛，快车道严格按窗口触发
MIN_BATCH_SLOW = 40

# 协同刷屏阈值：同一预览目标在 24h 内被多少个不同账号引用。
# 实测案例：两个账号交叉发单字母，指向同一个频道预览。
# 正常群友分享同一链接的账号数远低于此。
PREVIEW_SENDER_THRESHOLD = 3


@dataclass
class PendingMessage:
    """一条待判定的消息。"""

    row_id: int
    message_id: int
    chat_id: int
    user_id: int
    full_name: str
    text: str
    reply_to: int | None
    timestamp: str

    # 判定前算好的信号
    trust: str = "normal"
    history: list[str] = field(default_factory=list)
    fast: bool = False
    whitelist: str = ""      # "" | "advertiser"（trusted 的消息不会入队）
    report_id: int = 0       # >0 表示这条来自举报，判定后需回填举报状态
    origin: str = "group"    # group | forward（转发举报无法删除原消息）
    # 链接预览：广告号用"短文本 + 频道预览"引流，正文 Telegram 不下发，
    # 但预览目标（URL）是下发的，可作为协同刷屏的比对依据
    preview_fp: str = ""
    preview_url: str = ""
    has_preview: bool = False
    # inline 按钮：广告号把广告做成按钮，正文可以只是一个表情
    button_texts: list[str] = field(default_factory=list)
    button_urls: list[str] = field(default_factory=list)

    @property
    def marker(self) -> str:
        return trust_marker(self.trust)


@dataclass
class Verdict:
    """判定结果。"""

    message: PendingMessage
    is_spam: bool
    confidence: float
    source: str
    reason: str = ""


class ChatPipeline:
    """单个群的检测流水线。每个群一个实例，互不干扰。"""

    def __init__(self, chat_id: int, client: DeepSeekClient,
                 on_spam, notify=None, on_log=None) -> None:
        self.chat_id = chat_id
        self.client = client
        self.on_spam = on_spam          # async (Verdict) -> None，由 main 注入执行删除
        self.notify = notify            # async (list[Verdict]) -> None，通知管理员
        self.on_log = on_log            # async (list[Verdict], shadow) -> None，写记录频道
        # 判定完成后需要回填的举报：row_id -> report_id
        self._pending_reports: dict[int, int] = {}
        self._fast: deque[PendingMessage] = deque()
        self._slow: deque[PendingMessage] = deque()
        self._lock = asyncio.Lock()
        self._profile_cache: dict[int, tuple[float, dict, str]] = {}
        self._history_cache: dict[int, tuple[float, list[str]]] = {}
        self._admin_cache: tuple[float, set[int]] = (0.0, set())
        self._managed_cache: tuple[float, set[str]] = (0.0, set())
        self.last_error: str | None = None

    # -- 配置读取 ---------------------------------------------------------

    def _cfg(self) -> dict:
        row = db.get_chat(self.chat_id)
        if not row:
            return {
                "strictness": "strict", "confidence_cut": 0.5, "shadow_mode": 1,
                "fastlane_seconds": 15, "batch_seconds": 90, "batch_max": 100,
                "daily_token_cap": 300000, "chat_nature": "general",
                "notify_admin": 1, "title": "",
            }
        return dict(row)

    # -- 用户信号 ---------------------------------------------------------

    async def _profile(self, user_id: int, full_name: str) -> tuple[dict, str]:
        """取 (用户统计, 信任档位)。带 TTL 缓存。"""
        now = time.time()
        cached = self._profile_cache.get(user_id)
        if cached and cached[0] > now:
            return cached[1], cached[2]
        stats = await asyncio.to_thread(db.user_stats, self.chat_id, user_id)
        trust = compute_trust(stats["msg_n"], stats["del_n"], stats["span_days"])
        self._profile_cache[user_id] = (now + PROFILE_TTL, stats, trust)
        return stats, trust

    async def _history(self, user_id: int) -> list[str]:
        now = time.time()
        cached = self._history_cache.get(user_id)
        if cached and cached[0] > now:
            return cached[1]
        texts = await asyncio.to_thread(
            db.recent_user_texts, self.chat_id, user_id, REPEAT_WINDOW
        )
        self._history_cache[user_id] = (now + PROFILE_TTL, texts)
        return texts

    async def _managed_links(self) -> set[str]:
        """本群体系的频道标识。缓存 10 分钟——它只在配置变更时才会变。"""
        now = time.time()
        if self._managed_cache[0] > now:
            return self._managed_cache[1]
        names = await asyncio.to_thread(db.managed_links, self.chat_id)
        self._managed_cache = (now + 600, names)
        return names

    async def _is_admin(self, user_id: int) -> bool:
        now = time.time()
        if self._admin_cache[0] > now:
            return user_id in self._admin_cache[1]
        ids = await asyncio.to_thread(db.cached_admin_ids, self.chat_id)
        self._admin_cache = (now + ADMIN_TTL, ids)
        return user_id in ids

    # -- 入口：消息到达 ---------------------------------------------------

    async def submit(self, msg: PendingMessage) -> Verdict | None:
        """处理一条消息。返回 L0 立即判定结果，或 None（已入队等 LLM）。"""
        cfg = self._cfg()
        if not cfg:
            return None

        # 管理员免检
        if await self._is_admin(msg.user_id):
            logger.debug("跳过管理员 chat=%s user=%s", self.chat_id, msg.user_id)
            await asyncio.to_thread(db.mark_checked, [msg.row_id])
            return None

        stats, trust = await self._profile(msg.user_id, msg.full_name)
        msg.trust = trust

        # 白名单：trusted 完全跳过，advertiser 只删高置信
        level = await asyncio.to_thread(db.whitelist_level, self.chat_id, msg.user_id)
        if level == "trusted":
            logger.info("白名单(trusted)跳过 chat=%s user=%s", self.chat_id, msg.user_id)
            await asyncio.to_thread(db.mark_checked, [msg.row_id])
            return None
        msg.whitelist = level or ""

        # --- L0 确定性规则 ---
        blacklist = await asyncio.to_thread(db.blacklist_ids, self.chat_id)
        v = check_local(msg.text, msg.user_id, self.chat_id, blacklist)
        if not v.hit:
            # 以下检测都需要该用户的近期发言
            history = await self._history(msg.user_id)
            if history:
                v = check_repeat(msg.text, history, REPEAT_THRESHOLD)
            # 单字符刷屏：广告号用裸字母当载体，广告内容在频道预览里，
            # bot 只能拿到字母本身（实测漏判案例）。
            # 受信任群友豁免：真实群友之间偶尔发几个 ok/hi 很正常，
            # 而这个规则是"宁可严格"型的行为特征，用信任档位兜住误伤。
            if not v.hit and trust != "trusted":
                v = check_bare_letter_spam(msg.text, history)

        if v.hit:
            await asyncio.to_thread(db.mark_checked, [msg.row_id])
            return Verdict(msg, True, 1.0, "local_rule", v.reason)

        # --- inline 按钮检测 ---
        # 广告号把广告做成按钮（"加入频道｜领取福利"），正文可以完全干净。
        if msg.button_texts or msg.button_urls:
            bv = check_inline_buttons(msg.button_texts, msg.button_urls, managed)
            if bv.hit:
                await asyncio.to_thread(db.mark_checked, [msg.row_id])
                return Verdict(msg, True, 1.0, "local_rule", bv.reason)

        # --- 外部频道链接检测 ---
        # 这些群有"集团"式频道体系，群友转发自家频道是常态。广告号借这个习惯
        # 发**外部频道**的帖子：正文 Telegram 不下发（bot 只看到单字母），
        # 但链接指向哪里是可见的。指向陌生目标即引流。
        managed = await self._managed_links()
        ev = check_external_channel_link(msg.text, managed)
        if ev.hit:
            await asyncio.to_thread(db.mark_checked, [msg.row_id])
            return Verdict(msg, True, 1.0, "local_rule", ev.reason)

        # --- 链接预览协同检测 ---
        # 广告正文藏在下发不到的预览卡片里，但"多个账号指向同一预览目标"
        # 是协同引流的确凿特征，且广告号绕不过去——它必须让群友看到广告。
        if msg.preview_fp:
            senders = await asyncio.to_thread(
                db.record_preview, self.chat_id, msg.preview_fp,
                msg.preview_url, msg.user_id)
            if senders >= PREVIEW_SENDER_THRESHOLD:
                await asyncio.to_thread(db.mark_checked, [msg.row_id])
                return Verdict(
                    msg, True, 1.0, "local_rule",
                    f"协同引流（{senders} 个账号指向同一预览目标）")

        # --- 分流 ---
        cut = cfg["confidence_cut"]
        # 不可见字符占比高：广告号的强特征，插队到快车道
        suspicious = (len(msg.text) >= INVISIBLE_MIN_LENGTH
                      and invisible_ratio(msg.text) >= INVISIBLE_RATIO_THRESHOLD)
        msg.fast = trust in FASTLANE_TRUST_LEVELS or suspicious
        if level == "advertiser":
            msg.fast = False  # 付费商家不抢快车道资源

        msg.history = (await self._history(msg.user_id))[:5]

        async with self._lock:
            (self._fast if msg.fast else self._slow).append(msg)
        return None

    async def enqueue_backlog(self, limit: int = 200) -> int:
        """把库里未检测的消息回捞进队列。

        为什么需要：流水线只处理 submit() 推进来的实时消息。一旦出现
        "消息已标记 is_checked=1 但没过 LLM"（例如早先预算超限时被直接丢弃），
        它们既不在队列里、也不会被重新取——永久静默丢失。
        这个方法提供补判入口，也让运维可以手工回捞历史消息。

        返回实际入队的条数。
        """
        rows = await asyncio.to_thread(db.take_unchecked_batch, self.chat_id, limit)
        if not rows:
            return 0
        for r in rows:
            msg = PendingMessage(
                row_id=r["id"], message_id=r["message_id"], chat_id=r["chat_id"],
                user_id=r["user_id"], full_name=r["full_name"] or "",
                text=r["text"] or "", reply_to=r["reply_to"],
                timestamp=str(r["timestamp"] or ""),
            )
            # 补判走常规队列：这些是历史消息，不需要抢实时性
            async with self._lock:
                self._slow.append(msg)
        logger.info("chat=%s 回捞 %d 条未检测消息进入补判队列",
                    self.chat_id, len(rows))
        return len(rows)

    # -- 攒批循环 ---------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        """常驻循环：分别按各自的窗口排空两个队列。"""
        cfg = self._cfg()
        fast_win = max(3, int(cfg.get("fastlane_seconds", 15)))
        slow_win = max(10, int(cfg.get("batch_seconds", 90)))

        fast_deadline = time.time() + fast_win
        slow_deadline = time.time() + slow_win

        while not stop.is_set():
            now = time.time()
            async with self._lock:
                n_fast, n_slow = len(self._fast), len(self._slow)

            # 触发条件：攒够最小批 或 到时间窗。
            # 门槛为 0 时退化为纯粹的窗口触发（实时性优先）。
            fire_fast = ((MIN_BATCH_FAST > 0 and n_fast >= MIN_BATCH_FAST)
                         or (now >= fast_deadline and n_fast))
            fire_slow = ((MIN_BATCH_SLOW > 0 and n_slow >= MIN_BATCH_SLOW)
                         or (now >= slow_deadline and n_slow))

            if fire_fast or fire_slow:
                batch: list[PendingMessage] = []
                async with self._lock:
                    if fire_fast:
                        batch += list(self._fast)
                        self._fast.clear()
                    if fire_slow:
                        batch += list(self._slow)
                        self._slow.clear()
                if batch:
                    await self._judge(batch)
                if fire_fast:
                    fast_deadline = time.time() + fast_win
                if fire_slow:
                    slow_deadline = time.time() + slow_win

            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    # -- 判定 -------------------------------------------------------------

    async def _judge(self, batch: list[PendingMessage]) -> None:
        cfg = self._cfg()
        if not cfg:
            return

        # 预算护栏。
        # ⚠️ 早先的实现在这里直接 return，而这些消息**已经被标记 is_checked=1**
        # 且不会重试 —— 等于永久静默丢失。实测线上有 2,370 条消息是这样被吞掉的，
        # 其中混着真广告。现在改为：回队 + 告警（有重试上限兜底，不会无限堆积）。
        used = await asyncio.to_thread(db.tokens_used_today, self.chat_id)
        cap = int(cfg.get("daily_token_cap") or 0)
        # cap <= 0 表示不限量。测试期用它观察真实日消费，之后再定阈值。
        if cap > 0 and used >= cap:
            keep = await asyncio.to_thread(
                db.bump_retry, [m.row_id for m in batch])
            logger.warning(
                "chat=%s 已达每日 token 预算 %s/%s，本批 %d 条回队（%d 条将继续重试）",
                self.chat_id, used, cap, len(batch), len(keep))
            if keep:
                await asyncio.to_thread(
                    self._requeue, [m for m in batch if m.row_id in set(keep)])
            return

        messages = self._build_prompt(batch)
        try:
            parsed, usage = await self.client.judge(
                messages, user_id=f"chat_{self.chat_id}"
            )
        except DeepSeekError as exc:
            self.last_error = str(exc)
            keep = await asyncio.to_thread(
                db.bump_retry, [m.row_id for m in batch])
            logger.error("chat=%s 判定失败（本批 %d 条，%d 条将重试）: %s",
                         self.chat_id, len(batch), len(keep), exc)
            # 只对未超过重试上限的消息回队；超过的已在 bump_retry 里标记完成，
            # 否则 API 持续异常会让同一批消息每轮重复烧 token。
            if keep:
                await asyncio.to_thread(
                    self._requeue, [m for m in batch if m.row_id in set(keep)])
            return

        cost = estimate_cost_yuan(usage)  # 按时段自动判断
        await asyncio.to_thread(
            db.record_usage, self.chat_id, usage.calls,
            usage.cache_hit_tokens, usage.cache_miss_tokens,
            usage.completion_tokens, cost,
        )
        logger.info(
            "chat=%s 判定 %d 条：缓存命中 %.0f%%，成本 %.6f 元",
            self.chat_id, len(batch), usage.cache_hit_rate * 100, cost,
        )

        scores = {int(v["id"]): float(v.get("confidence", 0.0))
                  for v in parsed.get("verdicts", [])
                  if "id" in v}
        # 判定成功的部分立即标记已检测。
        # 漏掉这一步会让消息永远停在未检测状态，每轮被重新取出来判定。
        judged = [m.row_id for m in batch if m.row_id in scores]
        await asyncio.to_thread(db.mark_checked, judged)

        missing = [m for m in batch if m.row_id not in scores]
        if missing:
            keep = await asyncio.to_thread(db.bump_retry, [m.row_id for m in missing])
            logger.warning("chat=%s 有 %d 条缺判定（%d 条将重试）",
                           self.chat_id, len(missing), len(keep))
            if keep:
                await asyncio.to_thread(
                    self._requeue, [m for m in missing if m.row_id in set(keep)])

        cut = float(cfg.get("confidence_cut", 0.5))
        shadow = bool(cfg.get("shadow_mode", 1))

        verdicts: list[Verdict] = []
        for m in batch:
            conf = scores.get(m.row_id)
            if conf is None:
                continue
            # 付费商家用更高的门槛：只删"铁证"级广告（诈骗/黑产/外部推销），
            # 放过他们付费购买的招揽权。否则白名单等于投毒通道。
            threshold = max(cut, ADVERTISER_CUT) if m.whitelist == "advertiser" else cut
            is_spam = conf >= threshold
            verdicts.append(Verdict(m, is_spam, conf, "llm"))

        # 回填举报状态：举报要闭环，否则举报人看不到结果、也没法统计准确率
        for m in batch:
            rid = m.report_id or self._pending_reports.pop(m.row_id, 0)
            if not rid:
                continue
            conf = scores.get(m.row_id)
            if conf is None:
                continue
            threshold = (max(cut, ADVERTISER_CUT)
                         if m.whitelist == "advertiser" else cut)
            await asyncio.to_thread(
                db.mark_report_reviewed, rid,
                "delete" if conf >= threshold else "dismiss",
            )
            if conf >= threshold:
                await asyncio.to_thread(
                    db.add_fewshot, self.chat_id, "positive", m.text, "report")

        spam = [v for v in verdicts if v.is_spam]
        if not spam:
            return

        for v in spam:
            action = "observe" if shadow else "delete"
            await asyncio.to_thread(
                db.save_spam_record, self.chat_id, v.message.user_id,
                v.message.full_name, v.message.message_id, v.message.text,
                v.confidence, "llm", action,
            )

        if self.on_log:
            await self.on_log(spam, shadow)

        if shadow:
            logger.info("chat=%s 影子模式：本应删除 %d 条（未执行）",
                        self.chat_id, len(spam))
            if self.notify:
                await self.notify(spam)
            return

        for v in spam:
            try:
                await self.on_spam(v)
                # 付费商家的删除不计入 del_count：
                # 否则删几次就把付了钱的商家推成 repeat（惯犯）档，自相矛盾。
                if v.message.whitelist != "advertiser":
                    await asyncio.to_thread(
                        db.bump_user_deleted, self.chat_id, v.message.user_id,
                        v.message.full_name,
                    )
            except Exception as exc:  # noqa: BLE001
                if db.is_non_retryable_delete_error(exc):
                    # 消息已被自己/管理员删掉，或本就不可删。
                    # 这不是故障，压成 info 以免污染日志（线上曾有 84 次这类 WARNING）。
                    logger.info("删除跳过（不可重试）msg=%s: %s",
                                v.message.message_id, exc)
                else:
                    logger.warning("删除失败 msg=%s: %s", v.message.message_id, exc)

        if self.notify:
            await self.notify(spam)

    def _requeue(self, batch: list[PendingMessage]) -> None:
        """把未判定的消息退回未检测状态，下轮重新取。"""
        with db.tx() as conn:
            conn.executemany("UPDATE messages SET is_checked=0 WHERE id=?",
                             [(m.row_id,) for m in batch])

    def _build_prompt(self, batch: list[PendingMessage]) -> list[dict]:
        """构造请求。

        结构严格分三层，前两层稳定以获得缓存命中（命中价 0.02 元/M vs 未命中 1 元/M）：
            [GLOBAL_PROMPT]   跨群跨请求恒定
            [群画像 + few-shot] 同群内每天最多变 1 次
            [本批消息]         必然变化，放最后
        """
        cfg = self._cfg()
        positives = db.fewshots(self.chat_id, "positive")
        negatives = db.fewshots(self.chat_id, "negative")

        profile = build_group_profile(
            title=cfg.get("title") or f"群{self.chat_id}",
            chat_nature=cfg.get("chat_nature", "general"),
            now=time.strftime("%Y-%m-%d %H:%M"),
            strictness=cfg.get("strictness", "strict"),
            normal_words=db.normal_words(self.chat_id),
            blocked_words=db.blocked_words(self.chat_id),
        )
        shots = build_fewshots(positives, negatives)

        lines = []
        for m in batch:
            text = (m.text or "").replace("\n", " ")[:300]
            prefix = f"[ID={m.row_id}]"
            if m.marker:
                prefix += f" {m.marker}"
            hist = ""
            if m.history:
                past = [h.replace("\n", " ")[:40] for h in m.history[:3]]
                hist = f" | 历史:{' / '.join(past)}"
            lines.append(f"{prefix} 昵称={m.full_name[:20]} | 内容={text}{hist}")

        msgs = [{"role": "system", "content": GLOBAL_PROMPT}]
        if shots:
            msgs.append({"role": "system", "content": shots})
        msgs.append({"role": "system", "content": profile})
        msgs.append({"role": "user", "content": BATCH_INSTRUCTION.format(
            count=len(batch), messages="\n".join(lines))})
        return msgs


class PipelineManager:
    """所有群的流水线。一个群慢不会拖住其他群：每个群独立 asyncio 任务。"""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client
        self._pipelines: dict[int, ChatPipeline] = {}
        self._tasks: dict[int, asyncio.Task] = {}
        self._stop = asyncio.Event()

    def get(self, chat_id: int, on_spam, notify=None, on_log=None) -> ChatPipeline:
        p = self._pipelines.get(chat_id)
        if p is None:
            p = ChatPipeline(chat_id, self.client, on_spam, notify, on_log)
            self._pipelines[chat_id] = p
        return p

    async def start(self, chat_ids: list[int], on_spam, notify=None,
                    on_log=None) -> None:
        for cid in chat_ids:
            p = self.get(cid, on_spam, notify, on_log)
            self._tasks[cid] = asyncio.create_task(p.run(self._stop))
        logger.info("已启动 %d 个群的检测流水线", len(self._tasks))

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks.values():
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
