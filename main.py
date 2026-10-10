"""反广告机器人入口。

职责：Telegram 交互 + 定时任务。判定逻辑全在 pipeline.py，本文件不含判定规则。

设计要点：
- **每个群独立流水线**，一个群慢不会拖住其他群。
- **影子模式**（`chats.shadow_mode=1`）下判定照跑、记录照写、通知照发，只是不删消息。
  上线第一周应保持开启，用两个 bot 的判定分歧校准阈值。
- **多租户**：所有配置/白名单/黑名单/样本按 chat_id 隔离。
- **必须授予管理员权限**，否则删不掉别人的消息（Telegram API 的硬限制）。
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import sys
import time
from datetime import datetime, time as dtime, timedelta

import pytz
from dotenv import load_dotenv
from telegram import (
    ChatPermissions,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    ChatMemberHandler,
    filters,
)

import db
from deepseek import DeepSeekClient
from detector_rules import compute_trust, trust_marker
from link_preview import has_link_preview, preview_fingerprint
from pipeline import PendingMessage, PipelineManager, Verdict
from prompts import NATURE_PRESETS

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
# httpx 的 INFO 日志会把 Telegram API URL（含明文 token）写进日志。
# V1 就有这个问题，对外服务前必须压掉。
#
# 注意：httpx 被 openai 作为依赖装进来时，实际 logger 名可能是 `httpx2`
# （openai 的 vendored 版本）。按名字逐个压不可靠，这里扫描已存在的 logger。
for _name in list(logging.root.manager.loggerDict):
    if _name.split(".")[0].startswith("httpx"):
        logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger("telegram.ext.Application").setLevel(logging.WARNING)
logger = logging.getLogger("antispam")

BOT_TOKEN = os.environ["BOT_TOKEN"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-flash")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Shanghai")
DB_PATH = os.getenv("DB_NAME", "antispam.db")
MESSAGE_RETENTION_DAYS = int(os.getenv("MESSAGE_RETENTION_DAYS", "30"))

TZ = pytz.timezone(TIMEZONE)
HEARTBEAT_FILE = "heartbeat.touch"

FULL_PERMISSIONS = ChatPermissions(
    can_send_messages=True, can_send_audios=True, can_send_documents=True,
    can_send_photos=True, can_send_videos=True, can_send_video_notes=True,
    can_send_voice_notes=True, can_send_polls=True, can_send_other_messages=True,
    can_add_web_page_previews=True,
)

_manager: PipelineManager | None = None
_client: DeepSeekClient | None = None
# 举报 ID -> 消息 row_id。判定完成后据此回填举报状态。
_report_rows: dict[int, int] = {}


def _is_admin_user(user_id: int | None) -> bool:
    return bool(ADMIN_ID) and user_id == ADMIN_ID


async def _refresh_admins(app: Application, chat_id: int) -> set[int]:
    """刷新并缓存群管理员。TTL 由 pipeline 侧各自控制。"""
    try:
        admins = await app.bot.get_chat_administrators(chat_id)
        pairs = [(a.user.id, a.user.full_name) for a in admins]
        await asyncio.to_thread(db.cache_admins, chat_id, pairs)
        return {uid for uid, _ in pairs}
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取管理员失败 chat=%s: %s", chat_id, exc)
        return set()


# 管理员缓存的 TTL。超时后台刷新，消息路径永不阻塞在 API 调用上。
_ADMIN_TTL = 600.0
_admin_cache: dict[int, tuple[float, set[int]]] = {}


async def _is_cached_admin(chat_id: int, user_id: int) -> bool:
    now = time.time()
    cached = _admin_cache.get(chat_id)
    if cached and cached[0] > now:
        return user_id in cached[1]
    ids = await asyncio.to_thread(db.cached_admin_ids, chat_id)
    if ids or cached:
        _admin_cache[chat_id] = (now + _ADMIN_TTL, ids)
        return user_id in ids
    # 首次遇到该群：同步拉一次并落库
    return user_id in set()


async def _send_admin(bot, text: str, **kw) -> bool:
    """发消息给管理员。

    Telegram 不允许机器人主动给「从未与它对话过」的用户发消息，会抛
    Chat not found。这个错误对使用者毫无提示性，所以单独翻译成操作指引。
    所有管理员通知都走这里，避免每处都重复判断。
    """
    if not ADMIN_ID:
        return False
    try:
        await bot.send_message(chat_id=ADMIN_ID, text=text, **kw)
        return True
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        tail = ""
        if "Chat not found" in msg or "chat not found" in msg:
            try:
                me = await bot.get_me()
                tail = (f"\n  -> 请用你的账号打开 https://t.me/{me.username} "
                        f"并点 Start（机器人无法主动给从未对话过的用户发消息）")
            except Exception:  # noqa: BLE001
                tail = "\n  -> 请先与机器人私聊并点 Start"
        logger.warning("管理员通知发送失败: %s%s", msg, tail)
        return False



# ---------------------------------------------------------------------------
# 举报：群内 @机器人 / /report / 转发到私聊
# ---------------------------------------------------------------------------

# 判定一条举报内容用的最小请求。单条判定约 3,500 token（多数命中缓存）
# + 20 输出 token ≈ 0.00006 元，所以钱不是限制，误报占用注意力才是，
# 因此额度只是防刷，设得比较宽松。
REPORT_LIMIT_DEFAULT = 20

# 举报人历史准确率低于此值且样本足够时，降级为"仅记录不处理"
REPORTER_MIN_ACCURACY = 0.20
REPORTER_MIN_SAMPLES = 8


async def _pick_report_chat(update: Update, forwarded_user_id: int) -> tuple[int, str]:
    """判断这条举报属于哪个群。返回 (chat_id, 来源说明)。

    转发的消息拿不到原始 chat_id（Telegram 只给"谁发的"，不给"哪群的第几条"），
    所以按"被举报人所在群"匹配。若举报人只在一个已注册群里，直接用那个群。
    """
    chats = await asyncio.to_thread(db.list_active_chats)
    if not chats:
        return 0, ""

    # 优先：从群内发起的举报（回复 + @机器人 / /report），chat 是明确的
    chat = update.effective_chat
    if chat and chat.type in ("group", "supergroup"):
        return chat.id, "群内"

    # 转发路径：看被举报人在哪些已注册群里出现过
    if forwarded_user_id:
        for c in chats:
            st = await asyncio.to_thread(db.user_stats, c["chat_id"],
                                         forwarded_user_id)
            if st["msg_n"] > 0:
                return c["chat_id"], "转发"
    # 兜底：举报人只服务一个群时用那个
    if len(chats) == 1:
        return chats[0]["chat_id"], "转发"
    return 0, ""


async def _handle_forwarded_report(update: Update,
                                   context: ContextTypes.DEFAULT_TYPE) -> None:
    """私聊转发消息给机器人 = 举报。

    能拿到什么取决于转发来源（已实测）：
      · MessageOriginChannel 带 message_id -> 理论上可删原消息
      · MessageOriginUser / Chat / HiddenUser **不带 message_id** -> 删不掉原消息
    所以转发举报的产出是：判定内容 + 累加违规 + 必要时拉黑，
    而不是补删某一条。这与举报的定位一致——抓漏网广告的账号。
    """
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return

    origin = getattr(msg, "forward_origin", None)
    if origin is None and not getattr(msg, "forward_from", None):
        await msg.reply_text(
            "把要举报的消息**转发**给我即可。\n\n"
            "注意：转发只能用于识别内容与累加违规，**无法删除原消息**"
            "（Telegram 不提供原消息位置）。如需直接删除，"
            "请在群里回复那条消息并 @我。",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    from telegram import (MessageOriginChannel, MessageOriginChat,
                          MessageOriginHiddenUser, MessageOriginUser)

    target_id = 0
    target_name = "未知"
    origin_note = ""
    if isinstance(origin, MessageOriginUser):
        target_id = origin.sender_user.id
        target_name = origin.sender_user.full_name
        origin_note = "用户转发"
    elif isinstance(origin, MessageOriginHiddenUser):
        target_name = origin.sender_user_name or "隐藏用户"
        origin_note = "隐藏转发（对方设置了隐藏转发来源）"
    elif isinstance(origin, MessageOriginChannel):
        target_id = origin.chat.id
        target_name = origin.chat.title or str(origin.chat.id)
        origin_note = f"频道转发 msg_id={origin.message_id}"
    elif isinstance(origin, MessageOriginChat):
        target_id = origin.sender_chat.id
        target_name = origin.sender_chat.title or str(origin.sender_chat.id)
        origin_note = "群组转发"

    content = (msg.text or msg.caption or "[图片/媒体]")[:400]

    chat_id, how = await _pick_report_chat(update, target_id)
    if not chat_id:
        await msg.reply_text(
            "无法判断这条举报属于哪个群。\n"
            "请把机器人拉进相关群，或在群内直接回复消息并 @我。"
        )
        return

    # 额度与准确率
    if not _is_admin_user(user.id):
        left = await asyncio.to_thread(db.report_quota_left, chat_id, user.id)
        if left <= 0:
            await msg.reply_text(
                "今日举报次数已用完，明天再来。\n"
                "（额度是为了防刷，不是省钱——判定本身几乎不花钱）"
            )
            return
        valid, total = await asyncio.to_thread(db.reporter_accuracy, user.id, chat_id)
        if total >= REPORTER_MIN_SAMPLES and valid / total < REPORTER_MIN_ACCURACY:
            await msg.reply_text(
                f"你的历史举报准确率偏低（{valid}/{total}），"
                "本条已记录但暂不处理。请确认确实是广告再举报。"
            )
            await asyncio.to_thread(db.report_quota_consume, chat_id, user.id)
            return

    report_id = await asyncio.to_thread(
        db.save_report, chat_id, user.id, user.full_name,
        target_id, target_name,
        # 转发路径没有原消息 ID，用 0 占位（不可删除）
        origin.message_id if isinstance(origin, MessageOriginChannel) else 0,
        content,
    )
    if report_id is None:
        return
    await asyncio.to_thread(db.report_quota_consume, chat_id, user.id)

    logger.info("转发举报 chat=%s reporter=%s target=%s(%s) %s",
                chat_id, user.id, target_name, target_id, origin_note)

    # 送判定：转发没有原消息，无法删除，所以只做内容判定
    pipeline = _manager.get(chat_id, on_spam=_make_on_spam(context.application),
                            notify=_make_notify(context.application),
                            on_log=_make_on_log(context.application))
    pending = PendingMessage(
        row_id=-(10 ** 9 + report_id),   # 负 ID 表示举报，且不与群内举报冲突
        message_id=0, chat_id=chat_id, user_id=target_id,
        full_name=target_name, text=content, reply_to=None, timestamp="",
        fast=True, report_id=report_id, origin="forward",
    )
    async with pipeline._lock:  # noqa: SLF001
        pipeline._fast.appendleft(pending)  # noqa: SLF001

    await msg.reply_text(
        f"✅ 已受理举报：{target_name}\n"
        f"归属群：{chat_id}（{how}匹配）\n"
        f"剩余今日额度：{await asyncio.to_thread(db.report_quota_left, chat_id, user.id)}\n\n"
        f"判定结果会异步给出；若确认是广告，会累加该账号的违规次数。"
        + ("" if target_id else "\n⚠️ 未能识别被举报人 ID，只能记录内容。")
    )


# ---------------------------------------------------------------------------
# 记录频道
# ---------------------------------------------------------------------------

def _make_on_log(app: Application):
    """把删除记录写到频道。这是 V1 缺失的**可回溯证据链**：
    V1 的 spam_records 没有正文列，导致事后无法复核任何一条删除。

    注意这是**同步工厂**，返回 async 回调。误加 async 会让调用处拿到一个
    未被 await 的协程，功能静默失效（只有 RuntimeWarning 提示）。
    _make_on_spam / _make_notify 同样是同步工厂，保持一致。
    """
    async def on_log(verdicts: list[Verdict], shadow: bool) -> None:
        if not verdicts:
            return
        by_chat: dict[int, list[Verdict]] = {}
        for v in verdicts:
            by_chat.setdefault(v.message.chat_id, []).append(v)

        for chat_id, items in by_chat.items():
            row = await asyncio.to_thread(db.get_chat, chat_id)
            if not row or not row["log_channel_id"]:
                continue
            title = row["title"] or str(chat_id)
            head = ("👻 影子（未删）" if shadow else "🗑️ 已删除")
            lines = [f"{head} {len(items)} 条 · {html.escape(str(title))}"]
            kb_rows = []
            for v in items[:15]:
                m = v.message
                rec_id = await asyncio.to_thread(
                    _latest_record_id, chat_id, m.message_id)
                lines.append(
                    f"\n#{m.message_id} [{v.confidence:.2f}] "
                    f"{html.escape(str(m.full_name)[:20])}"
                    + (f" {m.marker}" if m.marker else "")
                    + f"\n{html.escape((m.text or '')[:200])}"
                )
                if rec_id and not shadow:
                    kb_rows.append([InlineKeyboardButton(
                        f"↩️ 恢复 {str(m.text or '')[:24]}",
                        callback_data=f"rev_{rec_id}")])
            if len(items) > 15:
                lines.append(f"\n…另有 {len(items) - 15} 条")
            kb = InlineKeyboardMarkup(kb_rows) if kb_rows else None
            try:
                await app.bot.send_message(
                    chat_id=row["log_channel_id"], text="\n".join(lines),
                    reply_markup=kb, disable_web_page_preview=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("写记录频道失败 channel=%s: %s",
                               row["log_channel_id"], exc)
    return on_log



# ---------------------------------------------------------------------------
# 删除执行
# ---------------------------------------------------------------------------

async def _delete_verdict(bot, v: Verdict) -> None:
    await bot.delete_message(chat_id=v.message.chat_id,
                             message_id=v.message.message_id)


def _spam_keyboard(chat_id: int, user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("↩️ 误删，恢复并加白",
                             callback_data=f"revive_{chat_id}_{user_id}"),
        InlineKeyboardButton("🚫 拉黑此人", callback_data=f"bl_{chat_id}_{user_id}"),
    ]])


# 一批删除过多时，逐条通知会把私聊刷爆，只对前 N 条发逐条通知，
# 其余合并成一条摘要。摘要里的条目仍可用 /recent 逐个处理。
MAX_DETAIL_NOTIFY = 6


async def _notify_spam(bot, verdicts: list[Verdict], shadow: bool) -> None:
    """逐条通知管理员，每条带自己的操作按钮。

    早先是"一批一条通知、一个按钮"，导致同批第二条无法单独恢复——
    按钮只指向第一项的发布者，且 revive 逻辑按用户找记录而非按消息。
    现在改为每条独立通知 + 按记录 ID 精确操作。
    """
    if not ADMIN_ID or not verdicts:
        return

    head = "👻 影子（未执行）" if shadow else "🗑️ 已删除"
    detail, rest = verdicts[:MAX_DETAIL_NOTIFY], verdicts[MAX_DETAIL_NOTIFY:]

    for v in detail:
        m = v.message
        row = await asyncio.to_thread(db.get_chat, m.chat_id)
        title = (row["title"] if row else str(m.chat_id)) or str(m.chat_id)
        # 取刚写入的记录 ID，用于精确恢复
        rec_id = await asyncio.to_thread(_latest_record_id, m.chat_id, m.message_id)
        text = (
            f"{head}\n"
            f"群：{html.escape(str(title))}\n"
            f"用户：{html.escape(str(m.full_name)[:24])}"
            + (f" {m.marker}" if m.marker else "") + "\n"
            f"置信度：{v.confidence:.2f}\n"
            f"内容：{html.escape((m.text or '')[:120])}"
        )
        kb = None
        if not shadow and rec_id:
            # 按**记录 ID** 操作，而不是按用户——同批多条时才能各自区分
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("↩️ 误删恢复", callback_data=f"rev_{rec_id}"),
                InlineKeyboardButton("🚫 拉黑此人", callback_data=f"blk_{rec_id}"),
            ]])
        await _send_admin(bot, text, reply_markup=kb,
                          disable_web_page_preview=True)

    if rest:
        by_chat: dict[int, list[Verdict]] = {}
        for v in rest:
            by_chat.setdefault(v.message.chat_id, []).append(v)
        lines = [f"{head} 另有 {len(rest)} 条（未逐条通知）"]
        for chat_id, items in by_chat.items():
            row = await asyncio.to_thread(db.get_chat, chat_id)
            title = (row["title"] if row else str(chat_id)) or str(chat_id)
            lines.append(f"\n【{html.escape(str(title))}】")
            for v in items[:8]:
                lines.append(f"• [{v.confidence:.2f}] "
                             f"{html.escape((v.message.text or '')[:44])}")
        lines.append("\n用 /recent 查看并逐条恢复。")
        await _send_admin(bot, "\n".join(lines), disable_web_page_preview=True)


def _latest_record_id(chat_id: int, message_id: int) -> int | None:
    """取某条消息最近一次删除记录的 ID。"""
    row = db.query_one("""SELECT id FROM spam_records
                          WHERE chat_id=? AND message_id=? AND action='delete'
                          ORDER BY id DESC LIMIT 1""", (chat_id, message_id))
    return row["id"] if row else None


# 自动注册时使用的默认值。
# 阈值 0.50 是离线实测 F1 最优档（召回 95.5% / 精确 89.3%）。
# 群性质默认 general（最保守的表征），群主确认后可改。
AUTO_REGISTER_CUT = 0.50
AUTO_REGISTER_NATURE = "general"


async def _auto_register(app: Application, chat, added_by: int | None) -> bool:
    """机器人被拉进群时自动注册并通知管理员。

    为什么不要求手动注册：拉群后如果不注册，handle_message 会直接 return，
    一个字的上下文都不会收集——用户以为在跑，实际什么都没记。
    自动注册后可用 manage.py 调整性质与严格度。
    """
    if chat.type not in ("group", "supergroup"):
        return False
    row = await asyncio.to_thread(db.get_chat, chat.id)
    if row:
        return False

    await asyncio.to_thread(db.register_chat, chat.id, chat.title or "",
                            AUTO_REGISTER_NATURE, shadow_mode=False)
    await asyncio.to_thread(db.set_chat_field, chat.id, "confidence_cut",
                            AUTO_REGISTER_CUT)
    await asyncio.to_thread(db.set_chat_field, chat.id, "strictness", "standard")

    # 立刻拉一次管理员列表，避免首条消息被群主自己触发判定
    await _refresh_admins(app, chat.id)

    logger.info("自动注册新群 chat=%s title=%s", chat.id, chat.title)
    if ADMIN_ID:
        try:
            me = await app.bot.get_me()
            await _send_admin(app.bot, ("🆕 已自动注册新群\n"
                      f"群名：{chat.title}\n"
                      f"ID：{chat.id}\n"
                      f"阈值：{AUTO_REGISTER_CUT}（实测最优档）\n"
                      f"性质：{AUTO_REGISTER_NATURE}\n\n"
                      f"必须把 @{me.username} 设为管理员并给「删除消息」权限，"
                      f"否则判定无法执行。\n\n"
                      f"调整：python manage.py set {chat.id} --nature adult_service"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("自动注册通知失败: %s", exc)
    return True


async def on_my_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """机器人自身的成员状态变化：被拉进群 / 被提升为管理员 / 被踢出。"""
    cm = update.my_chat_member
    if not cm or not update.effective_chat:
        return
    chat = update.effective_chat
    new_status = cm.new_chat_member.status
    actor = cm.from_user.id if cm.from_user else None

    if new_status in ("member", "administrator"):
        await _auto_register(context.application, chat, actor)
    elif new_status in ("left", "kicked"):
        row = await asyncio.to_thread(db.get_chat, chat.id)
        if row and row["is_active"]:
            await asyncio.to_thread(db.set_chat_field, chat.id, "is_active", 0)
            logger.info("机器人已离开群 chat=%s，已停用", chat.id)
            await _send_admin(
                context.bot,
                f"⚠️ 机器人已离开群 {chat.title}（{chat.id}），已停用该群。")

    # 成为管理员后立刻刷新管理员列表
    if new_status == "administrator":
        await _refresh_admins(context.application, chat.id)



# ---------------------------------------------------------------------------
# 消息处理
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not update.effective_chat or not update.effective_user:
        return

    chat = update.effective_chat
    user = update.effective_user

    if user.is_bot:
        return

    # 私聊：转发消息 = 举报
    if chat.type == "private":
        await _handle_forwarded_report(update, context)
        return

    # 群组以外不处理（频道等）
    if chat.type not in ("group", "supergroup"):
        logger.info("跳过非群组消息 chat=%s type=%s", chat.id, chat.type)
        return

    row = await asyncio.to_thread(db.get_chat, chat.id)
    if not row or not row["is_active"]:
        # 这是最危险的静默丢弃点：群已加入但未注册（my_chat_member 事件丢失、
        # 或注册被停用），此时消息全部被丢弃而日志里毫无痕迹，
        # 表现为"机器人不工作"但看不出原因。
        logger.warning("丢弃消息：群未注册 chat=%s title=%r user=%s",
                       chat.id, chat.title, user.id)
        try:
            await asyncio.to_thread(db.save_dropped, chat.id, chat.title,
                                    user.id, user.full_name,
                                    msg.text or msg.caption or "",
                                    "chat_not_registered")
        except Exception:  # noqa: BLE001
            pass
        return

    text = msg.text or msg.caption
    if not text:
        logger.info("跳过无文本消息 chat=%s msg=%s（纯媒体，无 caption）",
                    chat.id, msg.message_id)
        return

    # 诊断：极短文本 + 带转发/链接预览的消息，客户端显示的卡片内容与
    # bot 收到的 text 可能完全不同（实测：客户端看到频道广告卡片，
    # bot 只收到单个字母 't'）。这类场景要能看到完整结构才查得动。
    if len(text.strip()) <= 3:
        origin = getattr(msg, "forward_origin", None)
        _fp, _purl = preview_fingerprint(msg)
        logger.warning(
            "低信息量消息 chat=%s msg=%s text=%r "
            "is_auto_fwd=%s forward_origin=%s "
            "has_preview=%s preview_url=%s entities=%s reply_to=%s",
            chat.id, msg.message_id, text,
            getattr(msg, "is_automatic_forward", None),
            type(origin).__name__ if origin else None,
            has_link_preview(msg), (_purl or "(无)")[:90],
            [(e.type, e.offset, e.length, getattr(e, "url", None))
             for e in (msg.entities or [])],
            msg.reply_to_message.message_id if msg.reply_to_message else None,
        )

    ts = msg.date.astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")

    # 举报：回复某条消息并 @机器人，或 /report
    bot_username = context.bot.username
    if bot_username and f"@{bot_username}" in text and msg.reply_to_message:
        await _handle_report(update, context, chat.id, user, text)
        return

    # 入画像 + 落库（先落库才能拿到 row_id，判定记录需要它）
    await asyncio.to_thread(db.upsert_user_seen, chat.id, user.id,
                            user.full_name, user.username)
    row_id = await asyncio.to_thread(
        db.save_message, chat.id, msg.message_id, user.id, user.full_name,
        text, msg.reply_to_message.message_id if msg.reply_to_message else None, ts,
    )
    if row_id is None:
        return

    # 管理员消息免检但已记录。
    # 用带 TTL 的本地缓存，绝不能每条消息都调 get_chat_administrators——
    # 那是一次 Telegram API 往返，会直接拖垮吞吐。
    if await _is_cached_admin(chat.id, user.id):
        return

    # 提取链接预览信息。正文可能只有一个字母，但预览目标是下发的，
    # 它是识别"多账号协同引流"的唯一抓手（实测漏判案例的关键）。
    fp, purl = preview_fingerprint(msg)
    if fp:
        logger.info("预览指纹 chat=%s msg=%s fp=%s url=%s",
                    chat.id, msg.message_id, fp, purl[:80])

    pending = PendingMessage(
        row_id=row_id, message_id=msg.message_id, chat_id=chat.id,
        user_id=user.id, full_name=user.full_name, text=text,
        reply_to=msg.reply_to_message.message_id if msg.reply_to_message else None,
        timestamp=ts,
        preview_fp=fp, preview_url=purl, has_preview=has_link_preview(msg),
    )

    pipeline = _manager.get(chat.id, on_spam=_make_on_spam(context.application),
                            notify=_make_notify(context.application),
                            on_log=_make_on_log(context.application))
    verdict = await pipeline.submit(pending)

    # L0 命中：立即删除，不等 LLM
    if verdict and verdict.is_spam:
        await _apply_verdict(context.application, verdict)


async def _handle_report(update: Update, context: ContextTypes.DEFAULT_TYPE,
                         chat_id: int, reporter, text: str) -> None:
    """群友回复广告消息并 @机器人。"""
    msg = update.effective_message
    replied = msg.reply_to_message
    target = replied.from_user
    if not target or target.id == context.bot.id:
        return

    content = (replied.text or replied.caption or "[图片/媒体]")[:300]
    report_id = await asyncio.to_thread(
        db.save_report, chat_id, reporter.id, reporter.full_name,
        target.id, target.full_name, replied.message_id, content,
    )
    if report_id is None:
        return

    await msg.reply_text(f"已记录举报：{target.full_name}", quote=True)

    # 举报消息插到判定队列最前（负 row_id 便于回填）
    row_id = -report_id
    _report_rows[report_id] = row_id
    await asyncio.to_thread(
        db.save_message, chat_id, replied.message_id, target.id,
        target.full_name, content, None,
        replied.date.astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S"),
    )
    pending = PendingMessage(
        row_id=row_id, message_id=replied.message_id, chat_id=chat_id,
        user_id=target.id, full_name=target.full_name, text=content,
        reply_to=None, timestamp="",
    )
    pipeline = _manager.get(chat_id, on_spam=_make_on_spam(context.application),
                            notify=_make_notify(context.application),
                            on_log=_make_on_log(context.application))
    pending.report_id = report_id
    pending.fast = True
    async with pipeline._lock:  # noqa: SLF001 - 有意插队
        pipeline._fast.appendleft(pending)  # noqa: SLF001


def _make_on_spam(app: Application):
    async def on_spam(v: Verdict) -> None:
        await _delete_verdict(app.bot, v)
    return on_spam


def _make_notify(app: Application):
    async def notify(verdicts: list[Verdict]) -> None:
        chat_id = verdicts[0].message.chat_id if verdicts else 0
        row = await asyncio.to_thread(db.get_chat, chat_id) if chat_id else None
        shadow = bool(row["shadow_mode"]) if row else False
        if row and row["notify_admin"]:
            await _notify_spam(app.bot, verdicts, shadow)
    return notify


async def _apply_verdict(app: Application, v: Verdict) -> None:
    """执行一条 L0 判定（立即删除）。"""
    row = await asyncio.to_thread(db.get_chat, v.message.chat_id)
    shadow = bool(row["shadow_mode"]) if row else True
    action = "observe" if shadow else "delete"
    await asyncio.to_thread(
        db.save_spam_record, v.message.chat_id, v.message.user_id,
        v.message.full_name, v.message.message_id, v.message.text,
        1.0, v.source, action,
    )
    if shadow:
        logger.info("影子模式：本应删除 msg=%s reason=%s",
                    v.message.message_id, v.reason)
    else:
        try:
            await _delete_verdict(app.bot, v)
            await asyncio.to_thread(db.bump_user_deleted, v.message.chat_id,
                                    v.message.user_id, v.message.full_name)
            logger.info("L0 删除 msg=%s reason=%s", v.message.message_id, v.reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("L0 删除失败 msg=%s: %s", v.message.message_id, exc)
    if row and row["notify_admin"]:
        await _notify_spam(app.bot, [v], shadow)


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = update.effective_chat
    if chat and chat.type in ("group", "supergroup"):
        row = await asyncio.to_thread(db.get_chat, chat.id)
        if not row:
            # my_chat_member 事件可能丢失（例如机器人先被拉进群、后重启）。
            # /start 兜底注册，避免"以为在跑实际什么都没记"。
            await _auto_register(context.application, chat,
                                 update.effective_user.id if update.effective_user else None)
            row = await asyncio.to_thread(db.get_chat, chat.id)
        if not row:
            await update.effective_message.reply_text(
                "本群尚未注册，且自动注册未成功。请联系机器人所有者。"
            )
            return
        state = "影子模式（只记录不删除）" if row["shadow_mode"] else "已启用删除"
        await update.effective_message.reply_text(
            f"反广告机器人已就绪。\n当前状态：{state}\n"
            f"判定阈值：{row['confidence_cut']}\n\n"
            "群友发现漏网广告时，回复该消息并 @我 即可举报。"
        )
    else:
        await update.effective_message.reply_text(
            "把本机器人拉进群并设为管理员即可自动开启反广告。\n"
            "必须授予「删除消息」权限，否则无法清理他人消息。"
        )


async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin_user(update.effective_user.id if update.effective_user else None):
        return
    if update.message:
        await update.message.reply_text(
            "🛡️ 反广告管理面板", reply_markup=_admin_kb()
        )


def _admin_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 各群概览", callback_data="adm_overview"),
         InlineKeyboardButton("💾 用量成本", callback_data="adm_usage")],
        [InlineKeyboardButton("📋 白名单", callback_data="adm_wl"),
         InlineKeyboardButton("🚫 黑名单", callback_data="adm_bl")],
        [InlineKeyboardButton("📨 举报统计", callback_data="adm_reports"),
         InlineKeyboardButton("📜 判定记录", callback_data="adm_recent")],
    ])


async def cmd_wl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """白名单管理。用法：回复某人 + /wl [advertiser]，或 /wl remove。"""
    if not _is_admin_user(update.effective_user.id if update.effective_user else None):
        return
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return
    args = context.args or []

    if args and args[0] == "list":
        rows = await asyncio.to_thread(db.whitelist_list, chat.id)
        if not rows:
            await msg.reply_text("白名单为空。回复某人消息 + /wl 添加。")
            return
        lines = [f"📋 白名单（{len(rows)} 人）"]
        for r in rows:
            exp = f" 到期 {str(r['expires_at'])[:10]}" if r["expires_at"] else ""
            lines.append(f"• {r['full_name'] or r['user_id']} "
                         f"[{r['level']}]{exp}")
        await msg.reply_text("\n".join(lines[:60]))
        return

    if args and args[0] == "remove":
        target = None
        if msg.reply_to_message and msg.reply_to_message.from_user:
            target = msg.reply_to_message.from_user.id
        elif len(args) > 1 and args[1].isdigit():
            target = int(args[1])
        if not target:
            await msg.reply_text("用法：回复某人消息 + /wl remove")
            return
        ok = await asyncio.to_thread(db.whitelist_remove, chat.id, target)
        await msg.reply_text("✅ 已移出白名单" if ok else "ℹ️ 不在白名单中")
        return

    # 添加
    level = "advertiser" if "advertiser" in args else "trusted"
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply_text(
            "用法：\n"
            "回复某人消息 + /wl              加入白名单（完全跳过检测）\n"
            "回复某人消息 + /wl advertiser   仅删高置信（适合付费商家）\n"
            "/wl list                        查看\n"
            "/wl remove                      回复某人移除"
        )
        return
    target = msg.reply_to_message.from_user
    await asyncio.to_thread(db.whitelist_set, chat.id, target.id, level,
                            target.full_name, update.effective_user.id, "管理员添加")
    desc = "仅删高置信广告（付费商家）" if level == "advertiser" else "完全跳过检测"
    await msg.reply_text(f"✅ {target.full_name} 已加入白名单\n档位：{level}（{desc}）")


async def cmd_bl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """黑名单：不看内容直接删除。用法：回复某人 + /bl，或 /bl remove。"""
    if not _is_admin_user(update.effective_user.id if update.effective_user else None):
        return
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return
    args = context.args or []

    if args and args[0] == "remove":
        target = None
        if msg.reply_to_message and msg.reply_to_message.from_user:
            target = msg.reply_to_message.from_user.id
        elif len(args) > 1 and args[1].isdigit():
            target = int(args[1])
        if not target:
            await msg.reply_text("用法：回复某人消息 + /bl remove")
            return
        ok = await asyncio.to_thread(db.blacklist_remove, chat.id, target)
        await msg.reply_text("✅ 已移出黑名单" if ok else "ℹ️ 不在黑名单中")
        return

    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply_text("用法：回复某人消息 + /bl")
        return
    t = msg.reply_to_message.from_user
    await asyncio.to_thread(db.blacklist_add, chat.id, t.id, t.full_name,
                            update.effective_user.id, "管理员拉黑")
    await msg.reply_text(f"🚫 {t.full_name} 已拉黑，其消息将被直接删除。")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """举报。也可用「回复消息 + @机器人」。"""
    msg = update.effective_message
    if not msg or not msg.reply_to_message:
        await msg.reply_text("请回复要举报的消息再发送 /report")
        return
    await _handle_report(update, context, update.effective_chat.id,
                        update.effective_user, "report")


async def cmd_logchannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """设置/取消删除记录频道。

    用法：
      /logchannel off                    取消
      /logchannel -1001234567890         设为该频道（机器人需是频道管理员）
    在频道里发 /logchannel 也可，会取当前频道 ID。
    """
    if not _is_admin_user(update.effective_user.id if update.effective_user else None):
        return
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return
    args = context.args or []

    if args and args[0] == "off":
        for c in await asyncio.to_thread(db.list_active_chats):
            await asyncio.to_thread(db.set_log_channel, c["chat_id"], None)
        await msg.reply_text("✅ 已取消所有群的记录频道。")
        return

    if args and args[0].lstrip("-").isdigit():
        cid = int(args[0])
    elif chat.type == "channel":
        cid = chat.id
    else:
        rows = await asyncio.to_thread(db.chats_with_log_channel)
        cur = rows[0]["log_channel_id"] if rows else "未设置"
        await msg.reply_text(
            f"当前记录频道：{cur}\n\n"
            "用法：\n"
            "/logchannel -1001234567890   设为该频道\n"
            "/logchannel off              取消\n"
            "或在频道内直接发 /logchannel\n\n"
            "频道会收到每条删除记录：消息ID、置信度、用户、信任标记、正文。\n"
            "这是 V1 缺失的证据链——V1 的记录表没有正文，事后无法复核。\n\n"
            "注意：把机器人设为该频道的管理员才能发帖。"
        )
        return

    # 先验证能不能发
    try:
        probe = await context.bot.send_message(chat_id=cid, text="记录频道连通性测试")
        await probe.delete()
    except Exception as exc:  # noqa: BLE001
        await msg.reply_text(f"❌ 无法向 {cid} 发消息：{exc}\n"
                            "请先把机器人设为该频道管理员。")
        return

    # 单群场景直接绑；多群时绑到举报来源匹配的那个（默认全部）
    chats = await asyncio.to_thread(db.list_active_chats)
    for c in chats:
        await asyncio.to_thread(db.set_log_channel, c["chat_id"], cid)
    await msg.reply_text(f"✅ 已把 {len(chats)} 个群的删除记录指向频道 {cid}")


async def cmd_recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """列出最近删除记录，每条带独立恢复按钮。

    存在的理由：一批删除超过 MAX_DETAIL_NOTIFY 条时，多出的部分只发摘要，
    没有按钮。这个命令是那些条目的补漏入口。
    """
    if not _is_admin_user(update.effective_user.id if update.effective_user else None):
        return
    msg = update.effective_message
    if not msg:
        return
    limit = 10
    if context.args and context.args[0].isdigit():
        limit = max(1, min(30, int(context.args[0])))

    recs = await asyncio.to_thread(db.recent_spam_records_all, limit)
    if not recs:
        await msg.reply_text("暂无删除记录。")
        return

    chats = {c["chat_id"]: (c["title"] or str(c["chat_id"]))
             for c in await asyncio.to_thread(db.list_active_chats)}
    lines = [f"📜 最近 {len(recs)} 条删除记录", ""]
    kb_rows = []
    for r in recs:
        title = chats.get(r["chat_id"], str(r["chat_id"]))
        conf = r["confidence"] or 0
        lines.append(
            f"#{r['id']} [{conf:.2f}] {title}\n"
            f"   {str(r['full_name'] or r['user_id'])[:20]}: "
            f"{str(r['text_snapshot'] or '')[:52]}"
        )
        kb_rows.append([InlineKeyboardButton(
            f"↩️ 恢复 #{r['id']}", callback_data=f"rev_{r['id']}")])

    text = "\n".join(lines)
    if len(text) > 3800:
        text = text[:3800] + "\n…（已截断，用 /recent <数量> 调整）"
    await msg.reply_text(text, reply_markup=InlineKeyboardMarkup(kb_rows),
                         disable_web_page_preview=True)


async def cmd_revive(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """按记录 ID 恢复。/revive 123

    消息已被删除无法找回，这个命令的作用是：把该内容记为反例（收紧尺度）
    + 撤销违规计数。这是比事后翻日志更直接的反馈方式。
    """
    if not _is_admin_user(update.effective_user.id if update.effective_user else None):
        return
    msg = update.effective_message
    if not msg:
        return
    if not context.args or not context.args[0].isdigit():
        await msg.reply_text("用法：/revive <记录ID>\n记录 ID 见删除通知或 /recent")
        return
    rec_id = int(context.args[0])
    rec = await asyncio.to_thread(db.get_spam_record, rec_id)
    if not rec:
        await msg.reply_text(f"❌ 找不到记录 {rec_id}")
        return
    if rec["action"] == "revive":
        await msg.reply_text(f"ℹ️ 记录 {rec_id} 已经是恢复状态")
        return

    if rec["text_snapshot"]:
        await asyncio.to_thread(db.add_fewshot, rec["chat_id"], "negative",
                                rec["text_snapshot"], "revive")
    await asyncio.to_thread(db.execute,
                            "UPDATE spam_records SET action='revive' WHERE id=?",
                            (rec_id,))
    await asyncio.to_thread(db.execute,
                            """UPDATE users SET del_count = MAX(0, del_count - 1)
                               WHERE chat_id=? AND user_id=?""",
                            (rec["chat_id"], rec["user_id"]))
    await msg.reply_text(
        f"✅ 已恢复记录 {rec_id}\n"
        f"用户：{rec['full_name'] or rec['user_id']}\n"
        f"内容：{str(rec['text_snapshot'] or '')[:80]}\n\n"
        f"该内容已记为反例。消息本身无法找回。"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_admin_user(update.effective_user.id if update.effective_user else None):
        return
    if not update.message:
        return
    rows = await asyncio.to_thread(db.list_active_chats)
    lines = ["🤖 反广告机器人状态", f"运行群数：{len(rows)}", ""]
    for r in rows[:15]:
        usage = await asyncio.to_thread(db.usage_today, r["chat_id"])
        mode = "影子" if r["shadow_mode"] else "删除"
        lines.append(
            f"• {r['title'] or r['chat_id']} [{mode}] "
            f"阈值{r['confidence_cut']} 今日删{await asyncio.to_thread(db.count_deleted_since, r['chat_id'], 24)}条 "
            f"缓存{usage['cache_hit_rate']:.0%} ¥{usage['cost_yuan']:.4f}"
        )
    await update.message.reply_text("\n".join(lines))


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    if not _is_admin_user(query.from_user.id):
        await query.answer("🚫 权限不足", show_alert=True)
        return
    await query.answer()
    data = query.data or ""

    if data == "adm_overview":
        rows = await asyncio.to_thread(db.list_active_chats)
        lines = ["📊 各群概览", ""]
        for r in rows[:20]:
            n24 = await asyncio.to_thread(db.count_deleted_since, r["chat_id"], 24)
            mode = "👻影子" if r["shadow_mode"] else "🗑️删除"
            lines.append(f"{mode} {r['title'] or r['chat_id']}: 24h删{n}条 "
                         f"阈值{r['confidence_cut']} 预算{r['daily_token_cap']}")
        await query.edit_message_text("\n".join(lines), reply_markup=_admin_kb())

    elif data == "adm_usage":
        rows = await asyncio.to_thread(db.list_active_chats)
        lines = ["💾 今日用量", ""]
        total = 0.0
        for r in rows[:20]:
            u = await asyncio.to_thread(db.usage_today, r["chat_id"])
            total += u["cost_yuan"]
            lines.append(f"• {r['title'] or r['chat_id']}: {u['calls']}次调用 "
                         f"命中{u['cache_hit_rate']:.0%} ¥{u['cost_yuan']:.4f}")
        lines += ["", f"合计：¥{total:.4f}", "（按高峰时段单价估算）"]
        await query.edit_message_text("\n".join(lines), reply_markup=_admin_kb())

    elif data == "adm_wl":
        rows = await asyncio.to_thread(db.list_active_chats)
        lines = ["📋 白名单", ""]
        for r in rows[:10]:
            wl = await asyncio.to_thread(db.whitelist_list, r["chat_id"])
            if wl:
                lines.append(f"【{r['title'] or r['chat_id']}】")
                for w in wl[:10]:
                    lines.append(f"  • {w['full_name'] or w['user_id']} [{w['level']}]")
        if len(lines) == 2:
            lines.append("暂无")
        await query.edit_message_text("\n".join(lines), reply_markup=_admin_kb())

    elif data == "adm_bl":
        lines = ["🚫 黑名单", ""]
        for r in await asyncio.to_thread(db.list_active_chats):
            ids = await asyncio.to_thread(db.blacklist_ids, r["chat_id"])
            if ids:
                lines.append(f"【{r['title'] or r['chat_id']}】{len(ids)} 人")
        if len(lines) == 2:
            lines.append("暂无")
        await query.edit_message_text("\n".join(lines), reply_markup=_admin_kb())

    elif data == "adm_reports":
        lines = ["📨 举报统计", ""]
        for r in (await asyncio.to_thread(db.list_active_chats))[:10]:
            o = await asyncio.to_thread(db.report_overview, r["chat_id"])
            lines.append(f"【{r['title'] or r['chat_id']}】总{o['total']} "
                         f"有效{o['deleted']} 驳回{o['dismissed']} 待审{o['pending']}")
        await query.edit_message_text("\n".join(lines), reply_markup=_admin_kb())

    elif data == "adm_recent":
        lines = ["📜 最近判定记录", ""]
        for r in (await asyncio.to_thread(db.list_active_chats))[:5]:
            recs = await asyncio.to_thread(db.recent_spam_records, r["chat_id"], 5)
            if recs:
                lines.append(f"【{r['title'] or r['chat_id']}】")
                for s in recs:
                    lines.append(f"  [{s['action']}] {s['confidence'] or 0:.2f} "
                                 f"{(s['text_snapshot'] or '')[:34]}")
        if len(lines) == 2:
            lines.append("暂无")
        await query.edit_message_text("\n".join(lines), reply_markup=_admin_kb())

    elif data.startswith("revive_"):
        # 兼容旧版按钮（revive_{chat}_{user}），新消息用 rev_{record_id}
        await query.edit_message_text(
            "这条按钮来自旧版通知，无法精确定位。请用 /recent 重新操作。")

    elif data.startswith("rev_"):
        # 按记录 ID 恢复：同批多条时能精确区分，且不误伤同一用户的其他消息
        try:
            rec_id = int(data.split("_", 1)[1])
        except (IndexError, ValueError):
            await query.edit_message_text("❌ 无效的恢复请求")
            return
        rec = await asyncio.to_thread(db.get_spam_record, rec_id)
        if not rec:
            await query.edit_message_text("❌ 找不到该记录")
            return
        uid, cid = rec["user_id"], rec["chat_id"]

        # 1) 该内容记为反例（最有价值的反馈信号）
        if rec["text_snapshot"]:
            await asyncio.to_thread(db.add_fewshot, cid, "negative",
                                    rec["text_snapshot"], "revive")
        # 2) 把该条记录标记为已恢复
        await asyncio.to_thread(db.execute,
                                "UPDATE spam_records SET action='revive' WHERE id=?",
                                (rec_id,))
        # 3) 撤销计数（这条本不该算违规）
        await asyncio.to_thread(db.execute,
                                """UPDATE users SET del_count = MAX(0, del_count - 1)
                                   WHERE chat_id=? AND user_id=?""", (cid, uid))
        await query.edit_message_text(
            f"✅ 已恢复这条（记录 {rec_id}）\n"
            f"内容已记为反例，同类判定的尺度会随之收紧。\n"
            f"用户：{rec['full_name'] or uid}\n\n"
            f"注意：消息本身已被删除，无法找回；"
            f"如该用户是长期误删对象，用 /wl 加入白名单。")

    elif data.startswith("blk_"):
        try:
            rec_id = int(data.split("_", 1)[1])
        except (IndexError, ValueError):
            await query.edit_message_text("❌ 无效请求")
            return
        rec = await asyncio.to_thread(db.get_spam_record, rec_id)
        if not rec:
            await query.edit_message_text("❌ 找不到该记录")
            return
        await asyncio.to_thread(db.blacklist_add, rec["chat_id"],
                                rec["user_id"], rec["full_name"],
                                query.from_user.id, "管理员从通知拉黑")
        await query.edit_message_text(
            f"🚫 {rec['full_name'] or rec['user_id']} 已拉黑，其消息将被直接删除。")


# ---------------------------------------------------------------------------
# 定时任务
# ---------------------------------------------------------------------------

async def job_keepalive(context: ContextTypes.DEFAULT_TYPE) -> None:
    with open(HEARTBEAT_FILE, "w") as f:
        f.write(str(time.time()))


async def job_cleanup(context: ContextTypes.DEFAULT_TYPE) -> None:
    """按时间硬删过期消息。V1 只删"已总结"的，导致积压 114 天。"""
    n = await asyncio.to_thread(db.purge_old_messages, MESSAGE_RETENTION_DAYS)
    if n:
        logger.info("清理过期消息 %d 条（保留 %d 天）", n, MESSAGE_RETENTION_DAYS)


async def job_report_board(context: ContextTypes.DEFAULT_TYPE) -> None:
    """每日举报榜。门槛按群配置（V1 硬编码 50%）。"""
    for r in await asyncio.to_thread(db.list_active_chats):
        if not r["report_board"]:
            continue
        cid = r["chat_id"]
        await asyncio.to_thread(db.snapshot_report_stats, cid)
        stats = await asyncio.to_thread(db.report_stats_today, cid,
                                        r["report_min_rate"])
        if not stats:
            continue
        medals = ["🥇", "🥈", "🥉"]
        lines = ["📊 今日举报榜（感谢共同维护群环境）", ""]
        for i, s in enumerate(stats[:5]):
            medal = medals[i] if i < 3 else f"{i+1}."
            lines.append(f"{medal} {s['reporter_name']}  有效{s['valid']}/{s['total']}")
        try:
            await context.bot.send_message(chat_id=cid, text="\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            logger.warning("发送举报榜失败 chat=%s: %s", cid, exc)


async def job_refresh_admins(context: ContextTypes.DEFAULT_TYPE) -> None:
    for r in await asyncio.to_thread(db.list_active_chats):
        await _refresh_admins(context.application, r["chat_id"])


async def job_heartbeat_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """自检：判定连续失败时告警，避免静默失效。"""
    if not ADMIN_ID or not _manager:
        return
    for cid, p in _manager._pipelines.items():  # noqa: SLF001
        if p.last_error:
            await _send_admin(context.bot,
                              f"⚠️ 群 {cid} 判定异常：{p.last_error}")
            p.last_error = None


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

async def post_init(app: Application) -> None:
    global _manager, _client
    with open(HEARTBEAT_FILE, "w") as f:
        f.write(str(time.time()))

    _client = DeepSeekClient(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
                             model=MODEL_NAME)
    _manager = PipelineManager(_client)

    chats = await asyncio.to_thread(db.list_active_chats)
    ids = [c["chat_id"] for c in chats]
    await _manager.start(
        ids,
        on_spam=_make_on_spam(app),
        notify=_make_notify(app),
        on_log=_make_on_log(app),
    )
    for cid in ids:
        await _refresh_admins(app, cid)

    logger.info("已上线，模型=%s，服务 %d 个群（未注册的群会被自动注册）",
                MODEL_NAME, len(ids))
    await _send_admin(
        app.bot,
        (f"🤖 反广告机器人已上线\n"
         f"模型：{MODEL_NAME}\n"
         f"服务群数：{len(ids)}\n"
         f"判定阈值：{AUTO_REGISTER_CUT}（实测最优档）\n"
         f"数据库：{DB_PATH}\n\n"
         f"把机器人拉进群即可自动注册，无需手动操作。"),
    )


async def post_shutdown(app: Application) -> None:
    if _manager:
        await _manager.stop()
    if _client:
        await _client.close()
    logger.info("已下线")


async def preflight() -> int:
    """启动前自检：验证 token、模型、数据库、权限，不进入轮询。

    用途：部署前确认配置正确。尤其要确认机器人在目标群里有删除权限——
    Telegram 的硬限制是普通成员只能删自己的消息，没有管理员权限就等于不工作。
    """
    ok = True
    print("=" * 60)
    print("  反广告机器人 启动自检")
    print("=" * 60)

    # 1. 数据库
    try:
        _d = os.path.dirname(os.path.abspath(DB_PATH))
        if _d and not os.path.isdir(_d):
            os.makedirs(_d, exist_ok=True)
        db.init_db(DB_PATH)
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        print(f"  [1] 数据库      OK  {DB_PATH}  journal={mode}")
        if mode.lower() != "wal":
            print("      ⚠️ 非 WAL 模式，并发写入会阻塞")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"  [1] 数据库      FAIL  {exc}")
        return 1

    # 2. Telegram
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    try:
        async with app:
            me = await app.bot.get_me()
            print(f"  [2] Telegram    OK  @{me.username} ({me.first_name})")
            if not me.can_read_all_group_messages:
                print("      ⚠️ 隐私模式仍开启（can_read_all_group_messages=False），"
                      "机器人只能收到 @提及 与回复，会漏掉绝大多数广告！")
                print("         修复：@BotFather -> /setprivacy -> 选本机器人 -> Disable")
                ok = False
            else:
                print("      隐私模式已关闭，可收到群内全部消息")
    except Exception as exc:  # noqa: BLE001
        print(f"  [2] Telegram    FAIL  {exc}")
        return 1

    # 3. 已注册的群与权限
    chats = db.list_active_chats()
    print(f"  [3] 注册群      {len(chats)} 个")
    for c in chats:
        cid = c["chat_id"]
        try:
            member = await app.bot.get_chat_member(cid, app.bot.id)
            can_delete = getattr(member, "can_delete_messages", False)
            status = member.status
            mark = "OK  " if can_delete else "WARN"
            print(f"      [{mark}] {c['title'] or cid}  status={status}  "
                  f"可删消息={can_delete}  "
                  f"模式={'影子' if c['shadow_mode'] else '删除'}  "
                  f"阈值={c['confidence_cut']}")
            if not can_delete:
                print("            ⚠️ 缺少删除权限，判定结果无法执行！"
                      "  请把机器人设为群管理员")
                ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"      [FAIL] {cid}: {exc}（机器人可能不在该群）")
            ok = False

    # 4. DeepSeek
    try:
        client = DeepSeekClient(api_key=DEEPSEEK_API_KEY,
                                base_url=DEEPSEEK_BASE_URL, model=MODEL_NAME)
        messages = [
            {"role": "system", "content": "只输出 JSON：{\"verdicts\":[]}"},
            {"role": "user", "content": "[ID=1] 昵称=x | 内容=测试"},
        ]
        parsed, usage = await client.judge(messages, user_id="preflight")
        await client.close()
        print(f"  [4] DeepSeek    OK  {MODEL_NAME}  "
              f"cache_hit={usage.cache_hit_tokens} "
              f"miss={usage.cache_miss_tokens} "
              f"out={usage.completion_tokens} "
              f"reasoning={usage.reasoning_tokens}")
        if usage.reasoning_tokens:
            print("      ⚠️ 出现推理 token，thinking 关闭参数未生效，会白烧输出成本")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"  [4] DeepSeek    FAIL  {exc}")
        ok = False

    print("=" * 60)
    print("  自检结果：" + ("全部通过，可以启动" if ok else "存在问题，见上方 ⚠️"))
    print("=" * 60)
    return 0 if ok else 1


def main() -> None:
    if "--check" in sys.argv:
        sys.exit(asyncio.run(preflight()))

    if not ADMIN_ID:
        logger.warning("未设置 ADMIN_ID，将无法接收通知")

    # SQLite 不会自动创建父目录。compose 里 DB_NAME=/app/data/antispam.db，
    # 挂载的空卷首次启动时 data/ 可能不存在，不建就会直接崩。
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    if db_dir and not os.path.isdir(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info("已创建数据库目录: %s", db_dir)

    # 廉价自检：这三个必须是同步工厂（返回协程函数）。
    # 误加 async 会让它们返回未 await 的协程，功能静默失效。
    import inspect as _inspect
    for _fn in (_make_on_spam, _make_notify, _make_on_log):
        if _inspect.iscoroutinefunction(_fn):
            raise RuntimeError(
                f"{_fn.__name__} 被误声明为 async。它应当是同步工厂，"
                f"返回 async 回调；否则调用处会拿到未 await 的协程，功能静默失效。"
            )

    db.init_db(DB_PATH)

    app = (ApplicationBuilder()
           .token(BOT_TOKEN)
           .post_init(post_init)
           .post_shutdown(post_shutdown)
           .build())

    # 机器人自身的成员变化（被拉进群/成为管理员/被踢出）用 ChatMemberHandler
    # 配 MY_CHAT_MEMBER 过滤器。没有 MyChatMemberHandler 这个类。
    app.add_handler(ChatMemberHandler(on_my_chat_member,
                                     ChatMemberHandler.MY_CHAT_MEMBER))
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CommandHandler("wl", cmd_wl))
    app.add_handler(CommandHandler("bl", cmd_bl))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("logchannel", cmd_logchannel))
    app.add_handler(CommandHandler("recent", cmd_recent))
    app.add_handler(CommandHandler("revive", cmd_revive))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.CAPTION) & ~filters.COMMAND, handle_message))

    jq = app.job_queue
    if jq:
        jq.run_repeating(job_keepalive, interval=60, first=5)
        jq.run_repeating(job_heartbeat_check, interval=300, first=120)
        jq.run_repeating(job_refresh_admins, interval=3600, first=30)
        jq.run_daily(job_cleanup, time=dtime(hour=4, minute=30, tzinfo=TZ))
        jq.run_daily(job_report_board, time=dtime(hour=23, minute=55, tzinfo=TZ))
    else:
        logger.warning("JobQueue 不可用，定时任务不会运行")

    logger.info("启动轮询…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
