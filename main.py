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


async def _notify_spam(bot, verdicts: list[Verdict], shadow: bool) -> None:
    """通知管理员。影子模式标注"未执行"。"""
    if not ADMIN_ID or not verdicts:
        return
    by_chat: dict[int, list[Verdict]] = {}
    for v in verdicts:
        by_chat.setdefault(v.message.chat_id, []).append(v)

    for chat_id, items in by_chat.items():
        row = await asyncio.to_thread(db.get_chat, chat_id)
        title = (row["title"] if row else str(chat_id)) or str(chat_id)
        head = (f"👻 影子模式：本应删除 {len(items)} 条（未执行）" if shadow
                else f"🗑️ 已删除 {len(items)} 条广告")
        lines = [head, f"群：{html.escape(str(title))}", ""]
        for v in items[:10]:
            m = v.message
            mark = m.marker or "—"
            lines.append(
                f"• {html.escape(str(m.full_name)[:18])} {mark} "
                f"[{v.confidence:.2f}] {html.escape((m.text or '')[:60])}"
            )
        if len(items) > 10:
            lines.append(f"…另有 {len(items) - 10} 条")

        kb = None
        if not shadow and items:
            kb = _spam_keyboard(chat_id, items[0].message.user_id)
        await _send_admin(bot, "\n".join(lines), reply_markup=kb,
                          disable_web_page_preview=True)


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

    # 只处理群组
    if chat.type not in ("group", "supergroup"):
        return
    if user.is_bot:
        return

    row = await asyncio.to_thread(db.get_chat, chat.id)
    if not row or not row["is_active"]:
        logger.debug("未注册的群 chat=%s title=%s", chat.id, chat.title)
        return

    text = msg.text or msg.caption
    if not text:
        return

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

    pending = PendingMessage(
        row_id=row_id, message_id=msg.message_id, chat_id=chat.id,
        user_id=user.id, full_name=user.full_name, text=text,
        reply_to=msg.reply_to_message.message_id if msg.reply_to_message else None,
        timestamp=ts,
    )

    pipeline = _manager.get(chat.id, on_spam=_make_on_spam(context.application),
                            notify=_make_notify(context.application))
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
                            notify=_make_notify(context.application))
    # 举报直接进快车道：群友主动举报的应当尽快处理
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
        _, cid, uid = data.split("_", 2)
        cid_i, uid_i = int(cid), int(uid)
        await asyncio.to_thread(db.whitelist_set, cid_i, uid_i, "trusted",
                                None, query.from_user.id, "误删恢复")
        # 撤销信号是最有价值的反馈：记为反例，避免同类内容再被误删
        recs = await asyncio.to_thread(db.recent_spam_records, cid_i, 5)
        for s in recs:
            if s["user_id"] == uid_i and s["text_snapshot"]:
                await asyncio.to_thread(db.add_fewshot, cid_i, "negative",
                                        s["text_snapshot"], "revive")
                break
        await query.edit_message_text(
            f"✅ 已把 {uid_i} 加入白名单，并把最近一条内容记为反例。")

    elif data.startswith("bl_"):
        _, cid, uid = data.split("_", 2)
        cid_i, uid_i = int(cid), int(uid)
        await asyncio.to_thread(db.blacklist_add, cid_i, uid_i, None,
                                query.from_user.id, "管理员从通知拉黑")
        await query.edit_message_text(f"🚫 {uid_i} 已拉黑。")


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
