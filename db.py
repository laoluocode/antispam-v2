"""SQLite 数据层。

多租户设计：**每个群是独立租户**，配置/白名单/黑名单/模式库/样本全部带 chat_id。
V1 的致命缺陷是全局 `bot_settings` 与全局白名单——A 群加白会让 B 群也放行。

两个必须遵守的工程约束：

1. **必须用 WAL 模式。**
   V1 用 SQLite 默认的 `journal_mode=DELETE`，写操作互相阻塞。线上 3 天内
   出现 8 次 `实时删除失败: database is locked`、2 次定时任务异常。
   2026-10-10 我又在开发中踩了同一个坑：在生产库上跑重查询导致 bot 写入失败。
   WAL 允许读写并发，是本项目的地基。

2. **长事务是禁忌。**
   任何持锁超过秒级的操作都必须先复制快照再执行（见 export_eval_data.py）。
   这里所有函数都是短事务。

消息正文保留：产品需要"按记录ID回溯"、举报核实、以及给模型提供用户近期发言
作为判定上下文。保留期由 `messages` 表的清理任务控制，默认 30 天可配。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# 连接超时。WAL 下写冲突会等待而非立即失败。
DB_TIMEOUT = 30.0

_local = threading.local()
_db_path = "antispam.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- 租户注册表
CREATE TABLE IF NOT EXISTS chats (
    chat_id          INTEGER PRIMARY KEY,
    title            TEXT,
    chat_nature      TEXT    DEFAULT 'general',   -- general | adult_service | trade
    strictness       TEXT    DEFAULT 'strict',    -- strict | standard | loose
    confidence_cut   REAL    DEFAULT 0.50,        -- 判定阈值（可调，不动 prompt）
    shadow_mode      INTEGER DEFAULT 1,           -- 1 = 只记录不删除
    notify_admin     INTEGER DEFAULT 1,
    report_board     INTEGER DEFAULT 1,           -- 每日举报榜开关
    report_min_rate  REAL    DEFAULT 0.50,        -- 上榜门槛
    fastlane_seconds INTEGER DEFAULT 15,          -- L1 快车道攒批窗口
    batch_seconds    INTEGER DEFAULT 90,          -- L2 常规攒批窗口
    batch_max        INTEGER DEFAULT 50,          -- L2 单批上限（也是攒够即发的阈值）
                                                  -- 设太大在稀疏群会导致攒不满就等满窗口
    daily_token_cap  INTEGER DEFAULT 300000,      -- 每群每日 token 预算
    report_limit     INTEGER DEFAULT 20,          -- 每人每日举报额度
    log_channel_id   INTEGER,                     -- 删除记录频道（0/NULL=不发）
    normal_words     TEXT,                        -- 本群正常话题词，逗号分隔
                                                  -- 防止把群内惯例词误判为广告
    blocked_words    TEXT,                        -- 本群明确要删的词，逗号分隔
    is_active        INTEGER DEFAULT 1,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 群管理员（免检）
CREATE TABLE IF NOT EXISTS chat_admins (
    chat_id   INTEGER NOT NULL,
    user_id   INTEGER NOT NULL,
    full_name TEXT,
    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, user_id)
);

-- 用户画像：信任信号的数据来源
-- 实测依据：发言量无区分度（发言 1000+ 的用户里广告号占 2.4%，发言 <10 的是 1.7%），
-- 而删除比极强（21.4% 用户从未被删，24.3% 删除比 >90%）。
CREATE TABLE IF NOT EXISTS users (
    chat_id    INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    full_name  TEXT,
    username   TEXT,
    msg_count  INTEGER DEFAULT 0,
    del_count  INTEGER DEFAULT 0,      -- 被删消息数（去重）
    first_seen DATETIME,
    last_seen  DATETIME,
    PRIMARY KEY (chat_id, user_id)
);

-- 消息流水。is_checked 控制是否已送判定。
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL,
    message_id  INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    full_name   TEXT,
    text        TEXT,
    reply_to    INTEGER,
    timestamp   DATETIME,
    is_checked  INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,          -- 判定失败重试次数，超过上限则放弃
    UNIQUE (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_msg_unchecked ON messages (is_checked, chat_id);
CREATE INDEX IF NOT EXISTS idx_msg_chat_time ON messages (chat_id, timestamp);

-- 用户近期发言的内存缓存来源（判定上下文用）
CREATE INDEX IF NOT EXISTS idx_msg_user ON messages (chat_id, user_id, id DESC);

-- 判定与删除记录
CREATE TABLE IF NOT EXISTS spam_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    user_id       INTEGER NOT NULL,
    full_name     TEXT,
    message_id    INTEGER,
    text_snapshot TEXT,                      -- V1 缺此列，导致事后无法回溯
    confidence    REAL,
    source        TEXT,                      -- local_rule | llm | report | blacklist
    action        TEXT DEFAULT 'delete',     -- delete | observe | revive | ignore
    notified      INTEGER DEFAULT 0,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chat_id, message_id, action)
);
CREATE INDEX IF NOT EXISTS idx_spam_chat_time ON spam_records (chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_spam_user ON spam_records (chat_id, user_id, action);

-- 每群设置（替代 V1 的全局 bot_settings）
CREATE TABLE IF NOT EXISTS chat_settings (
    chat_id INTEGER NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT,
    PRIMARY KEY (chat_id, key)
);

-- 白名单三级：advertiser（付费商家，仅删 high）/ trusted（完全跳过）
CREATE TABLE IF NOT EXISTS spam_whitelist (
    chat_id    INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    level      TEXT DEFAULT 'trusted',       -- advertiser | trusted
    full_name  TEXT,
    added_by   INTEGER,
    reason     TEXT,
    expires_at DATETIME,                     -- 付费有期限，避免永久技术债
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, user_id)
);

-- 黑名单：零成本零误伤的第一道墙
CREATE TABLE IF NOT EXISTS spam_blacklist (
    chat_id    INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    full_name  TEXT,
    added_by   INTEGER,
    reason     TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, user_id)
);

-- 确定性模式库（shadow -> active）
CREATE TABLE IF NOT EXISTS spam_patterns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id           INTEGER,                -- NULL = 全局
    pattern           TEXT NOT NULL,
    mode              TEXT DEFAULT 'shadow',  -- shadow | active
    hit_count         INTEGER DEFAULT 0,
    shadow_hits       INTEGER DEFAULT 0,
    shadow_spam_hits  INTEGER DEFAULT 0,
    first_seen_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chat_id, pattern)
);

-- 群画像 few-shot（取代 V1 的"毕业规则"）
CREATE TABLE IF NOT EXISTS chat_fewshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    kind       TEXT NOT NULL,                 -- positive（广告）| negative（正常反例）
    content    TEXT NOT NULL,
    source     TEXT,                          -- report | revive | llm
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chat_id, kind, content)
);

-- 举报
CREATE TABLE IF NOT EXISTS spam_reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    reporter_id   INTEGER NOT NULL,
    reporter_name TEXT,
    user_id       INTEGER NOT NULL,
    full_name     TEXT,
    message_id    INTEGER NOT NULL,
    content       TEXT,
    source        TEXT DEFAULT 'in_group',    -- in_group | forward
    confidence    REAL,                       -- AI 判定置信度
    reviewed      INTEGER DEFAULT 0,
    action        TEXT,                       -- delete | dismiss
    reviewed_at   DATETIME,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_report_chat ON spam_reports (chat_id, reviewed);
CREATE INDEX IF NOT EXISTS idx_report_reporter ON spam_reports (reporter_id, created_at);

-- 被丢弃的消息（诊断用，回答"为什么这条没被处理"）
CREATE TABLE IF NOT EXISTS dropped_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER,
    chat_title TEXT,
    user_id    INTEGER,
    full_name  TEXT,
    text       TEXT,
    reason     TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_dropped_time ON dropped_messages (created_at);

-- 举报额度用量（每人每日）
CREATE TABLE IF NOT EXISTS report_usage (
    usage_date TEXT NOT NULL,
    chat_id    INTEGER NOT NULL,
    reporter_id INTEGER NOT NULL,
    used       INTEGER DEFAULT 0,
    PRIMARY KEY (usage_date, chat_id, reporter_id)
);

-- 每日榜统计（支持周榜/月榜）
CREATE TABLE IF NOT EXISTS daily_report_stats (
    stat_date TEXT NOT NULL,
    chat_id   INTEGER NOT NULL,
    user_id   INTEGER NOT NULL,
    full_name TEXT,
    total     INTEGER DEFAULT 0,
    valid     INTEGER DEFAULT 0,
    PRIMARY KEY (stat_date, chat_id, user_id)
);

-- Token 用量与成本
CREATE TABLE IF NOT EXISTS token_usage (
    usage_date    TEXT NOT NULL,
    chat_id       INTEGER NOT NULL,
    calls         INTEGER DEFAULT 0,
    hit_tokens    INTEGER DEFAULT 0,
    miss_tokens   INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_yuan     REAL DEFAULT 0,
    PRIMARY KEY (usage_date, chat_id)
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """为已存在的库补齐后加的列。SQLite 的 ADD COLUMN 是幂等的（先查再改）。"""
    additions = {
        "chats": [
            ("report_limit", "INTEGER DEFAULT 20"),
            ("log_channel_id", "INTEGER"),
            ("fastlane_seconds", "INTEGER DEFAULT 15"),
            ("normal_words", "TEXT"),
            ("blocked_words", "TEXT"),
        ],
        "spam_reports": [
            ("source", "TEXT DEFAULT 'in_group'"),
            ("confidence", "REAL"),
        ],
        "messages": [
            ("retry_count", "INTEGER DEFAULT 0"),
        ],
    }
    for table, cols in additions.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                logger.info("迁移：%s 新增列 %s", table, name)


def init_db(path: str) -> None:
    """建库。必须在任何连接之前调用一次。"""
    global _db_path
    _db_path = path
    conn = sqlite3.connect(path, timeout=DB_TIMEOUT)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if mode.lower() != "wal":
            logger.warning("journal_mode 不是 WAL 而是 %s，并发写入会阻塞", mode)
        else:
            logger.info("数据库就绪（WAL 模式）: %s", path)
    finally:
        conn.close()


def _conn() -> sqlite3.Connection:
    """每线程一个连接。SQLite 连接不可跨线程共享。"""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_db_path, timeout=DB_TIMEOUT)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


@contextmanager
def tx():
    """短事务。所有写操作走这里，异常时回滚。"""
    conn = _conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def query(sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
    return _conn().execute(sql, tuple(args)).fetchall()


def query_one(sql: str, args: Iterable = ()):
    return _conn().execute(sql, tuple(args)).fetchone()


def execute(sql: str, args: Iterable = ()) -> int:
    with tx() as conn:
        cur = conn.execute(sql, tuple(args))
        return cur.lastrowid or cur.rowcount


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# 租户
# --------------------------------------------------------------------------

def register_chat(chat_id: int, title: str, nature: str = "general",
                  shadow_mode: bool = True) -> bool:
    """注册群。已存在则只更新标题。返回是否新建。"""
    existing = query_one("SELECT 1 FROM chats WHERE chat_id=?", (chat_id,))
    if existing:
        execute("UPDATE chats SET title=? WHERE chat_id=?", (title, chat_id))
        return False
    execute(
        """INSERT INTO chats (chat_id, title, chat_nature, shadow_mode)
           VALUES (?,?,?,?)""",
        (chat_id, title, nature, 1 if shadow_mode else 0),
    )
    return True


def get_chat(chat_id: int):
    return query_one("SELECT * FROM chats WHERE chat_id=?", (chat_id,))


def list_active_chats() -> list[sqlite3.Row]:
    return query("SELECT * FROM chats WHERE is_active=1")


def set_chat_field(chat_id: int, field: str, value: Any) -> None:
    allowed = {
        "title", "chat_nature", "strictness", "confidence_cut", "shadow_mode",
        "notify_admin", "report_board", "report_min_rate", "fastlane_seconds",
        "batch_seconds", "batch_max", "daily_token_cap", "is_active",
        "report_limit", "log_channel_id", "normal_words", "blocked_words",
    }
    if field not in allowed:
        raise ValueError(f"不允许修改字段: {field}")
    with tx() as conn:
        cur = conn.execute(f"UPDATE chats SET {field}=? WHERE chat_id=?",
                           (value, chat_id))
        if cur.rowcount == 0:
            # 裸 UPDATE 影响 0 行是静默失败：调用方以为改好了，实际没改。
            # 这在"群 ID 打错"或"群还没注册"时很难排查，所以显式报错。
            raise ValueError(
                f"群 {chat_id} 不存在，无法设置 {field}。"
                f"先用 register_chat / manage.py add 注册"
            )


# --------------------------------------------------------------------------
# 每群设置
# --------------------------------------------------------------------------

def get_setting(chat_id: int, key: str, default: str = "") -> str:
    row = query_one("SELECT value FROM chat_settings WHERE chat_id=? AND key=?",
                    (chat_id, key))
    return row["value"] if row else default


def set_setting(chat_id: int, key: str, value: str) -> None:
    execute("""INSERT INTO chat_settings (chat_id, key, value) VALUES (?,?,?)
               ON CONFLICT(chat_id, key) DO UPDATE SET value=excluded.value""",
            (chat_id, key, value))


# --------------------------------------------------------------------------
# 用户画像
# --------------------------------------------------------------------------

def upsert_user_seen(chat_id: int, user_id: int, full_name: str,
                     username: str | None) -> None:
    """记录一次发言，推断首末时间。"""
    now = _now()
    execute("""
        INSERT INTO users (chat_id, user_id, full_name, username, msg_count,
                           first_seen, last_seen)
        VALUES (?,?,?,?,1,?,?)
        ON CONFLICT(chat_id, user_id) DO UPDATE SET
            msg_count = msg_count + 1,
            full_name = excluded.full_name,
            username  = excluded.username,
            last_seen = excluded.last_seen,
            first_seen = COALESCE(users.first_seen, excluded.first_seen)
    """, (chat_id, user_id, full_name, username, now, now))


def bump_user_deleted(chat_id: int, user_id: int,
                      full_name: str | None = None) -> None:
    """记一次删除。

    用 UPSERT 而非裸 UPDATE：如果该用户还没进过 users 表（消息先于画像落库、
    或服务重启后首次遇到），裸 UPDATE 会静默影响 0 行，删除计数就丢了——
    而 del_count 是信任信号的主分量，丢了会让惯犯被误判成正常用户。
    """
    now = _now()
    execute("""
        INSERT INTO users (chat_id, user_id, full_name, msg_count, del_count,
                           first_seen, last_seen)
        VALUES (?,?,?,0,1,?,?)
        ON CONFLICT(chat_id, user_id) DO UPDATE SET
            del_count = del_count + 1,
            full_name = COALESCE(excluded.full_name, users.full_name)
    """, (chat_id, user_id, full_name, now, now))


def user_stats(chat_id: int, user_id: int) -> dict:
    """取用户的信任信号分量。缺失时返回零值而非 None。"""
    row = query_one("""SELECT msg_count, del_count, first_seen, last_seen
                       FROM users WHERE chat_id=? AND user_id=?""",
                    (chat_id, user_id))
    if not row:
        return {"msg_n": 0, "del_n": 0, "span_days": 0.0}
    span = 0.0
    if row["first_seen"] and row["last_seen"]:
        try:
            a = datetime.fromisoformat(str(row["first_seen"]))
            b = datetime.fromisoformat(str(row["last_seen"]))
            span = round((b - a).total_seconds() / 86400, 1)
        except (ValueError, TypeError):
            span = 0.0
    return {"msg_n": row["msg_count"] or 0,
            "del_n": row["del_count"] or 0,
            "span_days": span}


# --------------------------------------------------------------------------
# 消息
# --------------------------------------------------------------------------

def save_dropped(chat_id: int, chat_title: str | None, user_id: int,
                 full_name: str | None, text: str, reason: str) -> None:
    """记录被丢弃的消息，用于回答"为什么这条广告没被处理"。"""
    execute("""INSERT INTO dropped_messages
               (chat_id, chat_title, user_id, full_name, text, reason)
               VALUES (?,?,?,?,?,?)""",
            (chat_id, chat_title, user_id, full_name, (text or "")[:400], reason))


def recent_dropped(limit: int = 30) -> list[sqlite3.Row]:
    return query("""SELECT * FROM dropped_messages
                    ORDER BY id DESC LIMIT ?""", (limit,))


def dropped_summary(since_hours: int = 24) -> list[sqlite3.Row]:
    cutoff = (datetime.now() - timedelta(hours=since_hours)).strftime("%Y-%m-%d %H:%M:%S")
    return query("""SELECT reason, COUNT(*) n FROM dropped_messages
                    WHERE created_at >= ? GROUP BY reason ORDER BY n DESC""",
                 (cutoff,))


def save_message(chat_id: int, message_id: int, user_id: int, full_name: str,
                 text: str, reply_to: int | None, timestamp: str) -> int | None:
    """入库一条待判定消息。重复 (chat_id, message_id) 会被忽略。"""
    try:
        return execute("""
            INSERT INTO messages (chat_id, message_id, user_id, full_name, text,
                                  reply_to, timestamp, is_checked)
            VALUES (?,?,?,?,?,?,?,0)
            ON CONFLICT(chat_id, message_id) DO NOTHING
        """, (chat_id, message_id, user_id, full_name, text, reply_to, timestamp))
    except sqlite3.Error as exc:
        logger.warning("save_message 失败 chat=%s msg=%s: %s", chat_id, message_id, exc)
        return None


def take_unchecked(chat_id: int, limit: int = 100) -> list[sqlite3.Row]:
    return query("""SELECT * FROM messages WHERE chat_id=? AND is_checked=0
                    ORDER BY id LIMIT ?""", (chat_id, limit))


def take_unchecked_batch(chat_id: int, limit: int) -> list[sqlite3.Row]:
    """取一批并把它们标记为已检测（原子）。

    必须先标记再删除：若删除中途失败，消息不会无限重试，
    但会留下 spam_records 记录供人工复核。
    """
    with tx() as conn:
        rows = conn.execute(
            """SELECT * FROM messages WHERE chat_id=? AND is_checked=0
               ORDER BY id LIMIT ?""", (chat_id, limit)).fetchall()
        if rows:
            ids = [r["id"] for r in rows]
            conn.executemany("UPDATE messages SET is_checked=1 WHERE id=?",
                             [(i,) for i in ids])
        return rows


# 判定失败的重试上限。超过则放弃并标记已检测，避免失败消息无限重试、
# 每轮都重新烧一遍 token。放弃的消息仍可在 spam_records / 日志里排查。
MAX_JUDGE_RETRIES = 3


def mark_checked(row_ids: list[int]) -> None:
    """把消息标记为已检测。

    ⚠️ 这一步在流水线里必须有调用点。漏掉会导致消息永远停在未检测状态，
    而取批逻辑每轮都会把它们重新送判定 —— 静默重复烧 token。
    （2026-10-10 线上就踩了这个坑：151 条消息全部 is_checked=0。）
    """
    if not row_ids:
        return
    with tx() as conn:
        conn.executemany("UPDATE messages SET is_checked=1 WHERE id=?",
                         [(i,) for i in row_ids])


def bump_retry(row_ids: list[int]) -> list[int]:
    """重试计数 +1，返回仍未超过上限、需要重试的 row_id。

    失败消息不能无限重试：API 若持续异常，每轮重试都是重复的 token 开销。
    """
    if not row_ids:
        return []
    with tx() as conn:
        conn.executemany(
            "UPDATE messages SET retry_count = retry_count + 1 WHERE id=?",
            [(i,) for i in row_ids])
        placeholders = ",".join("?" * len(row_ids))
        rows = conn.execute(
            f"""SELECT id FROM messages
                WHERE id IN ({placeholders}) AND retry_count < ?""",
            (*row_ids, MAX_JUDGE_RETRIES)).fetchall()
        keep = [r["id"] for r in rows]
        give_up = [i for i in row_ids if i not in set(keep)]
        if give_up:
            conn.executemany("UPDATE messages SET is_checked=1 WHERE id=?",
                             [(i,) for i in give_up])
    return keep


def recent_user_texts(chat_id: int, user_id: int, limit: int = 6) -> list[str]:
    """该用户近期发言，用作判定上下文。新到旧。"""
    rows = query("""SELECT text FROM messages WHERE chat_id=? AND user_id=?
                    ORDER BY id DESC LIMIT ?""", (chat_id, user_id, limit))
    return [r["text"] for r in rows if r["text"]]


def purge_old_messages(days: int) -> int:
    """按时间硬删。V1 的教训：只删"已总结"的消息导致积压 114 天。"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with tx() as conn:
        cur = conn.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
        return cur.rowcount


# --------------------------------------------------------------------------
# 判定记录
# --------------------------------------------------------------------------

# Telegram 的删除错误里，这几类是"重试也不会成功"的。
# V1 有这个判断（_is_non_retryable_delete_error），但线上仍有 84 次
# "Message to delete not found" 被当成普通失败，反复重试。
_NON_RETRYABLE_DELETE = (
    "message to delete not found",
    "message can't be deleted",
    "message identifier is not specified",
    "message is not modified",
)


def is_non_retryable_delete_error(err: Exception) -> bool:
    """判断删除失败是否不值得重试（消息已被自己/管理员删掉等）。"""
    text = str(err).lower()
    return any(p in text for p in _NON_RETRYABLE_DELETE)


def save_spam_record(chat_id: int, user_id: int, full_name: str,
                     message_id: int | None, text: str, confidence: float,
                     source: str, action: str = "delete") -> int | None:
    try:
        return execute("""
            INSERT INTO spam_records (chat_id, user_id, full_name, message_id,
                                      text_snapshot, confidence, source, action)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(chat_id, message_id, action) DO NOTHING
        """, (chat_id, user_id, full_name, message_id, text, confidence,
              source, action))
    except sqlite3.Error as exc:
        logger.warning("save_spam_record 失败: %s", exc)
        return None


def recent_spam_records(chat_id: int, limit: int = 10) -> list[sqlite3.Row]:
    return query("""SELECT * FROM spam_records WHERE chat_id=? AND action='delete'
                    ORDER BY id DESC LIMIT ?""", (chat_id, limit))


def get_spam_record(record_id: int):
    return query_one("SELECT * FROM spam_records WHERE id=?", (record_id,))


def recent_spam_records_all(limit: int = 20) -> list[sqlite3.Row]:
    """跨群取最近记录，供 /recent 使用。"""
    return query("""SELECT * FROM spam_records WHERE action='delete'
                    ORDER BY id DESC LIMIT ?""", (limit,))


def count_deleted_since(chat_id: int, hours: int = 24) -> int:
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    row = query_one("""SELECT COUNT(*) n FROM spam_records
                       WHERE chat_id=? AND action='delete' AND created_at>=?""",
                    (chat_id, cutoff))
    return row["n"] if row else 0


# --------------------------------------------------------------------------
# 白名单 / 黑名单
# --------------------------------------------------------------------------

def whitelist_set(chat_id: int, user_id: int, level: str, full_name: str | None,
                  added_by: int | None, reason: str | None = None,
                  expires_at: str | None = None) -> None:
    """level: advertiser（付费商家，仅删 high）| trusted（完全跳过）"""
    if level not in ("advertiser", "trusted"):
        raise ValueError(f"未知白名单档位: {level}")
    execute("""
        INSERT INTO spam_whitelist (chat_id, user_id, level, full_name,
                                    added_by, reason, expires_at)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(chat_id, user_id) DO UPDATE SET
            level=excluded.level, full_name=excluded.full_name,
            reason=excluded.reason, expires_at=excluded.expires_at
    """, (chat_id, user_id, level, full_name, added_by, reason, expires_at))


def whitelist_remove(chat_id: int, user_id: int) -> bool:
    with tx() as conn:
        cur = conn.execute("DELETE FROM spam_whitelist WHERE chat_id=? AND user_id=?",
                           (chat_id, user_id))
        return cur.rowcount > 0


def whitelist_level(chat_id: int, user_id: int) -> str | None:
    """取白名单档位。过期的自动忽略。"""
    row = query_one("""SELECT level, expires_at FROM spam_whitelist
                       WHERE chat_id=? AND user_id=?""", (chat_id, user_id))
    if not row:
        return None
    if row["expires_at"]:
        try:
            if datetime.fromisoformat(str(row["expires_at"])) < datetime.now():
                return None
        except (ValueError, TypeError):
            pass
    return row["level"]


def whitelist_list(chat_id: int, limit: int = 100) -> list[sqlite3.Row]:
    return query("""SELECT * FROM spam_whitelist WHERE chat_id=?
                    ORDER BY created_at DESC LIMIT ?""", (chat_id, limit))


def blacklist_add(chat_id: int, user_id: int, full_name: str | None,
                  added_by: int | None, reason: str | None = None) -> None:
    execute("""INSERT INTO spam_blacklist (chat_id, user_id, full_name, added_by, reason)
               VALUES (?,?,?,?,?)
               ON CONFLICT(chat_id, user_id) DO UPDATE SET reason=excluded.reason""",
            (chat_id, user_id, full_name, added_by, reason))


def blacklist_remove(chat_id: int, user_id: int) -> bool:
    with tx() as conn:
        cur = conn.execute("DELETE FROM spam_blacklist WHERE chat_id=? AND user_id=?",
                           (chat_id, user_id))
        return cur.rowcount > 0


def blacklist_ids(chat_id: int) -> set[int]:
    return {r["user_id"] for r in
            query("SELECT user_id FROM spam_blacklist WHERE chat_id=?", (chat_id,))}


# --------------------------------------------------------------------------
# 举报
# --------------------------------------------------------------------------

def save_report(chat_id: int, reporter_id: int, reporter_name: str, user_id: int,
                full_name: str, message_id: int, content: str) -> int | None:
    try:
        return execute("""
            INSERT INTO spam_reports (chat_id, reporter_id, reporter_name,
                                      user_id, full_name, message_id, content)
            VALUES (?,?,?,?,?,?,?)
        """, (chat_id, reporter_id, reporter_name, user_id, full_name,
              message_id, content))
    except sqlite3.Error as exc:
        logger.warning("save_report 失败: %s", exc)
        return None


def report_quota_left(chat_id: int, reporter_id: int) -> int:
    """今日剩余举报额度。额度是防刷的手段，不是省钱手段——
    单条判定约 0.00006 元，钱不是问题；真正的成本是误报占用管理员注意力。"""
    limit = 20
    row = query_one("SELECT report_limit FROM chats WHERE chat_id=?", (chat_id,))
    if row and row["report_limit"] is not None:
        limit = row["report_limit"]
    if limit <= 0:
        return 10 ** 6  # 0 表示不限
    today = datetime.now().strftime("%Y-%m-%d")
    r = query_one("""SELECT used FROM report_usage
                     WHERE usage_date=? AND chat_id=? AND reporter_id=?""",
                  (today, chat_id, reporter_id))
    return max(0, limit - (r["used"] if r else 0))


def report_quota_consume(chat_id: int, reporter_id: int) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    execute("""
        INSERT INTO report_usage (usage_date, chat_id, reporter_id, used)
        VALUES (?,?,?,1)
        ON CONFLICT(usage_date, chat_id, reporter_id)
        DO UPDATE SET used = used + 1
    """, (today, chat_id, reporter_id))


def reporter_accuracy(reporter_id: int, chat_id: int = 0) -> tuple[int, int]:
    """举报人的历史准确率 (有效数, 总数)。用于识别乱举报的人。"""
    if chat_id:
        row = query_one("""
            SELECT COUNT(*) total,
                   SUM(CASE WHEN action='delete' THEN 1 ELSE 0 END) valid
            FROM spam_reports WHERE reporter_id=? AND chat_id=?""",
            (reporter_id, chat_id))
    else:
        row = query_one("""
            SELECT COUNT(*) total,
                   SUM(CASE WHEN action='delete' THEN 1 ELSE 0 END) valid
            FROM spam_reports WHERE reporter_id=?""", (reporter_id,))
    if not row:
        return 0, 0
    return (row["valid"] or 0), (row["total"] or 0)


def _split_words(raw: str | None) -> list[str]:
    """逗号/顿号/空格分隔的词表。去空去重，保序。"""
    if not raw:
        return []
    out, seen = [], set()
    for w in raw.replace("，", ",").replace("、", ",").replace(" ", ",").split(","):
        w = w.strip()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def normal_words(chat_id: int) -> list[str]:
    """本群正常话题词。用于告诉模型"这些词在这里不算广告"。"""
    row = query_one("SELECT normal_words FROM chats WHERE chat_id=?", (chat_id,))
    return _split_words(row["normal_words"] if row else None)


def blocked_words(chat_id: int) -> list[str]:
    """本群明确要删的词。用于补充模型的领域知识。"""
    row = query_one("SELECT blocked_words FROM chats WHERE chat_id=?", (chat_id,))
    return _split_words(row["blocked_words"] if row else None)


def add_word(chat_id: int, word: str, kind: str = "normal") -> list[str]:
    """往词表加一个词（幂等）。返回加完后的完整词表。"""
    if kind not in ("normal", "blocked"):
        raise ValueError(f"未知词表类型: {kind}")
    col = "normal_words" if kind == "normal" else "blocked_words"
    cur = normal_words(chat_id) if kind == "normal" else blocked_words(chat_id)
    if word not in cur:
        cur.append(word)
    value = ",".join(cur)
    with tx() as conn:
        r = conn.execute(f"UPDATE chats SET {col}=? WHERE chat_id=?",
                         (value, chat_id))
        if r.rowcount == 0:
            raise ValueError(f"群 {chat_id} 不存在")
    return cur


def remove_word(chat_id: int, word: str, kind: str = "normal") -> list[str]:
    if kind not in ("normal", "blocked"):
        raise ValueError(f"未知词表类型: {kind}")
    col = "normal_words" if kind == "normal" else "blocked_words"
    cur = normal_words(chat_id) if kind == "normal" else blocked_words(chat_id)
    cur = [w for w in cur if w != word]
    with tx() as conn:
        conn.execute(f"UPDATE chats SET {col}=? WHERE chat_id=?",
                     (",".join(cur), chat_id))
    return cur


def set_log_channel(chat_id: int, channel_id: int | None) -> bool:
    """设置记录频道。群不存在时返回 False 而非静默无操作。"""
    with tx() as conn:
        cur = conn.execute("UPDATE chats SET log_channel_id=? WHERE chat_id=?",
                           (channel_id, chat_id))
        if cur.rowcount == 0:
            logger.warning("set_log_channel: 群 %s 不存在", chat_id)
            return False
    return True


def chats_with_log_channel() -> list[sqlite3.Row]:
    return query("""SELECT * FROM chats
                    WHERE is_active=1 AND log_channel_id IS NOT NULL
                      AND log_channel_id != 0""")


def mark_report_reviewed(report_id: int, action: str) -> None:
    execute("""UPDATE spam_reports SET reviewed=1, action=?, reviewed_at=?
               WHERE id=?""", (action, _now(), report_id))


def get_report(report_id: int):
    return query_one("SELECT * FROM spam_reports WHERE id=?", (report_id,))


def report_overview(chat_id: int) -> dict:
    row = query_one("""
        SELECT COUNT(*) total,
               SUM(CASE WHEN reviewed=0 THEN 1 ELSE 0 END) pending,
               SUM(CASE WHEN action='delete' THEN 1 ELSE 0 END) deleted,
               SUM(CASE WHEN action='dismiss' THEN 1 ELSE 0 END) dismissed
        FROM spam_reports WHERE chat_id=?
    """, (chat_id,))
    return {k: (row[k] or 0) for k in ("total", "pending", "deleted", "dismissed")} \
        if row else {"total": 0, "pending": 0, "deleted": 0, "dismissed": 0}


def report_stats_today(chat_id: int, min_rate: float = 0.5,
                       min_reports: int = 2) -> list[sqlite3.Row]:
    """今日举报榜数据源。门槛按群配置（V1 硬编码 50%）。"""
    today = datetime.now().strftime("%Y-%m-%d")
    return query("""
        SELECT reporter_name, COUNT(*) total,
               SUM(CASE WHEN action='delete' THEN 1 ELSE 0 END) valid
        FROM spam_reports
        WHERE chat_id=? AND DATE(created_at)=?
        GROUP BY reporter_id
        HAVING total >= ?
           AND (CAST(SUM(CASE WHEN action='delete' THEN 1 ELSE 0 END) AS REAL)
                / COUNT(*)) >= ?
        ORDER BY valid DESC, total DESC LIMIT 5
    """, (chat_id, today, min_reports, min_rate))


def snapshot_report_stats(chat_id: int) -> int:
    """把今日举报榜落库，支持周榜/月榜。"""
    today = datetime.now().strftime("%Y-%m-%d")
    rows = query("""
        SELECT reporter_id, reporter_name, COUNT(*) total,
               SUM(CASE WHEN action='delete' THEN 1 ELSE 0 END) valid
        FROM spam_reports WHERE chat_id=? AND DATE(created_at)=?
        GROUP BY reporter_id
    """, (chat_id, today))
    with tx() as conn:
        for r in rows:
            conn.execute("""
                INSERT INTO daily_report_stats
                    (stat_date, chat_id, user_id, full_name, total, valid)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(stat_date, chat_id, user_id) DO UPDATE SET
                    total=excluded.total, valid=excluded.valid
            """, (today, chat_id, r["reporter_id"], r["reporter_name"],
                  r["total"], r["valid"] or 0))
    return len(rows)


# --------------------------------------------------------------------------
# 模式库 / few-shot
# --------------------------------------------------------------------------

def active_patterns() -> list[sqlite3.Row]:
    return query("SELECT * FROM spam_patterns WHERE mode='active'")


def patterns_for_shadow() -> list[sqlite3.Row]:
    return query("SELECT * FROM spam_patterns WHERE mode='shadow'")


def add_pattern(chat_id: int | None, pattern: str, mode: str = "shadow") -> None:
    execute("""INSERT INTO spam_patterns (chat_id, pattern, mode) VALUES (?,?,?)
               ON CONFLICT(chat_id, pattern) DO NOTHING""",
            (chat_id, pattern, mode))


def bump_pattern(pattern: str, shadow_hit: bool, spam_hit: bool) -> None:
    execute("""UPDATE spam_patterns
               SET hit_count = hit_count + 1,
                   shadow_hits = shadow_hits + ?,
                   shadow_spam_hits = shadow_spam_hits + ?
               WHERE pattern=?""",
            (1 if shadow_hit else 0, 1 if spam_hit else 0, pattern))


def promote_pattern(pattern_id: int) -> None:
    execute("UPDATE spam_patterns SET mode='active' WHERE id=?", (pattern_id,))


def fewshots(chat_id: int, kind: str, limit: int = 8) -> list[str]:
    rows = query("""SELECT content FROM chat_fewshots
                    WHERE chat_id=? AND kind=? ORDER BY id DESC LIMIT ?""",
                 (chat_id, kind, limit))
    return [r["content"] for r in rows]


def add_fewshot(chat_id: int, kind: str, content: str, source: str) -> None:
    if not content or len(content.strip()) < 2:
        return
    execute("""INSERT INTO chat_fewshots (chat_id, kind, content, source)
               VALUES (?,?,?,?)
               ON CONFLICT(chat_id, kind, content) DO NOTHING""",
            (chat_id, kind, content.strip()[:200], source))


# --------------------------------------------------------------------------
# Token 用量
# --------------------------------------------------------------------------

def record_usage(chat_id: int, calls: int, hit: int, miss: int, output: int,
                 cost: float) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    execute("""
        INSERT INTO token_usage (usage_date, chat_id, calls, hit_tokens,
                                 miss_tokens, output_tokens, cost_yuan)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(usage_date, chat_id) DO UPDATE SET
            calls = calls + excluded.calls,
            hit_tokens = hit_tokens + excluded.hit_tokens,
            miss_tokens = miss_tokens + excluded.miss_tokens,
            output_tokens = output_tokens + excluded.output_tokens,
            cost_yuan = cost_yuan + excluded.cost_yuan
    """, (today, chat_id, calls, hit, miss, output, cost))


def usage_today(chat_id: int) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    row = query_one("""SELECT * FROM token_usage
                       WHERE usage_date=? AND chat_id=?""", (today, chat_id))
    if not row:
        return {"calls": 0, "hit_tokens": 0, "miss_tokens": 0,
                "output_tokens": 0, "cost_yuan": 0.0,
                "cache_hit_rate": 0.0}
    d = dict(row)
    total = (d["hit_tokens"] or 0) + (d["miss_tokens"] or 0)
    d["cache_hit_rate"] = (d["hit_tokens"] or 0) / total if total else 0.0
    return d


def tokens_used_today(chat_id: int) -> int:
    u = usage_today(chat_id)
    return (u["hit_tokens"] or 0) + (u["miss_tokens"] or 0) + (u["output_tokens"] or 0)


# --------------------------------------------------------------------------
# 管理员缓存
# --------------------------------------------------------------------------

def cache_admins(chat_id: int, admins: Iterable[tuple[int, str]]) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM chat_admins WHERE chat_id=?", (chat_id,))
        conn.executemany(
            "INSERT INTO chat_admins (chat_id, user_id, full_name) VALUES (?,?,?)",
            [(chat_id, uid, name) for uid, name in admins])


def cached_admin_ids(chat_id: int) -> set[int]:
    return {r["user_id"] for r in
            query("SELECT user_id FROM chat_admins WHERE chat_id=?", (chat_id,))}


def admins_cached_at(chat_id: int):
    row = query_one("""SELECT MAX(cached_at) t FROM chat_admins WHERE chat_id=?""",
                    (chat_id,))
    return row["t"] if row else None
