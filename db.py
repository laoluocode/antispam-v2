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
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
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
    username         TEXT,                        -- 群的公开 username（用于判定 t.me 链接归属）
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
    daily_token_cap  INTEGER DEFAULT 3000000,     -- 每群每日 token 预算
                                                  -- 实测校准：原先 300k 只够 144 次调用
                                                  -- （平均批仅 2 条），下午就烧光。
                                                  -- 配合最小批门槛（20 条）后单次成本
                                                  -- 摊薄 10 倍，3M 可覆盖高流量群全天。
                                                  -- 费用量级：约 0.2-1 元/天/群
    report_limit     INTEGER DEFAULT 20,          -- 每人每日举报额度
    log_channel_id   INTEGER,                     -- 删除记录频道（0/NULL=不发）
    managed_links    TEXT,                        -- 本群体系的频道标识（逗号分隔）
                                                  -- 指向这些目标的 t.me 链接属正常
    linked_chat_id   INTEGER,                     -- 本群绑定的频道 ID。
                                                  -- 该频道的自动转发（is_automatic_forward）
                                                  -- 是本群自己的内容，必须放行；
                                                  -- 外部频道没有这个身份，照常判定。
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

-- 链接预览指纹：广告号用"短文本 + 频道预览"引流，
-- 正文 Telegram 不下发（实测 message.text 只有单字母），
-- 但"多个账号在短时间窗内指向同一个预览目标"这个协同特征改不掉。
CREATE TABLE IF NOT EXISTS preview_fingerprints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,          -- URL 归一化后的指纹
    url         TEXT,
    first_seen  DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen   DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chat_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_preview_fp ON preview_fingerprints (chat_id, fingerprint);

-- 指纹 -> 发布账号。用于计算"几个账号在发同一预览"
CREATE TABLE IF NOT EXISTS preview_senders (
    chat_id     INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    user_id     INTEGER NOT NULL,
    seen_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, fingerprint, user_id)
);

-- 全局黑名单：跨群联动。
-- 同一批广告号会同时打多个群（实测：南京苏州集团与武汉集团同期出现
-- 同一套"招募探花"广告）。在 A 群确认的广告号，B 群应立刻生效，
-- 不必各自重新踩一遍。
CREATE TABLE IF NOT EXISTS global_blacklist (
    user_id    INTEGER PRIMARY KEY,
    full_name  TEXT,
    reason     TEXT,
    origin_chat INTEGER,                 -- 首次确认的群
    hits       INTEGER DEFAULT 0,        -- 在多少个群被确认过
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 判定记忆：管理员裁决过的内容，同类再来直接复用结论。
--
-- 来源（AstrBot QQ 群广告拦截插件的设计）：
--   "疑似广告自动发给群管理员人工确认；**确认结果记录进学习库，
--    后续相同内容直接复用结论**"
--
-- 这解决了一个重复劳动问题：同一条广告被 ABC 三个账号发三遍，
-- LLM 要判三次、管理员要看三次。人工裁决一次之后就该自动处理。
--
-- 两种记忆：
--   false_positive（管理员说误删）-> 同类内容永不删（保护真人）
--   confirmed（管理员确认是广告）-> 同类内容直接删（0ms，且抗变体）
CREATE TABLE IF NOT EXISTS verdict_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL,
    norm_hash   TEXT NOT NULL,             -- 归一化后的内容哈希
    sample      TEXT,                      -- 原文样本，供人工复核
    label       TEXT NOT NULL,             -- false_positive | confirmed
    source      TEXT,                      -- admin_revive | admin_report | auto
    hits        INTEGER DEFAULT 0,         -- 复用次数
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_verdict_memory
    ON verdict_memory(chat_id, norm_hash, label);
CREATE INDEX IF NOT EXISTS idx_verdict_memory_hash
    ON verdict_memory(norm_hash);

-- 抽样复核批次。
--
-- 为什么要"批次"而不是逐条发通知：管理员实测反馈"一次 40-50 条，
-- 在机器人端几乎无法操作"。逐条发 50 条消息会把对话冲垮，只能一条条点，
-- 实际结果就是没人复核（上线以来 1844 次删除只被复核过 1 次）。
-- 批次化后：一次推送一个入口，点进去一条条翻，每屏 3 条 + 固定 4 个按钮。
CREATE TABLE IF NOT EXISTS review_batches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL DEFAULT 0,
    items       TEXT NOT NULL,             -- JSON: [{rec, chat, text, conf}]
    cursor      INTEGER DEFAULT 0,         -- 当前翻到第几条
    done        INTEGER DEFAULT 0,         -- 已处理的条数
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 全部判定记录（含"判定为正常"的）。
--
-- 为什么不能只用 spam_records 做分布分析：那张表只记删除/观察/恢复，
-- 也就是**只有被判定为广告的消息**。用它看"预测分布"是幸存者偏差——
-- 低分区（被判为正常）的样本压根不存在，于是永远看不到阈值附近堆积。
-- 而阈值漂移最早的表现恰恰是"中间置信度样本比例上升"。
CREATE TABLE IF NOT EXISTS verdict_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL,
    message_id  INTEGER NOT NULL,
    user_id     INTEGER,
    confidence  REAL,                    -- LLM 给的原始置信度
    source      TEXT,                    -- llm | learned | local_rule
    deleted     INTEGER DEFAULT 0,       -- 最终是否删除
    triggers    TEXT,                    -- 触发重新检查的原因（空=免AI额度）
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_verdict_log_chat
    ON verdict_log(chat_id, created_at);

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
            ("managed_links", "TEXT"),
            ("username", "TEXT"),
            ("linked_chat_id", "INTEGER"),
        ],
        "spam_reports": [
            ("source", "TEXT DEFAULT 'in_group'"),
            ("confidence", "REAL"),
        ],
        "messages": [
            ("retry_count", "INTEGER DEFAULT 0"),
        ],
        "users": [
            # 自适应检查强度（借鉴 StopSpam）：长期干净发言的成员不必每条都送 AI，
            # 但**只是跳过 AI**，L0 确定性规则与所有触发条件仍然生效。
            ("last_clean_at", "DATETIME"),
            ("grace_left", "INTEGER DEFAULT 0"),
        ],
    }
    for table, cols in additions.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                logger.info("迁移：%s 新增列 %s", table, name)


# 所有应该存在的表。启动时自检用——
# 背景：schema 里的 CREATE TABLE IF NOT EXISTS 只在启动时执行，如果容器
# 没有真正重启（docker compose up -d 在配置未变时可能不重建容器），
# 新加的表就不会被创建，而调用方只是静默失败（被 except 吞掉），
# 表现为"功能上线了但数据是空的"，很难发现。线上确实踩到过这一点。
REQUIRED_TABLES = (
    "chats", "chat_admins", "users", "messages", "spam_records",
    "chat_settings", "spam_whitelist", "spam_blacklist", "spam_patterns",
    "chat_fewshots", "spam_reports", "daily_report_stats", "token_usage",
    "report_usage", "dropped_messages", "preview_fingerprints",
    "preview_senders", "global_blacklist", "verdict_log", "verdict_memory",
    "review_batches",
)


def verify_schema(path: str) -> list[str]:
    """返回缺失的表名列表。空列表 = 结构完整。

    这个自检的价值在于把"静默缺表"变成启动日志里的一行明确告警。
    """
    conn = sqlite3.connect(path, timeout=DB_TIMEOUT)
    try:
        have = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    return [t for t in REQUIRED_TABLES if t not in have]


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

    # 建表后立即自检。缺表说明 SCHEMA 与代码不一致（或迁移没跑到），
    # 必须显式喊出来，不能让它变成静默失败。
    missing = verify_schema(path)
    if missing:
        logger.error("数据库结构不完整，缺少表: %s —— "
                     "相关功能会静默失效，请检查 SCHEMA 与容器是否真正重启",
                     "、".join(missing))
    else:
        logger.info("数据库结构自检通过（%d 张表）", len(REQUIRED_TABLES))


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
        "managed_links", "username", "linked_chat_id",
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


# 长期干净发言的成员享有多少条"免 AI"额度。
# 额度耗尽后下一条消息会重新送 AI 复核——这样即使账号被慢慢养熟后变坏，
# 也最多漏过 GRACE_MESSAGES 条，不会永久失守。
GRACE_MESSAGES = 30

# 额度可刷新的条件：这次送 AI 判定的结果是"干净"
def refresh_grace(chat_id: int, user_id: int, reset: bool = False) -> None:
    """刷新免检额度。

    reset=True 用于"刚被 AI 复核且判为干净"的情形——重新给满额度。
    额度是计数的而非计时的：既避免按时间窗口漏判，也让行为可预期。
    """
    now = _now()
    if reset:
        execute("""UPDATE users SET grace_left=?, last_clean_at=?
                   WHERE chat_id=? AND user_id=?""",
                (GRACE_MESSAGES, now, chat_id, user_id))
    else:
        execute("""UPDATE users SET grace_left = MAX(0, grace_left - 1)
                   WHERE chat_id=? AND user_id=?""", (chat_id, user_id))


def consume_grace(chat_id: int, user_id: int) -> int:
    """消耗一次额度，返回剩余额度。"""
    with tx() as conn:
        conn.execute("""UPDATE users SET grace_left = MAX(0, grace_left - 1)
                        WHERE chat_id=? AND user_id=?""", (chat_id, user_id))
        row = conn.execute("""SELECT grace_left FROM users
                              WHERE chat_id=? AND user_id=?""",
                           (chat_id, user_id)).fetchone()
    return row["grace_left"] if row else 0


def grace_left(chat_id: int, user_id: int) -> int:
    row = query_one("""SELECT grace_left FROM users
                       WHERE chat_id=? AND user_id=?""", (chat_id, user_id))
    return (row["grace_left"] or 0) if row else 0


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

def record_preview(chat_id: int, fingerprint: str, url: str,
                   user_id: int) -> int:
    """记录一次预览指纹，返回**不同账号**的累计数量。

    这是本项目的关键反制手段：广告正文拿不到，但"几个账号在发同一预览"
    是协同刷屏的强证据，且广告号无法规避——它必须让群友看到广告。
    """
    now = _now()
    with tx() as conn:
        conn.execute("""
            INSERT INTO preview_fingerprints
                (chat_id, fingerprint, url, first_seen, last_seen)
            VALUES (?,?,?,?,?)
            ON CONFLICT(chat_id, fingerprint) DO UPDATE SET last_seen=excluded.last_seen
        """, (chat_id, fingerprint, url, now, now))
        conn.execute("""
            INSERT INTO preview_senders (chat_id, fingerprint, user_id, seen_at)
            VALUES (?,?,?,?)
            ON CONFLICT(chat_id, fingerprint, user_id) DO UPDATE SET seen_at=excluded.seen_at
        """, (chat_id, fingerprint, user_id, now))
        row = conn.execute("""
            SELECT COUNT(*) n FROM preview_senders
            WHERE chat_id=? AND fingerprint=?
              AND seen_at >= datetime('now', '-1 day')
        """, (chat_id, fingerprint)).fetchone()
    return row["n"] if row else 0


def preview_sender_count(chat_id: int, fingerprint: str, hours: int = 24) -> int:
    row = query_one("""
        SELECT COUNT(*) n FROM preview_senders
        WHERE chat_id=? AND fingerprint=?
          AND seen_at >= datetime('now', ?)
    """, (chat_id, fingerprint, f"-{hours} hours"))
    return row["n"] if row else 0


def list_previews(chat_id: int, hours: int = 24, limit: int = 20) -> list[sqlite3.Row]:
    """列出近期预览指纹及其发布账号数，用于排查协同刷屏。"""
    return query("""
        SELECT f.fingerprint, f.url, f.last_seen,
               (SELECT COUNT(*) FROM preview_senders p
                 WHERE p.chat_id=f.chat_id AND p.fingerprint=f.fingerprint
                   AND p.seen_at >= datetime('now', ?)) senders
        FROM preview_fingerprints f
        WHERE f.chat_id=? AND f.last_seen >= datetime('now', ?)
        ORDER BY senders DESC, f.last_seen DESC LIMIT ?
    """, (f"-{hours} hours", chat_id, f"-{hours} hours", limit))


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


def get_message(chat_id: int, message_id: int):
    return query_one("SELECT * FROM messages WHERE chat_id=? AND message_id=?",
                     (chat_id, message_id))


def remark_message(chat_id: int, message_id: int, text: str) -> bool:
    """编辑后更新正文并重置为未检测，让流水线重新判。

    The Edit Trick：先发正常消息，过段时间编辑成广告。不在编辑后重新判定
    就等于给广告号留了稳定的后门。
    """
    with tx() as conn:
        cur = conn.execute(
            """UPDATE messages SET text=?, is_checked=0, retry_count=0
               WHERE chat_id=? AND message_id=?""",
            (text, chat_id, message_id))
        return cur.rowcount > 0


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


def normalize_for_hash(text: str) -> str:
    """归一化后取哈希键。

    用归一化文本而不是原文：同一句广告的谐音/不可见字符变体应当命中
    同一条记忆，否则广告号改一个字符就绕过了人工裁决。
    """
    try:
        from utils.normalize import normalize
        return normalize(text or "")
    except Exception:  # noqa: BLE001
        return (text or "").strip().lower()


def log_verdict(chat_id: int, message_id: int, user_id: int,
                confidence: float | None, source: str, deleted: bool,
                triggers: str = "") -> None:
    """记录一次判定（无论删还是留）。

    这是分布漂移监控的数据基础：只有把"判为正常"的样本也存下来，
    才能看到置信度分布的真实形状。
    """
    execute("""INSERT INTO verdict_log
        (chat_id, message_id, user_id, confidence, source, deleted, triggers)
        VALUES (?,?,?,?,?,?,?)""",
        (chat_id, message_id, user_id, confidence, source,
         1 if deleted else 0, triggers))


def prune_verdict_log(days: int = 30) -> int:
    """清理过旧的判定记录，避免无限增长。"""
    with tx() as conn:
        cur = conn.execute("DELETE FROM verdict_log WHERE created_at < datetime('now', ?)",
                           (f"-{days} days",))
        return cur.rowcount


def remember_verdict(chat_id: int, text: str, label: str,
                     source: str = "admin") -> bool:
    """记住一条人工裁决，供后续同类内容复用。

    label: false_positive（误删，以后别删）| confirmed（确认是广告，以后直接删）
    """
    norm = normalize_for_hash(text)
    if len(norm) < 3:
        return False
    try:
        execute("""INSERT INTO verdict_memory
            (chat_id, norm_hash, sample, label, source)
            VALUES (?,?,?,?,?)
            ON CONFLICT(chat_id, norm_hash, label)
            DO UPDATE SET hits = hits""",
            (chat_id, norm, (text or "")[:200], label, source))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("写入判定记忆失败 chat=%s label=%s: %s", chat_id, label, exc)
        return False


def recall_verdict(chat_id: int, text: str) -> str | None:
    """查这条内容有没有被人工裁决过。返回 label 或 None。

    这是"人工裁决一次、后续自动复用"的入口——既省 LLM 调用，
    也保证同一条内容不会因为模型随机性而被反复改判。
    """
    norm = normalize_for_hash(text)
    if len(norm) < 3:
        return None
    row = query_one("""SELECT label FROM verdict_memory
                       WHERE chat_id=? AND norm_hash=?
                       ORDER BY CASE label WHEN 'false_positive' THEN 0 ELSE 1 END
                       LIMIT 1""", (chat_id, norm))
    if row is None:
        return None
    execute("UPDATE verdict_memory SET hits = hits + 1 "
            "WHERE chat_id=? AND norm_hash=?", (chat_id, norm))
    return row["label"]


def verdict_memory_list(chat_id: int = 0, limit: int = 50) -> list[sqlite3.Row]:
    if chat_id:
        return query("""SELECT * FROM verdict_memory WHERE chat_id=?
                        ORDER BY id DESC LIMIT ?""", (chat_id, limit))
    return query("SELECT * FROM verdict_memory ORDER BY id DESC LIMIT ?", (limit,))


def count_all_deletions(hours: int = 12) -> int:
    row = query_one("""SELECT COUNT(*) n FROM spam_records
                       WHERE action='delete'
                         AND created_at >= datetime('now', ?)""",
                    (f"-{hours} hours",))
    return row["n"] if row else 0


# 抽样分层。**这是抽样复核有没有价值的关键。**
#
# 早先按"随机抽已删除的记录"实现，实测抽到的全是置信度 0.9-1.0 的显然广告：
#   #698  [1.00] '帮收赌资 赚3千'
#   #1399 [0.95] '出勤名单…'
# 复核 30 条这样的样本，对"阈值 0.50 选得对不对"提供**零信息**——
# 它们在 0.30 或 0.70 的阈值下都会被删。
#
# 真正有信息量的是**阈值附近**的判定：那里的每一条都在检验判断尺度，
# 一条误删就能说明阈值偏低。所以按置信度分层，并把配额向边界倾斜。
SAMPLE_BANDS = (
    ("near_cut", "threshold-adjacent(0.5-0.7)", 0.50, 0.70, 0.5),
    ("mid", "mid(0.7-0.9)", 0.70, 0.90, 0.3),
    ("clear", "clear(>=0.9)", 0.90, 1.01, 0.2),
)


def sample_deletions(chat_id: int = 0, hours: int = 12,
                     limit: int = 5) -> list[sqlite3.Row]:
    """分层抽样最近的删除记录，供管理员复核。

    为什么需要（实际使用数据）：上线以来 1844 次删除，管理员只复核了 1 次。
    系统在"盲信"状态下运行——而"边用边更新"需要反馈才能更新。

    为什么分层：均匀抽样几乎全是 0.9+ 的显然广告，对阈值标定没有信息量。
    配额向 0.5-0.7 的边界区间倾斜，因为那里的判断才真正需要人来确认。
    """
    where = ["action='delete'", "text_snapshot IS NOT NULL",
             "confidence IS NOT NULL",
             "created_at >= datetime('now', ?)"]
    args: list = [f"-{hours} hours"]
    if chat_id:
        where.append("chat_id = ?")
        args.append(chat_id)
    clause = " AND ".join(where)

    sampled = sampled_ids()
    out: list = []
    seen: set[int] = set()
    for _key, _label, lo, hi, share in SAMPLE_BANDS:
        quota = max(1, int(limit * share))
        rows = query(f"""
            SELECT id, chat_id, user_id, full_name, text_snapshot, confidence,
                   source, created_at
            FROM spam_records
            WHERE {clause} AND confidence >= ? AND confidence < ?
            ORDER BY RANDOM() LIMIT ?
        """, (*args, lo, hi, quota * 4))
        picked = [r for r in rows if r["id"] not in sampled
                  and r["id"] not in seen][:quota]
        for r in picked:
            seen.add(r["id"])
        out.extend(picked)

    # 若某一层样本不足，用其他层补齐，保证条数稳定
    if len(out) < limit:
        rows = query(f"""
            SELECT id, chat_id, user_id, full_name, text_snapshot, confidence,
                   source, created_at
            FROM spam_records WHERE {clause}
            ORDER BY RANDOM() LIMIT ?
        """, (*args, limit * 4))
        for r in rows:
            if len(out) >= limit:
                break
            if r["id"] in seen or r["id"] in sampled:
                continue
            seen.add(r["id"])
            out.append(r)
    return out


def sampled_ids() -> set[int]:
    row = query_one("SELECT value FROM chat_settings "
                    "WHERE chat_id=0 AND key='sampled_ids'")
    if not row or not row["value"]:
        return set()
    out = set()
    for x in str(row["value"]).split(","):
        if x.strip().isdigit():
            out.add(int(x))
    return out


def create_review_batch(items: list[dict]) -> int:
    """把一批待复核项存起来，返回批次 ID。"""
    import json as _json
    cur = execute("""INSERT INTO review_batches (items) VALUES (?)""",
                  (_json.dumps(items, ensure_ascii=False),))
    return cur or 0


def get_review_batch(batch_id: int):
    row = query_one("SELECT * FROM review_batches WHERE id=?", (batch_id,))
    if not row:
        return None
    import json as _json
    try:
        row = dict(row)
        row["items"] = _json.loads(row["items"] or "[]")
        return row
    except Exception as exc:  # noqa: BLE001
        logger.warning("复核批次 %s 解析失败: %s", batch_id, exc)
        return None


def advance_review_batch(batch_id: int, cursor: int, done: int) -> None:
    execute("UPDATE review_batches SET cursor=?, done=? WHERE id=?",
            (cursor, done, batch_id))


def mark_sampled(record_ids: list[int]) -> None:
    """记录已抽样过的 ID，避免同一批反复推送。"""
    if not record_ids:
        return
    existing = query_one("SELECT value FROM chat_settings "
                         "WHERE chat_id=0 AND key='sampled_ids'")
    ids = set((existing["value"] if existing else "").split(","))
    ids.discard("")
    ids.update(str(i) for i in record_ids)
    # 只保留最近 2000 个，避免无限增长
    keep = sorted(ids, key=lambda x: int(x) if x.isdigit() else 0)[-2000:]
    execute("""INSERT INTO chat_settings (chat_id, key, value) VALUES (0,?,?)
               ON CONFLICT(chat_id,key) DO UPDATE SET value=excluded.value""",
            ("sampled_ids", ",".join(keep)))


def verdict_health(chat_id: int = 0, hours: int = 24) -> dict:
    """判定健康度：预测分布 + 语料倾斜 + 人工推翻率。

    设计依据（Gmail 垃圾邮件分类器故障 / 阈值漂移监控）：
    阈值漂移往往发生在**模型本身没变**的情况下——用户行为或广告手法的
    分布变了，原本合适的阈值就突然不合适了，表现为"误报机器"。
    文章给出的三个可落地信号：
      1. **预测分布**：稳定时呈双峰（低分区正常、高分区广告）。
         中间区间（0.3-0.7）样本比例**突然上升**是漂移的早期信号。
      2. **语料倾斜**：正负样本比例失衡（贝叶斯机器人就栽在这里）。
      3. **业务指标**：用户把误删消息"恢复"回来的比例——
         这正是我们的 revive 动作，是最直接的用户不满信号。
    """
    where = ["created_at >= datetime('now', ?)"]
    args: list = [f"-{hours} hours"]
    if chat_id:
        where.append("chat_id = ?")
        args.append(chat_id)
    clause = " AND ".join(where)

    # 预测分布：用 verdict_log（含判为正常的样本）。
    # 必须用这张表——spam_records 只记删除，用它做分布分析是幸存者偏差，
    # 低分区永远为空，看不到真实形状。
    vwhere = ["created_at >= datetime('now', ?)", "source='llm'"]
    vargs: list = [f"-{hours} hours"]
    if chat_id:
        vwhere.append("chat_id = ?")
        vargs.append(chat_id)
    buckets = {"low(<0.3)": 0, "mid(0.3-0.7)": 0, "high(>0.7)": 0,
               "very_high(>0.9)": 0}
    for r in query(f"""SELECT confidence FROM verdict_log
                       WHERE {' AND '.join(vwhere)}
                         AND confidence IS NOT NULL""", tuple(vargs)):
        c = r["confidence"] or 0
        if c < 0.3:
            buckets["low(<0.3)"] += 1
        elif c <= 0.7:
            buckets["mid(0.3-0.7)"] += 1
        elif c <= 0.9:
            buckets["high(>0.7)"] += 1
        else:
            buckets["very_high(>0.9)"] += 1
    llm_total = sum(buckets.values())

    # 人工推翻率：删除后被恢复的比例，最直接的用户不满信号
    row = query_one(f"""
        SELECT SUM(CASE WHEN action='delete' THEN 1 ELSE 0 END) d,
               SUM(CASE WHEN action='revive' THEN 1 ELSE 0 END) r
        FROM spam_records WHERE {clause}
    """, tuple(args))
    d, r = (row["d"] or 0), (row["r"] or 0)

    # 语料倾斜
    spam_n = query_one(f"""SELECT COUNT(*) n FROM spam_records
        WHERE {clause} AND action='delete' AND text_snapshot IS NOT NULL""",
        tuple(args))["n"]
    mwhere = ["timestamp >= datetime('now', ?)"]
    margs: list = [f"-{hours} hours"]
    if chat_id:
        mwhere.append("chat_id = ?")
        margs.append(chat_id)
    ham_n = query_one(f"""SELECT COUNT(*) n FROM messages m
        WHERE {' AND '.join(mwhere)}
          AND NOT EXISTS (SELECT 1 FROM spam_records s
              WHERE s.chat_id=m.chat_id AND s.message_id=m.message_id
                AND s.action='delete')""", tuple(margs))["n"]

    return {
        "buckets": buckets, "llm_total": llm_total,
        "deleted": d, "revived": r,
        "revive_rate": (r / d) if d else 0.0,
        "spam_n": spam_n, "ham_n": ham_n,
        "skew": (spam_n / ham_n) if ham_n else float("inf"),
    }


def audit_window(chat_id: int = 0, hours: int = 24, decisions: int = 0,
                 limit: int = 500) -> dict:
    """审计某个时间窗内的判定情况。

    存在的理由（对标市面实现的 audit log）：管理员需要能回答
      · 这条为什么被删？什么时候删的？谁删的？
      · 这段时间机器人一共做了多少决定？有多少是我不同意的？
    统计类命令（/status、manage.py cost）只能给总数，
    不能下钻到"某一条具体消息为什么是这个结果"。
    """
    where = ["created_at >= datetime('now', ?)"]
    args: list = [f"-{hours} hours"]
    if chat_id:
        where.append("chat_id = ?")
        args.append(chat_id)
    if decisions:
        where.append("action = ?")
        args.append(decisions)
    clause = " AND ".join(where)

    rows = query(f"""
        SELECT id, chat_id, user_id, full_name, message_id, text_snapshot,
               confidence, source, action, created_at
        FROM spam_records WHERE {clause}
        ORDER BY id DESC LIMIT ?
    """, (*args, limit))

    summary = query_one(f"""
        SELECT COUNT(*) total,
               SUM(CASE WHEN action='delete' THEN 1 ELSE 0 END) deleted,
               SUM(CASE WHEN action='observe' THEN 1 ELSE 0 END) observed,
               SUM(CASE WHEN action='revive' THEN 1 ELSE 0 END) revived,
               SUM(CASE WHEN source='local_rule' THEN 1 ELSE 0 END) by_rule,
               SUM(CASE WHEN source='llm' THEN 1 ELSE 0 END) by_llm
        FROM spam_records WHERE {clause}
    """, tuple(args))

    # 消息层面的统计（用来判断"漏没漏"）
    msg_where = ["timestamp >= datetime('now', ?)"]
    msg_args: list = [f"-{hours} hours"]
    if chat_id:
        msg_where.append("chat_id = ?")
        msg_args.append(chat_id)
    msg = query_one(f"""
        SELECT COUNT(*) total,
               SUM(CASE WHEN is_checked=1 THEN 1 ELSE 0 END) checked
        FROM messages WHERE {' AND '.join(msg_where)}
    """, tuple(msg_args))

    return {"rows": rows, "summary": dict(summary) if summary else {},
            "messages": dict(msg) if msg else {}}


def decision_stats(chat_id: int = 0, hours: int = 24) -> dict:
    """判定分布，用于观察"AI 与规则各承担了多少"。"""
    where = ["created_at >= datetime('now', ?)"]
    args: list = [f"-{hours} hours"]
    if chat_id:
        where.append("chat_id = ?")
        args.append(chat_id)
    clause = " AND ".join(where)
    out = {}
    for r in query(f"""
        SELECT source, confidence, COUNT(*) n FROM spam_records
        WHERE {clause} GROUP BY source, ROUND(confidence, 1)
        ORDER BY source, confidence
    """, tuple(args)):
        out.setdefault(r["source"], []).append((r["confidence"], r["n"]))
    return out


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
    """本群黑名单 + 全局黑名单。

    跨群联动的价值（实测依据）：同一批广告号会同时打多个群，
    A 群确认过的号在 B 群当场就生效，不必各自重新踩一遍。
    这是"网络效应"——群越多，每个群越安全。
    """
    local = {r["user_id"] for r in
             query("SELECT user_id FROM spam_blacklist WHERE chat_id=?", (chat_id,))}
    glob = {r["user_id"] for r in query("SELECT user_id FROM global_blacklist")}
    return local | glob


def global_blacklist_add(user_id: int, full_name: str | None, reason: str,
                         origin_chat: int) -> None:
    """加入全局黑名单（跨群生效）。"""
    execute("""
        INSERT INTO global_blacklist (user_id, full_name, reason, origin_chat, hits)
        VALUES (?,?,?,?,1)
        ON CONFLICT(user_id) DO UPDATE SET
            hits = hits + 1,
            reason = excluded.reason,
            full_name = COALESCE(excluded.full_name, global_blacklist.full_name)
    """, (user_id, full_name, reason, origin_chat))


def global_blacklist_remove(user_id: int) -> bool:
    with tx() as conn:
        return conn.execute("DELETE FROM global_blacklist WHERE user_id=?",
                            (user_id,)).rowcount > 0


def global_blacklist_list(limit: int = 100) -> list[sqlite3.Row]:
    return query("SELECT * FROM global_blacklist ORDER BY hits DESC, created_at DESC "
                 "LIMIT ?", (limit,))


def global_blacklist_has(user_id: int) -> bool:
    return query_one("SELECT 1 FROM global_blacklist WHERE user_id=?",
                     (user_id,)) is not None


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


# 从群描述里抓 @username。这些群主运营方习惯把自家频道列在描述里
# （实测：南京群/武汉群的 description 里写全了榜单、报告、资讯频道）。
_DESC_MENTION = re.compile(r"@([A-Za-z][A-Za-z0-9_]{3,31})")


def mine_channel_hints(description: str | None, own_username: str | None) -> str:
    """从群描述抽取频道标识，返回逗号分隔串，用于写入 managed_links。

    为什么这么做：这些群形成了"集团"式频道体系，运营方把自家频道
    都写在群描述里。自动抓下来后，"链接指向自家还是外部"的判断
    就不需要人工维护白名单。
    """
    names: list[str] = []
    if own_username:
        names.append(own_username.lstrip("@"))
    for m in _DESC_MENTION.finditer(description or ""):
        n = m.group(1)
        if n not in names:
            names.append(n)
    return ",".join(names)


def managed_links(chat_id: int) -> set[str]:
    """本群体系的频道标识集合，用于判断 t.me 链接是"自家"还是"外部"。

    组成：群自己的 username + 从群描述抓到的频道 + 显式配置 + 其他已注册群。
    只要域名对得上其中之一，就认为不是外部引流。
    """
    names: set[str] = set()
    row = query_one("SELECT username, managed_links FROM chats WHERE chat_id=?",
                    (chat_id,))
    if row:
        if row["username"]:
            names.add(str(row["username"]).lstrip("@").lower())
        for w in (row["managed_links"] or "").split(","):
            w = w.strip().lstrip("@").lower()
            if w:
                names.add(w)
    # 其他已注册群也视作自家（同一条消息可能同时提到多个群）
    for r in query("SELECT username FROM chats WHERE is_active=1 AND username IS NOT NULL"):
        if r["username"]:
            names.add(str(r["username"]).lstrip("@").lower())
    return names


def set_chat_username(chat_id: int, username: str | None) -> None:
    with tx() as conn:
        conn.execute("UPDATE chats SET username=? WHERE chat_id=?",
                     (username, chat_id))


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
    _refresh_local_pattern_cache()


def _refresh_local_pattern_cache() -> None:
    """刷新**当前进程**的规则缓存。

    ⚠️ 边界要说清楚：这只对本进程有效，**不会**让服务端进程立刻看到变化。
    learner 通常跑在独立进程（manage.py 或定时任务）里，它调这个函数
    刷的是自己的缓存，服务端完全不知情。

    服务端能拿到新规则的路径有两条：
      · load_patterns() 的 300 秒 TTL（每次判定都会检查并到期重载）
      · job_self_evolve 在毕业之后的显式 load_patterns(force=True)
    所以毕业后最多 5 分钟生效——能工作，但不是"立即"。

    早先这个函数叫 _invalidate_pattern_cache，名字让人以为它跨进程生效，
    实际不能。名字和注释如实描述比"看起来更强"重要。
    """
    try:
        import detector_rules
        detector_rules.load_patterns(force=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("刷新本进程规则缓存失败: %s", exc)


def bump_pattern(pattern: str, shadow_hit: bool, spam_hit: bool,
                 chat_id: int | None = None) -> None:
    """累加某条规则的命中统计。

    **必须限定 chat_id。** 主键是 UNIQUE(chat_id, pattern)，同一个 pattern
    可能在多个群各有一行；不加限定的 `WHERE pattern=?` 会让一次观测写进
    所有群，实测两个群的计数器逐位相同（hits=[20,20]）。

    这破坏的不是美观而是**闸门的前提**：毕业条件是"该规则**在本群**观察期内
    零误命中"，串扰后退化成"全局聚合零误命中"——一个群可以凭别群的证据毕业，
    在自己从未验证过该 pattern 的情况下开始直删本群消息。

    chat_id 为 None 时只更新全局行（chat_id IS NULL）；调用方拿不到群时
    （例如运维手工 bump）才应省略。
    """
    if chat_id is None:
        # 不传群时更新**所有**同名行。
        # 早先这里只更新 chat_id IS NULL 的行——而实际上没有全局规则
        # （全部由 refresh_candidates 按群创建），于是这个分支变成静默无操作，
        # 让调用方以为计数成功。运维手工 bump 和测试都踩过。
        execute("""UPDATE spam_patterns
                   SET hit_count = hit_count + 1,
                       shadow_hits = shadow_hits + ?,
                       shadow_spam_hits = shadow_spam_hits + ?
                   WHERE pattern=?""",
                (1 if shadow_hit else 0, 1 if spam_hit else 0, pattern))
        return
    execute("""UPDATE spam_patterns
               SET hit_count = hit_count + 1,
                   shadow_hits = shadow_hits + ?,
                   shadow_spam_hits = shadow_spam_hits + ?
               WHERE pattern=? AND chat_id=?""",
            (1 if shadow_hit else 0, 1 if spam_hit else 0, pattern, chat_id))


def promote_pattern(pattern_id: int) -> None:
    execute("UPDATE spam_patterns SET mode='active' WHERE id=?", (pattern_id,))
    _refresh_local_pattern_cache()


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


def usage_by_day(days: int = 7) -> list[sqlite3.Row]:
    """近 N 天各群的用量，用于观察真实日消费。"""
    return query("""
        SELECT usage_date, chat_id, calls, hit_tokens, miss_tokens,
               output_tokens, cost_yuan
        FROM token_usage
        ORDER BY usage_date DESC, cost_yuan DESC
        LIMIT ?
    """, (days * 20,))


def usage_daily_total(days: int = 7) -> list[sqlite3.Row]:
    """近 N 天合计（按天）。"""
    return query("""
        SELECT usage_date,
               SUM(calls) calls,
               SUM(hit_tokens) hit_tokens,
               SUM(miss_tokens) miss_tokens,
               SUM(output_tokens) output_tokens,
               SUM(cost_yuan) cost_yuan
        FROM token_usage
        GROUP BY usage_date
        ORDER BY usage_date DESC
        LIMIT ?
    """, (days,))


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
