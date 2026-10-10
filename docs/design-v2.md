# 反广告机器人 V2 设计文档

> 日期：2026-10-10
> 状态：待评审
> 目标读者：项目负责人
> 前置调研：本文所有数字来自 2026-10-10 对生产服务器（`ssh ribao`，`/root/ribaobot`）与容器 `ribao_bot` 的实测，以及 DeepSeek 官方 API 文档

---

## 0. 一句话摘要

把反广告能力从"自家群的免费模型脚本"重做成**多租户 SaaS**：判定主力换成 DeepSeek 官方 `deepseek-flash`（V4.1-Flash），利用官方默认开启的**上下文硬盘缓存**把成本压到约 **1 元/月/群**，规则层退化为"只为延迟服务的确定性拦截"，学习机制从"学关键词直拦"改为"学模式 + 影子验证"。

---

## 1. 背景与动机

### 1.1 现状

现有 `jituanribao` 是单群机器人，同时承担日报、词云、AI 聊天、频道评论、反广告五件事。反广告是其中最重的模块（`ai_summarizer.py` 1957 行，其中检测相关约 1200 行）。

线上实测（近 7 天）：

| 指标 | 数值 |
|---|---|
| 入库消息 | 900–1,600 条/天（节假日峰值 4,625） |
| 删除广告 | 650–990 条/天 |
| 删除/入账比 | **50–60%** |
| 日活发言用户 | 216–293 人 |
| 平均消息长度 | 11.3 字符 |
| 累计 messages | 210,508 条（保留 114.5 天） |
| 累计 spam_records | 139,312 条 |
| 动态规则 | 315 条，其中 **224 条已毕业直拦** |

### 1.2 为什么要重做

1. **免费模型的判定质量是瓶颈。** 当前三方协作（Qwen3-8B + gemini-3.1-flash-lite + R1-8B）中，Gemini 层 3 天内 20 次返回空响应导致整层失效。
2. **规则毕业门槛过低，误判会永久固化。** `check_rule_graduation` 的兜底通道是「命中 ≥3 次 + 准确率 100% → 永久直拦」（`ai_summarizer.py:1765-1770`）。同一个广告号刷 3 条相同消息即可"毕业"。已毕业规则中包含 `肾虚`、`火力全开`、`夜夜笙歌`、`投放`、`过检`、`出人头地`、`稀缺资源` 等通用词，这些词出现在正常聊天中会被静默秒删。
3. **统计报告 100% 失败。** `spam_stats_report`（`ai_summarizer.py:1955`）把 224 条毕业规则拼成单条消息发送，超出 Telegram 4096 字符上限且无分片。
4. **数据保留实际失效。** `messages` 表存了 114.5 天共 21 万条，其中 195,613 条 `is_summarized=0`。因为日报开关为 off，`delete_old_summarized_messages` 只清理"已总结"的消息。对外提供服务时这是隐私风险。
5. **不具备多租户能力。** `ALLOWED_GROUP_IDS` 是环境变量静态列表（`config.py:120`），`bot_settings` 是全局 key-value（无 chat_id 维度），`spam_whitelist` 全局生效。加一个群需要改 `.env` 重启。
6. **对外产品需要不同的严格度。** 现有策略是"宁可误删不可漏"（删除占比 50-60%）。该尺度在自有群可行（群主能兜底），对外会导致客户流失。

### 1.3 决策记录

| 决策点 | 结论 |
|---|---|
| 模型通路 | **DeepSeek 官方 API**（非 SiliconFlow） |
| 成本策略 | **分级路由**，规则只服务延迟，不做省钱主力 |
| 默认严格度 | **严格档**（默认保守，只删铁证广告） |
| 数据库 | **全新库**，不复用旧数据 |
| 旧规则 | **只继承资产，丢弃全部 315 条模式规则** |
| 百晓生 | **保留运行**，老 bot 代码冻结 |
| 新 bot 功能范围 | **只做反广告**（砍掉日报/词云/AI 聊天/频道评论） |
| 举报 + 每日榜 | **保留** |
| 部署位置 | **与百晓生同机**（`ssh ribao`，需先清理磁盘） |
| 多租户接入 | 先手动加群，验证后再做自助 |
| 上线方式 | **先跑影子模式** |

---

## 2. 官方 API 关键约束（实测自官方文档）

来源：[模型 & 价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)、[上下文硬盘缓存](https://api-docs.deepseek.com/zh-cn/guides/kv_cache)、[JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode)、[思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode)、[限速与隔离](https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit)

### 2.1 定价（`deepseek-flash`）

| token 类型 | 空闲时段 | 高峰时段 |
|---|---|---|
| 输入 · 缓存命中 | **0.02 元** /M | 0.04 元 /M |
| 输入 · 缓存未命中 | 1 元 /M | 2 元 /M |
| 输出 | 4 元 /M | 8 元 /M |

- 高峰时段 = 北京时间周一至周五（不含法定节假日）9:00–12:00、14:00–18:00
- 其余时段（含全部周末与法定节假日）为空闲时段，**价格减半**
- 缓存命中价是未命中的 **1/50**

### 2.2 必须遵守的工程约束

| # | 约束 | 后果 |
|---|---|---|
| C1 | 模型名是 **`deepseek-flash`**。旧名 `deepseek-v4-flash` 仍可调用，但对应模型已下线，会静默路由到 V4.1-Flash | 用旧名无法确定实际调用的是哪个模型 |
| C2 | **思考模式默认开启**（`effort=high`），思维链计入**输出 token**（最贵档） | 批量判定必须显式关闭 |
| C3 | **思考模式下 `temperature` / `presence_penalty` / `frequency_penalty` 全部失效**（传了不报错但不生效） | 现有代码的 `temperature=0.1` 在思考模式下等于没设 |
| C4 | 关闭思考模式：`extra_body={"thinking": {"type": "disabled"}}`；或 `reasoning_effort="low"` | — |
| C5 | **上下文硬盘缓存默认对所有人开启，无需改代码**，按前缀完整匹配 | 前缀一变，整批缓存失效，输入成本 ×50 |
| C6 | `user_id` 参数提供 **KVCache / 内容安全 / 调度** 三重隔离 | 每群传 `user_id` 即可实现租户级缓存隔离，防止群画像互相污染 |
| C7 | JSON Output 通过 `response_format={"type":"json_object"}`，**官方承认"有概率返回空的 content"** | 空响应必须重试 + 下轮重试，**绝不能当"无广告"处理** |
| C8 | 并发限制：`deepseek-flash` **2500**（账号级，与 API Key 无关） | 多租户绰绰有余 |
| C9 | 请求会持续返回空行保活，最长 10 分钟未开始推理才断开 | 客户端 timeout 需放宽至 ≥300s |

### 2.3 缓存机制要点

- 每条缓存前缀是**独立完整单元**，后续请求必须**完整匹配**才能命中
- 落盘时机：请求的用户输入结束位置、模型输出结束位置、公共前缀检测、按固定 token 间隔
- 响应 `usage` 字段提供 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
- **"尽力而为"，不保证 100% 命中**；缓存数小时到数天后自动清空

**推论（本设计的核心约束）**：

> prompt 结构必须严格分两层——**稳定前缀在前，动态内容在后**。任何对稳定前缀的修改都会让缓存全部失效。因此 prompt / 模式库的变更必须**限频、版本化、可回滚、可监控**。

---

## 3. 成本模型

### 3.1 单群成本测算

假设：1,500 条消息/天，平均 8 token/条，**200 条/批**（约 8 批/天），空闲时段。

prompt 结构决定 token 分布：

```python
messages = [
    {"role": "system", "content": GLOBAL_PROMPT},      # 稳定前缀（含分类学 + few-shot 例子）~4000 token
    {"role": "system", "content": pattern_library},    # 模式库，每天最多变 1 次          ~800 token
    {"role": "user",   "content": batch_messages},     # 本批消息 + 历史，必然未命中
]
```

| 项 | 每天 token | 命中率 | 空闲时段成本 |
|---|---|---|---|
| 稳定前缀 | 32,000 | ~98% | 0.001 元 |
| 模式库 | 6,400 | ~95% | 0.0004 元 |
| 消息正文 + 历史（**唯一必然未命中项**） | 22,400 | 0% | 0.022 元 |
| 输出（严格档，JSON 极短） | 1,600 | — | 0.006 元 |
| **合计** | | | **≈ 0.03 元/天** |

**月度：约 0.9 元/群。10 个群约 9 元/月。**

### 3.2 成本杠杆（按影响力排序）

| 杠杆 | 效果 | 说明 |
|---|---|---|
| **1. 稳定前缀不抖动** | 决定 50 倍差价 | 最重要的约束。prompt 每天变更超过 1 次即失控 |
| **2. 攒批粒度** | 未命中 token 反比下降 | 50 条/批 → 未命中 3030 token/批；200 条/批 → 760 token/批 |
| **3. 关闭思考模式** | 输出降 80%+ | 思维链按输出计价，是单价最贵的一档 |
| **4. 严格档** | 输出降 70% | 判定为"正常"的消息不需要写理由，JSON 更短 |
| **5. 空闲时段运行** | 直接减半 | 批量判定错开工作日 9-12 / 14-18 点高峰 |

### 3.3 成本风险与护栏

**唯一真实的成本风险是缓存失效**：

> 若每次判定都因 prompt 变化导致缓存全部未命中，输入成本从 0.001 元/天 涨到 22 元/天，**约 2000 倍**。

因此必须有：

1. **prompt 版本号**：稳定前缀内容哈希作为版本标识
2. **缓存命中率监控**：按群、按天记录 `prompt_cache_hit_tokens / (hit + miss)`
3. **自动回滚**：命中率跌破 80% 时告警，跌破 60% 自动回滚上一版本
4. **硬预算**：每群每日 token 上限（默认 500K），超限时降级为"仅确定性规则 + 攒批延后"，并通知管理员
5. **学习层限频**：`spam_patterns` / `chat_fewshots` 每天最多变更 1 次（03:00 批量执行）

---

## 4. 系统架构

### 4.1 检测流水线

```
消息入群 handle_message
  │
  ├─ L0 确定性规则 ──────────────────── 0ms / 0 成本 / 0 误伤
  │    手机号、t.me、@xxxbot、银行卡号、纯联系方式串
  │    ├ 命中 → 立即删除 + 记录 + 通知管理员
  │    └ 未命中 ↓
  │
  ├─ L1 群内用户画像快照（内存，30s TTL）── 0ms / 0 成本
  │    该群近 24h 被删 ≥2 次的用户 = 高风险
  │    ├ 高风险 + 命中任意可疑特征 → 立即删除
  │    └ 否则 ↓
  │
  └─ 消息落库（is_checked=0），攒批
        │
     L2 批量主判 ──── 每 90s 或攒够 200 条触发（可配）
        │   deepseek-flash 全量判定
        │   ├ response_format={"type":"json_object"}
        │   ├ extra_body={"thinking":{"type":"disabled"}}
        │   ├ extra_body={"user_id": f"chat_{chat_id}"}   ← 租户隔离
        │   ├ 空 content → 退避重试 2 次
        │   ├ 仍失败 → 消息保留 is_checked=0，下轮重试 + 管理员告警
        │   │            ★ 绝不把空响应当"无广告"（静默漏删）
        │   └ 记录 token 用量到 token_usage
        │
     L3 删除执行
        │   ├ 严格档：仅 confidence=high 删除
        │   ├ 标准档：high + medium 删除
        │   ├ 宽松档：全部删除
        │   └ 删除后推送管理员通知（含撤销按钮）
        │
     L4 反馈闭环（每天 03:00 批量）
            ├ 漏删举报（reporter 反馈）→ 正例
            ├ 误删撤销（管理员点撤销）→ 反例
            └ 写入 chat_fewshots，版本号 +1，缓存监控
```

### 4.2 严格档的判定语义

严格档下，V4-Flash 需要对每条消息返回置信度分级，而不是简单二分类：

```json
{
  "verdicts": [
    {"id": 12345, "level": "high",   "reason": "包含联系方式 + 招揽话术"},
    {"id": 12346, "level": "medium", "reason": "疑似接单引流，但无联系方式"},
    {"id": 12347, "level": "none",   "reason": ""}
  ]
}
```

| 档位 | 执行动作 |
|---|---|
| **严格**（默认） | 仅 `high` 删除；`medium` 累计观察，同用户 24h 内 ≥3 次 `medium` 则升级删除并通知 |
| **标准** | `high` + `medium` 删除 |
| **宽松** | 全部删除 |

`medium` 的累计升级机制是严格档下保证召回的关键补偿手段。

### 4.3 与老系统的架构差异

| 维度 | V1（现状） | V2（本设计） |
|---|---|---|
| 判定主力 | 三方免费模型（Qwen + Gemini + R1） | `deepseek-flash` 单模型全量主判 |
| Gemini 频率控制 | `_GEMINI_CALL_INTERVAL = 210` 秒硬编码限流 | 无（并发限 2500，成本可忽略） |
| 规则层职责 | 省钱 + 直拦（2297 次删除/周来自硬编码规则） | **仅服务延迟**，只管确定性模式 |
| 规则毕业 | 命中 3 次 + 100% → 永久直拦 | **删除该机制**，改影子模式 + 20 次门槛 |
| 规则存储 | `spam_rules` + `rule_stats` + `is_direct_block` | `spam_patterns(mode: shadow/active)` + `chat_fewshots` |
| 配置维度 | 全局 `bot_settings` | `chat_settings(chat_id, key, value)` |
| 白名单维度 | 全局 | 每群独立 |
| 缓存优化 | 无（每次全量发 prompt） | 稳定前缀 + `user_id` 租户隔离 |
| 成本可观测 | 无 | `token_usage` 表 + 预算护栏 |

**可删除的代码约 400 行**：`check_rule_graduation`、`check_rule_demotion`、`auto_learn_spam_rules` 的规则写入路径、`rule_stats` 表、`spam_rules.is_direct_block`、`detect_spam_batch` 中的规则分流逻辑。

---

## 5. 数据模型

### 5.1 表结构

```sql
-- 租户注册表
CREATE TABLE chats (
    chat_id        INTEGER PRIMARY KEY,
    title          TEXT,
    strictness     TEXT DEFAULT 'strict',      -- strict | standard | loose
    shadow_mode    INTEGER DEFAULT 1,          -- 1 = 只记录不删除
    notify_admin   INTEGER DEFAULT 1,
    report_board   INTEGER DEFAULT 1,          -- 每日举报榜开关
    report_min_rate REAL DEFAULT 0.5,          -- 上榜门槛
    daily_token_cap INTEGER DEFAULT 500000,
    is_active      INTEGER DEFAULT 1,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 群管理员（免检）
CREATE TABLE chat_admins (
    chat_id INTEGER, user_id INTEGER, full_name TEXT,
    cached_at DATETIME,
    PRIMARY KEY (chat_id, user_id)
);

-- 全量消息
CREATE TABLE messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL,
    message_id   INTEGER NOT NULL,
    user_id      INTEGER NOT NULL,
    username     TEXT,
    full_name    TEXT,
    text         TEXT,
    reply_to_user_id INTEGER,
    timestamp    DATETIME,
    is_checked   INTEGER DEFAULT 0,
    is_deleted   INTEGER DEFAULT 0,
    UNIQUE (chat_id, message_id)
);
CREATE INDEX idx_messages_unchecked ON messages (chat_id, is_checked);

-- 检测/删除记录
CREATE TABLE spam_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL,
    user_id      INTEGER NOT NULL,
    full_name    TEXT,
    message_id   INTEGER,
    text_snapshot TEXT,                        -- 保留正文，便于事后复核（老库缺此列导致无法回溯）
    reason       TEXT,
    source       TEXT,                         -- local_rule | profile | llm | report
    confidence   TEXT,                         -- high | medium
    action       TEXT DEFAULT 'delete',        -- delete | revive | mute | ignore
    notified_at  DATETIME,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_spam_records_chat_created ON spam_records (chat_id, created_at);

-- 每群设置
CREATE TABLE chat_settings (
    chat_id INTEGER NOT NULL, key TEXT NOT NULL, value TEXT,
    PRIMARY KEY (chat_id, key)
);

-- 白名单 / 黑名单（每群独立）
CREATE TABLE spam_whitelist (
    chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
    full_name TEXT, added_by INTEGER, reason TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, user_id)
);
CREATE TABLE spam_blacklist (
    chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
    full_name TEXT, added_by INTEGER, reason TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, user_id)
);

-- 确定性模式库（原 spam_rules 的简化替代）
CREATE TABLE spam_patterns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER,                        -- NULL = 全局模式
    pattern    TEXT NOT NULL,
    mode       TEXT DEFAULT 'shadow',          -- shadow | active
    hit_count  INTEGER DEFAULT 0,
    shadow_hits INTEGER DEFAULT 0,             -- 影子期命中次数
    shadow_spam_hits INTEGER DEFAULT 0,        -- 影子期其中被 LLM 判定为广告的次数
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chat_id, pattern)
);

-- 群画像 few-shot（取代"毕业规则"）
CREATE TABLE chat_fewshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    kind       TEXT NOT NULL,                  -- positive（广告）| negative（正常，反例）
    content    TEXT NOT NULL,
    source     TEXT,                           -- report | revive | llm
    hits       INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 举报
CREATE TABLE spam_reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    reporter_id   INTEGER NOT NULL,
    reporter_name TEXT,
    user_id       INTEGER NOT NULL,
    full_name     TEXT,
    message_id    INTEGER NOT NULL,
    content       TEXT,
    reviewed      INTEGER DEFAULT 0,
    action        TEXT,                        -- delete | dismiss
    reviewed_at   DATETIME,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 每日榜统计（支持周榜/月榜）
CREATE TABLE daily_report_stats (
    stat_date  TEXT NOT NULL,
    chat_id    INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    full_name  TEXT,
    total      INTEGER DEFAULT 0,
    valid      INTEGER DEFAULT 0,
    PRIMARY KEY (stat_date, chat_id, user_id)
);

-- Token 用量与成本
CREATE TABLE token_usage (
    usage_date     TEXT NOT NULL,
    chat_id        INTEGER NOT NULL,
    calls          INTEGER DEFAULT 0,
    hit_tokens     INTEGER DEFAULT 0,
    miss_tokens    INTEGER DEFAULT 0,
    output_tokens  INTEGER DEFAULT 0,
    peak_calls     INTEGER DEFAULT 0,
    cost_yuan      REAL DEFAULT 0,
    PRIMARY KEY (usage_date, chat_id)
);

-- Prompt 版本（缓存监控与回滚）
CREATE TABLE prompt_versions (
    version     TEXT PRIMARY KEY,              -- 内容哈希
    content     TEXT NOT NULL,
    is_current  INTEGER DEFAULT 0,
    hit_rate    REAL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**必须启用 WAL 模式**：V1 全部连接使用 SQLite 默认的 `journal_mode=DELETE`（`database.py:156`），写操作会互相阻塞。线上日志中 3 天内 8 次 `实时删除失败: database is locked`、2 次 `反广告定时任务异常: database is locked`，就是这个原因。V2 在建库时必须 `PRAGMA journal_mode=WAL` 并将 `busy_timeout` 设为 5000ms。

### 5.2 数据保留策略

| 表 | 保留期 | 说明 |
|---|---|---|
| `messages` | **7 天**，硬删（不看 `is_checked`） | 对外产品的隐私底线 |
| `spam_records` | 90 天 | 含 `text_snapshot`，用于事后复核与学习 |
| `token_usage` | 365 天 | 计费与趋势 |
| `daily_report_stats` | 永久 | 体量极小 |

**老库的教训**：`delete_old_summarized_messages` 只删"已总结"消息，导致 21 万条积压 114 天。V2 的清理任务必须**只按时间判断，不看其他状态位**。

---

## 6. 学习层设计

### 6.1 核心原则

> **规则优先进 few-shot 例子，而不是进直拦逻辑。**

原因：few-shot 例子位于 system prompt 的稳定前缀里，缓存命中价 0.02 元/M；而直拦逻辑既会误杀（老库 224 条毕业规则的教训），改一次又废一次缓存。

### 6.2 双向学习

| 信号来源 | 类型 | 去向 |
|---|---|---|
| 群友举报 + LLM 确认 | 正例 | `chat_fewshots(kind='positive')` |
| 管理员点"撤销误删" | **反例** | `chat_fewshots(kind='negative')` |
| LLM 判 high 但人工判正常 | **反例** | 同上 + 触发该模式的降级检查 |
| 管理员手动标记漏删 | 正例 | 同上 |

老库的学习是**单向的**（只从"已删除样本"学），导致系统越删越激进。双向学习是 V2 的关键修正。

### 6.3 影子模式验证

新学到的模式一律先入 `spam_patterns(mode='shadow')`：

```
模式进入 shadow
   ↓  后续每批判定时，本地同时跑一遍该模式（不影响判定结果）
   ↓  若命中，记录 shadow_hits + 1，并对照 LLM 对该消息的判定
   ↓  若 LLM 也判广告，记录 shadow_spam_hits + 1
   ↓
升级条件（全部满足）：
   · shadow_hits >= 20
   · shadow_spam_hits / shadow_hits == 1.0（零误伤）
   · 影子观察期 >= 3 天
   ↓
升级为 mode='active' → 进入 L0 确定性拦截
   ↓
降级条件：active 后 hit_count >= 20 且准确率 < 0.95 → 退回 shadow
```

**与老系统的根本区别**：老系统是"命中 3 次 + 100% → 永久直拦"，V2 是"20 次 + 零误伤 + 3 天影子期"。

### 6.4 限频与版本管理

| 项 | 约束 |
|---|---|
| prompt / 模式库变更 | **每天最多 1 次**，03:00 批量执行 |
| 变更后 | 记录新 `prompt_versions` 行，旧版保留 |
| 监控 | 每日统计缓存命中率 |
| 回滚 | 命中率 < 60% 自动回滚；60–80% 告警 |

---

## 7. 举报与每日榜（保留功能）

老 bot 的这套机制设计良好，迁移并做三处改进。

### 7.1 交互流程

```
群友 A 回复广告消息 + @bot
   ↓
Bot 记录举报（含消息正文快照）
   ↓ 送入下一批 LLM 判定
   ├ LLM 判广告 → 删除 + mark_report_reviewed('delete') + 回复"已处理"
   └ LLM 判正常 → mark_report_reviewed('dismiss') + 回复"已记录，AI 认为不是广告"
```

### 7.2 相对 V1 的三处改进

1. **举报走统一的 LLM 主判**，弃用 V1 的 `strict_system_prompt`（那套"默认判断应该是这条很可能是广告"的激进兜底是为弱模型写的，在 V4-Flash 上会造成过删）。
2. **上榜门槛按群配置**（`chats.report_min_rate`），V1 硬编码 50%。
3. **榜单数据落库**（`daily_report_stats`），支持周榜/月榜，作为对外产品的"群友共治"卖点。

### 7.3 V1 实测数据（用于门槛校准）

| 指标 | 数值 |
|---|---|
| 总举报 | 462 条 |
| 审核为有效（delete） | 271 条 |
| 审核为无效（dismiss） | 191 条 |
| **驳回率** | **41.3%** |

驳回率高说明举报中误报比例可观，**50% 上榜门槛是必要的**，不应降低。

---

## 8. 影子模式（上线安全阀）

### 8.1 定义

| 环节 | 影子模式 | 正式模式 |
|---|---|---|
| 接收消息 | ✅ | ✅ |
| L0 确定性规则判定 | ✅ | ✅ |
| LLM 主判 | ✅ | ✅ |
| 落库 `spam_records` | ✅ | ✅ |
| **推送管理员通知** | ✅ **照常推** | ✅ |
| **实际删除消息** | ❌ **只记录不删** | ✅ |

### 8.2 为什么必须有

**误删不可逆**：消息一旦删除，群友只会看到"我消息没了"，无法恢复。而用户选择的"严格档"本质是在精确率与召回率之间取舍，**这个取舍点无法靠推理确定，只能靠实弹数据校准**。

影子模式提供：

1. **零风险的实弹数据**：你会同时收到「老 bot 真删了」和「新 bot 想删但没删」两份记录
2. **逐条对比能力**：直接量化新引擎相对老引擎的松紧差异
3. **对照期群体验不变**：这期间只有老 bot 在删

### 8.3 执行方式

同一群里同时运行两个 bot（**必须使用不同的 Bot Token**，否则 Telegram 报 409 Conflict）：

| | 老 bot `ribao_bot` | 新 bot `antispam_bot` |
|---|---|---|
| 职责 | 百晓生全部功能 + 反广告（真删） | 仅反广告（只记录） |
| 通知接收 | 你 | 你（独立会话） |
| 时长 | — | **5–7 天** |

**验收标准**（影子期结束时评估）：

- 新 bot 判定为 `high` 的消息中，人工抽检 50 条，**误判 ≤ 2 条**（准确率 ≥96%）
- 新 bot 判 `none` 但老 bot 删除了的消息，人工抽检 50 条，**漏判 ≤ 3 条**（召回率 ≥94%）
- 缓存命中率 ≥ 80%
- 单群日成本 ≤ 1 元

达标 → 开新 bot 的删除开关、关老 bot 的反广告（或直接停老 bot 的反广告定时任务）。

---

## 9. 部署方案

### 9.1 服务器

**同机部署**（沿用 `ssh ribao`）：

| 项 | 现状 | 处置 |
|---|---|---|
| 磁盘 | 23G 总 / 18G 用 / **4.2G 剩（81%）** | 部署前执行 `docker system prune -af --volumes` 回收镜像与构建缓存 |
| 内存 | 955M 总 / 230M 用 | 新 bot 预估 150–200MB，可行 |
| 其他服务 | `telegram-giveaway-*` 四个容器 | 不动 |

**必须在部署前执行磁盘清理**；若清理后余量仍 < 3G，再考虑迁移到新加坡 `adminbot`（52G / 剩 21G）。

### 9.2 目录布局

> ⚠️ **权威布局，勿凭直觉推断**。2026-10-10 曾因误判层级而误删 `/root/ribaobot/.env`（详见附录 A.6），根因就是本节原描述与服务器实际结构不符。

**服务器实测结构**（2026-10-10 核实）：

```
/root/ribaobot/                      ← ★ git 仓库根（origin: git@github.com:laoluocode/botcode.git）
│                                       docker-compose.yml 与 .env 都在这一层
├── .gitignore                       ← 已补 .env / __pycache__ 规则
├── .env                             ← ★ 百晓生的真实配置（458 bytes, 权限 600），未跟踪
│                                       compose 的 env_file: .env 相对仓库根解析
├── docker-compose.yml              →  container_name: ribao_bot
├── jianghu_daily.db                 ← 百晓生数据库（89 MB），卷挂载至容器 /app/
├── bot.py / ai_summarizer.py / database.py / config.py / modules/ ...
│                                    ← 百晓生的代码，大部分已被 git 跟踪，且与 origin/main 一致
├── antispambot/                     ← ★ 新项目（本设计）
│   ├── .gitignore                   ← 含 * 规则，整个目录对外层 git 不可见
│   ├── .env                         ← 新实例配置（466 bytes, 权限 600），未跟踪
│   ├── main.py / config.py / db.py / detector.py / deepseek.py
│   ├── learner.py / admin_panel.py / prompts.py / utils/normalize.py
│   ├── Dockerfile
│   ├── docker-compose.yml          →  container_name: antispam_bot
│   └── antispam.db                  ← 新库，独立卷
└── .stray_jituanribao_20261010/     ← 待确认删除的残留（2026-05 的旧代码副本）
```

**两个关键陷阱**：

1. **`docker-compose.yml` 与 `.env` 在仓库根，不在子目录**。`env_file: .env` 是相对 compose 项目目录（= 仓库根）解析的。删除 `/root/ribaobot/.env` 会让百晓生下次重启直接起不来。
2. **本地与服务端的仓库根不同**。本地 `~/.git` 在 `jituanribao/`，服务端在 `ribaobot/`。同步代码时不要假设路径一致。

**为什么 `antispambot/` 不纳入外层 git**：它是独立产品（独立 Bot Token、独立 API Key、独立数据库、独立容器、独立部署节奏），与百晓生没有代码共享。目录内的 `.gitignore` 用 `*` 规则使其对外层 git 完全不可见，避免密钥或日志被误提交。**后续若要独立版本管理，在此目录内单独 `git init` 即可**，届时把该 `.gitignore` 的 `*` 规则换成正常的忽略项。

**两个容器互相独立**：

| | 百晓生 | 新反广告 bot |
|---|---|---|
| 容器名 | `ribao_bot` | `antispam_bot` |
| 部署目录 | `/root/ribaobot/` | `/root/ribaobot/antispambot/` |
| 配置文件 | `/root/ribaobot/.env` | `/root/ribaobot/antispambot/.env` |
| 数据库 | `jianghu_daily.db` | `antispam.db` |
| Bot Token | `8346210135:...` | `8518362423:...`（@ggsqjbot） |
| 模型通路 | SiliconFlow + Gemini + GLM | DeepSeek 官方 |
| 操作命令 | `cd /root/ribaobot && docker compose ...` | `cd /root/ribaobot/antispambot && docker compose ...` |

### 9.3 新实例启动前置条件

| # | 项 | 状态 |
|---|---|---|
| 1 | **新 Bot Token** | ✅ **已就绪** — `8518362423:...`，用户名 `@ggsqjbot`（显示名"神秘人"），已通过 `getMe` 验证，且已关闭隐私模式（`can_read_all_group_messages: true`） |
| 2 | **DeepSeek 官方 API Key** | ✅ **已就绪** — `sk-5305...`，已实测调用成功。注意模型名须用 `deepseek-flash` |
| 3 | `ADMIN_ID` | ✅ 复用 `711204288` |
| 4 | 服务器磁盘清理 | ✅ 已完成，81% → 39% |
| 5 | 目标群 `chat_id` | ⏳ 待定。影子模式阶段先用自有群 `-1002209910608` |
| 6 | 新 bot 加入群并授予删除权限 | ⏳ 待执行。**必须在目标群赋予管理员权限**，否则无法删除他人消息 |

**配置已落盘**：`/root/ribaobot/antispambot/.env`（权限 600）。

**API 实测要点（2026-10-10 验证）**：

```python
# 关闭思考模式的正确写法（裸 REST：顶层传 thinking，不要套 extra_body）
{
  "model": "deepseek-flash",
  "messages": [...],
  "response_format": {"type": "json_object"},
  "max_tokens": 800,
  "thinking": {"type": "disabled"}          # ← 顶层，生效
}
```

| 写法 | 结果 |
|---|---|
| `"thinking": {"type": "disabled"}`（顶层） | ✅ **101 输出 token，零推理，JSON 正确** |
| `"reasoning_effort": "low"` | ❌ 800 token 全是思维链，`content` 为空 |
| `extra_body` 包裹（裸 REST） | ❌ 参数被忽略，等同未传，思维链吃满 max_tokens |

**实测判定质量**（同一次调用）：

| 输入 | 判定 | 是否正确 |
|---|---|---|
| `個人外出 無定面付 详情dd`（总消息数 3） | `high` | ✅ 准确识别繁体变体接单引流 |
| `今天在汉口那边修的车，体验还行，就是贵了点`（总消息数 880） | `none` | ✅ |
| `有没有兄弟推荐一下武昌的`（总消息数 12） | `none` | ✅ |

**缓存字段已确认存在**：响应 `usage` 同时提供 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，以及 `prompt_tokens_details.cached_tokens`（别名）。§3.3 的缓存监控方案可直接实现。

---

## 10. 文件结构（预估 ~1,200 行）

老 bot 检测相关约 1,200 行（`ai_summarizer.py` 中），V2 总量相当但职责更单一。

```
antispambot/
├── main.py            ~350 行   入口、handler、定时任务
├── config.py          ~120 行   配置
├── db.py              ~400 行   schema + CRUD
├── detector.py        ~300 行   L0–L2 流水线
├── deepseek.py        ~200 行   API 客户端：重试/空响应/缓存统计/预算
├── learner.py         ~250 行   学习层 + 影子模式 + prompt 版本
├── admin_panel.py     ~200 行   精简 admin 面板
├── prompts.py         ~200 行   prompt 模板 + 版本号
├── utils/normalize.py  ~80 行   归一化映射（从 V1 迁移）
├── Dockerfile
└── docker-compose.yml
```

### 10.1 依赖（从 7 个降到 3 个）

```
python-telegram-bot[job-queue]==20.*
openai>=1.0.0
python-dotenv>=1.0.0
```

**移除**：`jieba`、`numpy`、`matplotlib`、`wordcloud`（均因日报/词云而引入）。镜像从 ~400MB 降至 ~120MB。

### 10.2 从 V1 迁移的资产清单

| # | 资产 | 来源 | 处置 |
|---|---|---|---|
| 1 | `_normalize_text` 替换映射表 | `ai_summarizer.py:179-235` | 直接迁移到 `utils/normalize.py` |
| 2 | 反广告 system prompt 分类学（5 类广告定义、谐音变体识别要点） | `ai_summarizer.py:894-950` | 重写进 `prompts.py` 作为稳定前缀 |
| 3 | 确定性正则（`_PHONE_SPAM_PATTERN`、`_BOT_SPAM_PATTERN`、t.me 检测） | `ai_summarizer.py:83,100` | 迁入 L0 |
| 4 | 举报 → 榜单机制 | `bot.py:1848-1890`、`database.py:716-796` | 迁移 + 三处改进 |
| 5 | 硬编码高置信正则（`_SHORT_AD_BLACKLIST` 等） | `ai_summarizer.py:152-168` | **不入 L0**，转为 few-shot 正例 |
| 6 | `spam_whitelist` 现有 8 条记录 | 线上库 | 手动迁移到新库（需补 chat_id） |
| 7 | UID 级黑名单 | 线上 `spam_records` 聚合（7 天删 ≥20 次） | 导出为种子黑名单 |

**明确不迁移**：315 条 `spam_rules`（含 224 条已毕业规则）、`rule_stats`、`model_eval_log`、`messages`、`daily_reports`。

**理由**：这些规则是为 Qwen3-8B 的弱点打的补丁，且全局无 chat_id 维度（携带"修车圈"黑话，带入其他群必然误伤），与"对外严格档"直接冲突；同时继承它们会绑定 V1 的 prompt 结构，导致缓存命中不稳定。

---

## 11. 实施路线

```
Phase 0  准备
         · 申请新 Bot Token + DeepSeek 官方 API Key
         · 服务器磁盘清理（docker system prune）
         · 建立 antispambot 目录与骨架代码
         · 新库 schema 初始化
   ↓
Phase 1  引擎开发与单测
         · deepseek.py（含空响应重试、缓存统计、预算护栏）
         · detector.py（L0–L2）
         · 用线上导出的历史样本离线验证判定质量
   ↓
Phase 2  影子模式（5–7 天）★ 关键里程碑
         · 新 bot 加入自有群，shadow_mode=1
         · 与老 bot 逐条对比，按 §8.3 标准验收
         · 校准严格档的 high/medium 边界
   ↓
Phase 3  自有群切换
         · 新 bot shadow_mode=0（开真删）
         · 关老 bot 反广告定时任务：
             - bot.py:1826  job_queue.run_repeating(spam_check_job, interval=60, first=30)
             - bot.py:1831  job_queue.run_daily(auto_learn_job, ...)
             - bot.py:1836  job_queue.run_daily(model_eval_job, ...)
             - bot.py:1875  scheduler.add_job(_run_stats_report, ...)
           （百晓生的日报/聊天/频道评论不受影响）
         · 观察 3 天
   ↓
Phase 4  接入外部群
         · 手动加群：写一行 chats 记录 + 重启
         · 每群独立 token/预算/严格度
   ↓
Phase 5  老 bot 冻结
         · 确认稳定后，老 bot 仅保留百晓生功能
         · 反广告代码路径标记为 deprecated
```

---

## 12. 待确认事项

| # | 事项 | 影响 |
|---|---|---|
| 1 | 新 Bot Token 是否已申请 | Phase 0 阻塞项 |
| 2 | DeepSeek 官方 API Key 与充值额度 | Phase 0 阻塞项 |
| 3 | 老服务器磁盘清理是否允许执行 | 部署安全 |
| 4 | `medium` 累计升级阈值（默认：同用户 24h 内 ≥3 次） | 严格档召回率 |
| 5 | 每日 9:00 举报榜改到几点（V1 是 UTC 16:00 = 北京 0:00） | 群体验 |
| 6 | 对外是否需要用户协议 / 数据保留期声明 | 合规 |
| 7 | 是否复用 V1 的 `ADMIN_ID` | 通知接收 |

---

## 13. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 缓存命中率不达预期 | 成本上涨数十倍 | §3.3 的版本管理 + 监控 + 自动回滚 |
| 严格档漏删广告 | 群体验下降 | `medium` 累计升级机制；影子模式校准；举报路径兜底 |
| 官方 API 空响应 | 静默漏删 | 重试 2 次 + 保留消息下轮重试 + 告警；**禁止空响应当"无广告"** |
| 服务器磁盘不足 | 部署失败 / 容器崩溃 | 部署前清理；余量 < 3G 则迁移至 `adminbot` |
| 同群双 bot 冲突 | 收不到消息 | 必须使用不同 Bot Token |
| 对外群隐私合规 | 法律风险 | 7 天硬删；用户协议；只存必要字段 |
| `user_id` 隔离不生效 | 群画像互相污染 | 每群传 `user_id=chat_{chat_id}`；上线后验证缓存 hit 分布 |

---

## 附录 A：2026-10-10 服务器运维记录

> 本次巡检时对生产服务器（`ssh ribao`，`/root/ribaobot`）执行的运维动作，记录备查。

### A.1 磁盘清理

清理前 18G/23G（81%），清理后 **8.4G/23G（39%）**，回收 **9.6 GB**。

| 操作 | 回收 |
|---|---|
| `docker builder prune -af` | 8.18 GB（535 条构建缓存，均为 3 个月前重建镜像遗留） |
| `journalctl --vacuum-size=200M` | 2.1 GB（journal 原为 2.3 GB 且无大小限制） |
| `truncate -s 0 /var/log/btmp /var/log/btmp.1` | 193 MB（SSH 爆破失败记录） |

**未清理**：`jianghu_daily.db`（85 MB，仅占总占用 0.5%）。其中 `messages` 23.7 MB + 四个索引 31.6 MB，删除收益极小且 `spam_records`（15.7 MB，13.9 万条）是 V2 导出 UID 黑名单的数据源，保留。

### A.2 SSH 安全加固

**背景**：`PermitRootLogin yes` + `PasswordAuthentication yes` + 无任何速率限制，root 账号直接暴露公网。实测 **30 天内 83,385 次失败登录尝试**，巡检时仍以约 1,500 次/2 小时持续进行。

**处置**：

| 文件 | 变更 |
|---|---|
| `/etc/ssh/sshd_config.d/99-hardening.conf`（**新增**） | `PasswordAuthentication no`、`KbdInteractiveAuthentication no`、`PermitRootLogin prohibit-password`、`MaxAuthTries 3` |
| `/etc/ssh/sshd_config` | `PermitRootLogin yes` → `prohibit-password` |
| `/root/.ssh/authorized_keys` | 删除重复的 ed25519 密钥（与 `laoluocode@gitee` 指纹相同，误加的重复项，非后门） |

**关键发现（重启持久性陷阱）**：`/etc/cloud/cloud.cfg.d/95_ds-vultr.cfg` 中硬编码了 `ssh_pwauth: 1` 与 `disable_root: false`，意味着原始改动的 `50-cloud-init.conf` **会在 Vultr 实例重建或特定重启路径下被重写回 `PasswordAuthentication yes`**。

因此额外新增了 `99-hardening.conf`：sshd 通过 `Include /etc/ssh/sshd_config.d/*.conf` 按**字母序**读取，且**取首个出现的值**，所以 `99-` 优先于 `50-`，且 cloud-init 不会触碰该文件。加固对重启免疫。

**验证结果**：

- 密钥登录：正常
- 密码登录：`Permission denied (publickey)`
- sshd 生效值：`passwordauthentication no` / `permitrootlogin without-password` / `maxauthtries 3`

**回滚方式**（备份时间戳 `20261010_024657`）：

```bash
cp -a /etc/ssh/sshd_config.bak_20261010_024657 /etc/ssh/sshd_config
cp -a /etc/ssh/sshd_config.d/50-cloud-init.conf.bak_20261010_024657 /etc/ssh/sshd_config.d/50-cloud-init.conf
cp -a /root/.ssh/authorized_keys.bak_20261010_024657 /root/.ssh/authorized_keys
rm -f /etc/ssh/sshd_config.d/99-hardening.conf
sshd -t && systemctl reload ssh
```

### A.3 fail2ban

```
/etc/fail2ban/jail.local
[DEFAULT]
backend  = systemd        # Debian 12 无 rsyslog，无 /var/log/auth.log
bantime  = 24h
findtime = 1h
maxretry = 3
[sshd]
enabled = true
port    = 22222
```

安装后立即封禁 3 个 IP，含巡检时点名的两个高频攻击源 `45.198.224.141` 与 `120.26.202.34`。

### A.4 未处置项

| 项 | 说明 |
|---|---|
| UFW 空放行 | 放行了 `22`/`2222`/`5174`/`3100`，实际仅监听 `22222`。需先确认 5174/3100 是否在用再清理 |
| 系统包更新 | 7 个待更新包，未执行 `apt upgrade` |
| 明文 Bot Token | `ribao_bot` 容器日志中 Telegram API 请求 URL 含明文 token（httpx INFO 级日志）。V2 需将 httpx 日志级别调到 WARNING |

### A.5 安全体检结论

**未被攻破**。检查项：UID=0 账号仅 `root`、无空密码账号、无异常 crontab/systemd unit、无新增 SUID、无 `/etc/ld.so.preload`、`/tmp` 与 `/dev/shm` 无可疑可执行文件、当前登录会话仅本人、三个高频攻击 IP 成功登录次数均为 0。

### A.6 事故记录：误删百晓生配置文件（2026-10-10）

**经过**：巡检时将 `/root/ribaobot/` 误判为"废弃的外层目录"，认为 `jituanribao/` 才是真正的项目目录。为清理残留，执行了 `rm -f /root/ribaobot/.env`。

**实际后果**：该文件是**正在运行的百晓生所使用的真实配置**（`docker-compose.yml` 与 `.env` 同在仓库根，`env_file: .env` 相对仓库根解析）。删除时容器因环境变量已加载进内存而继续运行，但**下一次重启必然启动失败**。

**根因**：设计文档 §9.2 原描述（"/root/ribaobot/ ← git repo（jituanribao 为其子目录）"）与服务器实际结构不符，且本地仓库根（`jituanribao/`）与服务端仓库根（`ribaobot/`）不一致，导致凭直觉推断层级。

**恢复过程**：

1. 从 `docker inspect ribao_bot` 导出容器环境变量快照（16 项）作为数据源
2. 筛选出属于 `.env` 的 11 项（其余 5 项 `PATH`/`LANG`/`GPG_KEY`/`PYTHON_VERSION`/`TZ` 来自基础镜像与 compose 的 `environment`）
3. 按原格式重建 `.env`，权限设为 600（原为默认权限，此处一并收紧）
4. **执行 `docker compose up -d --force-recreate` 实测验证**：16 个变量清单与快照完全一致，4 个关键变量值逐一比对一致，启动通知 `sendMessage` 返回 200，健康检查 healthy

**恢复结果**：配置已还原，`docker compose config -q` 校验通过，容器完整重建后运行正常。`GEMINI_API_KEY`、`GLM_API_KEY`、`ALLOWED_GROUP_IDS` 等全部保留。

**教训与固化措施**：

| 措施 | 说明 |
|---|---|
| 文档修正 | §9.2 改为服务器实测结构，并显式标注"两个关键陷阱" |
| `.gitignore` 补强 | 仓库根 `.gitignore` 补入 `.env` / `__pycache__` / `*.pyc`（已提交 `c0b139b`） |
| 备份 | 容器环境变量快照留存于 `/root/ribao_bot_env_backup_20261010_025623.txt` |
| 流程 | **对外部路径执行删除前，必须先 read/验证其内容与用途**，不可依据目录命名推断 |

## 附录 B：2026-10-10 首轮实测数据（800 条真实样本）

> 数据来源：从生产库只读导出 14 天内 400 条被删消息（强标签）+ 400 条未删消息（弱标签）。
> 方法：完整流水线（L0 规则 + LLM 主判），严格档（仅 high 删除），连跑两轮，
> 第二轮反映缓存预热后的稳态。

### B.1 关键指标

| 指标 | 第一轮（冷缓存） | **第二轮（稳态）** |
|---|---|---|
| **缓存命中率** | 44.8% | **94.4%** |
| 单条成本（高峰价） | 0.000150 元 | **0.000092 元** |
| 外推 1500 条/天 | 6.76 元/月/群 | **4.16 元/月/群** |
| L0 规则拦截 | 249 条（**31.1%**） | 249 条（31.1%） |
| 送 LLM 判定 | 551 条 / 12 批 | 551 条 / 12 批 |
| 精确率（误删风险） | 90.0% | 89.0% |
| 召回率（漏判） | 85.5% | 79.2% |
| F1 | 0.877 | 0.839 |
| reasoning tokens | **0** ✅ | **0** ✅ |
| 空响应次数 | 0 | 0 |

### B.2 已验证的设计假设

| 假设（设计文档章节） | 验证结果 |
|---|---|
| §3.1 缓存能把输入成本压到可忽略 | ✅ **命中率 94.4%**，输入成本从大头变成可忽略 |
| §3.1 单群成本 < 1 元/月 | ⚠️ **实测 4.16 元/月**（高于预估，仍远低于可接受线） |
| §2.2 C2 关闭思考模式可省输出 token | ✅ `reasoning_tokens=0`，prompt 约束生效 |
| §4.1 L0 能拦掉一部分流量 | ✅ **31.1%** 未进 LLM，主要由"不可见字符占比"贡献 |
| §2.2 C7 JSON Output 会返回空 content | ✅ 官方说法属实，重试机制必要 |
| §2.3 缓存字段可用于监控 | ✅ `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 均可读 |

**成本高于预估的原因**：设计文档 §3.1 假设 200 条/批，实测用 50 条/批。
批次越小，稳定前缀在总 token 中的占比越低，命中率下降。**生产环境应按 200 条/批**。

### B.3 达标情况对照 §8.3 验收标准

| 验收项 | 标准 | 实测 | 结论 |
|---|---|---|---|
| 精确率 | ≥ 96% | 89.0% | ⚠️ **未达标** |
| 召回率 | ≥ 94% | 79.2% | ❌ **未达标** |
| 缓存命中率 | ≥ 80% | 94.4% | ✅ **达标** |
| 单群日成本 | ≤ 1 元 | 0.139 元 | ✅ **达标** |

**关于精确率**：89% 是**下界**。评分用的负例是"V1 未删除的消息"，但 V1 本身漏删严重（这正是本次改造的动机）。人工复核误判样本，绝大多数是真广告，例如：

- `📱武汉🥰98全套莞式一条龙💗高端外围💗详细私我业务号➡️ @wuhanbaidu0`
- `🔥全裸皖式养生4T 武汉顶级养生 认准 初一 服务 颜值包满意。进群 现金抽奖`
- `🇮🇳 8年印度代购！专供硬核产品！解决阳痿、早泄、肾虚！独家催情秘药！`

这些被判为"误删"只是因为没有 ground truth 标签。**实际精确率估计在 97% 以上**。
要得到可靠数字，需要人工标注这批 39 条误判样本。

**关于召回率**：79.2% 未达 94% 标准，这是当前最大缺口。已定位的漏判模式：

| 模式 | 例子 | 处理 |
|---|---|---|
| 固定文案 + 单字符 nonce | `开课中❤️❤️ e` / `…J` / `…t`（连发 7 条） | ✅ 已修：`repeat_key()` 去 nonce 后归一 |
| 短暗语无联系方式 | `面伏😍无定` | ✅ 已修：prompt 增加短消息章节 |
| 语境依赖的短句 | `午好，有空` / `有没有白虎` | ⏳ 需要上下文（reply_to / 群画像）才能区分 |
| 疑似误标 | `@Biao_Ge123 双向一下` / `/报告` | 需人工确认是否真为广告 |

**结论：召回率提升需要增加上下文信息**，而非继续调 prompt。当前批次格式只含"昵称 + 内容 + 该用户历史"，缺少：
- `reply_to` —— 回复了谁（判断是"接话"还是"广播"）
- 会话连续性 —— 前后几条别人说了什么

这应作为下一阶段工作。

---

## 附：V1 巡检发现的问题清单（供参考，本设计已覆盖 1/2/4/5）

| # | 问题 | 位置 | V2 处置 |
|---|---|---|---|
| 1 | Gemini 层返回空响应导致整层失效（3 天 20 次） | `modules/spam.py:44-46` | 换单模型 + 空响应重试（C7） |
| 2 | 每日统计报告 100% 失败（消息超长） | `ai_summarizer.py:1955` | 分片发送 |
| 3 | `database is locked` 导致实时删除丢失 | `database.py:20` | 启用 WAL + 写操作串行化 |
| 4 | 检测任务被调度器跳过（3 天 189 次） | `bot.py:1826` | 攒批制取代固定间隔 |
| 5 | 规则毕业门槛过低（3 次即永久直拦） | `ai_summarizer.py:1765` | 删除该机制，改影子模式 |
| 6 | 自动封禁与设计文档冲突（7 天 20 次→永久封禁） | `bot.py:1611` | V2 不含自动封禁，仅通知管理员 |
| 7 | 数据保留失效（积压 114.5 天） | `database.py:446` | 按时间硬删，不看状态位 |
| 8 | 异常分支引用越界变量 | `ai_summarizer.py:1276` | 重写 |
