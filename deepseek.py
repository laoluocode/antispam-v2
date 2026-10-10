"""DeepSeek 官方 API 客户端。

职责边界：只负责"把消息发出去、把 JSON 拿回来、把用量记下来"。
不含任何判定逻辑，便于单独测试。

三个必须处理的官方行为（均经 2026-10-10 实测确认）：

1. **思考模式默认开启**，思维链计入输出 token（最贵档），且开启时
   `temperature` 会被静默忽略。必须在**顶层**传 `thinking={"type":"disabled"}`。
   实测：套在 `extra_body` 里（OpenAI SDK 的写法）对裸 REST 请求无效，
   参数被丢弃后思维链会吃满 max_tokens 且 `content` 返回空。

2. **JSON Output 有概率返回空 content**（官方文档明确承认）。
   绝不能把空响应当成"没有广告"处理——那是静默漏删。
   处理方式：退避重试，仍失败则抛异常让调用方保留消息下轮重试。

3. **请求会持续返回空行保活**，最长 10 分钟未开始推理才断开连接。
   因此超时必须放宽，默认 60s 偏短。

缓存用量从 `usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 读取，
用于设计文档 §3.3 的缓存命中率监控与自动回滚。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# 官方并发限制：deepseek-flash 为 2500（账号级），无需主动限流。
# 但单请求超时需放宽——官方最长会保活 10 分钟。
DEFAULT_TIMEOUT = 300.0
# 输出上限。改为置信度格式后每条 verdict 约 10 token
# （实测 14 条 = 147 token），200 条批次约 2k token。
# 截断有挽救机制兜底，这里留 4 倍余量即可。
DEFAULT_MAX_TOKENS = 8000

# 空响应重试次数。官方 JSON Output 偶发返回空 content，
# 重试 2 次可覆盖绝大多数瞬时情况。
EMPTY_RESPONSE_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0


class DeepSeekError(RuntimeError):
    """调用失败。调用方必须保留消息待下轮重试，不可视为"无广告"。"""


@dataclass
class Usage:
    """单次调用的用量，用于成本核算与预算护栏。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    empty_responses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return self.cache_hit_tokens / total if total else 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cache_hit_tokens += other.cache_hit_tokens
        self.cache_miss_tokens += other.cache_miss_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.calls += other.calls
        self.empty_responses += other.empty_responses

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "Usage":
        details = raw.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=raw.get("prompt_tokens", 0) or 0,
            completion_tokens=raw.get("completion_tokens", 0) or 0,
            # 官方两个字段名互为别名，取其一即可（实测两者都存在）
            cache_hit_tokens=raw.get("prompt_cache_hit_tokens", 0) or 0,
            cache_miss_tokens=raw.get("prompt_cache_miss_tokens", 0) or 0,
            reasoning_tokens=details.get("reasoning_tokens", 0) or 0,
            calls=1,
        )


# 官方定价（元 / 百万 token，高峰时段价）。
# 高峰 = 周一至周五 9:00-12:00 与 14:00-18:00（北京时间）；
# **其余时段（含全部周末与法定节假日）价格减半**。
PRICE_YUAN_PER_MILLION = {
    "cache_hit": 0.04,
    "cache_miss": 2.00,
    "output": 8.00,
}


def is_peak_now(now=None) -> bool:
    """当前是否处于高峰计费时段。

    早先的实现在所有调用处硬编码 peak=True，等于**按最贵时段估算**，
    而空闲时段（含全部周末）实际价格减半——估算值约为真实值的两倍。
    这会让预算判断偏保守，但也会让我们误判成本规模。
    """
    try:
        import pytz
    except ImportError:
        return True
    tz = pytz.timezone("Asia/Shanghai")
    t = now or datetime.now(tz)
    if t.tzinfo is None:
        t = tz.localize(t)
    else:
        t = t.astimezone(tz)
    # 周末整体算空闲
    if t.weekday() >= 5:
        return False
    h = t.hour
    return (9 <= h < 12) or (14 <= h < 18)


def estimate_cost_yuan(usage: Usage, peak: bool | None = None) -> float:
    """按官方定价估算成本。

    peak=None 时**自动按当前时段**判断——这才能反映真实支出。
    显式传 True/False 用于回测与对比。
    """
    if peak is None:
        peak = is_peak_now()
    factor = 1.0 if peak else 0.5
    return factor * (
        usage.cache_hit_tokens * PRICE_YUAN_PER_MILLION["cache_hit"]
        + usage.cache_miss_tokens * PRICE_YUAN_PER_MILLION["cache_miss"]
        + usage.completion_tokens * PRICE_YUAN_PER_MILLION["output"]
    ) / 1_000_000


@dataclass
class DailyBudget:
    """每群每日 token 预算护栏。

    触发后调用方应降级为"仅 L0 规则 + 攒批延后"，并通知管理员。
    防止某个群被刷屏时 token 无上限燃烧。
    """

    cap: int
    used: int = 0

    @property
    def exceeded(self) -> bool:
        return self.used >= self.cap

    @property
    def remaining_ratio(self) -> float:
        return max(0.0, 1.0 - self.used / self.cap) if self.cap else 1.0


def parse_verdicts(content: str) -> dict[int, float]:
    """从模型返回中提取 {记录ID: 置信度}。

    **容忍截断**：输出被 max_tokens 切断时，完整的那部分 verdict 依然有效。
    逐个提取 `"id":N` 与紧随其后的 `"confidence":X`，不依赖整体 JSON 可解析。

    这样做的意义：截断只会让**尾部若干条**缺少判定，由调用方重试这些具体消息；
    而不是整批报废。若把截断当"无广告"处理，就成了静默漏删。
    """
    verdicts: dict[int, float] = {}
    pattern = re.compile(
        r'"id"\s*:\s*(-?\d+)\s*,\s*"confidence"\s*:\s*([0-9]*\.?[0-9]+)'
    )
    for match in pattern.finditer(content or ""):
        try:
            verdicts[int(match.group(1))] = float(match.group(2))
        except (TypeError, ValueError):
            continue
    return verdicts


def to_verdict_list(scores: dict[int, float]) -> list[dict]:
    """把 {id: 置信度} 转回 verdicts 列表，供调用方统一处理。"""
    return [{"id": rid, "confidence": conf} for rid, conf in scores.items()]


class DeepSeekClient:
    """官方 API 客户端。

    每个群传独立的 user_id，官方据此做 KVCache / 内容安全 / 调度三重隔离，
    因此不同群的缓存与限流天然互不干扰。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.usage = Usage()

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "DeepSeekClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def judge(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> tuple[dict[str, Any], Usage]:
        """发送一批消息做判定，返回 (解析后的 JSON, 本次用量)。

        messages 的前缀必须稳定（见 prompts.py 的说明），最后一条是待判消息。
        失败抛 DeepSeekError —— 调用方**不得**把它当"无广告"，必须留待下轮重试。
        """
        if not messages:
            raise ValueError("messages 不能为空")

        request_usage = Usage()
        last_error: Exception | None = None

        for attempt in range(EMPTY_RESPONSE_RETRIES + 1):
            if attempt > 0:
                delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "空响应重试 %d/%d，%.1fs 后重试",
                    attempt, EMPTY_RESPONSE_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    # JSON Output。prompt 中必须含 "json" 字样并给出格式样例（已满足）。
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    timeout=timeout,
                    # 官方参数：顶层传 thinking。关闭思考模式以省下最贵的输出 token。
                    # 不能套 extra_body —— 那是 OpenAI SDK 的用法，裸 REST 会被忽略。
                    extra_body={
                        "thinking": {"type": "disabled"},
                        # 租户隔离：官方按 user_id 做 KVCache / 内容安全 / 调度隔离。
                        "user_id": user_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - 网络/限流/服务端错误统一处理
                last_error = exc
                logger.warning("API 调用失败（第 %d 次）: %s", attempt + 1, exc)
                continue

            call_usage = Usage.from_api(
                response.usage.model_dump() if response.usage else {}
            )
            request_usage.add(call_usage)

            content = None
            finish_reason = None
            if response.choices:
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

            # 输出被 max_tokens 截断：JSON 不完整。
            # 不直接放弃——先尝试挽救已完整的部分，缺失的由调用方重试。
            if finish_reason == "length":
                salvaged = parse_verdicts(content or "")
                self.usage.add(request_usage)
                logger.warning(
                    "输出被截断（finish_reason=length, completion_tokens=%s），"
                    "已挽救 %d 条判定，其余需重试",
                    call_usage.completion_tokens, len(salvaged),
                )
                return (
                    {"verdicts": to_verdict_list(salvaged), "truncated": True},
                    request_usage,
                )

            if not content or not content.strip():
                # 官方已知问题：JSON Output 偶发空 content。
                # 关键：这里绝不能返回空判定，否则等于静默漏删。
                request_usage.empty_responses += 1
                reasoning = (
                    response.usage.completion_tokens_details.reasoning_tokens
                    if response.usage and response.usage.completion_tokens_details
                    else 0
                )
                logger.warning(
                    "收到空 content（第 %d 次）finish_reason=%s reasoning_tokens=%s",
                    attempt + 1,
                    response.choices[0].finish_reason if response.choices else "?",
                    reasoning,
                )
                if reasoning:
                    # 出现推理 token 说明 thinking 没被关掉，值得单独告警：
                    # 这会让每次调用白烧最贵的输出 token。
                    logger.error(
                        "检测到 reasoning_tokens=%s，thinking 关闭参数可能未生效！", reasoning
                    )
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                # 整体 JSON 损坏（截断或格式错乱）——用逐个提取挽救。
                salvaged = parse_verdicts(content)
                if salvaged:
                    self.usage.add(request_usage)
                    logger.warning(
                        "JSON 解析失败但已挽救 %d 条判定（第 %d 次）: %s",
                        len(salvaged), attempt + 1, exc,
                    )
                    return (
                        {"verdicts": to_verdict_list(salvaged), "truncated": True},
                        request_usage,
                    )
                last_error = exc
                logger.warning("JSON 解析失败且无法挽救（第 %d 次）: %s | 原文: %s",
                               attempt + 1, exc, content[:300])
                continue

            self.usage.add(request_usage)
            return parsed, request_usage

        raise DeepSeekError(
            f"调用失败，已重试 {EMPTY_RESPONSE_RETRIES} 次: {last_error}"
        ) from last_error


if __name__ == "__main__":
    import os

    from prompts import GLOBAL_PROMPT

    async def _selftest() -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            print("需要设置 DEEPSEEK_API_KEY 环境变量")
            return

        messages = [
            {"role": "system", "content": GLOBAL_PROMPT},
            {
                "role": "user",
                "content": (
                    "待判定消息（共 3 条）：\n"
                    "[ID=1] [发言数=3] 昵称=新人 | 内容=個人外出 無定面付 详情dd | 历史:无\n"
                    "[ID=2] [发言数=880] 昵称=老王 | 内容=今天在汉口那边修的车，体验还行，就是贵了点 | 历史:无\n"
                    "[ID=3] [发言数=12] 昵称=路人 | 内容=约课吗 有空 | 历史:无"
                ),
            },
        ]

        async with DeepSeekClient(api_key=key) as client:
            for i in range(2):
                result, usage = await client.judge(messages, user_id="chat_selftest")
                print(f"\n--- 第 {i+1} 次调用 ---")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print(f"用量: prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
                      f"cache_hit={usage.cache_hit_tokens} cache_miss={usage.cache_miss_tokens} "
                      f"reasoning={usage.reasoning_tokens}")
                print(f"缓存命中率: {usage.cache_hit_rate:.1%}")
                print(f"本次成本（高峰价）: {estimate_cost_yuan(usage, peak=True):.6f} 元")

            print(f"\n=== 累计 ===")
            print(f"缓存命中率: {client.usage.cache_hit_rate:.1%}")
            print(f"总成本（高峰价）: {estimate_cost_yuan(client.usage, peak=True):.6f} 元")

    asyncio.run(_selftest())
