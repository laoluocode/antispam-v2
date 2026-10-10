"""链接预览解析。

## 解决什么问题

实测漏判案例（2026-10-10）：广告号发**单个字母**，广告正文藏在 Telegram 的
**链接预览卡片**里。客户端显示：

    Jfdueyp Kvdzehcm
    📢 wdgx
    招募探花 提供设备 收探花视频9000一单
    n

而 bot 收到的 `message.text` 只有 `n`。预览卡片的内容 Telegram **不通过 API 下发**。

但卡片**指向的 URL 是下发的**（两个来源）：
  · `Message.link_preview_options.url`
  · `Message.entities` 里 type 为 `url` / `text_link` 的实体

拿到 URL 后自己把页面的 `og:` 元信息抓回来，就能补上广告正文。

## 为什么这是决定性的

靠"单字母刷屏"这类**行为特征**容易被绕过（换成 3 字母词、中文单字即可）。
而抓预览内容是**从信息源头补齐**，广告号改不掉——它必须让群友看到广告，
就必须把广告放在能被抓取的地方（否则群友也看不到）。

## 成本控制

只在可疑时抓取（短文本 / 低信任 / 带预览），不对每条消息发外部请求。
带超时与体积上限，失败不影响主判定流程。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass


logger = logging.getLogger(__name__)

# 只读页面头部即可拿到 og: 标签，不必下载整个页面
MAX_BYTES = 200_000
FETCH_TIMEOUT = 6.0

# og: / twitter: / <title> —— 按信息量排序
_META_PATTERNS = (
    re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']', re.I),
    re.compile(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:description["\']', re.I),
    re.compile(r'<meta[^>]+name=["\']twitter:title["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r"<title[^>]*>([^<]+)</title>", re.I),
)

# Telegram 链接的形态
_TME_PATTERN = re.compile(r"(?:https?://)?t\.me/(?:s/)?([A-Za-z0-9_]+)(?:/(\d+))?")
_TME_C_PATTERN = re.compile(r"(?:https?://)?t\.me/c/(\d+)(?:/(\d+))?")

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.I)


@dataclass
class Preview:
    """从预览卡片来源取回的内容。"""

    url: str = ""
    title: str = ""
    description: str = ""
    # Telegram 频道/群组链接解析出的用户名
    tg_username: str = ""
    error: str = ""

    @property
    def text(self) -> str:
        """拼成可判定的文本。"""
        parts = [p for p in (self.title, self.description) if p]
        return " ".join(parts).strip()

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error


def preview_fingerprint(msg) -> tuple[str, str]:
    """给消息的预览目标算一个指纹。返回 (fingerprint, url)。

    为什么用指纹而不是 URL 原文：广告号会在 URL 上加随机参数绕过比对
    （t.me/xxx?a=1 / t.me/xxx?a=2）。归一化掉查询参数与末尾斜杠后，
    同一个引流目标会归到同一个指纹。

    优先用实体里的 URL（更可靠），其次用预览选项的 URL。
    """
    urls = urls_from_message(msg)
    if not urls:
        return "", ""
    url = urls[0]
    norm = url.split("?")[0].split("#")[0].rstrip("/").lower()
    # t.me 链接统一成 t.me/username 形式（去掉可能存在的 /s/ 前缀与帖子号）
    m = _TME_PATTERN.search(norm)
    if m:
        norm = f"t.me/{m.group(1).lower()}"
    import hashlib
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16], url


def has_link_preview(msg) -> bool:
    """消息是否带链接预览。预览卡片的存在本身就是"文本之外还有内容"的信号。"""
    try:
        lpo = getattr(msg, "link_preview_options", None)
        if lpo is not None:
            if getattr(lpo, "is_disabled", False):
                return False
            if getattr(lpo, "url", None):
                return True
    except Exception:  # noqa: BLE001
        pass
    for field_name in ("entities", "caption_entities"):
        for ent in (getattr(msg, field_name, None) or []):
            if str(getattr(ent, "type", "")).lower() in ("url", "text_link"):
                return True
    return False


def extract_urls(text: str) -> list[str]:
    """从纯文本里抽 URL。"""
    return [u.rstrip(".,;)]}") for u in _URL_PATTERN.findall(text or "")]


def urls_from_message(msg) -> list[str]:
    """从 Telegram Message 抽取所有可能的预览来源 URL。

    取两个来源：预览选项、文本实体。去重保序。
    """
    urls: list[str] = []

    # 1) 预览卡片 URL
    try:
        lpo = getattr(msg, "link_preview_options", None)
        if lpo is not None and getattr(lpo, "url", None):
            urls.append(lpo.url)
    except Exception:  # noqa: BLE001
        pass

    # 2) 文本实体：url 与 text_link
    for field_name in ("entities", "caption_entities"):
        for ent in (getattr(msg, field_name, None) or []):
            etype = str(getattr(ent, "type", "")).lower()
            if etype == "text_link" and getattr(ent, "url", None):
                urls.append(ent.url)
            elif etype == "url":
                # url 实体的目标就在正文对应片段里
                body = getattr(msg, "text", None) or getattr(msg, "caption", None) or ""
                try:
                    frag = body[ent.offset:ent.offset + ent.length]
                    urls.extend(extract_urls(frag))
                except Exception:  # noqa: BLE001
                    pass

    # 3) 兜底：正文里裸链接
    urls.extend(extract_urls(getattr(msg, "text", None)
                             or getattr(msg, "caption", None) or ""))

    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def parse_telegram_link(url: str) -> tuple[str, str]:
    """解析 t.me 链接，返回 (用户名, 帖子ID)。非 t.me 返回空。"""
    m = _TME_PATTERN.search(url or "")
    if m:
        return m.group(1), (m.group(2) or "")
    m = _TME_C_PATTERN.search(url or "")
    if m:
        # t.me/c/<internal_id>/<msg_id>：私有频道，无法用用户名访问
        return "", (m.group(2) or "")
    return "", ""


def _extract_meta(html_text: str) -> tuple[str, str]:
    """从 HTML 里抽标题与描述。按模式顺序取第一个非空。"""
    title = ""
    desc = ""
    for pat in _META_PATTERNS:
        for m in pat.finditer(html_text):
            val = (m.group(1) or "").strip()
            if not val:
                continue
            if not title:
                title = val
            elif not desc and val != title:
                desc = val
                break
        if title and desc:
            break
    return title[:300], desc[:500]


async def fetch_preview(session, url: str, timeout: float = FETCH_TIMEOUT) -> Preview:
    """抓取 URL 的 og: 元信息。

    session 由调用方传入（复用 aiohttp.ClientSession）。
    任何失败都返回带 error 的 Preview，不抛异常——预览解析是增强信号，
    失败不应影响主判定流程。
    """
    pv = Preview(url=url)
    if not url:
        pv.error = "空 URL"
        return pv

    username, post_id = parse_telegram_link(url)
    if username:
        pv.tg_username = username

    try:
        async with session.get(
            url, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TelegramBot)"},
            allow_redirects=True,
        ) as resp:
            if resp.status >= 400:
                pv.error = f"HTTP {resp.status}"
                return pv
            raw = await resp.content.read(MAX_BYTES)
            charset = resp.charset or "utf-8"
            html_text = raw.decode(charset, errors="ignore")
    except Exception as exc:  # noqa: BLE001
        pv.error = f"{type(exc).__name__}: {exc}"[:120]
        return pv

    pv.title, pv.description = _extract_meta(html_text)
    if not pv.text:
        pv.error = "页面无 og: 元信息"
    return pv


async def resolve_preview_content(session, msg, max_urls: int = 2) -> Preview:
    """给定 Telegram Message，取回其预览卡片的内容（若可获得）。"""
    urls = urls_from_message(msg)
    if not urls:
        return Preview(error="消息不含 URL")
    for url in urls[:max_urls]:
        pv = await fetch_preview(session, url)
        if pv.ok:
            return pv
        logger.debug("预览解析失败 url=%s err=%s", url, pv.error)
    return Preview(url=urls[0], error="全部 URL 解析失败")


def should_resolve(text: str, trust: str = "normal", min_len: int = 15) -> bool:
    """是否值得为这条消息去抓预览。

    成本控制：只对"文本过短无法判定"的情况抓取。
    正常聊天的长消息不需要外部请求。
    """
    return len((text or "").strip()) < min_len


if __name__ == "__main__":
    import asyncio

    import aiohttp

    async def _demo():
        logging.basicConfig(level=logging.INFO)
        async with aiohttp.ClientSession() as session:
            for url in ("https://t.me/telegram", "https://example.com"):
                pv = await fetch_preview(session, url)
                print(f"  {url}")
                print(f"    ok={pv.ok} title={pv.title[:60]!r}")
                print(f"    desc={pv.description[:80]!r} err={pv.error}")

    # 纯函数自测（不需要网络）
    class FakeEnt:
        def __init__(self, t, off, ln, url=None):
            self.type, self.offset, self.length, self.url = t, off, ln, url

    class FakeMsg:
        text = "看看 https://t.me/wdgx/123 这个"
        caption = None
        caption_entities = None
        entities = [FakeEnt("url", 3, 22)]
        link_preview_options = type("L", (), {"url": "https://t.me/wdgx/123"})()

    print("=== 从消息抽 URL ===")
    print("  ", urls_from_message(FakeMsg()))
    print("=== 解析 t.me 链接 ===")
    for u in ("https://t.me/wdgx/123", "t.me/abc", "https://t.me/c/1234/56"):
        print(f"  {u} -> {parse_telegram_link(u)}")
    print("=== should_resolve ===")
    for t in ("t", "n", "招募探花 提供设备 收视频9000一单", "今天修车体验还行啊啊啊啊"):
        print(f"  {t[:20]!r:24s} -> {should_resolve(t)}")
    print()
    asyncio.run(_demo())
