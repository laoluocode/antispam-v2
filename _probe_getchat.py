"""测试用 Bot API 解析频道信息（比对网页预览是否更有效）。"""
import asyncio
import os
import sys

sys.path.insert(0, "/root/ribaobot/antispambot")

from telegram import Bot


async def main():
    bot = Bot(os.environ["BOT_TOKEN"])
    async with bot:
        for target in ("@wdgx", "@telegram"):
            print("--- %s ---" % target)
            try:
                c = await bot.get_chat(target)
                print("  type       : %s" % c.type)
                print("  id         : %s" % c.id)
                print("  title      : %r" % c.title)
                uname = ("@" + c.username) if c.username else "(无)"
                print("  username   : %s" % uname)
                desc = (c.description or "")[:200]
                print("  description: %r" % desc)
                print("  linked_chat: %s" % c.linked_chat_id)
            except Exception as exc:  # noqa: BLE001
                print("  FAIL %s: %s" % (type(exc).__name__, exc))
            print()


asyncio.run(main())
