"""测这类"招募探花"广告的判定能力。"""
import asyncio
import os
import sys

sys.path.insert(0, "/root/ribaobot/antispambot")

from deepseek import DeepSeekClient
from prompts import BATCH_INSTRUCTION, GLOBAL_PROMPT, build_group_profile

SAMPLES = [
    ("招募探花 提供设备 收探花视频9000一单", "用户给的漏判样本"),
    ("招募探花 提供设备 收探花视频9000一单", "同上，测一致性"),
    ("招嫖客 提供场地 收视频5000一单", "同结构变体"),
    ("招募主播 提供设备 收视频3000一单", "同结构变体（可能合法招聘）"),
    ("收视频 9000一单", "简写变体"),
    ("招募探花", "只留核心词"),
    ("今天在汉口修的车体验还行", "正常对照"),
    ("有没有兄弟推荐一下武昌的", "正常对照"),
]


async def main():
    prof = build_group_profile(
        "武汉集团", "adult_service", "2026-10-10 14:20", "standard",
        normal_words=["导航"], blocked_words=[],
    )
    lines = []
    for i, (text, _) in enumerate(SAMPLES):
        lines.append("[ID=%d] [信任=低] 昵称=某号 | 内容=%s" % (i + 1, text))
    msgs = [
        {"role": "system", "content": GLOBAL_PROMPT},
        {"role": "system", "content": prof},
        {"role": "user", "content": BATCH_INSTRUCTION.format(
            count=len(lines), messages="\n".join(lines))},
    ]
    async with DeepSeekClient(api_key=os.environ["DEEPSEEK_API_KEY"]) as client:
        parsed, usage = await client.judge(msgs, user_id="chat_test")

    scores = {int(v["id"]): float(v["confidence"])
              for v in parsed.get("verdicts", [])}
    print("  %7s  %4s  样本" % ("置信度", "判定"))
    print("  " + "-" * 74)
    for i, (text, note) in enumerate(SAMPLES, 1):
        s = scores.get(i, -1.0)
        mark = "删除" if s >= 0.5 else "放过"
        print("  %7.2f  %4s  %s" % (s, mark, text[:44]))
        print("  %7s  %4s  ^ %s" % ("", "", note))
    print()
    print("  一致性检查（前两条同文本）: %.2f vs %.2f"
          % (scores.get(1, -1), scores.get(2, -1)))
    print("  usage: hit=%d miss=%d out=%d"
          % (usage.cache_hit_tokens, usage.cache_miss_tokens,
             usage.completion_tokens))


asyncio.run(main())
