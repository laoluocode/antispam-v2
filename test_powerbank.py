"""用线上真实的"漏掉"样本测当前 prompt 的判定能力。"""
import asyncio
import os
import sys

sys.path.insert(0, "/root/ribaobot/antispambot")

from deepseek import DeepSeekClient
from prompts import BATCH_INSTRUCTION, GLOBAL_PROMPT, build_group_profile

CASES = [
    ("充电宝投放执行力强的来", "漏掉的（msg 99732）", "低"),
    ("充电宝投放，日入过w的新项目", "同类（已删）", "低"),
    ("来几个充电宝投放手400一个", "同类（已删）", "低"),
    ("充电宝投放一个月提宝马", "同类（已删）", "低"),
    ("抓住这波风口项目了", "同类（已删）", "低"),
    ("小白也能翻身的项目", "同类（已删）", "低"),
    ("今天在汉口修的车体验还行", "正常对照", "高"),
    ("有没有兄弟推荐一下武昌的", "正常对照", "高"),
]


async def main():
    prof = build_group_profile(
        "南京苏州集团", "adult_service", "2026-10-10 14:45", "standard",
        normal_words=["导航", "签到", "中文", "汉化", "友群", "修车机器人",
                      "此消息包含广告"],
        blocked_words=[],
    )
    lines = []
    for i, (text, _note, trust) in enumerate(CASES):
        lines.append("[ID=%d] %s 昵称=某号 | 内容=%s"
                     % (i + 1, ("[信任=%s]" % trust) if trust != "低" else "[信任=低]",
                        text))
    msgs = [
        {"role": "system", "content": GLOBAL_PROMPT},
        {"role": "system", "content": prof},
        {"role": "user", "content": BATCH_INSTRUCTION.format(
            count=len(lines), messages="\n".join(lines))},
    ]
    async with DeepSeekClient(api_key=os.environ["DEEPSEEK_API_KEY"]) as client:
        parsed, _usage = await client.judge(msgs, user_id="chat_probe")

    scores = {int(v["id"]): float(v["confidence"])
              for v in parsed.get("verdicts", [])}
    print("  %8s  %5s  %s" % ("置信度", "阈值0.5", "样本"))
    print("  " + "-" * 70)
    for i, (text, note, _t) in enumerate(CASES, 1):
        s = scores.get(i, -1.0)
        mark = "删除" if s >= 0.5 else "放过"
        print("  %8.2f  %5s  %s" % (s, mark, text[:40]))
        print("  %8s  %5s  ^ %s" % ("", "", note))


asyncio.run(main())
