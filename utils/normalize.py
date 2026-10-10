"""文本归一化。

从 V1 `ai_summarizer.py:_normalize_text` 迁移，是 V1 调了几百小时才攒出来的资产：
广告号用不可见字符（U+E0000 段变体选择符）、emoji、谐音字把关键词拆开绕过过滤，
这里把它们还原成可比对的形态。

两个入口：
- normalize(text)      : 完整归一化，用于模糊匹配谐音变体
- strip_invisibles(t)  : 只去掉零宽/变体选择符，保留原有字面，用于日志展示与 few-shot

注意：normalize 会做有损替换（如 l/o -> 1/0），不要用它回写数据库原文。
"""

import re

# 零宽字符与变体选择符：广告号用来隔断关键词的主力手段。
# U+E0000-U+E01EF 是"变体选择符补充"段，Telegram 会原样显示但不占视觉宽度。
_INVISIBLE_PATTERN = re.compile(
    "["
    "\u00ad"          # soft hyphen
    "\u061c"          # arabic letter mark
    "\u180e"          # mongolian vowel separator
    "\u200b-\u200f"   # zero width space/joiner/non-joiner + LRM/RLM
    "\u202a-\u202e"   # bidi embedding/override
    "\u2060-\u2064"   # word joiner 等
    "\u2066-\u2069"   # bidi isolates
    "\ufeff"          # BOM
    "\ufe00-\ufe0f"   # variation selectors
    "\U000e0000-\U000e01ef"  # variation selectors supplement（广告号最爱）
    "]"
)

# emoji 变体映射：广告号用这些 emoji 代替汉字
_EMOJI_MAP = {
    "\U0001f251": "可",  # 🉑
    "\U0001f4af": "百",  # 💯
    "\u2705": "是",      # ✅
}

# 谐音/形近字替换表
_REPLACEMENT_MAP = {
    # 门
    "門": "门", "闷": "门", "闩": "门", "菛": "门", "们": "门", "扪": "门", "閅": "门",
    # 上
    "仩": "上", "丄": "上",
    # 可
    "坷": "可", "苛": "可", "砢": "可", "柯": "可", "岢": "可",
    # 出
    "岀": "出", "粗": "出",
    # 订 / 免 / 无 / 尚 / 有
    "訂": "订", "锭": "定",
    "偭": "免", "缅": "免", "湎": "免", "面": "面",
    "芜": "无",
    "尙": "尚",
    "侑": "有",
    # 约
    "掖": "约", "悦": "约", "棵": "课", "颗": "课",
    # 价
    "介": "价",
    # 字母缓冲字符
    "i": "1", "l": "1", "o": "0",
    # 灰产隐晦用语
    "恢": "灰",   # 恢产 -> 灰产
    "煮": "主",   # 煮业 -> 主业
    "橡": "项",   # 橡木 -> 项目
    "幕": "目",   # 橡幕 -> 项目
    "薇": "微",   # 薇信 -> 微信
    "威": "微",   # 威信 -> 微信
    # 繁体
    "個人": "个人", "無": "无", "錢": "钱", "賺": "赚", "費": "费",
}

# 噪声：空白与所有非单词字符（含 emoji、标点、符号）
_NOISE_PATTERN = re.compile(r"[\s\W_]+", re.UNICODE)


def strip_invisibles(text: str) -> str:
    """去掉零宽字符与变体选择符，保留其余字面。

    用于日志展示与 few-shot 样本 —— 广告号的混淆手段本身就是特征，
    存原文才能在事后复核时看出它用了什么手法。
    """
    return _INVISIBLE_PATTERN.sub("", text or "")


def normalize(text: str) -> str:
    """完整归一化：去不可见字符 -> 小写 -> emoji/谐音替换 -> 去字母缓冲 -> 去噪声。

    结果用于模糊匹配。示例：
        "叚i钞"    -> "假钞" 前体（"叚" 由调用方另行映射）
        "上 门"    -> "上门"
        "無定 面付" -> "无定面付"
        "🉑丄 門"  -> "可上门"
    """
    t = strip_invisibles(text or "").lower()

    for emoji, char in _EMOJI_MAP.items():
        t = t.replace(emoji, char)

    for src, dst in _REPLACEMENT_MAP.items():
        t = t.replace(src.lower(), dst)

    # 去掉汉字之间夹的单字母/单数字缓冲：叚i钞 -> 叚钞
    t = re.sub(
        r"([\u4e00-\u9fff]?)([iIl1])([\u4e00-\u9fff]?)",
        lambda m: m.group(1) + m.group(3) if (m.group(1) or m.group(3)) else m.group(2),
        t,
    )

    # 去掉汉字之间的空格，使跨空格关键词可以匹配：上 门 -> 上门
    t = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", t)

    return _NOISE_PATTERN.sub("", t)


def invisible_ratio(text: str) -> float:
    """不可见字符占比。

    这是 V1 没有显式使用的信号：正常群友几乎不会打出变体选择符，
    而广告号为了绕过滤会把半条消息塞满不可见字符。
    占比高本身就是强特征，可作为送 AI 判定的前置加权项。
    """
    t = text or ""
    if not t:
        return 0.0
    return len(_INVISIBLE_PATTERN.findall(t)) / len(t)


if __name__ == "__main__":
    cases = [
        ("\u5e94\u7528\u573a\u666f", "普通文本"),
        ("\u80fd\ufe0f\u200d\U000e0068\u6536\U0001f9e7\u5c31\u884c\uff0c\u66f0\U000e00775K", "广告号混淆"),
        ("\u4e2a\u4eba\u5916\u51fa \u7121\u5b9a\u9762\u4ed8", "繁体变体"),
        ("\u53ef\u4e0a\u95e8", "emoji 变体"),
        ("\u4e0a \u95e8", "空格拆分"),
        ("\u53c1i\u94de", "字母缓冲"),
        ("\u4eca\u5929\u5728\u6c49\u53e3\u4fee\u7684\u8f66\uff0c\u4f53\u9a8c\u8fd8\u884c", "正常聊天"),
    ]
    for raw, label in cases:
        print(f"[{label}]")
        print(f"  原文      : {raw!r}")
        print(f"  去不可见  : {strip_invisibles(raw)!r}")
        print(f"  归一化    : {normalize(raw)!r}")
        print(f"  不可见占比: {invisible_ratio(raw):.1%}")
        print()
