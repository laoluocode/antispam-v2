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

# 全角数字 -> 半角。
#
# 这是一个**现成的绕过口子**：全角数字（U+FF10-U+FF19）的 \w 属性为真，
# 所以不会被 _NOISE_PATTERN 删掉，也没有 ASCII 映射，导致
#   "我电话13812345678"   -> 命中手机号规则
#   "我电话１３８１２３４５６７８" -> 放过（绕过）
#
# 实测决策依据（全库 spam=1766 / ham=3040）：
#   含全角数字的广告 0 条，含全角数字的正常消息 0 条
#   -> 双方都从未使用，映射零误伤风险。
# 只映射数字、不动全角标点：全角标点（？，！）是中文用户的正常输入习惯，
# 正常消息里占比 13%，改它们会影响其他模式的匹配。
_FULLWIDTH_DIGITS = {
    "\uff10": "0", "\uff11": "1", "\uff12": "2", "\uff13": "3",
    "\uff14": "4", "\uff15": "5", "\uff16": "6", "\uff17": "7",
    "\uff18": "8", "\uff19": "9",
}

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

# 汉字之间夹的单字母缓冲：叚i钞 -> 叚钞。
# 零宽断言（不消费左右字符）是关键，理由见 normalize() 的注释。
_BUFFER_PATTERN = re.compile(r"(?<=[\u4e00-\u9fff])[iIl1](?=[\u4e00-\u9fff])")


def strip_invisibles(text: str) -> str:
    """去掉零宽字符与变体选择符，保留其余字面。

    用于日志展示与 few-shot 样本 —— 广告号的混淆手段本身就是特征，
    存原文才能在事后复核时看出它用了什么手法。
    """
    return _INVISIBLE_PATTERN.sub("", text or "")


def _normalize_pass(text: str) -> str:
    """一趟归一化。不要直接调用——用 normalize()。"""
    t = strip_invisibles(text or "").lower()

    # 全角数字先归一到半角，否则手机号/金额等 ASCII 模式会被绕过
    if any(d in t for d in _FULLWIDTH_DIGITS):
        for fw, half in _FULLWIDTH_DIGITS.items():
            t = t.replace(fw, half)

    for emoji, char in _EMOJI_MAP.items():
        t = t.replace(emoji, char)

    for src, dst in _REPLACEMENT_MAP.items():
        t = t.replace(src.lower(), dst)

    # 去掉汉字之间夹的单字母/单数字缓冲：叚i钞 -> 叚钞。
    # 用**零宽断言**而不是捕获组：捕获组会消费左右字符，导致
    # '洗m1挣' 里的 '1' 因为左边的 'm' 被消费掉而漏掉。
    t = _BUFFER_PATTERN.sub("", t)

    # 去掉汉字之间的空格，使跨空格关键词可以匹配：上 门 -> 上门
    t = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", t)

    return _NOISE_PATTERN.sub("", t)


# 归一化最多迭代几轮。实测 2 轮足够收敛（绝大多数 1 轮）。
_NORMALIZE_MAX_PASSES = 3


def normalize(text: str) -> str:
    """完整归一化，**保证幂等**。

    幂等很重要：广告号靠"多插一个缓冲字符"制造变体，
    如果 normalize('来洗mi 挣一万') 与 normalize('来洗m1挣一万') 得到不同结果，
    同一条广告就有了两种指纹，规则、判定记忆、预览指纹都会各算一份。

    单趟变换做不到幂等——替换表把 i 变成 1 之后，
    原先"字母夹在汉字间"的判断条件会变（'mi 挣' 变成 'm1 挣'），
    需要再走一趟才能收敛。所以这里迭代到不动点，最多 _NORMALIZE_MAX_PASSES 轮。
    """
    cur = _normalize_pass(text or "")
    for _ in range(_NORMALIZE_MAX_PASSES - 1):
        nxt = _normalize_pass(cur)
        if nxt == cur:
            break
        cur = nxt
    return cur


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
