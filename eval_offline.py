"""离线评测。

用线上真实样本跑完整检测流水线（L0 规则 + LLM 主判），测量：

1. **L0 命中率** —— 多少广告在进 LLM 之前就被拦掉（延迟与成本的双重节省）
2. **判定准确率** —— 严格档下 high 的精确率，这是误删风险的直接度量
3. **缓存命中率** —— 成本模型 §3.3 的基石假设
4. **单条成本** —— 外推到生产规模

负例是弱标签（V1 漏删的广告会混在"未删除"消息里），因此精确率是**下界**，
召回率是**上界**。正例是强标签（确实被 V1 删除过）。

用法：
    python eval_offline.py --samples eval_samples.jsonl --batch-size 50 --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepseek import DeepSeekClient, DeepSeekError, Usage, estimate_cost_yuan
from detector_rules import check_local, trust_marker
from prompts import BATCH_INSTRUCTION, GLOBAL_PROMPT, build_group_profile

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval")


def build_batch_messages(rows: list[dict], chat_title: str) -> list[dict]:
    """按生产格式构造请求。前缀稳定，批次在最后一条。

    信任标记来自样本自带的字段（导出时按生产规则算好），
    这样评测才能验证"信任信号到底有没有用"。
    """
    lines = []
    for r in rows:
        text = (r["text"] or "").replace("\n", " ")
        trust = r.get("trust", "normal")
        marker = trust_marker(trust)
        prefix = f"[ID={r['msg_id']}]"
        if marker:
            prefix += f" {marker}"
        lines.append(f"{prefix} 昵称=群友 | 内容={text} | 历史:无")

    return [
        {"role": "system", "content": GLOBAL_PROMPT},
        {"role": "system", "content": build_group_profile(
            title=chat_title, chat_nature="adult_service",
            now="2026-10-10", strictness="strict",
        )},
        {"role": "user", "content": BATCH_INSTRUCTION.format(
            count=len(rows), messages="\n".join(lines),
        )},
    ]


async def run_once(
    client: DeepSeekClient,
    rows: list[dict],
    batch_size: int,
    chat_title: str,
) -> dict:
    """跑一轮。返回逐样本判定结果与累计用量。"""
    predictions: dict[int, str] = {}
    l0_hits: dict[int, str] = {}
    usage_total = Usage()
    errors = 0

    # --- L0 规则先过一遍，命中的不进 LLM ---
    llm_rows = []
    for r in rows:
        v = check_local(r["text"], user_id=0, chat_id=r["chat_id"], blacklist_ids=set())
        if v.hit:
            l0_hits[r["msg_id"]] = v.reason
        else:
            llm_rows.append(r)

    # --- 分批送 LLM ---
    batches = [llm_rows[i:i + batch_size] for i in range(0, len(llm_rows), batch_size)]
    for idx, batch in enumerate(batches, 1):
        messages = build_batch_messages(batch, chat_title)
        try:
            parsed, usage = await client.judge(messages, user_id="chat_eval")
            usage_total.add(usage)
        except DeepSeekError as exc:
            errors += 1
            logger.error("批次 %d/%d 失败: %s", idx, len(batches), exc)
            continue

        for v in parsed.get("verdicts", []):
            try:
                predictions[int(v["id"])] = float(v.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue

        if idx % 4 == 0 or idx == len(batches):
            print(f"    批次 {idx}/{len(batches)} | "
                  f"累计缓存命中率 {usage_total.cache_hit_rate:.1%} | "
                  f"累计成本 {estimate_cost_yuan(usage_total, peak=True):.4f} 元")

    # --- 补批：截断或被丢弃的判定，用小批次重试 ---
    missing = [r for r in llm_rows if r["msg_id"] not in predictions]
    if missing:
        print(f"    补批：{len(missing)} 条缺少判定，重试中…")
        retry_batches = [missing[i:i + 10] for i in range(0, len(missing), 10)]
        for batch in retry_batches:
            try:
                parsed, usage = await client.judge(
                    build_batch_messages(batch, chat_title), user_id="chat_eval"
                )
                usage_total.add(usage)
                for v in parsed.get("verdicts", []):
                    try:
                        predictions[int(v["id"])] = float(v.get("confidence", 0.0))
                    except (KeyError, TypeError, ValueError):
                        continue
            except DeepSeekError as exc:
                errors += 1
                logger.error("补批失败: %s", exc)

    return {
        "predictions": predictions,
        "l0_hits": l0_hits,
        "usage": usage_total,
        "errors": errors,
        "batches": len(batches),
        "llm_count": len(llm_rows),
        "unjudged": sum(1 for r in llm_rows if r["msg_id"] not in predictions),
    }


def score(rows: list[dict], result: dict, threshold: float) -> dict:
    """按置信度阈值计算混淆矩阵。

    预测为广告 = L0 命中 或 置信度 >= threshold。
    """
    preds = result["predictions"]
    l0 = result["l0_hits"]

    tp = fp = fn = tn = 0
    fp_samples, fn_samples = [], []

    for r in rows:
        mid = r["msg_id"]
        actual_spam = r["label"] == "spam"

        if mid in l0:
            predicted_spam = True
        else:
            conf = preds.get(mid)
            if conf is None:
                continue  # 未判定（API 失败），不计入
            predicted_spam = conf >= threshold

        if actual_spam and predicted_spam:
            tp += 1
        elif actual_spam:
            fn += 1
            if len(fn_samples) < 12:
                fn_samples.append(r["text"][:70])
        elif predicted_spam:
            fp += 1
            if len(fp_samples) < 12:
                fp_samples.append(r["text"][:70])
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "fp_samples": fp_samples, "fn_samples": fn_samples,
        "l0_count": len(l0),
        "by_trust": _by_trust(rows, result, threshold),
    }


def _by_trust(rows: list[dict], result: dict, threshold: float) -> dict:
    """按信任档位分层统计，用于验证信任信号是否真的起作用。

    如果 [信任=高] 组的精确率明显更高，说明信号有效。
    如果 [信任=低] 组的召回率没有提升，说明信号没用上。
    """
    preds = result["predictions"]
    l0 = result["l0_hits"]
    groups: dict[str, dict] = {}

    for r in rows:
        mid = r["msg_id"]
        g = groups.setdefault(r.get("trust", "normal"),
                              {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0})
        g["n"] += 1
        actual_spam = r["label"] == "spam"
        if mid in l0:
            predicted_spam = True
        else:
            conf = preds.get(mid)
            if conf is None:
                continue
            predicted_spam = conf >= threshold

        if actual_spam and predicted_spam:
            g["tp"] += 1
        elif actual_spam:
            g["fn"] += 1
        elif predicted_spam:
            g["fp"] += 1
        else:
            g["tn"] += 1

    out = {}
    for k, g in groups.items():
        p = g["tp"] / (g["tp"] + g["fp"]) if (g["tp"] + g["fp"]) else 0.0
        rc = g["tp"] / (g["tp"] + g["fn"]) if (g["tp"] + g["fn"]) else 0.0
        out[k] = {**g, "precision": p, "recall": rc}
    return out


def sweep(rows: list[dict], result: dict, thresholds) -> list[dict]:
    """扫描阈值，返回各切点的指标。用于用数据选阈值而非凭感觉。"""
    return [score(rows, result, t) for t in thresholds]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="eval_samples.jsonl")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="限制样本数（0=全部）")
    ap.add_argument("--chat-title", default="测试群")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="选定阈值，默认 0.85（严格档）")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        print("需要 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    # 按时间排序，模拟真实到达顺序（缓存命中这才真实）
    rows.sort(key=lambda r: r.get("ts", ""))

    n_spam = sum(1 for r in rows if r["label"] == "spam")
    n_ham = len(rows) - n_spam
    print("=" * 78)
    print(f"  离线评测  样本 {len(rows)} 条（广告 {n_spam} / 正常 {n_ham}）")
    print(f"  批次大小 {args.batch_size}  运行 {args.runs} 轮  选定阈值 {args.threshold}")
    print("=" * 78)

    async with DeepSeekClient(api_key=key) as client:
        for run_idx in range(1, args.runs + 1):
            print(f"\n--- 第 {run_idx} 轮 ---")
            result = await run_once(client, rows, args.batch_size, args.chat_title)
            usage = result["usage"]

            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
            table = sweep(rows, result, thresholds)

            print(f"\n  阈值扫描（预测为广告 = 置信度 >= 阈值）:")
            print(f"  {'阈值':>6} {'精确率':>8} {'召回率':>8} {'F1':>7}  {'TP':>5} {'FP':>5} {'FN':>5}")
            print("  " + "-" * 58)
            for t in table:
                print(f"  {t['threshold']:>6.2f} {t['precision']:>7.1%} {t['recall']:>7.1%} "
                      f"{t['f1']:>7.3f}  {t['tp']:>5} {t['fp']:>5} {t['fn']:>5}")

            # 选定阈值：默认严格档 0.85
            chosen = args.threshold
            sc = score(rows, result, chosen)

            l0_n = sc["l0_count"]
            print(f"\n  === 选定阈值 {chosen} ===")
            print(f"  L0 规则拦截      : {l0_n} 条（占 {l0_n/len(rows):.1%}，未进 LLM）")
            err_note = f"，失败 {result['errors']} 批" if result["errors"] else ""
            unj = result.get("unjudged", 0)
            if unj:
                err_note += f"，最终未判定 {unj} 条"
            print(f"  送 LLM 判定      : {result['llm_count']} 条 / {result['batches']} 批{err_note}")
            print(f"  混淆矩阵         : TP={sc['tp']} FP={sc['fp']} FN={sc['fn']} TN={sc['tn']}")
            print(f"  精确率(误删风险) : {sc['precision']:.1%}")
            print(f"  召回率(漏判)     : {sc['recall']:.1%}")
            print(f"  F1               : {sc['f1']:.3f}")
            print(f"  缓存命中率       : {usage.cache_hit_rate:.1%} "
                  f"(hit={usage.cache_hit_tokens} miss={usage.cache_miss_tokens})")
            print(f"  reasoning tokens : {usage.reasoning_tokens} "
                  f"{'✅ thinking 已关闭' if usage.reasoning_tokens == 0 else '❌ thinking 未关闭！'}")
            print(f"  空响应次数       : {usage.empty_responses}")
            cost = estimate_cost_yuan(usage, peak=True)
            print(f"  本轮成本(高峰价) : {cost:.4f} 元")
            per_msg = cost / len(rows) if rows else 0
            print(f"  单条成本         : {per_msg:.6f} 元")
            print(f"  外推 1500 条/天  : {per_msg*1500:.3f} 元/天 ≈ {per_msg*1500*30:.2f} 元/月/群")

            print(f"\n  按信任档位分层（验证信任信号是否有效）:")
            print(f"  {'档位':10s} {'样本':>5} {'精确率':>8} {'召回率':>8}")
            print("  " + "-" * 36)
            for k in ("trusted", "normal", "new", "repeat"):
                g = sc["by_trust"].get(k)
                if not g:
                    continue
                print(f"  {k:10s} {g['n']:>5} {g['precision']:>7.1%} {g['recall']:>7.1%}")

            if sc["fp_samples"]:
                print(f"\n  ⚠️  误删样本（判 high 实为正常，需人工复核）:")
                for s in sc["fp_samples"]:
                    print(f"      {s!r}")
            if sc["fn_samples"]:
                print(f"\n  漏判样本（实为广告但未判 high）:")
                for s in sc["fn_samples"]:
                    print(f"      {s!r}")

            if run_idx == args.runs:
                report = {
                    "samples": len(rows), "spam": n_spam, "ham": n_ham,
                    "batch_size": args.batch_size, "threshold": chosen,
                    "threshold_sweep": [
                        {k: v for k, v in t.items() if k not in ("fp_samples", "fn_samples")}
                        for t in table
                    ],
                    "l0_hits": result["l0_hits"], "l0_count": sc["l0_count"],
                    "llm_count": result["llm_count"],
                    "batches": result["batches"], "errors": result["errors"],
                    "unjudged": result.get("unjudged", 0),
                    **{k: v for k, v in sc.items() if k not in ("fp_samples", "fn_samples")},
                    "cache_hit_rate": usage.cache_hit_rate,
                    "cache_hit_tokens": usage.cache_hit_tokens,
                    "cache_miss_tokens": usage.cache_miss_tokens,
                    "output_tokens": usage.completion_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                    "cost_yuan_peak": cost,
                    "cost_per_msg_peak": per_msg,
                    "extrapolated_daily_1500": per_msg * 1500,
                    "fp_samples": sc["fp_samples"], "fn_samples": sc["fn_samples"],
                }
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                print(f"\n  报告已保存: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
