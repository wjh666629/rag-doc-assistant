"""Day 5 B2：检索指标（自己实现，不调现成库）。

指标定义：
- **Hit@k（命中率）**：标准答案来源是否出现在检索结果前 k 条中。
  公式：Hit@k = (命中样本数) / (参与统计的样本数)，单条命中记 1，否则 0。
  直觉：「找没找到」，不管排第几。

- **MRR（Mean Reciprocal Rank，平均倒数排名）**：对每条样本取第一个正确来源的排名 r，
  记 1/r（没命中记 0），全体取平均。
  公式：MRR = (1/N) * Σ 1/rank_i
  直觉：「排得前不前」。Hit@k 相同的两个系统，MRR 能看出谁把好结果排得更靠前。

两者互补：Hit@5 衡量召回覆盖，MRR 衡量排序质量。
"""
from __future__ import annotations

from typing import Any


def hit_at_k(retrieved_sources: list[str], golden_sources: list[str], k: int) -> float:
    """单条样本的 Hit@k：golden 中任意一个出现在前 k 位则记 1.0，否则 0.0。

    边界：golden 为空（越界样本）→ 返回 0.0 且不参与分母（由上层过滤）；
          retrieved 为空 → 0.0；k > len(retrieved) 时按实际长度截断。
    """
    if not golden_sources or not retrieved_sources:
        return 0.0
    top = retrieved_sources[:k]
    return 1.0 if any(g in top for g in golden_sources) else 0.0


def reciprocal_rank(retrieved_sources: list[str], golden_sources: list[str]) -> float:
    """单条样本的倒数排名：第一个 golden 来源出现在第 r 位则记 1/r，未命中记 0.0。"""
    if not golden_sources or not retrieved_sources:
        return 0.0
    for rank, src in enumerate(retrieved_sources, start=1):
        if src in golden_sources:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(results: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """批量评估检索质量。

    Args:
        results: 每条形如 {"retrieved": [来源...], "golden": [来源...], "type": 类型}
        k: Hit@k 的 k

    Returns:
        {"hit@k": float, "mrr": float,
         "by_type": {类型: {"hit@k":..., "mrr":..., "count":...}},
         "n": 参与统计的样本数,
         "out_of_scope": {"count": int, "avg_top1_score": float}}

    说明：越界样本（type="越界" 或 golden 为空）不计入 hit/mrr 分母——
    它们没有"正确来源"，只能用拒答率/检索分数分布来评估。
    """
    scored: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, Any]] = {}
    oos_count = 0
    oos_scores: list[float] = []

    for r in results:
        retrieved = list(r.get("retrieved", []))
        golden = list(r.get("golden", []))
        qtype = r.get("type", "未知")

        # 越界样本：单独统计，不进分母
        if qtype == "越界" or not golden:
            oos_count += 1
            if r.get("top1_score") is not None:
                oos_scores.append(float(r["top1_score"]))
            continue

        hit = hit_at_k(retrieved, golden, k)
        rr = reciprocal_rank(retrieved, golden)
        scored.append({"hit": hit, "rr": rr, "type": qtype})

        bucket = by_type.setdefault(qtype, {"hit@k": 0.0, "mrr": 0.0, "count": 0})
        bucket["hit@k"] += hit
        bucket["mrr"] += rr
        bucket["count"] += 1

    n = len(scored)
    summary: dict[str, Any] = {
        "hit@k": round(sum(h["hit"] for h in scored) / n, 4) if n else 0.0,
        "mrr": round(sum(h["rr"] for h in scored) / n, 4) if n else 0.0,
        "by_type": {},
        "n": n,
        "out_of_scope": {
            "count": oos_count,
            "avg_top1_score": round(sum(oos_scores) / len(oos_scores), 4) if oos_scores else 0.0,
        },
    }
    for qtype, bucket in by_type.items():
        c = bucket["count"]
        summary["by_type"][qtype] = {
            "hit@k": round(bucket["hit@k"] / c, 4),
            "mrr": round(bucket["mrr"] / c, 4),
            "count": c,
        }
    return summary
