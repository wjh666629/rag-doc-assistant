"""Day 5 B2：检索指标单元测试（覆盖边界 case）。"""
from __future__ import annotations

from eval.metrics import evaluate_retrieval, hit_at_k, reciprocal_rank


def test_hit_all_and_none():
    assert hit_at_k(["a.md", "b.md"], ["a.md"], 5) == 1.0
    assert hit_at_k(["a.md", "b.md"], ["c.md"], 5) == 0.0


def test_hit_partial_and_k_boundary():
    # 命中在第 2 位，k=1 时不算命中，k=2 时算命中
    assert hit_at_k(["x.md", "a.md"], ["a.md"], 1) == 0.0
    assert hit_at_k(["x.md", "a.md"], ["a.md"], 2) == 1.0


def test_hit_empty_inputs():
    assert hit_at_k([], ["a.md"], 5) == 0.0          # retrieved 为空
    assert hit_at_k(["a.md"], [], 5) == 0.0           # golden 为空（越界样本）
    assert hit_at_k(["a.md"], ["a.md"], 0) == 0.0     # k=0


def test_reciprocal_rank_positions():
    assert reciprocal_rank(["a.md", "b.md"], ["a.md"]) == 1.0     # 第 1 位
    assert reciprocal_rank(["x.md", "a.md"], ["a.md"]) == 0.5      # 第 2 位
    assert abs(reciprocal_rank(["x", "y", "z", "p", "a.md"], ["a.md"]) - 0.2) < 1e-9  # 第 5 位
    assert reciprocal_rank(["x.md"], ["a.md"]) == 0.0              # 未命中


def test_reciprocal_rank_beyond_k_still_counts():
    """超出 k 不计 hit，但 rr 仍按其真实位置计算（本例中第 3 位）。"""
    assert hit_at_k(["x", "y", "a.md"], ["a.md"], 2) == 0.0
    assert abs(reciprocal_rank(["x", "y", "a.md"], ["a.md"]) - 1 / 3) < 1e-9


def test_evaluate_retrieval_mixed():
    results = [
        {"retrieved": ["a.md", "b.md"], "golden": ["a.md"], "type": "简单事实"},   # hit, rr=1
        {"retrieved": ["b.md", "a.md"], "golden": ["a.md"], "type": "简单事实"},   # hit, rr=0.5
        {"retrieved": ["c.md"], "golden": ["a.md"], "type": "简单事实"},           # miss, rr=0
        {"retrieved": ["d.md"], "golden": ["d.md"], "type": "综合"},               # hit, rr=1
    ]
    out = evaluate_retrieval(results, k=5)
    assert out["n"] == 4
    assert out["hit@k"] == 0.75
    assert abs(out["mrr"] - (1 + 0.5 + 0 + 1) / 4) < 1e-9
    assert out["by_type"]["简单事实"]["count"] == 3
    assert out["by_type"]["综合"]["hit@k"] == 1.0


def test_out_of_scope_excluded_from_denominator():
    """越界样本不进 hit/mrr 分母，只统计数量与平均分。"""
    results = [
        {"retrieved": ["a.md"], "golden": ["a.md"], "type": "简单事实", "top1_score": 0.8},
        {"retrieved": ["z.md"], "golden": [], "type": "越界", "top1_score": 0.35},
    ]
    out = evaluate_retrieval(results, k=5)
    assert out["n"] == 1
    assert out["hit@k"] == 1.0
    assert out["out_of_scope"]["count"] == 1
    assert out["out_of_scope"]["avg_top1_score"] == 0.35
