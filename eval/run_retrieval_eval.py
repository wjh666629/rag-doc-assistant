"""Day 5 B3：基线检索评估（只评检索层，不调生成，成本≈0）。

逐条用 golden_qa.json 的问题去检索 top N，算 Hit@5 / MRR，
输出报告 eval/reports/retrieval_baseline_{date}.md：
- 指标总表 + 分类型指标
- 越界组 vs 正常组 top1 分数对比
- 未命中清单（留"人工归因"空栏）

运行：python -m eval.run_retrieval_eval
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.retriever import get_vectorstore  # noqa: E402
from eval.metrics import evaluate_retrieval  # noqa: E402

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_qa.json"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


def run(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """跑一次检索评估并返回结果 dict。"""
    config = config or load_config()
    eval_cfg = config.get("eval", {})
    top_k = int(eval_cfg.get("top_k_eval", 8))
    hit_k = int(eval_cfg.get("hit_k", 5))

    items = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = [it for it in items if it.get("verified", False)]
    print(f"评估集条数（verified）: {len(items)}")

    store = get_vectorstore(config)
    results: list[dict[str, Any]] = []
    per_item: list[dict[str, Any]] = []

    for it in items:
        scored = store.similarity_search_with_relevance_scores(it["question"], k=top_k)
        retrieved = [d.metadata.get("source", "") for d, _ in scored]
        top1_score = scored[0][1] if scored else 0.0
        golden = list(it.get("source", []))

        results.append({"retrieved": retrieved, "golden": golden,
                        "type": it.get("type", "未知"), "top1_score": top1_score})
        per_item.append({
            "id": it["id"], "question": it["question"], "type": it.get("type"),
            "golden": golden, "top1_source": retrieved[0] if retrieved else "",
            "top1_score": round(top1_score, 4),
            "hit": any(g in retrieved[:hit_k] for g in golden) if golden else None,
        })

    summary = evaluate_retrieval(results, k=hit_k)

    # 越界组 vs 正常组的 top1 分数对比
    oos_scores = [r["top1_score"] for r in results if r["type"] == "越界"]
    normal_scores = [r["top1_score"] for r in results if r["type"] != "越界"]
    summary["score_compare"] = {
        "越界组平均 top1 分数": round(statistics.mean(oos_scores), 4) if oos_scores else 0.0,
        "正常组平均 top1 分数": round(statistics.mean(normal_scores), 4) if normal_scores else 0.0,
    }
    return {"summary": summary, "per_item": per_item}


def write_report(result: dict[str, Any]) -> Path:
    """输出 baseline 报告 markdown。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"retrieval_baseline_{date.today().isoformat()}.md"
    s = result["summary"]
    cmp_ = s.get("score_compare", {})

    lines = [
        f"# 检索基线评估（{date.today().isoformat()}）",
        "",
        f"- 样本数（进分母）: {s['n']}",
        f"- 越界样本数: {s['out_of_scope']['count']}",
        "",
        "## 指标总表",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| Hit@5 | {s['hit@k']} |",
        f"| MRR | {s['mrr']} |",
        f"| 越界组平均 top1 分数 | {cmp_.get('越界组平均 top1 分数', 0)} |",
        f"| 正常组平均 top1 分数 | {cmp_.get('正常组平均 top1 分数', 0)} |",
        "",
        "## 分类型指标",
        "",
        "| 类型 | 样本数 | Hit@5 | MRR |",
        "|---|---|---|---|",
    ]
    for t, v in s["by_type"].items():
        lines.append(f"| {t} | {v['count']} | {v['hit@k']} | {v['mrr']} |")

    lines.append("\n## 未命中清单（人工归因待填）\n")
    lines.append("| 问题 | golden 来源 | 实际 top1 | top1 分数 | 人工归因 |")
    lines.append("|---|---|---|---|---|")
    for it in result["per_item"]:
        if it["hit"] is False or it["hit"] is None and it["type"] != "越界":
            lines.append(f"| {it['question'][:50]} | {','.join(it['golden']) or '-'} | "
                         f"{it['top1_source']} | {it['top1_score']} |  |")

    lines.append("\n## 结论\n（人工补充）\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    config = load_config()
    result = run(config)
    path = write_report(result)
    print(f"\nHit@5 = {result['summary']['hit@k']}  MRR = {result['summary']['mrr']}  "
          f"n = {result['summary']['n']}")
    print(f"报告: {path}")


if __name__ == "__main__":
    main()
