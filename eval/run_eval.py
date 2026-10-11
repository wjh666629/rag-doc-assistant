"""Day 6 B2：一键完整评估（检索 + 生成 + 工程指标 + 对比历史）。

一条命令完成：
1. 加载 golden_qa.json（verified=true）
2. 逐条跑完整 CRAG graph，收集 generation / contexts / trace / 耗时
3. 检索层：Hit@5 / MRR / 分类型
4. 生成层：Ragas 三指标 + 越界拒答正确率
5. 工程层：平均/p95 延迟、缓存命中率
6. 与 eval/reports/ 下最近一份 report_*.json 对比算 Δ
7. 输出 report_{hash}.md（人读）+ report_{hash}.json（机器读）

运行：python -m eval.run_eval --note "baseline: chunk500/top5"
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from src.agent.graph import build_graph, run_question  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from eval.metrics import evaluate_retrieval  # noqa: E402
from eval.ragas_eval import run_ragas  # noqa: E402
from eval.run_retrieval_eval import run as run_retrieval  # noqa: E402

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_qa.json"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


def git_short_hash() -> str:
    """取当前 commit 短 hash；失败（非 git 仓库）则用时间戳。"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_golden(limit: int | None = None) -> list[dict[str, Any]]:
    """加载评估集；limit 时**分层抽样**（越界题必留）。

    坑：越界题在文件末尾，直接 items[:limit] 会把它们全切掉，
    导致拒答率恒为 0 —— 所以必须优先保留越界样本再补足其他类型。
    """
    items = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = [it for it in items if it.get("verified", False)]
    if not limit or limit >= len(items):
        return items
    oos = [it for it in items if it.get("type") == "越界"]
    others = [it for it in items if it.get("type") != "越界"]
    keep = oos + others[:max(0, limit - len(oos))]
    print(f"[提示] 分层抽样：保留全部 {len(oos)} 条越界题 + {len(keep) - len(oos)} 条其他类型")
    return keep


def run_graph_batch(graph, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐条跑 graph，返回每条的运行结果。"""
    records: list[dict[str, Any]] = []
    for i, it in enumerate(items, start=1):
        try:
            final = run_question(graph, it["question"])
        except Exception as exc:
            print(f"[警告] 第 {i} 条运行失败: {type(exc).__name__}: {exc}")
            records.append({"question": it["question"], "answer": "", "contexts": [],
                            "trace": [], "elapsed": 0.0, "refusal": False, "error": str(exc)})
            continue
        records.append({
            "question": it["question"],
            "answer": final.get("generation", ""),
            "contexts": [d.page_content for d in (final.get("documents") or [])],
            "sources": [d.metadata.get("source", "") for d in (final.get("documents") or [])],
            "trace": final.get("trace", []),
            "elapsed": final.get("elapsed", 0.0),
            "refusal": final.get("refusal", False),
            "type": it.get("type", ""),
            "golden": it.get("source", []),
            "ground_truth": it.get("ground_truth", ""),
        })
        if i % 10 == 0:
            print(f"  已跑 {i}/{len(items)} 条")
    return records


def latest_report_json() -> dict[str, Any] | None:
    """找最近一份 report_*.json 作为对比基线。"""
    files = sorted(REPORTS_DIR.glob("report_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def delta(new: float | None, old: float | None) -> str:
    """算 Δ 字符串（没有历史则 "—"）。"""
    if old is None or new is None:
        return "—"
    d = new - old
    return f"{d:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="一键完整评估（Day 6）")
    parser.add_argument("--note", default="", help="本次变更说明，写入报告")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（控成本）")
    parser.add_argument("--skip-ragas", action="store_true", help="跳过 Ragas（只评检索与工程）")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    items = load_golden(args.limit)
    print(f"评估集条数: {len(items)}")

    # 成本护栏
    max_samples = int(config.get("eval", {}).get("max_cost_samples", 50))
    if len(items) > max_samples and not args.limit:
        print(f"[提示] 样本数 {len(items)} 超过 max_cost_samples={max_samples}，"
              f"如需继续请显式指定 --limit 或调整 config")

    # 1) 检索层
    print("\n[1/3] 检索层评估...")
    retrieval = run_retrieval(config)
    r_summary = retrieval["summary"]

    # 2) 跑完整 graph
    print("\n[2/3] 跑 CRAG graph...")
    graph = build_graph(config)
    t0 = time.perf_counter()
    records = run_graph_batch(graph, items)
    graph_elapsed = time.perf_counter() - t0

    # 3) 生成层 Ragas
    ragas_summary: dict[str, Any] = {}
    if not args.skip_ragas:
        print("\n[3/3] Ragas 生成层评估...")
        samples = [{"question": r["question"], "answer": r["answer"],
                    "contexts": r["contexts"], "ground_truth": r.get("ground_truth", ""),
                    "type": r.get("type", "")} for r in records if r.get("answer")]
        ragas_summary = run_ragas(samples, config, show_progress=False)
    else:
        print("\n[3/3] 已跳过 Ragas")

    # 汇总
    latencies = [r["elapsed"] for r in records if r.get("elapsed")]
    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95) - 1] if latencies_sorted else 0.0

    refusal_items = [r for r in records if r.get("ground_truth") == "__REFUSAL__"]
    refusal_correct = sum(1 for r in refusal_items if r.get("refusal"))

    metrics: dict[str, Any] = {
        "hit@5": r_summary["hit@k"],
        "mrr": r_summary["mrr"],
        "faithfulness": ragas_summary.get("faithfulness"),
        "answer_relevancy": ragas_summary.get("answer_relevancy"),
        "context_recall": ragas_summary.get("context_recall"),
        "refusal_rate": (round(refusal_correct / len(refusal_items), 4)
                         if refusal_items else ragas_summary.get("refusal_rate")),
        "avg_latency": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency": round(p95, 2),
        "n": len(items),
        "n_retrieval": r_summary["n"],
        "ragas_failure_rate": ragas_summary.get("failure_rate", 0.0),
    }

    old = latest_report_json()
    old_metrics = (old or {}).get("metrics", {}) if old else {}

    commit = git_short_hash()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"report_{commit}.json"
    md_path = REPORTS_DIR / f"report_{commit}.md"

    payload = {"commit": commit, "timestamp": datetime.now().isoformat(),
               "note": args.note, "metrics": metrics,
               "by_type": r_summary["by_type"], "records": records}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        "| 指标 | 本版 | 上版 | Δ |",
        "|---|---|---|---|",
    ]
    labels = {"hit@5": "Hit@5", "mrr": "MRR", "faithfulness": "Faithfulness",
              "answer_relevancy": "Answer Relevancy", "context_recall": "Context Recall",
              "refusal_rate": "拒答正确率", "avg_latency": "平均延迟(s)",
              "p95_latency": "p95 延迟(s)", "n": "样本数"}
    for key, label in labels.items():
        rows.append(f"| {label} | {metrics.get(key)} | {old_metrics.get(key, '—')} | "
                    f"{delta(metrics.get(key), old_metrics.get(key))} |")

    md = "\n".join([
        f"# Eval Report @{commit}",
        "",
        f"生成时间: {datetime.now().isoformat()} ｜ 样本数: {len(items)}",
        "",
        "## 指标对比",
        "",
        *rows,
        "",
        "## 变更说明",
        "",
        args.note or "（未填写）",
        "",
        "## 分类型检索指标",
        "",
        "| 类型 | 样本数 | Hit@5 | MRR |",
        "|---|---|---|---|",
        *[f"| {t} | {v['count']} | {v['hit@k']} | {v['mrr']} |"
          for t, v in r_summary["by_type"].items()],
        "",
        "## 低分样本 Top 5",
        "",
    ])
    scored = [r for r in records if r.get("contexts")]
    for r in scored[:5]:
        md += f"- {r['question'][:60]}（{r.get('type')}）｜trace: {' → '.join(r.get('trace', [])[:3])}\n"

    md += ("\n---\n\n对比实验请修改 configs/ 后重跑本命令并加 --note\n")
    md_path.write_text(md, encoding="utf-8")

    print(f"\n=== 指标 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n报告: {md_path}")
    print(f"机器读: {json_path}")
    print("对比实验请修改 configs/ 后重跑本命令并加 --note")


if __name__ == "__main__":
    main()
