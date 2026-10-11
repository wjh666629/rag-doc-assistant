"""Day 7 B1：批量对比实验（一次一变量）。

读 eval/experiments.yaml，对每个实验：
1. default.yaml + config_override → 临时 config
2. rebuild_index=true 时把语料重建到**独立 collection**（不污染主库）
3. 跑与 run_eval 同等逻辑的评估（检索 + 生成 + 工程）
4. 输出 report_{实验名}_{hash}.md/json，并与 baseline 对比
5. 全部跑完生成 experiments_summary.md 大表

运行：python -m eval.run_experiments [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import yaml  # noqa: E402

from src.config import PROJECT_ROOT, load_config, merge_config  # noqa: E402

EXPERIMENTS_PATH = PROJECT_ROOT / "eval" / "experiments.yaml"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


def git_short_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def rebuild_to_collection(cfg: dict[str, Any], collection_name: str) -> int:
    """按给定 config 重建一个独立 collection，返回 chunk 数。"""
    from langchain_chroma import Chroma
    from src.ingest.build_index import get_embeddings
    from src.ingest.loader import load_documents
    from src.ingest.splitter import split_documents

    chunks = split_documents(load_documents(), cfg)
    store = Chroma(persist_directory=str(PROJECT_ROOT / cfg["paths"]["chroma_dir"]),
                   embedding_function=get_embeddings(cfg),
                   collection_name=collection_name)
    try:
        store.delete_collection()
    except Exception:
        pass
    store = Chroma(persist_directory=str(PROJECT_ROOT / cfg["paths"]["chroma_dir"]),
                   embedding_function=get_embeddings(cfg),
                   collection_name=collection_name)
    batch = int(cfg.get("chroma", {}).get("batch_size", 64))
    for i in range(0, len(chunks), batch):
        store.add_documents(chunks[i:i + batch])
    return len(chunks)


def run_one(exp: dict[str, Any], base_cfg: dict[str, Any], limit: int | None,
            skip_ragas: bool) -> dict[str, Any]:
    """跑单个实验。"""
    name = exp["name"]
    print(f"\n{'=' * 70}\n实验: {name}\n说明: {exp.get('note', '')}\n{'=' * 70}")
    cfg = merge_config(base_cfg, exp.get("config_override", {}))

    if exp.get("rebuild_index"):
        collection = cfg["chroma"]["collection_name"] + exp.get("collection_suffix", "")
        n = rebuild_to_collection(cfg, collection)
        print(f"已重建独立 collection: {collection}（{n} chunks）")
        cfg["chroma"]["collection_name"] = collection

    # 通过子进程调用 run_eval，保证与单次评估走完全相同的代码路径
    cmd = [sys.executable, "-m", "eval.run_eval", "--note", exp.get("note", "")]
    if limit:
        cmd += ["--limit", str(limit)]
    if skip_ragas:
        cmd.append("--skip-ragas")
    # 把实验 config 写到临时文件传给子进程
    tmp_cfg = PROJECT_ROOT / "eval" / f".tmp_{name}.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    cmd += ["--config", str(tmp_cfg)]

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        print(f"[错误] 实验 {name} 失败: {proc.stderr[-500:]}")
        return {"name": name, "note": exp.get("note", ""), "error": proc.stderr[-500:]}

    # 取本次生成的报告（按 mtime 最近的一份）
    reports = sorted(REPORTS_DIR.glob("report_*.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        return {"name": name, "error": "未找到报告"}
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    # 重命名为带实验名的报告
    commit = payload.get("commit", "na")
    tagged_md = REPORTS_DIR / f"report_{name}_{commit}.md"
    tagged_json = REPORTS_DIR / f"report_{name}_{commit}.json"
    src_md = REPORTS_DIR / f"report_{commit}.md"
    src_json = REPORTS_DIR / f"report_{commit}.json"
    if src_md.exists():
        src_md.rename(tagged_md)
    if src_json.exists():
        src_json.rename(tagged_json)
    tmp_cfg.unlink(missing_ok=True)

    return {"name": name, "note": exp.get("note", ""),
            "metrics": payload.get("metrics", {}),
            "report_md": str(tagged_md.name), "report_json": str(tagged_json.name)}


def write_summary(results: list[dict[str, Any]], baseline: dict[str, Any] | None) -> Path:
    """输出所有实验与 baseline 的对比大表。"""
    path = REPORTS_DIR / "experiments_summary.md"
    keys = ["hit@5", "mrr", "faithfulness", "answer_relevancy", "context_recall",
            "refusal_rate", "avg_latency", "p95_latency"]
    lines = ["# 对比实验汇总（Day 7）", "",
             f"生成时间: {datetime.now().isoformat()}", "",
             "## 配置 diff", ""]
    for r in results:
        lines.append(f"- **{r['name']}**：{r.get('note', '')}")
    lines += ["", "## 指标对比", "", "| 指标 | baseline | " +
              " | ".join(r["name"] for r in results) + " |",
              "|---|" + "---|" * (len(results) + 1)]
    base_metrics = (baseline or {}).get("metrics", {})
    for k in keys:
        cells = []
        for r in results:
            v = (r.get("metrics") or {}).get(k)
            cells.append(str(v) if v is not None else "—")
        lines.append(f"| {k} | {base_metrics.get(k, '—')} | " + " | ".join(cells) + " |")

    lines += ["", "## 逐实验结论（人工补充）", ""]
    for r in results:
        lines.append(f"- **{r['name']}**：假设是否成立？为什么？（待填）")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="批量对比实验（Day 7）")
    parser.add_argument("--limit", type=int, default=None, help="每个实验只跑前 N 条")
    parser.add_argument("--skip-ragas", action="store_true", help="跳过 Ragas（快跑模式）")
    args = parser.parse_args()

    specs = yaml.safe_load(EXPERIMENTS_PATH.read_text(encoding="utf-8")) or {}
    experiments = specs.get("experiments", [])
    if not experiments:
        print("[错误] experiments.yaml 中没有实验定义")
        sys.exit(1)

    base_cfg = load_config()
    # baseline：取最近一份非实验报告（名称形如 report_{hash}.json）
    baseline_reports = [p for p in REPORTS_DIR.glob("report_*.json")]
    baseline = None
    if baseline_reports:
        baseline = json.loads(sorted(baseline_reports, key=lambda p: p.stat().st_mtime)[0]
                              .read_text(encoding="utf-8"))

    results = [run_one(exp, base_cfg, args.limit, args.skip_ragas) for exp in experiments]
    path = write_summary(results, baseline)
    print(f"\n汇总报告: {path}")


if __name__ == "__main__":
    main()
