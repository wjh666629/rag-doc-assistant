"""Day 3 B2：切分参数实验（只跑检索，不调 LLM，成本≈0）。

对比不同 chunk_size / top_k 组合对检索质量的影响：
每组参数建一个独立 collection（如 docs_300_3），用同一批测试问题检索，
记录 top1 分数、命中来源，输出对比 markdown。

重要：实验绝不碰主 collection "langchain_docs"，临时 collection 跑完保留供抽查。

运行：python scripts/chunk_experiment.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 直接以脚本方式运行时，把项目根目录加入模块搜索路径，保证能 import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

from src.config import PROJECT_ROOT, load_config, merge_config  # noqa: E402
from src.ingest.build_index import get_embeddings  # noqa: E402
from src.ingest.loader import load_documents  # noqa: E402
from src.ingest.splitter import split_documents  # noqa: E402

QUESTIONS_PATH: Path = PROJECT_ROOT / "eval" / "test_questions.json"
REPORTS_DIR: Path = PROJECT_ROOT / "eval" / "reports"

# 参数网格：chunk_size × top_k
CHUNK_SIZES: list[int] = [300, 500, 1000]
TOP_KS: list[int] = [3, 5]


def load_questions() -> list[dict[str, Any]]:
    """加载测试问题；文件不存在则生成示例文件并提示人工补充。"""
    if not QUESTIONS_PATH.exists():
        QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample = [{"id": i + 1, "question": q, "type": "简单事实",
                   "expected_source": "", "notes": "示例，请人工补充"}
                  for i, q in enumerate([
                      "如何定义 LangGraph 的 state？",
                      "什么是流式输出？",
                      "如何创建工具（tool）？"])]
        QUESTIONS_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[提示] 未找到测试问题文件，已生成示例: {QUESTIONS_PATH}，请人工补充后重跑")
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def build_temp_collection(chunks: list[Document], collection_name: str,
                          config: dict[str, Any]) -> Chroma:
    """把一组 chunk 写入独立的临时 collection（不污染主库）。"""
    persist_dir = PROJECT_ROOT / config["paths"]["chroma_dir"]
    store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(config),
        collection_name=collection_name,
    )
    batch = int(config.get("chroma", {}).get("batch_size", 64))
    for i in range(0, len(chunks), batch):
        store.add_documents(chunks[i:i + batch])
    return store


def run_one(chunk_size: int, top_k: int, docs: list[Document],
            questions: list[dict[str, Any]], base_config: dict[str, Any]) -> dict[str, Any]:
    """跑一组参数：切分 → 建临时库 → 逐题检索 → 汇总。"""
    override = {"ingest": {"chunk_size": chunk_size,
                           "chunk_overlap": int(chunk_size * 0.15)},
                "rag": {"top_k": top_k}}
    cfg = merge_config(base_config, override)
    chunks = split_documents(docs, cfg)
    name = f"exp_{chunk_size}_{top_k}"
    print(f"\n=== 参数组 chunk_size={chunk_size}, top_k={top_k} → {len(chunks)} chunks ===")

    store = build_temp_collection(chunks, name, cfg)
    retriever = store.as_retriever(search_kwargs={"k": top_k})

    scores: list[float] = []
    details: list[dict[str, Any]] = []
    for q in questions:
        hits = retriever.invoke(q["question"])
        # Chroma 的 similarity_search_with_score 用距离，这里用 retriever 只取文档；
        # 分数另用 store 的 similarity_search_with_relevance_scores 计算（0~1，越大越相关）
        scored = store.similarity_search_with_relevance_scores(q["question"], k=top_k)
        top1_score = scored[0][1] if scored else 0.0
        top1_source = hits[0].metadata.get("source", "") if hits else ""
        scores.append(top1_score)
        details.append({"id": q["id"], "question": q["question"], "type": q["type"],
                        "top1_source": top1_source, "top1_score": round(top1_score, 4)})

    return {
        "chunk_size": chunk_size,
        "top_k": top_k,
        "chunk_count": len(chunks),
        "avg_chunk_len": round(statistics.mean([len(c.page_content) for c in chunks]), 1),
        "avg_top1_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "details": details,
        "collection": name,
    }


def write_report(results: list[dict[str, Any]]) -> Path:
    """输出对比 markdown 报告。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"chunk_experiment_{date.today().isoformat()}.md"
    lines: list[str] = [
        f"# chunk_size / top_k 参数实验（{date.today().isoformat()}）",
        "",
        "> 说明：本实验只跑检索，不调 LLM；每组参数使用独立 collection，不污染主库。",
        "",
        "| chunk_size | top_k | chunk 总数 | 平均 chunk 长度 | 平均 top1 相关度 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['chunk_size']} | {r['top_k']} | {r['chunk_count']} | "
                     f"{r['avg_chunk_len']} | {r['avg_top1_score']} |")

    lines.append("\n## 逐题明细（最优参数组）\n")
    best = max(results, key=lambda r: r["avg_top1_score"])
    lines.append(f"最优组：chunk_size={best['chunk_size']}, top_k={best['top_k']} "
                 f"（平均 top1 相关度 {best['avg_top1_score']}）\n")
    lines.append("| 问题 | 类型 | top1 来源 | top1 分数 |")
    lines.append("|---|---|---|---|")
    for d in best["details"]:
        lines.append(f"| {d['question']} | {d['type']} | {d['top1_source']} | {d['top1_score']} |")

    lines.append("\n## 结论\n（人工补充：哪组参数最好，为什么）\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    base_config = load_config()
    questions = load_questions()
    print(f"测试问题数: {len(questions)}")

    print("加载文档...")
    docs = load_documents()
    print(f"文档数: {len(docs)}")

    results: list[dict[str, Any]] = []
    for chunk_size in CHUNK_SIZES:
        for top_k in TOP_KS:
            results.append(run_one(chunk_size, top_k, docs, questions, base_config))

    path = write_report(results)
    print(f"\n实验报告已生成: {path}")
    for r in results:
        print(f"  chunk_size={r['chunk_size']:5d} top_k={r['top_k']}  "
              f"chunks={r['chunk_count']:5d}  avg_top1={r['avg_top1_score']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[错误] 参数实验失败: {type(exc).__name__}: {exc}")
        sys.exit(1)
