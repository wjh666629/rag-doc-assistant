"""Day 2 B3：向量化入库（Chroma）。

流程：load_documents → split_documents → HuggingFaceEmbeddings 向量化 → 写入 Chroma。

设计要点：
- 幂等：不带 --rebuild 时若库已存在则提示并退出，防止重复入库；
- 批量入库（每批 64 条）+ 进度打印；
- 入库后自检：总数比对 + 3 个内置问题做 similarity_search，人工判断相关性。

运行：
    python -m src.ingest.build_index --rebuild
    python -m src.ingest.build_index            # 已存在则提示
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# 必须在 import sentence_transformers 之前设置镜像与禁用 xet
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from langchain_core.documents import Document  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.ingest.loader import load_documents  # noqa: E402
from src.ingest.splitter import split_documents  # noqa: E402

# 入库后自检用的内置问题（人工判断 top1 是否相关）
SELFTEST_QUESTIONS: list[str] = [
    "LangGraph 中如何定义 state？",
    "什么是 LCEL？",
    "如何做流式输出？",
]


def get_embeddings(config: dict[str, Any]) -> HuggingFaceEmbeddings:
    """构建本地 embedding 模型（优先用项目内模型目录，避免重复下载）。"""
    emb_cfg = config.get("embedding", {})
    model_name: str = emb_cfg.get("model_name", "models/bge-small-zh-v1.5")
    model_path = Path(model_name)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_name
    if not model_path.exists():
        # 本地目录不存在 → 回退到 HF 仓库名（会自动走 HF_ENDPOINT 镜像下载）
        model_name = emb_cfg.get("hf_model_name", "BAAI/bge-small-zh-v1.5")
        print(f"[提示] 本地模型目录不存在，改用 HF 模型名: {model_name}")
    else:
        model_name = str(model_path)

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": bool(emb_cfg.get("normalize", True))},
    )


def build_vectorstore(config: dict[str, Any], rebuild: bool = False) -> Chroma:
    """构建（或重建）Chroma 向量库。"""
    chroma_cfg = config.get("chroma", {})
    persist_dir = PROJECT_ROOT / config["paths"]["chroma_dir"]
    collection_name: str = chroma_cfg.get("collection_name", "langchain_docs")
    batch_size: int = int(chroma_cfg.get("batch_size", 64))

    embeddings = get_embeddings(config)

    # 幂等检查：库已存在且未要求重建 → 提示后退出
    if persist_dir.exists() and any(persist_dir.iterdir()) and not rebuild:
        probe = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        count = probe._collection.count()
        print(f"索引已存在（{count} 条），使用 --rebuild 重建")
        sys.exit(0)

    print("加载并切分文档...")
    docs = load_documents()
    chunks: list[Document] = split_documents(docs, config)
    print(f"待入库 chunk 数: {len(chunks)}")

    if rebuild and persist_dir.exists():
        # 清空重建：删除旧 collection
        try:
            old = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
                collection_name=collection_name,
            )
            old.delete_collection()
            print("已删除旧 collection")
        except Exception as exc:
            print(f"[提示] 删除旧 collection 失败（继续创建新的）: {exc}")

    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    print(f"批量入库（每批 {batch_size} 条）...")
    start = time.perf_counter()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        store.add_documents(batch)
        done = min(i + batch_size, len(chunks))
        print(f"  进度: {done}/{len(chunks)}（{done / len(chunks) * 100:.1f}%）", flush=True)
    elapsed = time.perf_counter() - start
    print(f"入库完成，耗时 {elapsed:.1f} 秒（{len(chunks) / max(elapsed, 1e-6):.1f} chunk/秒）")

    # 自检 1：总数比对
    actual = store._collection.count()
    print(f"\n[自检] 切分数 {len(chunks)} vs 库内总数 {actual} → "
          f"{'一致 ✔' if actual == len(chunks) else '不一致 ✘'}")

    # 自检 2：3 个内置问题检索，人工判断相关性
    print("\n--- 检索自检（top1）---")
    for q in SELFTEST_QUESTIONS:
        hits = store.similarity_search(q, k=3)
        if not hits:
            print(f"Q: {q}\n  无结果 ✘\n")
            continue
        top = hits[0]
        print(f"Q: {q}")
        print(f"  top1 来源: {top.metadata.get('source')}")
        print(f"  前 150 字符: {top.page_content[:150].replace(chr(10), ' ')}")
        print()
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Chroma 向量库")
    parser.add_argument("--rebuild", action="store_true", help="清空并重建索引")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        build_vectorstore(config, rebuild=args.rebuild)
    except Exception as exc:
        print(f"[错误] 入库失败: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
