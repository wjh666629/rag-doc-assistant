"""Day 3：检索层（与生成解耦，可单独 import 测试）。

提供：
- get_retriever(config)：返回 Chroma retriever（top_k / score_threshold 从 config 读）
- get_vectorstore(config)：返回 Chroma 实例（评估/实验用）
- format_docs(docs)：把检索结果拼成带 [n] 编号的参考资料文本
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.ingest.build_index import get_embeddings  # noqa: E402


def get_vectorstore(config: dict[str, Any] | None = None,
                    collection_name: str | None = None) -> Chroma:
    """获取 Chroma 向量库实例（不存在的目录会给出友好提示而非 traceback）。"""
    config = config or load_config()
    persist_dir = PROJECT_ROOT / config["paths"]["chroma_dir"]
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        raise FileNotFoundError(
            f"向量库不存在: {persist_dir}。请先运行: python -m src.ingest.build_index --rebuild"
        )
    name = collection_name or config["chroma"]["collection_name"]
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(config),
        collection_name=name,
    )


def get_retriever(config: dict[str, Any] | None = None,
                  k: int | None = None) -> BaseRetriever:
    """构建 retriever：top_k 与 score_threshold 全部来自 config，不硬编码。"""
    config = config or load_config()
    rag_cfg = config.get("rag", {})
    top_k: int = int(k or rag_cfg.get("top_k", 5))
    threshold: float = float(rag_cfg.get("score_threshold", 0.0))

    store = get_vectorstore(config)
    search_kwargs: dict[str, Any] = {"k": top_k}
    if threshold > 0:
        # Chroma 的距离越小越相似，score_threshold 用距离阈值过滤
        search_kwargs["score_threshold"] = threshold
    return store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def format_docs(docs: list[Document], max_chars: int | None = None,
                config: dict[str, Any] | None = None) -> str:
    """把检索结果拼成 prompt 用的参考资料文本。

    格式：
        [1] (来源: 05_models.md)
        <chunk 正文>

    超长时按 max_chars 截断（默认读 config 的 max_context_chars=4000）。
    """
    if max_chars is None:
        config = config or load_config()
        max_chars = int(config.get("rag", {}).get("max_context_chars", 4000))

    parts: list[str] = []
    total: int = 0
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        block = f"[{i}] (来源: {source})\n{doc.page_content}"
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            block = block[:remain] + " …（已截断）"
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "\n\n".join(parts)
