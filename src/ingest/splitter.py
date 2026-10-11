"""Day 2 B2：中文友好的文档切分（含代码块保护）。

策略：
1. 先把每个 ``` 代码块整体抠出来，换成占位符 \x00CODEBLOCK_{i}\x00，
   让切分器"看不见"代码，从而不会把函数从中间切断；
2. 用 RecursiveCharacterTextSplitter + 中文优先分隔符切分；
3. 还原占位符；若某个 chunk 还原后仍远超 chunk_size，则把该代码块
   单独拆成一个 chunk（metadata 标记 is_code=true）。

运行：python -m src.ingest.splitter
"""
from __future__ import annotations

import random
import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_config
from src.ingest.loader import load_documents

# 代码块占位符：\x00 是普通文本里几乎不会出现的字符，避免与正文冲突
CODE_PLACEHOLDER = "\x00CODEBLOCK_{i}\x00"
CODE_PLACEHOLDER_RE = re.compile(r"\x00CODEBLOCK_(\d+)\x00")
# 允许缩进的代码围栏（Mintlify 的 <CodeGroup> 内代码块是缩进的）
CODE_FENCE_RE = re.compile(r"^[ \t]*```.*$", re.M)
# Mintlify 多标签代码块容器：整块视为一个不可切分单元
CODE_GROUP_RE = re.compile(r"<CodeGroup>.*?</CodeGroup>", re.S)


def _extract_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    """把代码块替换为占位符。

    返回：(替换后的文本, {占位符: 原始代码块文本})
    """
    blocks: dict[str, str] = {}
    result_parts: list[str] = []
    pos: int = 0
    protected: list[tuple[int, int]] = []

    # 第 1 类：Mintlify 的 <CodeGroup> 多标签代码块，整块保护
    for m in CODE_GROUP_RE.finditer(text):
        protected.append((m.start(), m.end()))
    # 第 2 类：普通成对代码围栏（允许缩进）
    fence_starts: list[tuple[int, int]] = []
    for m in CODE_FENCE_RE.finditer(text):
        if any(s <= m.start() < e for s, e in protected):
            continue  # 已被 CodeGroup 覆盖的围栏跳过
        if not fence_starts or fence_starts[-1][1] is not None:
            fence_starts.append((m.start(), None))
        else:
            fence_starts[-1] = (fence_starts[-1][0], m.end())

    protected.extend((s, e) for s, e in fence_starts if e is not None)
    protected.sort()

    for i, (start, end) in enumerate(protected):
        result_parts.append(text[pos:start])
        placeholder = CODE_PLACEHOLDER.format(i=i)
        blocks[placeholder] = text[start:end]
        result_parts.append(placeholder)
        pos = end
    result_parts.append(text[pos:])
    return "".join(result_parts), blocks


def _restore_code_blocks(text: str, blocks: dict[str, str]) -> str:
    """把占位符还原成原始代码块文本。"""
    def repl(m: re.Match) -> str:
        placeholder = m.group(0)
        return blocks.get(placeholder, placeholder)
    return CODE_PLACEHOLDER_RE.sub(repl, text)


def split_documents(docs: list[Document], config: dict[str, Any]) -> list[Document]:
    """切分文档为 chunk。

    Args:
        docs: loader 输出的文档列表
        config: 完整配置 dict（读 config["ingest"]）

    Returns:
        chunk 列表，每个 chunk 的 metadata 继承父文档并追加
        chunk_id / chunk_index / parent_len / is_code
    """
    ingest_cfg = config.get("ingest", {})
    chunk_size: int = int(ingest_cfg.get("chunk_size", 500))
    chunk_overlap: int = int(ingest_cfg.get("chunk_overlap", 80))
    separators: list[str] = list(ingest_cfg.get("separators", ["\n\n", "\n", "。", ""]))
    min_chars: int = int(ingest_cfg.get("min_chunk_chars", 20))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,          # 中文按字符数计长，不用 token 数
        add_start_index=False,
    )

    chunks: list[Document] = []
    chunk_id: int = 0
    filtered: int = 0

    for doc in docs:
        # 1) 代码块替换为占位符
        masked_text, blocks = _extract_code_blocks(doc.page_content)
        # 2) 切分（占位符很短，不会被切断）
        pieces = splitter.split_text(masked_text)

        local_index: int = 0
        for piece in pieces:
            # 3) 还原代码块；is_code 以"还原后是否真的含代码围栏"为准，
            #    避免把只含占位符残片的正文误判为代码
            restored = _restore_code_blocks(piece, blocks)
            is_code = "```" in restored

            # 代码块超长时：整体独立成一个 chunk（代码完整性优先于长度上限），
            # 绝不按空行把函数切碎
            if is_code and len(restored) > chunk_size * 2:
                chunks.append(_make_chunk(doc, restored, chunk_id, local_index, is_code=True))
                chunk_id += 1
                local_index += 1
                continue

            # 4) 碎片过滤
            if len(restored.strip()) < min_chars:
                filtered += 1
                continue

            chunks.append(_make_chunk(doc, restored, chunk_id, local_index, is_code=is_code))
            chunk_id += 1
            local_index += 1

    print(f"[切分] 碎片过滤（<{min_chars} 字符）: {filtered} 个")
    return chunks


def _make_chunk(
    doc: Document,
    content: str,
    chunk_id: int,
    chunk_index: int,
    is_code: bool = False,
) -> Document:
    """构造 chunk 的 Document：继承父文档 metadata 并追加切分信息。"""
    metadata = dict(doc.metadata)
    metadata.update({
        "chunk_id": chunk_id,       # 全局递增唯一 ID
        "chunk_index": chunk_index,  # 在父文档中的序号
        "parent_len": len(doc.page_content),  # 父文档长度（用于判断该 chunk 出自长文还是短文）
        "is_code": is_code,
    })
    return Document(page_content=content, metadata=metadata)


def _percentile(values: list[int], p: float) -> int:
    """计算分位数（p 取 0~100），不依赖 numpy。"""
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    """主入口：切分全部文档并打印质量统计 + 随机抽查 5 个 chunk。"""
    config = load_config()
    print("加载文档...")
    docs = load_documents()
    print(f"文档数: {len(docs)}")

    chunks = split_documents(docs, config)
    lengths = [len(c.page_content) for c in chunks]

    print(f"\nchunk 总数: {len(chunks)}")
    print(f"长度分布: min={min(lengths)}  p50={_percentile(lengths, 50)}  "
          f"p95={_percentile(lengths, 95)}  max={max(lengths)}")
    code_chunks = sum(1 for c in chunks if c.metadata.get("is_code"))
    print(f"含代码块 chunk 数: {code_chunks}")

    print("\n--- 随机抽查 5 个 chunk ---")
    for i, chunk in enumerate(random.sample(chunks, min(5, len(chunks))), start=1):
        print("=" * 70)
        print(f"[样例 {i}] source={chunk.metadata['source']} "
              f"chunk_id={chunk.metadata['chunk_id']} "
              f"is_code={chunk.metadata['is_code']} 长度={len(chunk.page_content)}")
        print("-" * 70)
        print(chunk.page_content)
    print("=" * 70)


if __name__ == "__main__":
    main()
