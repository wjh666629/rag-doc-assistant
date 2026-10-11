"""Day 2 B1：文档加载与清洗。

职责：读取 data/raw/ 下的 markdown 语料 → 解析文件头元信息 → 清洗正文
      → 返回 langchain Document 列表。

运行：python -m src.ingest.loader
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document

from src.config import PROJECT_ROOT

# 文件头元信息形如：
# source_url: https://...
# title: xxx
# fetch_date: 2026-10-11
HEADER_PATTERNS: dict[str, str] = {
    "url": r"^source_url:\s*(.+)$",
    "title": r"^title:\s*(.+)$",
    "fetch_date": r"^fetch_date:\s*(.+)$",
}

# 文档站导航残留行（整行丢弃）
NAV_LINE_PATTERNS: list[str] = [
    r"^\[编辑此页\].*$",
    r"^\[上一页.*$",
    r"^\[下一页.*$",
    r"^上一页\s*/\s*下一页.*$",
    r"^Copy page$",
    r"^Edit this page.*$",
    r"^Was this page helpful\?.*$",
    r"^\[?上一页\]?\s*\|?\s*\[?下一页\]?.*$",
    # Mintlify 原生 markdown 导出时自带的「文档索引」引用块，属爬虫噪声
    r"^>\s*##\s*Documentation Index.*$",
    r"^>\s*Fetch the complete documentation index.*$",
    r"^>\s*Use this file to discover all available pages.*$",
    # Mintlify 组件标签行（如 </ParamField>、<Accordion>），纯结构噪声无语义
    r"^</?[A-Za-z][A-Za-z0-9_\-]*(\s[^>]*)?/?>$",
]

# 纯标签/符号噪声行：一行里全是标签与空白（如 "</ParamField> </Accordion>"）
TAG_ONLY_LINE_RE = re.compile(r"^\s*(</?[A-Za-z][^>]*>\s*)+\s*$")

# 代码块围栏：``` 或 ```python 等（允许缩进，Mintlify CodeGroup 内是缩进的）
CODE_FENCE_RE = re.compile(r"^[ \t]*```.*$", re.M)


def _parse_header(lines: list[str]) -> tuple[dict[str, str], int]:
    """解析文件头三行元信息，返回 (metadata 片段, 正文起始行号)。"""
    meta: dict[str, str] = {}
    idx: int = 0
    for i, line in enumerate(lines[:5]):  # 元信息只可能在前几行
        matched = False
        for key, pattern in HEADER_PATTERNS.items():
            m = re.match(pattern, line.strip())
            if m:
                meta[key] = m.group(1).strip()
                matched = True
                break
        if matched:
            idx = i + 1
        elif i >= 3:
            break
    return meta, idx


def _split_code_blocks(text: str) -> list[tuple[str, bool]]:
    """把文本切成 [普通片段 / 代码块片段] 序列，便于"保护代码块不做任何删改"。

    返回列表元素为 (片段文本, 是否代码块)。
    """
    parts: list[tuple[str, bool]] = []
    pos: int = 0
    in_code: bool = False
    for line in text.splitlines(keepends=True):
        if CODE_FENCE_RE.match(line.rstrip("\n")):
            in_code = not in_code
            parts.append((line, True))
            continue
        parts.append((line, in_code))
        pos += len(line)
    # 合并相邻同类型片段，减少碎片
    merged: list[tuple[str, bool]] = []
    for seg, is_code in parts:
        if merged and merged[-1][1] == is_code:
            merged[-1] = (merged[-1][0] + seg, is_code)
        else:
            merged.append((seg, is_code))
    return merged


def clean_text(text: str) -> str:
    """清洗正文：压缩多余空行、删除导航残留行，代码块内容原样保留。

    规则：
    1. 连续 3 个以上空行压缩为 1 个空行；
    2. 删除文档站导航残留行（[编辑此页]、上一页/下一页 等）；
    3. ``` 包裹的代码块不做任何删改（含围栏行本身）。
    """
    out_parts: list[str] = []
    for segment, is_code in _split_code_blocks(text):
        if is_code:
            out_parts.append(segment)  # 代码块：原样保留
            continue
        # 普通文本：逐行过滤
        kept_lines: list[str] = []
        for line in segment.splitlines(keepends=True):
            stripped = line.strip()
            if any(re.match(p, stripped) for p in NAV_LINE_PATTERNS):
                continue  # 导航残留行，丢弃
            if TAG_ONLY_LINE_RE.match(line):
                continue  # 纯标签噪声行，丢弃
            kept_lines.append(line)
        cleaned = "".join(kept_lines)
        # 连续 3+ 空行压缩为 1 个空行（即 "\n\n\n+" → "\n\n"）
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        out_parts.append(cleaned)
    return "".join(out_parts)


def iter_documents(raw_dir: str | Path) -> Iterator[Document]:
    """逐文件生成器：读一个处理一个，避免一次性把全部语料读入内存。"""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"语料目录不存在: {raw_path}（请先运行 python scripts/fetch_docs.py）")

    md_files = sorted(p for p in raw_path.glob("*.md") if not p.name.startswith("_"))
    if not md_files:
        raise FileNotFoundError(f"语料目录中没有可用的 .md 文件: {raw_path}")

    for path in md_files:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            print(f"[警告] 跳过无法解码的文件: {path}（{exc}）")
            continue

        lines = raw_text.splitlines()
        meta, body_start = _parse_header(lines)
        body = "\n".join(lines[body_start:]).strip()
        body = clean_text(body)

        metadata = {
            "source": path.name,
            "url": meta.get("url", ""),
            "title": meta.get("title", path.stem),
            "fetch_date": meta.get("fetch_date", ""),
        }
        yield Document(page_content=body, metadata=metadata)


def load_documents(raw_dir: str | Path | None = None) -> list[Document]:
    """加载目录下全部文档（生成器逐文件处理后汇总）。"""
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw"
    return list(iter_documents(raw_dir))


def main() -> None:
    """主入口：打印语料统计，并随机抽查 1 篇前 300 字符。"""
    docs = load_documents()
    if not docs:
        print("[错误] 未加载到任何文档")
        return

    lengths = [len(d.page_content) for d in docs]
    print(f"文件数: {len(docs)}")
    print(f"总字符数: {sum(lengths):,}")
    print(f"最长文档: {max(lengths):,} 字符  最短文档: {min(lengths):,} 字符")
    print(f"平均长度: {sum(lengths) // len(lengths):,} 字符")

    sample = random.choice(docs)
    print("\n--- 随机抽查 1 篇（前 300 字符）---")
    print(f"文件: {sample.metadata['source']}")
    print(f"标题: {sample.metadata['title']}")
    print(f"来源: {sample.metadata['url']}")
    print("-" * 60)
    print(sample.page_content[:300])
    print("-" * 60)


if __name__ == "__main__":
    main()
