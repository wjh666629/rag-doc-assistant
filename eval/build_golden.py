"""Day 5 B1：评估集半自动生成（机器出候选 + 人工校验）。

两个阶段：
1. `--generate`：从向量库抽样 chunk → LLM 扮演读者出题 → 写 golden_candidates.json
   （结果按 chunk_id 缓存到 eval/.cache/，重跑不重复扣费）
2. `--export`：合并候选 + 人工题（manual_questions.json）→ 去重 → 输出 golden_qa.json
   （只有 verified=true 的条目进入最终文件，未校验的进 golden_todo.json）

运行：
    python -m eval.build_golden --generate
    python -m eval.build_golden --export
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.llm import get_chat_model  # noqa: E402
from src.retriever import get_vectorstore  # noqa: E402

CANDIDATES_PATH = PROJECT_ROOT / "eval" / "golden_candidates.json"
MANUAL_PATH = PROJECT_ROOT / "eval" / "manual_questions.json"
GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_qa.json"
TODO_PATH = PROJECT_ROOT / "eval" / "golden_todo.json"

GEN_SYSTEM = """你是技术文档的读者。请根据给出的文档片段，提出 1 个读者最可能问的问题，
并给出仅基于该片段的标准答案。只输出 JSON：
{"question": "...", "answer": "...", "answer_span": "片段中支撑答案的原文句子"}"""

GEN_USER = """文档片段（来自 {source}）：
{chunk}

请出题并作答（JSON）："""

SAMPLE_PER_SOURCE = 2       # 每个源文件抽几个 chunk
MAX_CHUNKS = 60             # 候选总量上限（控成本）


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}.json"


def _load_cache(cache_dir: Path, key: str) -> dict | None:
    p = _cache_path(cache_dir, key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _save_cache(cache_dir: Path, key: str, value: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, key).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_chunks(config: dict[str, Any]) -> list[dict[str, Any]]:
    """按 source 分组抽样 chunk（保证来源覆盖度）。"""
    store = get_vectorstore(config)
    data = store._collection.get(include=["documents", "metadatas"])
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []

    by_source: dict[str, list[tuple[str, dict]]] = {}
    for text, meta in zip(docs, metas):
        meta = meta or {}
        # 跳过代码块 chunk：代码本身不适合出"读者会问的问题"
        if meta.get("is_code"):
            continue
        if len(text) < 200 or len(text) > 1200:
            continue
        by_source.setdefault(meta.get("source", "unknown"), []).append((text, meta))

    random.seed(42)
    sampled: list[dict[str, Any]] = []
    for source, items in sorted(by_source.items()):
        for text, meta in random.sample(items, min(SAMPLE_PER_SOURCE, len(items))):
            sampled.append({
                "chunk_id": meta.get("chunk_id"),
                "source": source,
                "text": text,
            })
        if len(sampled) >= MAX_CHUNKS:
            break
    return sampled[:MAX_CHUNKS]


def generate_candidates(config: dict[str, Any], force: bool = False) -> list[dict[str, Any]]:
    """阶段 1：LLM 出题，结果入库缓存。"""
    cache_dir = PROJECT_ROOT / config["paths"].get("cache_dir", "eval/.cache")
    chunks = sample_chunks(config)
    print(f"抽样 chunk 数: {len(chunks)}")

    est_tokens = len(chunks) * 900
    print(f"预估 token 用量: ~{est_tokens:,}（约 {est_tokens / 1_000_000 * 4:.2f} 元量级，按输入计）")
    if not force:
        ans = input("确认执行？(y/N): ").strip().lower()
        if ans != "y":
            print("已取消")
            sys.exit(0)

    llm = get_chat_model(temperature=0.3, config=config, json_mode=True)
    candidates: list[dict[str, Any]] = []
    for i, c in enumerate(chunks, start=1):
        key = f"gen::{c['chunk_id']}"
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            candidates.append(cached)
            print(f"[{i}/{len(chunks)}] 命中缓存 chunk_id={c['chunk_id']}")
            continue

        user = GEN_USER.replace("{source}", c["source"]).replace("{chunk}", c["text"][:1200])
        try:
            resp = llm.invoke([("system", GEN_SYSTEM), ("human", user)])
            content = str(resp.content)
            start = content.find("{")
            end = content.rfind("}")
            parsed = json.loads(content[start:end + 1]) if start >= 0 and end > start else None
        except Exception as exc:
            print(f"[{i}/{len(chunks)}] 生成失败 chunk_id={c['chunk_id']}: {exc}")
            parsed = None

        if not parsed or "question" not in parsed:
            continue

        record = {
            "question": parsed["question"].strip(),
            "ground_truth": parsed.get("answer", "").strip(),
            "answer_span": parsed.get("answer_span", ""),
            "source": [c["source"]],
            "type": "简单事实",
            "origin": "auto",
            "verified": False,
        }
        _save_cache(cache_dir, key, record)
        candidates.append(record)
        print(f"[{i}/{len(chunks)}] 生成: {record['question'][:40]}")
        time.sleep(0.3)  # 简单限流

    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n候选写入: {CANDIDATES_PATH}（{len(candidates)} 条）")
    return candidates


def auto_verify(record: dict[str, Any], chunk_texts: dict[str, str]) -> bool:
    """程序化初筛：标准答案的关键片段是否真的能在原文里找到（人工校验的预筛）。

    说明：这是**机器预筛**不是人工校验，用于减少人工工作量；
    answer_span 的显著片段能在对应 source 原文中出现才算通过。
    """
    span = (record.get("answer_span") or record.get("ground_truth") or "").strip()
    if not span:
        return False
    probe = span[:24]
    texts = [chunk_texts.get(s, "") for s in record.get("source", [])]
    return any(probe in t for t in texts if t)


def export(config: dict[str, Any]) -> None:
    """阶段 2：合并候选 + 人工题 → 去重 → 输出 golden_qa.json / golden_todo.json。"""
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8")) if CANDIDATES_PATH.exists() else []
    manual = json.loads(MANUAL_PATH.read_text(encoding="utf-8")) if MANUAL_PATH.exists() else []

    # 取回原文用于机器预筛
    store = get_vectorstore(config)
    data = store._collection.get(include=["documents", "metadatas"])
    chunk_texts: dict[str, str] = {}
    for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
        meta = meta or {}
        src = meta.get("source", "")
        chunk_texts[src] = chunk_texts.get(src, "") + "\n" + text

    merged: list[dict[str, Any]] = []
    for r in candidates:
        merged.append({**r, "verified": bool(r.get("verified")) or auto_verify(r, chunk_texts)})
    for r in manual:
        merged.append({**r, "verified": True})  # 人工书写的题目默认已校验

    # 去重：question 文本相似度高（>0.85）的只保留第一条
    unique: list[dict[str, Any]] = []
    for r in merged:
        if any(SequenceMatcher(None, r["question"], u["question"]).ratio() > 0.85 for u in unique):
            continue
        unique.append(r)

    for i, r in enumerate(unique, start=1):
        r["id"] = i

    verified_items = [r for r in unique if r.get("verified")]
    todo_items = [r for r in unique if not r.get("verified")]

    GOLDEN_PATH.write_text(json.dumps(verified_items, ensure_ascii=False, indent=2), encoding="utf-8")
    TODO_PATH.write_text(json.dumps(todo_items, ensure_ascii=False, indent=2), encoding="utf-8")

    stats: dict[str, int] = {}
    for r in verified_items:
        stats[r.get("type", "未知")] = stats.get(r.get("type", "未知"), 0) + 1
    origins: dict[str, int] = {}
    for r in verified_items:
        origins[r.get("origin", "?")] = origins.get(r.get("origin", "?"), 0) + 1

    print(f"golden_qa.json: {len(verified_items)} 条 → {GOLDEN_PATH}")
    print(f"golden_todo.json: {len(todo_items)} 条（未通过校验，需人工处理）")
    print(f"类型分布: {stats}")
    print(f"来源分布: {origins}")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估集半自动生成（Day 5）")
    parser.add_argument("--generate", action="store_true", help="阶段1：LLM 生成候选")
    parser.add_argument("--export", action="store_true", help="阶段2：合并导出")
    parser.add_argument("--force", action="store_true", help="跳过成本确认")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.generate:
        generate_candidates(config, force=args.force)
    elif args.export:
        export(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
