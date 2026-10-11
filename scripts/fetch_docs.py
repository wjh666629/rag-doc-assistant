"""Day 1 B6：语料准备 —— 抓取 LangChain 官方文档核心教程页，保存为 markdown。

- 正文抽取用 trafilatura（自动去导航/页脚/cookie 提示），失败回退 markdownify
- 每篇文件头加 source_url / title / fetch_date 三行元信息
- 频率礼仪：每次请求间隔 1 秒；失败记录到 data/raw/_failed.log

直接运行：python scripts/fetch_docs.py
"""
import hashlib
import re
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import trafilatura
from markdownify import markdownify as md

# 输出目录：data/raw/
RAW_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
FAILED_LOG: Path = RAW_DIR / "_failed.log"
REQUEST_INTERVAL: float = 1.0  # 请求间隔（秒），遵守抓取礼仪

# LangChain 官方新文档站（docs.langchain.com）核心页面，共 30 篇。
# 注意：python.langchain.com 旧版 /docs/tutorials/* 与 /docs/concepts/* 已整站
# 301 到新站，旧 URL 全部重定向到同一个页面，抓下来全是重复内容——
# 因此这里直接使用新站真实 URL（RAG/Agent 主题相关优先）。
URLS: list[str] = [
    "https://docs.langchain.com/oss/python/langchain/quickstart",
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/langchain/install",
    "https://docs.langchain.com/oss/python/langchain/agents",
    "https://docs.langchain.com/oss/python/langchain/models",
    "https://docs.langchain.com/oss/python/langchain/messages",
    "https://docs.langchain.com/oss/python/langchain/tools",
    "https://docs.langchain.com/oss/python/langchain/streaming",
    "https://docs.langchain.com/oss/python/langchain/structured-output",
    "https://docs.langchain.com/oss/python/langchain/short-term-memory",
    "https://docs.langchain.com/oss/python/langchain/long-term-memory",
    "https://docs.langchain.com/oss/python/langchain/knowledge-base",
    "https://docs.langchain.com/oss/python/langchain/sql-agent",
    "https://docs.langchain.com/oss/python/langchain/runtime",
    "https://docs.langchain.com/oss/python/langchain/middleware/overview",
    "https://docs.langchain.com/oss/python/langchain/middleware/built-in",
    "https://docs.langchain.com/oss/python/langchain/middleware/custom",
    "https://docs.langchain.com/oss/python/langchain/context-engineering",
    "https://docs.langchain.com/oss/python/langchain/guardrails",
    "https://docs.langchain.com/oss/python/langchain/human-in-the-loop",
    "https://docs.langchain.com/oss/python/langchain/observability",
    "https://docs.langchain.com/oss/python/langchain/studio",
    "https://docs.langchain.com/oss/python/langchain/test",
    "https://docs.langchain.com/oss/python/langchain/test/evals",
    "https://docs.langchain.com/oss/python/langchain/test/unit-testing",
    "https://docs.langchain.com/oss/python/langchain/test/integration-testing",
    "https://docs.langchain.com/oss/python/langchain/event-streaming",
    "https://docs.langchain.com/oss/python/langchain/component-architecture",
    "https://docs.langchain.com/oss/python/langchain/multi-agent",
    "https://docs.langchain.com/oss/python/langchain/multi-agent/subagents",
    # --- LangGraph 官方文档（补充说明：Day 3 发现原有语料缺少 LangGraph/state 主题，
    #     补充 30 篇使语料与「LangGraph 项目」定位匹配）---
    "https://docs.langchain.com/oss/python/langgraph/overview",
    "https://docs.langchain.com/oss/python/langgraph/quickstart",
    "https://docs.langchain.com/oss/python/langgraph/install",
    "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "https://docs.langchain.com/oss/python/langgraph/use-graph-api",
    "https://docs.langchain.com/oss/python/langgraph/functional-api",
    "https://docs.langchain.com/oss/python/langgraph/use-functional-api",
    "https://docs.langchain.com/oss/python/langgraph/use-subgraphs",
    "https://docs.langchain.com/oss/python/langgraph/persistence",
    "https://docs.langchain.com/oss/python/langgraph/checkpointers",
    "https://docs.langchain.com/oss/python/langgraph/add-memory",
    "https://docs.langchain.com/oss/python/langgraph/stores",
    "https://docs.langchain.com/oss/python/langgraph/streaming",
    "https://docs.langchain.com/oss/python/langgraph/event-streaming",
    "https://docs.langchain.com/oss/python/langgraph/interrupts",
    "https://docs.langchain.com/oss/python/langgraph/use-time-travel",
    "https://docs.langchain.com/oss/python/langgraph/human-in-the-loop",
    "https://docs.langchain.com/oss/python/langgraph/agentic-rag",
    "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "https://docs.langchain.com/oss/python/langgraph/choosing-apis",
    "https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph",
    "https://docs.langchain.com/oss/python/langgraph/application-structure",
    "https://docs.langchain.com/oss/python/langgraph/fault-tolerance",
    "https://docs.langchain.com/oss/python/langgraph/pregel",
    "https://docs.langchain.com/oss/python/langgraph/sql-agent",
    "https://docs.langchain.com/oss/python/langgraph/observability",
    "https://docs.langchain.com/oss/python/langgraph/studio",
    "https://docs.langchain.com/oss/python/langgraph/local-server",
    "https://docs.langchain.com/oss/python/langgraph/ui",
    "https://docs.langchain.com/oss/python/langgraph/test",
    "https://docs.langchain.com/oss/python/langgraph/backward-compatibility",
]


def slugify(url: str) -> str:
    """从 URL 提取最后一段作为文件名 slug，并做安全化。"""
    slug: str = url.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", slug) or "untitled"


def extract_markdown(html: str, url: str) -> str:
    """把 HTML 转为干净的 markdown 正文（保留代码块）。"""
    # 首选 trafilatura：能识别正文区域，去掉导航栏/页脚/cookie 横幅，
    # include_formatting=False 时保留代码块结构
    text: str | None = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=False,
        include_images=False,
    )
    if text and len(text.strip()) > 200:
        return text.strip()
    # 回退：markdownify 全量转换（可能带少量导航残留，但不至于丢正文）
    fallback: str = md(html, heading_style="ATX", code_language="python")
    return fallback.strip()


def fetch_one(client: httpx.Client, url: str) -> tuple[str, str, str]:
    """抓取单个页面，返回 (最终 URL, 标题, markdown 正文)。失败抛异常由调用方记录。

    docs.langchain.com 基于 Mintlify 构建，支持在 URL 后加 .md 直接返回原生
    markdown（代码块、表格结构完整，无导航残留），优先使用该端点；
    其他站点走 HTML + trafilatura 正文抽取。
    """
    if "docs.langchain.com" in url:
        md_url: str = url.rstrip("/") + ".md"
        resp = client.get(md_url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        body = resp.text.strip()
        if not body:
            raise ValueError("原生 markdown 端点返回为空")
        # 标题取 markdown 的第一个一级标题
        m = re.search(r"^#\s+(.+)$", body, re.M)
        title: str = m.group(1).strip() if m else slugify(url)
        return str(resp.url).removesuffix(".md"), title, body

    resp = client.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    html: str = resp.text
    # 从 <title> 提取标题
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else slugify(url)
    body = extract_markdown(html, url)
    if not body:
        raise ValueError("正文抽取结果为空")
    return str(resp.url), title, body


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok: int = 0
    failed: list[tuple[str, str]] = []
    seen_hashes: set[str] = set()  # 正文哈希去重：多个 URL 重定向到同一页时只保留一篇

    # User-Agent 标明身份，尊重对方服务
    headers = {"User-Agent": "rag-doc-assistant/0.1 (learning project)"}
    with httpx.Client(headers=headers) as client:
        for i, url in enumerate(URLS, start=1):
            print(f"[{i:02d}/{len(URLS)}] {url}")
            try:
                final_url, title, body = fetch_one(client, url)
                body_hash: str = hashlib.md5(body.encode("utf-8")).hexdigest()
                if body_hash in seen_hashes:
                    raise ValueError(f"内容与已抓取页面重复（重定向到 {final_url}）")
                seen_hashes.add(body_hash)
            except Exception as exc:
                reason: str = f"{type(exc).__name__}: {exc}"
                print(f"    失败: {reason}")
                failed.append((url, reason))
            else:
                # 文件头三行元信息，便于后续溯源与增量更新
                header = (f"source_url: {final_url}\n"
                          f"title: {title}\n"
                          f"fetch_date: {date.today().isoformat()}\n\n")
                path: Path = RAW_DIR / f"{i:02d}_{slugify(url)}.md"
                path.write_text(header + body + "\n", encoding="utf-8")
                print(f"    已保存: {path.name}（{len(body)} 字符）")
                ok += 1
            # 频率礼仪：无论成功失败，请求之间都间隔 1 秒
            if i < len(URLS):
                time.sleep(REQUEST_INTERVAL)

    # 失败的 URL 与原因写入 _failed.log
    if failed:
        with FAILED_LOG.open("a", encoding="utf-8") as f:
            for url, reason in failed:
                f.write(f"{date.today().isoformat()}\t{url}\t{reason}\n")

    print(f"\n抓取摘要: 成功 {ok} 篇，失败 {len(failed)} 篇"
          + (f"（详见 {FAILED_LOG}）" if failed else ""))
    if ok < 20:
        print("[警告] 成功篇数不足 20，请检查网络或考虑方案 b（中文博客源）")
        sys.exit(1)


if __name__ == "__main__":
    main()
