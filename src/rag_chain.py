"""Day 3 B1：最小 RAG 闭环（检索 → 拼接 → 生成），LCEL 直线链，不含 LangGraph。

设计说明（为什么这样组装）：
    chain = {"context": itemgetter("question") | retriever | format_docs,
             "question": itemgetter("question")} | prompt | llm | parser
这里用 RunnableParallel（那个 dict）而不是"先检索再单独调 chain 两步走"，
好处是整条链仍然是单个 Runnable，天然获得 batch/stream/追踪能力。
但本脚本需要打印引用来源，所以额外保留 `retrieve_only()` 单独取 docs：
检索结果既要进 prompt，也要给人看，这两件事分开做最清晰。

运行：
    python -m src.rag_chain                # 交互模式
    python -m src.rag_chain --selftest     # 内置 3 个问题自检
"""
from __future__ import annotations

import argparse
import sys
import time
from operator import itemgetter
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.config import load_config
from src.llm import get_chat_model
from src.agent.prompts import get_rag_prompt
from src.retriever import format_docs, get_retriever

SELFTEST_QUESTIONS: list[str] = [
    "LangGraph 中如何定义 state？",
    "什么是 LCEL？",
    "如何做流式输出？",
]


def retrieve_only(question: str, config: dict[str, Any] | None = None) -> list[Document]:
    """只做检索（供打印来源 / 评估复用），与生成解耦。"""
    config = config or load_config()
    retriever = get_retriever(config)
    return retriever.invoke(question)


def build_chain(config: dict[str, Any] | None = None):
    """构建最小 RAG 链：{context, question} | prompt | llm | parser。"""
    config = config or load_config()
    rag_cfg = config.get("rag", {})
    retriever = get_retriever(config)
    prompt = get_rag_prompt(rag_cfg.get("prompt_version", "v1"))
    llm = get_chat_model(temperature=float(rag_cfg.get("temperature", 0.1)), config=config)

    def to_context(question: str) -> str:
        docs = retriever.invoke(question)
        return format_docs(docs, config=config)

    chain = (
        {
            "context": itemgetter("question") | RunnableLambda(to_context),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def answer(question: str, config: dict[str, Any] | None = None,
           debug: bool = False) -> dict[str, Any]:
    """答一个问题：返回回答文本 + 检索到的 docs + 耗时。"""
    config = config or load_config()
    docs = retrieve_only(question, config)
    context = format_docs(docs, config=config)

    if debug:
        print("=" * 70)
        print(f"[DEBUG] 拼接后的参考资料（{len(context)} 字符）:")
        print(context)
        print("=" * 70)

    chain = build_chain(config)
    start = time.perf_counter()
    generation = chain.invoke({"question": question})
    elapsed = time.perf_counter() - start

    return {"question": question, "answer": generation, "documents": docs,
            "context": context, "elapsed": elapsed}


def print_result(result: dict[str, Any]) -> None:
    """打印回答 + 引用来源。"""
    print(f"\n回答: {result['answer']}\n")
    print("引用来源:")
    for i, doc in enumerate(result["documents"], start=1):
        print(f"  [{i}] {doc.metadata.get('source')}  "
              f"(chunk_id={doc.metadata.get('chunk_id')})  "
              f"{doc.page_content[:60].replace(chr(10), ' ')}…")
    print(f"\n耗时: {result['elapsed']:.2f} 秒")


def selftest(config: dict[str, Any]) -> None:
    """内置自检：3 个问题（含 1 条越界问题验证拒答）。"""
    questions = SELFTEST_QUESTIONS + ["如何做红烧肉？"]
    for q in questions:
        print("\n" + "#" * 70)
        print(f"Q: {q}")
        result = answer(q, config)
        print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="最小 RAG 闭环（Day 3）")
    parser.add_argument("--selftest", action="store_true", help="跑内置自检问题")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.selftest:
        selftest(config)
        return

    print("RAG 问答已就绪（/debug 切换调试模式，exit 退出）\n")
    debug = False
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not q:
            continue
        if q.lower() == "exit":
            print("再见！")
            break
        if q == "/debug":
            debug = not debug
            print(f"[调试模式: {'开' if debug else '关'}]")
            continue
        try:
            print_result(answer(q, config, debug=debug))
        except Exception as exc:
            print(f"[错误] 回答失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
