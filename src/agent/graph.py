"""Day 4 B4：CRAG 图装配 + CLI。

结构：
    router ─┬─(无需检索)─→ generate ─→ END
            └─(需要检索)─→ retrieve → grade ─┬─(relevant)─→ generate
                                             └─(irrelevant, retry<max)─→ rewrite → retrieve（回边，最多 1 次）
                                             └─(irrelevant, retry>=max)─→ generate（拒答）

运行：
    python -m src.agent.graph --selftest
    python -m src.agent.graph
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    generate_node,
    grade_node,
    retrieve_node,
    rewrite_node,
    route_after_grade,
    route_after_router,
    router_node,
)
from src.agent.state import GraphState
from src.config import PROJECT_ROOT, load_config

SELFTEST_CASES: list[tuple[str, str]] = [
    ("LangGraph 中如何定义 state？", "a. 正常文档问题"),
    ("DeepSeek 模型的训练数据规模是多少？", "b. 越界问题（期望触发 rewrite）"),
    ("你好", "c. 纯闲聊（期望直接回答）"),
    ("那个东西怎么用？", "d. 模糊问题（人工判断）"),
]


def build_graph(config: dict[str, Any] | None = None):
    """装配并编译 CRAG 图。"""
    config = config or load_config()
    graph = StateGraph(GraphState)

    # 节点：每个节点只干一件事，便于单独测试与定位
    graph.add_node("router", lambda s: router_node(s, config))
    graph.add_node("retrieve", lambda s: retrieve_node(s, config))
    graph.add_node("grade", lambda s: grade_node(s, config))
    graph.add_node("rewrite", lambda s: rewrite_node(s, config))
    graph.add_node("generate", lambda s: generate_node(s, config))

    # 边
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", lambda s: route_after_router(s, config),
                                {"retrieve": "retrieve", "generate": "generate"})
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", lambda s: route_after_grade(s, config),
                                {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")  # 回边：重写后再检索（循环由 max_retry 封顶）
    graph.add_edge("generate", END)

    return graph.compile()


def save_mermaid(compiled, out_path: Path | None = None) -> Path:
    """把图结构导出为 mermaid 文本，供 README 渲染架构图。"""
    out_path = out_path or (PROJECT_ROOT / "docs" / "architecture.mmd")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mermaid = compiled.get_graph().draw_mermaid()
    except Exception as exc:
        mermaid = f"%% mermaid 生成失败: {exc}"
    out_path.write_text(mermaid, encoding="utf-8")
    return out_path


def run_question(compiled, question: str) -> dict[str, Any]:
    """跑一次问答，返回 state 终值（含 generation/documents/trace）。"""
    init: dict[str, Any] = {
        "question": question,
        "current_query": question,
        "need_retrieval": True,
        "documents": [],
        "relevance": "",
        "retry_count": 0,
        "generation": "",
        "refusal": False,
        "trace": [],
    }
    start = time.perf_counter()
    final = compiled.invoke(init)
    elapsed = time.perf_counter() - start
    final["elapsed"] = elapsed
    return final


def print_result(final: dict[str, Any], debug: bool = False) -> None:
    """打印回答、引用来源，debug 模式下完整打印 trace。"""
    flag = " ⚠️ 拒答" if final.get("refusal") else ""
    print(f"\n回答{flag}: {final.get('generation', '')}\n")
    docs = final.get("documents") or []
    if docs:
        print("引用来源:")
        for i, d in enumerate(docs, start=1):
            print(f"  [{i}] {d.metadata.get('source')}  {d.page_content[:60].replace(chr(10), ' ')}…")
    print(f"\n耗时: {final.get('elapsed', 0):.2f} 秒")
    if debug:
        print("执行路径:")
        for line in final.get("trace", []):
            print(f"  - {line}")


def selftest(config: dict[str, Any]) -> None:
    """4 个用例自检，打印每个用例的执行路径。"""
    compiled = build_graph(config)
    for question, label in SELFTEST_CASES:
        print("\n" + "#" * 70)
        print(f"[{label}] Q: {question}")
        final = run_question(compiled, question)
        print_result(final, debug=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRAG 图（Day 4）")
    parser.add_argument("--selftest", action="store_true", help="跑 4 个内置用例")
    parser.add_argument("--save-mermaid", action="store_true", help="导出架构图到 docs/architecture.mmd")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # LangSmith 观测状态提示（接入只需环境变量，代码零改动）
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    print(f"LangSmith tracing: {'开（project=' + os.getenv('LANGSMITH_PROJECT', '') + '）' if tracing else '关'}")

    compiled = build_graph(config)
    if args.save_mermaid:
        path = save_mermaid(compiled)
        print(f"架构图已导出: {path}")

    if args.selftest:
        selftest(config)
        return

    print("CRAG 问答已就绪（/debug 显示执行路径，exit 退出）\n")
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
            print_result(run_question(compiled, q), debug=debug)
        except Exception as exc:
            print(f"[错误] 运行失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
