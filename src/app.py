"""Day 6 B3：Streamlit Demo 界面。

启动：streamlit run src/app.py

设计说明：
- 每轮问题**独立跑 graph，不共享 state**（即无多轮上下文）。这是刻意简化：
  本项目的核心是检索与诚实拒答，引入多轮会话会让评估不可复现；
  若要做真正的多轮，应把 messages 放进 GraphState 并配 checkpointer。
- @st.cache_resource 缓存编译后的图与向量库连接，避免每次交互重建（否则每轮都要加载模型）
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import streamlit as st  # noqa: E402

from src.agent.graph import build_graph  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402

REFUSAL_HINT = "根据现有资料无法回答"


@st.cache_resource
def get_graph(_config_key: str = "default"):
    """编译并缓存 graph（重资源，只建一次）。"""
    config = load_config()
    return build_graph(config)


@st.cache_data(ttl=60)
def load_latest_metrics() -> dict | None:
    """读取最新一份 report_*.json 的指标（60 秒缓存）。"""
    reports = sorted((PROJECT_ROOT / "eval" / "reports").glob("report_*.json"),
                     key=lambda p: p.stat().st_mtime)
    if not reports:
        return None
    try:
        return json.loads(reports[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def check_index_exists() -> bool:
    """运行前检查向量库是否存在，不存在给友好提示而非 traceback。"""
    chroma_dir = PROJECT_ROOT / "data" / "chroma"
    return chroma_dir.exists() and any(chroma_dir.iterdir())


def main() -> None:
    st.set_page_config(page_title="rag-doc-assistant", page_icon="📚")
    st.title("📚 rag-doc-assistant")
    st.caption("基于 LangGraph 的中文技术文档问答系统（CRAG：路由 → 检索 → 打分 → 重写 → 生成）")

    config = load_config()

    # 侧边栏：配置摘要 + 最新评估指标
    with st.sidebar:
        st.subheader("当前配置")
        st.json({
            "chunk_size": config["ingest"]["chunk_size"],
            "top_k": config["rag"]["top_k"],
            "prompt_version": config["rag"]["prompt_version"],
            "max_retry": config["agent"]["max_retry"],
        })
        st.subheader("最新评估指标")
        report = load_latest_metrics()
        if report:
            m = {k: v for k, v in report.get("metrics", {}).items()
                 if k in ("hit@5", "mrr", "faithfulness", "answer_relevancy", "refusal_rate")}
            st.table(m)
            st.caption(f"commit: {report.get('commit')}")
        else:
            st.caption("尚无评估报告，先跑 python -m eval.run_eval")
        if st.button("清空对话"):
            st.session_state.messages = []

    if not check_index_exists():
        st.error("向量库不存在。请先运行：`python -m src.ingest.build_index --rebuild`")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("问一个关于 LangChain / LangGraph 的问题"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            graph = get_graph()
            start = time.perf_counter()
            with st.spinner("检索与生成中..."):
                final = graph.invoke({
                    "question": question,
                    "current_query": question,
                    "need_retrieval": True,
                    "documents": [],
                    "relevance": "",
                    "retry_count": 0,
                    "generation": "",
                    "refusal": False,
                    "trace": [],
                })
            elapsed = time.perf_counter() - start

            refusal = final.get("refusal", False)
            if refusal:
                st.warning("⚠️ 系统判断现有资料不足以回答，已诚实拒答")
            st.markdown(final.get("generation", ""))
            st.caption(f"耗时 {elapsed:.2f} 秒")

            docs = final.get("documents") or []
            if docs:
                with st.expander("引用来源"):
                    for i, d in enumerate(docs, start=1):
                        st.markdown(f"**[{i}] {d.metadata.get('source')}**")
                        st.code(d.page_content[:300])
            with st.expander("执行轨迹"):
                for line in final.get("trace", []):
                    st.text(f"- {line}")

        st.session_state.messages.append(
            {"role": "assistant", "content": final.get("generation", "")})


if __name__ == "__main__":
    main()
