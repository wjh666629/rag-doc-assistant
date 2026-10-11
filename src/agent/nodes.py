"""Day 4 B3：CRAG 的五个节点 + 路由函数。

每个节点都是纯函数：输入 GraphState（dict），返回要更新进 state 的字段 dict，
并向 trace 追加一行日志。这样每个节点都能脱离图单独 import 测试。
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.agent.prompts import (
    DIRECT_ANSWER_PROMPT_V1,
    DIRECT_ANSWER_USER_TEMPLATE,
    GRADE_PROMPT_V1,
    GRADE_USER_TEMPLATE,
    REFUSAL_TEMPLATE,
    REWRITE_PROMPT_V1,
    REWRITE_USER_TEMPLATE,
    ROUTER_PROMPT_V1,
    ROUTER_USER_TEMPLATE,
)
from src.agent.state import GraphState
from src.config import load_config
from src.llm import get_chat_model, is_mock_mode
from src.retriever import format_docs, get_retriever

JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON 对象；失败返回 None。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_RE.search(text or "")
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _llm_json(system: str, user: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """调用 LLM 并要求 JSON 输出；解析失败返回 None（调用方决定兜底策略）。"""
    if is_mock_mode():
        return None
    llm = get_chat_model(temperature=0.0, config=config, json_mode=True)
    messages = [("system", system), ("human", user)]
    try:
        resp = llm.invoke(messages)
    except Exception:
        return None
    return _parse_json(str(resp.content))


# ------------------------------------------------------------------ 节点 ---
def router_node(state: GraphState, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """判断是否需要检索。JSON 解析失败时默认 True（宁可多检索一次）。"""
    config = config or load_config()
    question = state["question"]
    result = _llm_json(ROUTER_PROMPT_V1, ROUTER_USER_TEMPLATE.replace("{question}", question), config)

    if result is None or "need_retrieval" not in result:
        need = True  # 兜底：宁可多检索，也不要漏检
        reason = "JSON 解析失败，默认需要检索"
    else:
        need = bool(result["need_retrieval"])
        reason = str(result.get("reason", ""))

    return {
        "need_retrieval": need,
        "trace": [f"router: need_retrieval={need}{'（' + reason + '）' if reason else ''}"],
    }


def retrieve_node(state: GraphState, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """用 current_query（首次等于 question，重写后为新版查询）检索。"""
    config = config or load_config()
    query = state.get("current_query") or state["question"]
    docs = get_retriever(config).invoke(query)
    return {"documents": docs, "trace": [f"retrieve: {len(docs)} docs（query={query[:30]}）"]}


def grade_node(state: GraphState, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """判断检索结果相关性。空结果直接判 irrelevant（不调 LLM，省钱）。"""
    config = config or load_config()
    docs = state.get("documents") or []
    if not docs:
        # 短路：没有检索结果，不用问模型，必然 irrelevant
        return {"relevance": "irrelevant", "trace": ["grade: irrelevant（检索结果为空，短路不调 LLM）"]}

    context = format_docs(docs, config=config)
    question = state["question"]
    user = (GRADE_USER_TEMPLATE
            .replace("{context}", context)
            .replace("{question}", question))
    result = _llm_json(GRADE_PROMPT_V1, user, config)

    if result is None or "relevance" not in result:
        # 兜底默认 relevant：宁可带着资料生成（可能被判不忠实），
        # 也不要因为解析失败就白白拒答一次——拒答的用户体验代价更高
        relevance = "relevant"
        reason = "JSON 解析失败，默认 relevant"
    else:
        relevance = str(result["relevance"]).lower()
        if relevance not in ("relevant", "irrelevant"):
            relevance = "relevant"
        reason = str(result.get("reason", ""))

    return {"relevance": relevance,
            "trace": [f"grade: {relevance}{'（' + reason[:60] + '）' if reason else ''}"]}


def rewrite_node(state: GraphState, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """重写查询并递增 retry_count。"""
    config = config or load_config()
    question = state["question"]
    current = state.get("current_query") or question
    prev_reason = next((t for t in reversed(state.get("trace", [])) if t.startswith("grade:")), "")

    user = (REWRITE_USER_TEMPLATE
            .replace("{question}", question)
            .replace("{current_query}", current)
            .replace("{reason}", prev_reason))

    new_query = current  # 兜底：重写失败就沿用旧查询（外层有 max_retry 兜底）
    if is_mock_mode():
        new_query = f"{question}（重写版）"
    else:
        llm = get_chat_model(temperature=0.0, config=config)
        try:
            resp = llm.invoke([("system", REWRITE_PROMPT_V1), ("human", user)])
            text = str(resp.content).strip()
            if text:
                new_query = text
        except Exception as exc:
            print(f"[警告] 查询重写失败，沿用旧查询: {exc}")

    return {
        "current_query": new_query,
        "retry_count": int(state.get("retry_count", 0)) + 1,
        "trace": [f"rewrite: 新查询={new_query[:40]}"],
    }


def generate_node(state: GraphState, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """生成最终回答；必要时拒答。"""
    config = config or load_config()
    question = state["question"]
    docs = state.get("documents") or []
    need_retrieval = state.get("need_retrieval", True)
    relevance = state.get("relevance", "irrelevant")
    retry_count = int(state.get("retry_count", 0))

    # 情况 1：无需检索 → 直接回答（闲聊/常识）
    if not need_retrieval and not docs:
        if is_mock_mode():
            generation = "[MOCK] 这是无需检索的直接回答（MOCK_LLM 模式）"
        else:
            llm = get_chat_model(temperature=0.3, config=config)
            resp = llm.invoke([
                ("system", DIRECT_ANSWER_PROMPT_V1),
                ("human", DIRECT_ANSWER_USER_TEMPLATE.replace("{question}", question)),
            ])
            generation = str(resp.content)
        return {"generation": generation, "refusal": False, "trace": ["generate: 直接回答（无需检索）"]}

    # 情况 2：检索失败且重试已耗尽 → 诚实拒答
    if not docs or (relevance == "irrelevant" and retry_count >= int(config["agent"]["max_retry"])):
        generation = REFUSAL_TEMPLATE.format(attempts=retry_count + 1)
        return {"generation": generation, "refusal": True,
                "trace": [f"generate: 拒答（relevance={relevance}, retry={retry_count}）"]}

    # 情况 3：有可用资料 → 带引用生成
    context = format_docs(docs, config=config)
    if is_mock_mode():
        generation = f"[MOCK] 依据 {len(docs)} 个片段生成的回答（MOCK_LLM 模式）"
    else:
        from src.agent.prompts import get_rag_prompt
        rag_cfg = config.get("rag", {})
        prompt = get_rag_prompt(rag_cfg.get("prompt_version", "v1"))
        llm = get_chat_model(temperature=float(rag_cfg.get("temperature", 0.1)), config=config)
        resp = (prompt | llm).invoke({"context": context, "question": question})
        generation = str(resp.content)

    return {"generation": generation, "refusal": False,
            "trace": [f"generate: ok（{len(docs)} docs, relevance={relevance}）"]}


# ------------------------------------------------------------------ 路由 ---
def route_after_router(state: GraphState, config: dict[str, Any] | None = None) -> str:
    """router 之后的分支：需要检索 → retrieve；否则直接生成。"""
    return "retrieve" if state.get("need_retrieval", True) else "generate"


def route_after_grade(state: GraphState, config: dict[str, Any] | None = None) -> str:
    """grade 之后的分支：相关 → generate；不相关且未达上限 → rewrite。

    防死循环是**代码级硬保证**：retry_count >= max_retry 时强制走向 generate，
    绝不依赖模型"自觉"停止。
    """
    config = config or load_config()
    max_retry = int(config["agent"].get("max_retry", 1))
    if state.get("relevance") == "relevant":
        return "generate"
    if int(state.get("retry_count", 0)) >= max_retry:
        return "generate"
    return "rewrite"
