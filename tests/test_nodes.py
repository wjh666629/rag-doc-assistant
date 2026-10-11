"""Day 4 B3：节点单元测试（mock LLM，不真调 API）。

覆盖：
- router 的 JSON 解析兜底
- grade 的空 documents 短路
- rewrite 的计数递增
- route_after_grade 的上限强制（防死循环）
- generate 的拒答分支
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.agent import nodes
from src.agent.state import GraphState


def base_state(**overrides) -> GraphState:
    """构造一个最小可用 state，便于各用例只改关心的字段。"""
    state: GraphState = {
        "question": "测试问题",
        "current_query": "测试问题",
        "need_retrieval": True,
        "documents": [],
        "relevance": "",
        "retry_count": 0,
        "generation": "",
        "refusal": False,
        "trace": [],
    }
    state.update(overrides)
    return state


def test_router_json_fallback(monkeypatch):
    """JSON 解析失败时 router 必须默认 need_retrieval=True（宁可多检索）。"""
    monkeypatch.setattr(nodes, "_llm_json", lambda *a, **k: None)
    out = nodes.router_node(base_state())
    assert out["need_retrieval"] is True
    assert "trace" in out and out["trace"][0].startswith("router:")


def test_router_normal(monkeypatch):
    monkeypatch.setattr(nodes, "_llm_json",
                        lambda *a, **k: {"need_retrieval": False, "reason": "闲聊"})
    out = nodes.router_node(base_state())
    assert out["need_retrieval"] is False


def test_grade_empty_docs_short_circuit(monkeypatch):
    """空 documents 必须短路判 irrelevant，且不调用 LLM（省钱）。"""
    def boom(*a, **k):
        raise AssertionError("空结果时不应调用 LLM")

    monkeypatch.setattr(nodes, "_llm_json", boom)
    out = nodes.grade_node(base_state(documents=[]))
    assert out["relevance"] == "irrelevant"


def test_grade_json_fallback_defaults_relevant(monkeypatch):
    """解析失败默认 relevant（宁可生成，拒答代价更高）。"""
    monkeypatch.setattr(nodes, "_llm_json", lambda *a, **k: None)
    fake_doc = Document(page_content="假装有一个 chunk", metadata={"source": "x.md"})
    out = nodes.grade_node(base_state(documents=[fake_doc]))
    assert out["relevance"] == "relevant"


def test_rewrite_increments_retry(monkeypatch):
    monkeypatch.setattr(nodes, "is_mock_mode", lambda: True)
    out = nodes.rewrite_node(base_state(retry_count=0))
    assert out["retry_count"] == 1
    assert out["current_query"]


def test_route_after_grade_forces_generate_at_limit(monkeypatch):
    """retry 达到上限时强制走向 generate，防止无限重写循环。"""
    from src.config import load_config
    cfg = load_config()
    max_retry = cfg["agent"]["max_retry"]

    # 未达上限且 irrelevant → rewrite
    assert nodes.route_after_grade(base_state(relevance="irrelevant", retry_count=0), cfg) == "rewrite"
    # 达到上限且 irrelevant → 强制 generate
    assert nodes.route_after_grade(
        base_state(relevance="irrelevant", retry_count=max_retry), cfg) == "generate"
    # relevant → 无论 retry 多少都走 generate
    assert nodes.route_after_grade(
        base_state(relevance="relevant", retry_count=max_retry), cfg) == "generate"


def test_generate_refusal_when_exhausted(monkeypatch):
    """检索失败 + 重试耗尽 → 拒答话术 + refusal=True。"""
    out = nodes.generate_node(base_state(documents=[], relevance="irrelevant", retry_count=1))
    assert out["refusal"] is True
    assert "无法回答" in out["generation"]


def test_generate_direct_answer_when_no_retrieval(monkeypatch):
    """无需检索时走直答分支，不拒答。"""
    monkeypatch.setattr(nodes, "is_mock_mode", lambda: True)
    out = nodes.generate_node(base_state(need_retrieval=False, documents=[]))
    assert out["refusal"] is False
    assert out["generation"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
