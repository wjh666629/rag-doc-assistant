"""Day 4 B1：图的状态定义。

关键机制（LangGraph 最容易踩的坑）：
- 默认情况下，节点返回的字段会**覆盖** state 中的同名值；
- 用 `Annotated[list, add]` 标注的字段是**追加式（reducer）**，
  节点返回的内容会被 add 到已有列表尾部，而不是替换。

本项目中 documents 故意用覆盖式：查询重写后重检，旧结果必须丢弃，
否则新旧检索结果混在一起会让生成节点看到矛盾材料；
trace 用追加式：它记录"走过的路径"，历史不应被覆盖。
"""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """CRAG 图的共享状态。

    字段生命周期说明（谁写 → 谁读）：
    - question：入口写入，全程只读（用于最终展示与评估对账）
    - current_query：入口初始化为 question；rewrite_node 修改；retrieve_node 读取
    - need_retrieval：router_node 写入；route_after_router 读取决定分支
    - documents：retrieve_node 覆盖写入；grade_node / generate_node 读取
    - relevance：grade_node 写入；route_after_grade 读取决定分支
    - retry_count：rewrite_node 递增；route_after_grade 读取（防死循环硬上限）
    - generation：generate_node 写入，终端输出
    - refusal：generate_node 写入（True 表示拒答），UI 用于显示 ⚠️
    - trace：每个节点追加一行，最后打印执行路径
    """

    question: str
    current_query: str
    need_retrieval: bool
    documents: list[Document]        # 覆盖式：重检后丢弃旧结果
    relevance: str                   # "relevant" | "irrelevant"
    retry_count: int
    generation: str
    refusal: bool
    trace: Annotated[list[str], add]  # 追加式：记录完整执行路径
