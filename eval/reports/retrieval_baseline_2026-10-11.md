# 检索基线评估（2026-10-11）

- 样本数（进分母）: 52
- 越界样本数: 8

## 指标总表

| 指标 | 值 |
|---|---|
| Hit@5 | 0.5769 |
| MRR | 0.4427 |
| 越界组平均 top1 分数 | 0.3266 |
| 正常组平均 top1 分数 | 0.434 |

## 分类型指标

| 类型 | 样本数 | Hit@5 | MRR |
|---|---|---|---|
| 简单事实 | 44 | 0.6136 | 0.4815 |
| 综合 | 8 | 0.375 | 0.2292 |

## 未命中清单（人工归因待填）

| 问题 | golden 来源 | 实际 top1 | top1 分数 | 人工归因 |
|---|---|---|---|---|
| 如何将文档连接到所选的智能体？ | 03_install.md | 01_quickstart.md | 0.351 |  |
| LangChain 推荐使用什么方式来实现实时流式更新？ | 08_streaming.md | 03_install.md | 0.4669 |  |
| 在使用agent时如何实现结构化输出？ | 09_structured-output.md | 04_agents.md | 0.5245 |  |
| 片段中提到的复杂应用示例是什么？ | 12_knowledge-base.md | 29_multi-agent.md | 0.3666 |  |
| 该教程涵盖哪些主要步骤？ | 12_knowledge-base.md | 51_thinking-in-langgraph.md | 0.377 |  |
| 如果代理生成的SQL命令有错误，会发生什么？ | 13_sql-agent.md | 06_messages.md | 0.4556 |  |
| 这种提供反馈的模式有什么作用？ | 13_sql-agent.md | 05_models.md | 0.3016 |  |
| 中间件可以用于哪些场景？ | 15_overview.md | 27_event-streaming.md | 0.1967 |  |
| 在LangGraph工作流中使用中间件时，中间件是作为独立运行时运行的吗？ | 15_overview.md | 54_pregel.md | 0.3644 |  |
| 如何为摘要功能指定自定义提示模板？ | 16_built-in.md | 05_models.md | 0.3797 |  |
| 当非reducer状态字段发生冲突时，哪个中间件的值优先？ | 17_custom.md | 54_pregel.md | 0.3806 |  |
| 执行暂停时图状态是如何保存的？ | 20_human-in-the-loop.md | 46_use-time-travel.md | 0.32 |  |
| 为什么Agentic applications更倾向于集成？ | 23_test.md | 05_models.md | 0.4741 |  |
| `strict` 和 `unordered` 模式在工具调用顺序上有何区别？ | 24_evals.md | 07_tools.md | 0.3807 |  |
| 根据文档片段，Models 类别的主要用途是什么？ | 28_component-architecture.md | 05_models.md | 0.4897 |  |
| 根据表格，LangChain 的 Memory 组件的主要功能是什么？ | 28_component-architecture.md | 01_quickstart.md | 0.4814 |  |
| 该文档片段的主要内容是什么？ | 30_subagents.md | 05_models.md | 0.3361 |  |
| 如何给 LangGraph 加上跨会话的长期记忆？ | 38_persistence.md,39_checkpointers.md,40_add-memory.md,41_stores.md | 51_thinking-in-langgraph.md | 0.4275 |  |
| 子图（subgraph）如何与父图共享状态？ | 37_use-subgraphs.md | 38_use-subgraphs.md | 0.3458 |  |
| 流式输出在 LangGraph 里有哪几种粒度？ | 26_streaming.md,42_streaming.md,43_event-streaming.md | 54_pregel.md | 0.3692 |  |
| LangGraph 里如何做人工介入（human-in-the-loop）？ | 44_interrupts.md,46_use-time-travel.md | 31_overview.md | 0.4497 |  |
| 持久化（persistence）和容错（fault tolerance）在 LangGraph 中是 | 38_persistence.md,51_fault-tolerance.md,39_checkpointers.md | 40_checkpointers.md | 0.4242 |  |

## 结论
（人工补充）
