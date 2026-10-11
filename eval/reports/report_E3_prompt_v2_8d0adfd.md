# Eval Report @8d0adfd

生成时间: 2026-10-11T23:28:21.031797 ｜ 样本数: 30

## 指标对比

| 指标 | 本版 | 上版 | Δ |
|---|---|---|---|
| Hit@5 | 0.5769 | 0.5577 | +0.0192 |
| MRR | 0.4427 | 0.3781 | +0.0646 |
| Faithfulness | 0.5975 | 0.697 | -0.0995 |
| Answer Relevancy | 0.5692 | 0.5583 | +0.0109 |
| Context Recall | 0.5625 | 0.4271 | +0.1354 |
| 拒答正确率 | 0.5 | 0.375 | +0.1250 |
| 平均延迟(s) | 15.54 | 15.78 | -0.2400 |
| p95 延迟(s) | 29.25 | 33.89 | -4.6400 |
| 样本数 | 30 | 30 | +0.0000 |

## 变更说明

RAG prompt v1→v2（强化拒答约束措辞；假设：拒答正确率上升，faithfulness 上升）

## 分类型检索指标

| 类型 | 样本数 | Hit@5 | MRR |
|---|---|---|---|
| 简单事实 | 44 | 0.6136 | 0.4815 |
| 综合 | 8 | 0.375 | 0.2292 |

## 低分样本 Top 5
- 如何用 LangGraph 做红烧肉的菜谱推荐？（越界）｜trace: router: need_retrieval=True（问题涉及如何使用 LangGraph 构建菜谱推荐应用，属于 LangGraph 的具体技术实现与 API 用法，需要检索相关技术文档。） → retrieve: 5 docs（query=如何用 LangGraph 做红烧肉的菜谱推荐？） → grade: irrelevant（参考资料全部是关于 LangGraph 的技术文档，包括 Overwrite 类型、reducer 结合性、Pregel）
- 这份语料里有没有讲 Kubernetes 集群运维？（越界）｜trace: router: need_retrieval=True（用户询问语料中是否包含 Kubernetes 集群运维内容，需要检索技术文档以确认。） → retrieve: 5 docs（query=这份语料里有没有讲 Kubernetes 集群运维？） → grade: irrelevant（所有片段均围绕 LangChain/LangGraph 的技术内容（如流式输出、Functional API、模型调用、）
- BAAI/bge-small-zh-v1.5 模型的参数量是多少？（越界）｜trace: router: need_retrieval=True（问题是关于具体模型 BAAI/bge-small-zh-v1.5 的参数量，属于具体技术事实查询，需要检索模型文档或相关技术资料才能准确回答。） → retrieve: 5 docs（query=BAAI/bge-small-zh-v1.5 模型的参数量是） → grade: irrelevant（所有片段均来自 LangChain 相关文档，分别讲述重试参数配置、消息输出格式、多模态、模型类列表和集成测试建议，没有）
- LangGraph 支持哪些编程语言的官方 SDK（除 Python/JS 外）？（越界）｜trace: router: need_retrieval=True（问题涉及 LangGraph 官方 SDK 支持的具体编程语言，属于技术文档中的具体事实信息，需要检索确认。） → retrieve: 5 docs（query=LangGraph 支持哪些编程语言的官方 SDK（除 Py） → grade: irrelevant（参考资料仅提及 Python 和 TypeScript/JavaScript 的文档链接与 API 说明，未提供除 Py）
- 如果没有任何已设置的支持的 provider API key，应该怎么做？（简单事实）｜trace: router: need_retrieval=True（问题涉及缺少支持的 provider API key 时的配置或处理方式，属于具体技术配置场景，需要查阅相关技术文档。） → retrieve: 5 docs（query=如果没有任何已设置的支持的 provider API key） → grade: relevant（片段[1]明确指出：如果没有设置任何支持的 provider API key，应询问用户要使用哪个 provider，然）

---

对比实验请修改 configs/ 后重跑本命令并加 --note
