# Eval Report @8d0adfd

生成时间: 2026-10-11T23:11:57.065647 ｜ 样本数: 30

## 指标对比

| 指标 | 本版 | 上版 | Δ |
|---|---|---|---|
| Hit@5 | 0.5577 | 0.5769 | -0.0192 |
| MRR | 0.3781 | 0.4427 | -0.0646 |
| Faithfulness | 0.697 | 0.8514 | -0.1544 |
| Answer Relevancy | 0.5583 | 0.7781 | -0.2198 |
| Context Recall | 0.4271 | 0.6458 | -0.2187 |
| 拒答正确率 | 0.375 | 0.5 | -0.1250 |
| 平均延迟(s) | 15.78 | 18.16 | -2.3800 |
| p95 延迟(s) | 33.89 | 35.69 | -1.8000 |
| 样本数 | 30 | 30 | +0.0000 |

## 变更说明

chunk_size 500→300, overlap 45（假设：简单事实类 Hit@5 上升，综合类下降）

## 分类型检索指标

| 类型 | 样本数 | Hit@5 | MRR |
|---|---|---|---|
| 简单事实 | 44 | 0.6136 | 0.4014 |
| 综合 | 8 | 0.25 | 0.25 |

## 低分样本 Top 5
- 如何用 LangGraph 做红烧肉的菜谱推荐？（越界）｜trace: router: need_retrieval=True（问题涉及如何使用 LangGraph 构建菜谱推荐应用，属于框架/技术用法问题，需要参考 LangGraph 相关文档。） → retrieve: 5 docs（query=如何用 LangGraph 做红烧肉的菜谱推荐？） → grade: irrelevant（参考资料仅涉及 LangGraph 的 Graph API、reducer 覆盖和 checkpointer 等技术概念）
- 这份语料里有没有讲 Kubernetes 集群运维？（越界）｜trace: router: need_retrieval=True（用户询问当前语料中是否包含 Kubernetes 集群运维内容，需要检索文档库才能确认。） → retrieve: 5 docs（query=这份语料里有没有讲 Kubernetes 集群运维？） → grade: irrelevant（所有片段均来自 LangGraph/LangSmith 相关文档（interrupt、streaming、workflo）
- LangGraph 支持哪些编程语言的官方 SDK（除 Python/JS 外）？（越界）｜trace: router: need_retrieval=True（该问题涉及 LangGraph 官方 SDK 支持的具体编程语言，属于需要查阅技术文档才能准确回答的技术事实问题。） → retrieve: 5 docs（query=LangGraph 支持哪些编程语言的官方 SDK（除 Py） → grade: irrelevant（所有片段仅提及 Python 和 TypeScript/JavaScript 的文档链接，没有任何内容说明 LangGr）
- 如果没有任何已设置的支持的 provider API key，应该怎么做？（简单事实）｜trace: router: need_retrieval=True（问题涉及未设置支持的 provider API key 时的处理方式，属于具体技术配置与使用问题，需要查阅 LangChain/LangGraph 相关文档。） → retrieve: 5 docs（query=如果没有任何已设置的支持的 provider API key） → grade: relevant（片段[1]明确说明：如果没有任何支持的provider API key已设置，应询问用户使用哪个provider，然后停）
- 在快速入门中，第一步需要做什么？（简单事实）｜trace: router: need_retrieval=True（问题询问快速入门文档中的第一步，属于具体文档内容，需要检索 LangChain/LangGraph 技术文档才能准确回答。） → retrieve: 5 docs（query=在快速入门中，第一步需要做什么？） → grade: relevant（参考资料[1]中明确包含“## Step 1: Read the guide”，直接回答了快速入门的第一步需要做什么。）

---

对比实验请修改 configs/ 后重跑本命令并加 --note
