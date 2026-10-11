# chunk_size / top_k 参数实验（2026-10-11）

> 说明：本实验只跑检索，不调 LLM；每组参数使用独立 collection，不污染主库。

| chunk_size | top_k | chunk 总数 | 平均 chunk 长度 | 平均 top1 相关度 |
|---|---|---|---|---|
| 300 | 3 | 3183 | 581.1 | 0.5293 |
| 300 | 5 | 3183 | 581.1 | 0.5293 |
| 500 | 3 | 1777 | 1080.1 | 0.5151 |
| 500 | 5 | 1777 | 1080.1 | 0.5151 |
| 1000 | 3 | 824 | 2268.0 | 0.4795 |
| 1000 | 5 | 824 | 2268.0 | 0.4795 |

## 逐题明细（最优参数组）

最优组：chunk_size=300, top_k=3 （平均 top1 相关度 0.5293）

| 问题 | 类型 | top1 来源 | top1 分数 |
|---|---|---|---|
| LangGraph 中如何定义 state？ | 简单事实 | 34_graph-api.md | 0.5553 |
| What is a checkpointer used for in LangGraph? | 简单事实 | 40_checkpointers.md | 0.7621 |
| 如何中断一个 LangGraph 的执行并等待人工输入？ | 简单事实 | 54_pregel.md | 0.5167 |
| How to stream token-by-token output from a LangGraph agent? | 简单事实 | 43_streaming.md | 0.727 |
| create_agent 的 middleware 有哪些典型用法？ | 综合 | 15_overview.md | 0.6594 |
| LangGraph 的 Graph API 和 Functional API 分别适合什么场景？ | 综合 | 36_functional-api.md | 0.6671 |
| 子图（subgraph）如何与父图共享状态？ | 综合 | 38_use-subgraphs.md | 0.3541 |
| DeepSeek 模型的训练数据规模是多少？ | 越界 | 05_models.md | 0.335 |
| 如何用 LangGraph 做红烧肉的菜谱推荐？ | 越界 | 35_use-graph-api.md | 0.3429 |
| LangChain 2028 年的发布路线图是什么？ | 越界 | 40_checkpointers.md | 0.3729 |

## 结论
（人工补充：哪组参数最好，为什么）
