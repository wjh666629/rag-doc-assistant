# LangGraph State 与 reducer：一个让我调试半小时的机制

## 现象

图跑起来后，`trace` 里的内容越来越多，而 `documents` 在重写查询后却"莫名其妙只剩最新一轮的结果"。我一度以为 retrieve 节点有 bug。

## 真实原因

LangGraph 的 state 字段有两种更新语义：

- **默认（覆盖式）**：节点返回什么，state 里就变成什么。旧值消失。
- **`Annotated[list, add]`（追加式 reducer）**：节点返回的值被 `add` 到已有列表尾部。

我的 `GraphState` 里 `documents` 没标 reducer（覆盖式），`trace` 标了 `add`（追加式）。两者行为不同是**设计如此**，不是 bug。

## 为什么这样设计

- `documents`：重写查询后重新检索，旧检索结果已经失效。若追加，生成节点会同时看到两轮不同查询的片段，容易生成自相矛盾的答案。所以必须覆盖。
- `trace`：记录执行路径，历史天然应该累积，用来回答"这次走了哪条路"，覆盖掉就看不到循环次数了。

## 一句话教训

调试 LangGraph 时，先确认字段是覆盖式还是追加式——`Annotated[X, add]` 决定了"节点返回值"和"state 里的值"是不是一回事。官方文档把 reducer 放在进阶章节，但它是理解状态流转的第一课。
