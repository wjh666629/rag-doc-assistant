# LCEL 到底在链什么

## 一句话答案

LCEL（LangChain Expression Language）链的**不是字符串，而是实现了 Runnable 接口的对象**——把这些对象用 `|` 串起来，形成一条数据自动流动的类型化管道。

## Runnable 接口：管道能成立的地基

链上每一环（PromptTemplate / ChatModel / OutputParser / Retriever / 普通函数）都实现了同一套方法：

- `invoke(x)`：单条输入 → 单条输出；
- `batch([x1, x2, ...])`：批量输入，框架可自动并发；
- `stream(x)`：流式产出，边生成边消费（生成器模式）；
- `ainvoke(x)`：异步版本。

这意味着**组合出来的链自动获得批量、流式、异步能力**，不需要为每个组件单独写一遍。这也是为什么 `prompt | llm | parser` 这种写法不只是语法糖——它是把"调用协议"统一了。

## 管道怎么传数据

- `chain.invoke({"question": "..."})` 传入 dict，key 对应模板里的变量名；
- 上一个 Runnable 的**输出类型**必须能被下一个 Runnable 接受：PromptTemplate 输出 `PromptValue` → ChatModel 接受它并输出 `AIMessage` → `StrOutputParser` 把 `AIMessage` 转成 `str`；
- `RunnablePassthrough` 原样透传输入，RAG 场景用它实现"检索结果喂给 prompt 的同时，原始问题也传下去"：
  `{"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question")} | prompt | llm | parser`。

注意 `{...}` 这里是 `RunnableParallel`：dict 的每个 value 并行执行，结果合并成一个 dict 再交给下一步——所以 LCEL 表达的不只是"串联"，还有"并联"。

## 没有 LCEL 时的样子

不用框架时，每一步都要手工对齐格式：自己拼 prompt 字符串、自己把 `AIMessage` 里的 `.content` 取出来、想批量就自己写线程池、想流式就自己处理 SSE、想观测就自己埋点。用了 LCEL，这些横切能力由框架在 Runnable 层统一提供，业务代码只描述"数据从哪来到哪去"。

## 但 LCEL 不是万能的

它擅长**有向无环**的数据流。一旦需要"条件分支"或"循环"（例如：检索不到 → 重写查询 → 再检索一次），用 `|` 表达就非常别扭——这正是 Day 4 引入 LangGraph 的理由。工程判断力体现在：知道什么时候该用链，什么时候该用图。
