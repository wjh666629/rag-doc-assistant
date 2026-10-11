# CRAG 图执行路径记录（Day 4）

`python -m src.agent.graph --selftest` 的实测路径（模型 deepseek-v4-flash，max_retry=1）。

## a. 正常文档问题：「LangGraph 中如何定义 state？」

```
router: need_retrieval=True（涉及框架 API 用法）
retrieve: 5 docs
grade: relevant（资料[1]明确说明 state 可用 TypedDict/Pydantic/dataclass）
generate: ok（5 docs）
```
耗时 25.85 秒。回答正确引用了 `35_use-graph-api.md`。

## b. 越界问题：「DeepSeek 模型的训练数据规模是多少？」

```
router: need_retrieval=True
retrieve: 5 docs
grade: irrelevant（资料讲的是模型配置/重试/记忆，与 DeepSeek 训练数据无关）
rewrite: 新查询=「DeepSeek 大语言模型预训练数据集规模多少 token」
retrieve: 5 docs
grade: irrelevant
generate: 拒答（relevance=irrelevant, retry=1）
```
耗时 12.47 秒。**rewrite 节点确实被触发，且循环在第 1 次后由 max_retry 硬性截断**——这正是图相对 LCEL 直线链的价值所在。

## c. 纯闲聊：「你好」

```
router: need_retrieval=False
generate: 直接回答
```
未进入检索分支，省下一次向量检索与一次 LLM 调用。

## d. 模糊问题：「那个东西怎么用？」

```
router: need_retrieval=True（模型倾向保守：判定需要检索）
retrieve: 5 docs
grade: 视检索结果而定（实测多判 irrelevant → rewrite 一轮后拒答）
```
人工判断：模糊问题缺乏指代对象，路由层无法判断"那个东西"是什么，
最终诚实拒答属于合理行为；若要改善，应做**追问澄清**（而非重写查询），
这是已知的改进点（见 README 已知缺陷）。

---

## 三个自测问题的回答

**Q1：为什么 documents 用覆盖式而 trace 用追加式 reducer？**
documents 是"当前这一轮检索的结果"，查询重写后旧结果已经失效，
如果追加就会把两轮不同查询的片段混在一起，生成节点会看到互相矛盾的材料。
trace 是"走过的路径"，历史天然应该累积，覆盖掉就看不出循环次数了。
判断标准很简单：**这个字段的旧值在新一轮里还有没有意义？**

**Q2：如果去掉 max_retry 限制会发生什么？**
grade 判 irrelevant 就会无限 rewrite→retrieve 循环，直到超时或 token 烧完。
而且重写本身是 LLM 调用，循环里每次都要付钱。**循环上限必须在代码层硬保证**，
不能指望模型"自觉"判断"我已经试够了"——模型没有全局计数器的概念。

**Q3：grade 误判的代价是什么？**
- 误判为 relevant（漏检失败）：带着不相关资料去生成 → 幻觉风险上升，
  但 prompt 的强约束会兜一层"资料不足就拒答"，代价中等；
- 误判为 irrelevant（误检失败）：多花一轮 rewrite + retrieve（约 10 秒 + 两次 LLM），
  最终仍可能进入 generate，代价是延迟与成本，但不会错答。
本项目选择**宁可判 relevant**（解析失败时默认 relevant），因为拒答对用户是"什么都没得到"，
比"多等 10 秒"更贵。这是典型的误检/漏检权衡，取决于业务对"错答"和"慢"的容忍度。
