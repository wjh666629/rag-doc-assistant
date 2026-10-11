# 踩坑日志

> 模板：日期 / 主题 → [坑] 现象 → [因] 真实原因 → [解] 解法 + 来源 → [得] 一句话教训
> 每天至少 1 条，Day 7 目标 ≥12 条。

---

## Day 1 · 环境与语料

### 坑 1 · HF 镜像 + xet 协议导致模型文件下载成 0 字节
- **[坑]** `SentenceTransformer("BAAI/bge-small-zh-v1.5")` 报 `JSONDecodeError: Expecting value: line 1 column 1`。
- **[因]** 查缓存目录发现 `modules.json`、`config_sentence_transformers.json` 是 **0 字节空文件**（blob 里的内容却是正常的）。hf-mirror 镜像不支持 huggingface_hub 1.x 默认启用的 xet 加速协议，经镜像下载会拿到空响应。
- **[解]** 设置 `os.environ["HF_HUB_DISABLE_XET"] = "1"`（必须在 import 之前）；最终改为用 curl 直接从 `https://hf-mirror.com/<repo>/resolve/main/<file>` 把 9 个必要文件下载到项目内 `models/bge-small-zh-v1.5/`，脚本优先加载本地目录。
- **[得]** 镜像站的"部分协议不支持"不会报网络错误，只会静默给你空文件——看到 0 字节缓存先怀疑传输协议。

### 坑 2 · huggingface_hub 下载被本机文件安全钩子拦截
- **[坑]** `snapshot_download()` 下载中途失败，stderr 刷满 `safe-delete ... .lock` 相关错误。
- **[因]** hub 在下载过程中会创建并删除 `.lock` 临时文件，本机的安全删除钩子拦截了这些删除操作（trash 失败/需确认），导致下载中断在 85%。
- **[解]** 绕开 hub 的下载逻辑，用 curl 手动下载所需文件到固定目录。
- **[得]** 依赖库的"内部临时文件管理"可能与环境安全策略冲突，卡住时换一条更朴素的路更快。

### 坑 3 · CloseAI 中转平台：base_url 与模型名都对不上官方
- **[坑]** 用 DeepSeek 官方配置跑报 `404 page not found`；补 `/v1` 后报 `model deepseek-chat is no longer available`；换成模型列表里存在的 `deepseek-v3.2` 后仍报 `Model not exist.`。
- **[因]** ① 中转平台要求 OpenAI 兼容路径 `/v1/chat/completions`，裸域名 404；② 平台模型名与官方不同且 `/v1/models` 列表不等于该 key 实际可用模型。
- **[解]** base_url 改 `https://api.openai-proxy.org/v1`；用 curl 逐个试模型，确认 `deepseek-v4-flash` / `deepseek-v4-pro` 可用，写进 `.env` 的 `DEEPSEEK_MODEL`。
- **[得]** 中转站的模型名必须以"实际发一次请求"验证，`/v1/models` 只作参考。

### 坑 4 · 推理模型的流式结束 chunk 没有 choices
- **[坑]** 流式输出正常，但循环末尾抛 `IndexError: list index out of range`。
- **[因]** DeepSeek 推理模型（`deepseek-v4-flash`）流结束时会发一个不含 `choices`（或 choices 为空）的 chunk，直接 `chunk.choices[0]` 越界。
- **[解]** `for chunk in stream: if not chunk.choices: continue`。
- **[得]** 遍历流式响应必须防御"无 choices 的控制帧"，别假设每个 chunk 都有内容。

### 坑 5 · 提示词模板里的 JSON 花括号被 str.format 吃掉
- **[坑]** `structured_output.py` 报 `KeyError: '"summary"'`。
- **[因]** 模板里写着 `{"summary": ...}`，`EXTRACT_PROMPT.format(text=...)` 把 JSON 的字段名当成了占位符。
- **[解]** 含 JSON 示例的模板一律用 `.replace("{text}", text)` 注入变量，不用 format（或把所有 JSON 花括号写成 `{{ }}`）。
- **[得]** prompt 模板里出现 JSON 样例时，`str.format` 是雷区。

---

## Day 2 · 入库管道

### 坑 6 · python.langchain.com 旧文档 URL 全部重定向，抓下来全是重复内容
- **[坑]** 按旧地址抓 30 篇"成功"，但文件正文只有 2 种（md5 去重后 16 篇相同、13 篇相同）。
- **[因]** 官方文档已整站迁移到 `docs.langchain.com`，`python.langchain.com/docs/tutorials/*`、`/docs/concepts/*` 全部 301 到同一个页面，trafilatura 抓到的是重定向后的同一份骨架。
- **[解]** 从 `https://docs.langchain.com/sitemap.xml` 取真实 URL 列表（`/oss/python/langchain/*`），并**在脚本里加正文 md5 去重**，重复内容记为失败。
- **[得]** "抓取成功 30 篇"不等于"拿到 30 篇不同内容"——批量抓取必须做内容级去重校验。

### 坑 7 · trafilatura 丢失 Mintlify 的代码块
- **[坑]** 抓到的 RAG 教程里一个代码块都没有（`grep '```'` 为 0），而 agents 页面却有。
- **[因]** 新文档站是 Mintlify 渲染，代码块由 JS 组件（CodeGroup/ParamField）承载，trafilatura 对这类页面的正文抽取会漏掉代码；且代码块围栏是**缩进的**（`  ```python`），只匹配行首 `^``` 的正则也配对错位。
- **[解]** 发现 Mintlify 支持在 URL 后加 `.md` 直接返回**原生 markdown**（`content-type: text/markdown`），代码块 100% 完整 → `fetch_docs.py` 优先请求 `<url>.md`；同时把围栏正则放宽为 `^[ \t]*``` 并整块保护 `<CodeGroup>...</CodeGroup>`。
- **[得]** 现代文档站先找它的 markdown/llms.txt 原生出口，比解析 HTML 干净一个数量级。

### 坑 8 · 代码块被切碎（切分器的经典坑）
- **[坑]** 抽查 chunk 时出现以 ``` 开头却没有结尾的残段；is_code 标记还误判到纯正文上。
- **[因]** ① 大代码块被按空行二次拆分；② `is_code` 以"是否含占位符"判断，而占位符可能来自不完整的围栏配对。
- **[解]** 大代码块**整体作为一个 chunk 绝不拆分**（代码完整性优先于长度上限）；`is_code` 改为判断"还原后是否真的含 ``` 围栏"。
- **[得]** 切分的第一原则是"宁可 chunk 超长，也不能把代码/句子切断"——断掉的 chunk 对检索和生成都是噪声。

### 坑 9 · 语料里混进 Mintlify 标签噪声，污染检索
- **[坑]** 检索"什么是 LCEL？"的 top1 是 `</ParamField> </Accordion>` 这种纯标签文本。
- **[因]** 原生 markdown 里保留了 Mintlify 的 JSX 组件标签，切分后形成了"只有标签"的垃圾 chunk。
- **[解]** loader 的 `clean_text` 增加"纯标签行"与"纯标签噪声行"正则过滤；重建索引后 chunk 从 959 降到 831。
- **[得]** 清洗规则要针对"抓取源的实际渲染框架"定制，通用规则挡不住站点特有的组件标签。

---

## Day 3 · 最小 RAG

### 坑 10 · 语料与评估问题不匹配，指标全线偏低
- **[坑]** 自检问题「LangGraph 中如何定义 state？」「什么是 LCEL？」全部拒答，检索 top1 是 human-in-the-loop 之类无关文档。
- **[因]** 一开始按"中文 + 英文"的假设排查，实测中英文提问命中率**一样差** → 真正原因是 Day 1 抓的语料是 LangChain **新版 agent 文档**，几乎不含 LCEL / LangGraph state 内容，语料与问题主题不匹配。
- **[解]** 扩抓 30 篇 `docs.langchain.com/oss/python/langgraph/*` 后重建索引（60 篇 / 831→更多 chunk），state 类问题 top1 正确命中 `35_use-graph-api.md`。
- **[得]** 检索指标差时先做"中英对照 + 看 top1 内容"二分定位，别急着归因到语言或模型。

### 坑 11 · 脚本直跑时 import src 失败
- **[坑]** `python scripts/chunk_experiment.py` 报 `ModuleNotFoundError: No module named 'src'`。
- **[因]** 以脚本方式运行时 `sys.path[0]` 是 scripts/ 目录，项目根目录不在搜索路径里；而 `python -m src.xxx` 会把根目录加进去。
- **[解]** 脚本顶部 `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`。
- **[得]** 既要支持 `python -m` 又要支持直接跑脚本时，必须显式补根目录路径。

---

## Day 5 · 评估集与指标

### 坑 12 · `--limit` 截断把越界题全切掉，拒答率恒为 0
- **[坑]** 加 `--limit 20` 跑评估，`refusal_rate` 显示 0.0，看起来像"系统从不拒答"。
- **[因]** golden_qa.json 里人工书写的越界题排在文件末尾（auto 题在前），`items[:20]` 把它们全部切掉了——**样本被截断的方式改变了评估集的难度结构**。
- **[解]** `load_golden()` 改为分层抽样：优先保留全部越界题，再用其他类型补足。
- **[得]** 对评估集做任何抽样都必须保持难度结构（各类别比例），否则指标不可比。

### 坑 13 · LLM 自动出题会产出"指代不明"的废题
- **[坑]** 自动生成的候选题里混进"该教程涵盖哪些主要步骤？""片段中提到的复杂应用示例是什么？"这类问题，脱离原片段根本无法回答，拉低 Hit@5。
- **[因]** LLM 出题时看得到片段，会把"这个/该/上述"写进问题；评估时系统看不到片段，问题就失效了。这是**自动生成评估集的固有偏差**。
- **[解]** 用 `answer_span` 是否能在原文命中做程序化预筛，未通过的进 `golden_todo.json` 等人工处理；越界题与综合题一律人工书写。
- **[得]** 机器出题只能出"片段内事实题"，综合题与越界题必须人工补——这正是人工校验不可替代的地方。

---

## Day 6 · Ragas

### 坑 14 · ragas 0.4.3 的指标类在私有路径，且依赖的 n>1 采样不被中转平台支持
- **[坑]** `from ragas.metrics import Faithfulness` 直接 ImportError；跑起来后又刷 `Invalid n value (currently only n = 1 is supported)`。
- **[因]** ① ragas 0.4.x 把指标类挪到了 `ragas.metrics._Faithfulness` 等下划线私有路径，公开 API 与 0.2/0.3 完全不同；② ragas 内部某些指标会请求一个 prompt 生成 n 份结果，DeepSeek 中转只支持 n=1。
- **[解]** 按 `pip show ragas` 的实际版本写 import（并在 import 失败时给出带版本提示的报错）；指标列名动态捕获而非写死（列名在版本间也变）。n>1 的报错 ragas 内部会自动降级为 1 份继续，不影响主流程，但报告中标注了该警告。
- **[得]** 第三方评估库的版本差异是最大的时间黑洞——动手前先 `pip show` + 打印 `inspect.signature(evaluate)`，比查文档快。

### 坑 15 · 指标列名写死导致指标静默为 0
- **[坑]** `context_recall` 一直是 0.0，但明细里每条都是 None。
- **[因]** ragas 返回的 DataFrame 列名是 `llm_context_recall`，我写死读 `context_recall`，取不到就当 0。
- **[解]** 改为动态捕获 DataFrame 的全部指标列，并按候选列名依次匹配（`avg("llm_context_recall", "context_recall")`）。
- **[得]** 指标取不到时要报错或显式标记 None，**绝不能静默当 0**——静默的 0 会让人得出完全错误的结论。

## Day 7 · 对比实验

### 坑 16 · 改生成 prompt 救不了拒答率：router 短路绕过了整条检索路径
- **[坑]** E3 实验强化 RAG prompt 的拒答措辞，拒答正确率却原地踏步（0.5→0.5）。
- **[因]** 逐题看 trace 才发现：8 条越界题里 5 条在 **router 节点**就被判 `need_retrieval=False`，走"直接回答"分支——拒答机制只在检索路径生效（grade=irrelevant→refuse），router 短路完全绕过了它。改检索路径的 prompt 对这 5 条题零作用。
- **[解]** 消融实验失败后必须**逐题归因**再下结论；本项目的正确优化点是 router prompt（"涉及具体事实/数字/版本一律检索"）或给 direct-answer 分支加拒答约束。
- **[得]** 多节点系统里，"指标没动"往往是因为改动的作用点根本不在失效路径上——先看 trace 定位失效节点，再决定改哪里。

### 坑 17 · LLM-as-judge 单次运行的指标不可比较
- **[坑]** E3 完全没动检索（Hit@5/MRR 与 baseline 逐项相同），context_recall 却从 0.6458"跌"到 0.5625。
- **[因]** Ragas 指标由 LLM 裁判打分，非确定性；有效样本 n≈22 时，运行间噪声达 ±0.08 量级。
- **[解]** prompt 类消融必须跑多次取均值（或固定更稳定的裁判模型）才能比较；检索类指标（Hit@5/MRR 是确定性的）不受影响。
- **[得]** 同理，Day 3 用 10 题 quick 实验得出"chunk 300 最优"，被 Day 7 的 52 题评估推翻——**小样本实验的结论只能当线索，不能当结论**。
