"""提示词集中管理（Day 3 起，Day 4 扩充路由/打分/重写）。

约定：
1. 所有 prompt 放这里，版本号写在顶部注释，config 用 prompt_version 选择；
2. 含 JSON 示例的模板用 .replace() 注入变量，**不要**用 str.format
   （JSON 花括号会被当成占位符，见踩坑日志坑 5）。
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------- v1 · Day 3 ---------
# RAG_PROMPT v1：强约束 + 引用标注 + 拒答话术
RAG_SYSTEM_V1: str = """你是严谨的中文技术文档助手。规则：
1. 仅依据"参考资料"回答，禁止使用你自己的知识补充；
2. 参考资料不足以回答时，明确说"根据现有资料无法回答该问题"，不要硬答；
3. 回答中引用资料时用 [1] [2] 标注，编号对应资料的 [n] 编号；
4. 回答简洁，技术内容适当给出代码片段。"""

RAG_USER_TEMPLATE: str = """参考资料：
{context}

问题：{question}"""

RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_V1),
    ("human", RAG_USER_TEMPLATE),
])

# ------------------------------------------------- v1 · Day 4 ---------
# ROUTER_PROMPT v1：判断是否需要检索文档
ROUTER_PROMPT_V1: str = """你是问答系统的路由器。判断用户问题是否需要检索技术文档才能回答。

判断规则：
- 需要检索（true）：涉及具体技术概念、API 用法、配置项、代码写法、框架对比等问题；
- 不需要检索（false）：问候寒暄、纯闲聊、与 LangChain/LangGraph 技术文档无关的常识问题。

只输出 JSON，格式：{"need_retrieval": true/false, "reason": "..."}"""

ROUTER_USER_TEMPLATE: str = "用户问题：{question}"

# GRADE_PROMPT v1：判断检索结果是否足以回答问题
GRADE_PROMPT_V1: str = """你是检索质量评判员。判断"参考资料"是否包含回答"用户问题"所需的关键信息。

规则：
- 只要有一个片段包含回答问题所需的关键信息，就判 relevant；
- 若所有片段只是主题沾边、没有实质答案，判 irrelevant。

只输出 JSON，格式：{"relevance": "relevant"/"irrelevant", "reason": "..."}"""

GRADE_USER_TEMPLATE: str = """参考资料：
{context}

用户问题：{question}"""

# REWRITE_PROMPT v1：查询重写（让检索更可能命中）
REWRITE_PROMPT_V1: str = """你是查询优化助手。上一次检索没有找到能回答问题的资料，请重写检索查询。

重写策略：补充同义术语、把代词还原成具体名词、拆出核心技术关键词、或改用更贴近文档用词的说法。
只输出新的查询本身，不要解释、不要加引号。"""

REWRITE_USER_TEMPLATE: str = """原始问题：{question}
上一版查询：{current_query}
检索失败原因：{reason}

新查询："""

# DIRECT_ANSWER_PROMPT v1：无需检索时的直接回答
DIRECT_ANSWER_PROMPT_V1: str = """你是友好的中文技术助手。用户的问题不需要查阅文档，
请简洁作答（200 字以内）。若问题与技术无关，正常聊天即可。"""

DIRECT_ANSWER_USER_TEMPLATE: str = "用户问题：{question}"

# 拒答话术（generate_node 在检索失败且重试耗尽时使用）
REFUSAL_TEMPLATE: str = ("根据现有资料无法回答该问题。"
                         "（已尝试 {attempts} 次检索与查询重写，仍未找到相关资料。）"
                         "建议：换个更具体的问法，或补充相关文档到语料库。")

# ------------------------------------------------- v2 · Day 7 ---------
# RAG_PROMPT v2（E3 实验变量）：仅强化"拒答"约束的措辞与判定标准，
# 其他与 v1 完全一致 —— 保证实验是一次一变量。
RAG_SYSTEM_V2: str = """你是严谨的中文技术文档助手。必须严格遵守以下规则：
1. 你的回答只能来自"参考资料"，禁止使用任何你自己的知识、经验或推测来补充；
2. 判定标准（重要）：只有当参考资料中**明确写出**了问题的答案时才作答；
   若资料只是主题相关、提到类似概念、或需要你推断才能得出，一律视为资料不足；
3. 资料不足时，只回答"根据现有资料无法回答该问题"，不要给出任何猜测性内容，
   也不要说"可能是因为""我推测"之类的话；
4. 作答时用 [1] [2] 标注引用，编号对应资料的 [n] 编号；
5. 回答简洁，技术内容适当给出代码片段。"""

RAG_PROMPT_V2: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_V2),
    ("human", RAG_USER_TEMPLATE),
])

PROMPT_VERSIONS: dict[str, ChatPromptTemplate] = {"v1": RAG_PROMPT, "v2": RAG_PROMPT_V2}


def get_rag_prompt(version: str = "v1") -> ChatPromptTemplate:
    """按版本号取 RAG prompt（未知版本回退 v1，避免线上崩）。"""
    return PROMPT_VERSIONS.get(version, RAG_PROMPT)
