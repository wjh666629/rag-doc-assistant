"""Day 1 B4：结构化输出 —— 让 LLM 返回严格 JSON 并解析为 Python 对象。

为什么结构化输出是后续评估模块（LLM-as-judge）的基础：
    评估模块要让 LLM 对 RAG 的回答打分、给出理由，程序必须能"稳定地"把模型的
    输出解析成分数和字段，而不是一段自由文本。如果模型输出的 JSON 都无法可靠
    解析，那么 faithfulness、answer relevancy 这类自动评估指标就无从谈起。
    本脚本演练的"json_object 模式 + 解析失败重试"正是 LLM-as-judge 的最小闭环。

直接运行：python scripts/structured_output.py
"""
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from openai import APITimeoutError, RateLimitError

load_dotenv()  # 提前加载 .env，下面的常量才能读到其中的配置

# 默认 DeepSeek 官方接口，可在 .env 中改为中转平台地址/模型名
BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 内置两段技术文档示例文本
SAMPLE_TEXTS: list[str] = [
    "LangGraph 是 LangChain 生态中的 Agent 编排框架，它把智能体的工作流建模为"
    "有向图：节点代表一次 LLM 调用或工具执行，边代表状态流转，并且原生支持循环，"
    "因此可以实现反思、重试、多智能体协作等带状态的复杂逻辑。",

    "RAG（检索增强生成）先把文档切成块并计算向量存入向量数据库；用户提问时，"
    "系统检索最相关的若干文档块，连同问题一起塞进提示词交给大模型生成回答，"
    "从而缓解模型的知识截止与幻觉问题。",
]

EXTRACT_PROMPT: str = """请从下面的技术文档段落中抽取信息，并严格输出一个 JSON 对象，格式：
{"summary": "一句话摘要", "keywords": ["关键词1", "关键词2"], "difficulty": "入门|中级|高级"}
要求：只输出 JSON，不要输出任何其他文字。difficulty 只能是"入门""中级""高级"之一。

文档段落：
{text}"""


def build_client() -> OpenAI:
    """构建客户端，API key 只从环境变量读取。"""
    api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[错误] 请在 .env 中配置 DEEPSEEK_API_KEY，参考 .env.example")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=BASE_URL)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    reraise=True,
)
def call_llm(client: OpenAI, prompt: str) -> str:
    """调用模型并返回文本内容（JSON 模式）。"""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # 抽取任务要求确定性，关掉随机性
        max_tokens=512,
        response_format={"type": "json_object"},  # 强制 JSON 输出
    )
    return completion.choices[0].message.content or ""


def extract(client: OpenAI, text: str) -> dict:
    """对一段文本做结构化抽取；JSON 解析失败时换更强的提示词重试一次。"""
    # 注意：提示词模板含 JSON 花括号，不能用 str.format（会把 JSON 字段名当占位符），
    # 只能用 replace 注入文档文本
    prompt: str = EXTRACT_PROMPT.replace("{text}", text)
    raw: str = call_llm(client, prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 重试一次：在提示词中追加上次失败的反馈
        print("[提示] 首次输出不是合法 JSON，追加提示后重试一次")
        retry_prompt: str = prompt + "\n\n上次输出不是合法 JSON，请只输出 JSON。"
        raw = call_llm(client, retry_prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[错误] 重试后仍无法解析 JSON，原始输出: {raw[:200]}")
            raise RuntimeError(f"结构化抽取失败: {exc}") from exc


def main() -> None:
    client = build_client()
    for i, text in enumerate(SAMPLE_TEXTS, start=1):
        print(f"--- 文档 {i} ---")
        print(f"原文: {text[:40]}...")
        try:
            result: dict = extract(client, text)
        except Exception as exc:
            print(f"[错误] 文档 {i} 抽取失败: {type(exc).__name__}: {exc}")
            sys.exit(1)
        # 打印解析后的 Python 对象及各字段类型，确认字段完整
        print(f"解析结果: {result}")
        print(f"对象类型: {type(result).__name__}; "
              f"summary={type(result.get('summary')).__name__}, "
              f"keywords={type(result.get('keywords')).__name__}, "
              f"difficulty={type(result.get('difficulty')).__name__}")
        # 校验必需字段，字段缺失视为失败
        missing = {"summary", "keywords", "difficulty"} - result.keys()
        if missing:
            print(f"[错误] 输出缺少字段: {missing}")
            sys.exit(1)
        print()
    print("结构化输出验证通过 ✔")


if __name__ == "__main__":
    main()
