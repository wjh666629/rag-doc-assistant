"""Day 1 验证脚本 2/3：验证 DeepSeek API 连通性（OpenAI 兼容接口）。

直接运行：python scripts/test_llm.py
"""
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()  # 提前加载 .env，下面的常量才能读到其中的配置

# DeepSeek 的 OpenAI 兼容接口地址与默认模型（可在 .env 中覆盖为中转平台）
BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def build_llm() -> ChatOpenAI:
    """构建聊天模型客户端，key 只从环境变量读取。"""
    api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[错误] 未找到 DEEPSEEK_API_KEY，请先在 .env 中配置。")
        sys.exit(1)
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=api_key,
        temperature=0,
        max_tokens=64,  # 验证脚本只需短回复，节省 token
    )


@retry(
    stop=stop_after_attempt(3),  # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避：2s/4s/8s
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def ask(llm: ChatOpenAI, question: str) -> str:
    """向 DeepSeek 发送一条消息并返回文本内容。"""
    response = llm.invoke(question)
    return str(response.content)


def main() -> None:
    question: str = "请用一句中文回答：1+1 等于几？"
    print(f"提问: {question}")
    llm = build_llm()
    try:
        answer: str = ask(llm, question)
    except Exception as exc:
        # 打印带上下文的错误信息，便于定位网络/鉴权/配额问题
        print(f"[错误] 调用 DeepSeek API 失败（base_url={BASE_URL}, model={MODEL_NAME}）: "
              f"{type(exc).__name__}: {exc}")
        sys.exit(1)
    print(f"回答: {answer}")
    print("\nLLM 连通性验证通过 ✔")


if __name__ == "__main__":
    main()
