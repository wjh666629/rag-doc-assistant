"""Day 1 B2：单轮问答 —— 用 openai 官方 SDK 调用 DeepSeek。

用法：
    python scripts/hello_llm.py "什么是 RAG？"
    python scripts/hello_llm.py          # 运行后在提示符下输入问题
"""
import os
import sys

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()  # 提前加载 .env，下面的常量才能读到其中的配置

# OpenAI 兼容接口地址与模型名：默认 DeepSeek 官方，可在 .env 中改为中转平台
# （如 CloseAI 中转：DEEPSEEK_BASE_URL=https://api.openai-proxy.org）
BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
# system 提示词：限制回答风格与长度，控制成本
SYSTEM_PROMPT: str = "你是一个简洁的技术助手，回答控制在 200 字以内"


def build_client() -> OpenAI:
    """构建客户端，API key 只从环境变量读取。"""
    api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[错误] 请在 .env 中配置 DEEPSEEK_API_KEY，参考 .env.example")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=BASE_URL)


@retry(
    stop=stop_after_attempt(3),  # 最多 3 次（含首次）
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避 2s/4s/8s
    # 只对流式可恢复的瞬态错误重试：限流与超时
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    reraise=True,
)
def chat(client: OpenAI, question: str):
    """发送单轮问答请求，返回 ChatCompletion 对象。"""
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.3,  # 低随机性，技术问答更稳定
        max_tokens=512,
    )


def get_question() -> str:
    """优先从命令行参数取问题，否则交互式输入。"""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    question: str = input("请输入你的问题：").strip()
    if not question:
        print("[错误] 问题不能为空")
        sys.exit(1)
    return question


def main() -> None:
    question: str = get_question()
    client = build_client()
    try:
        completion = chat(client, question)
    except (RateLimitError, APITimeoutError) as exc:
        print(f"[错误] 重试 3 次后仍失败（限流/超时）: {type(exc).__name__}: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[错误] 调用 DeepSeek API 失败（base_url={BASE_URL}, model={MODEL_NAME}）: "
              f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    answer: str = completion.choices[0].message.content or ""
    usage = completion.usage
    print(f"回答:\n{answer}\n")
    # 打印 token 用量，建立成本直觉（DeepSeek 按 token 计费）
    if usage:
        print(f"Token 用量: prompt={usage.prompt_tokens}, "
              f"completion={usage.completion_tokens}, total={usage.total_tokens}")


if __name__ == "__main__":
    main()
