"""Day 1 B3：流式输出（stream=True）演示 —— 打字机效果 + TTFT 统计。

流式为什么能改善用户体验：
    非流式调用要等模型把整段回答生成完才一次性返回，用户面对"转圈"的空等待；
    流式调用让模型每生成几个 token 就推送一个 chunk，用户可以立刻开始阅读，
    感知延迟从"整个回答的生成时间"缩短到"第一个 token 的到达时间"（TTFT）。

生成器模式在这里的体现：
    client.chat.completions.create(stream=True) 返回的是一个迭代器（Stream），
    它不是把所有结果装进列表一次性给你，而是"来一个给一个"地惰性产出 chunk——
    我们用 for 循环逐个消费，内存占用恒定，且消费与生产并行进行。

用法：python scripts/stream_demo.py ["你的问题"]
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 提前加载 .env，下面的常量才能读到其中的配置

# 默认 DeepSeek 官方接口，可在 .env 中改为中转平台地址/模型名
BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def build_client() -> OpenAI:
    """构建客户端，API key 只从环境变量读取。"""
    api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[错误] 请在 .env 中配置 DEEPSEEK_API_KEY，参考 .env.example")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def stream_chat(client: OpenAI, question: str) -> None:
    """流式发送问题，逐 chunk 打印，并统计 TTFT 与总耗时。"""
    start: float = time.perf_counter()
    ttft: float | None = None  # 首 token 延迟（Time To First Token）

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": question}],
            temperature=0.3,
            max_tokens=512,
            stream=True,  # 关键参数：开启流式
        )
        # 消费生成器：每个 chunk 只含增量内容 delta.content
        for chunk in stream:
            if not chunk.choices:
                continue  # 结束标记等特殊 chunk 没有 choices，跳过
            delta = chunk.choices[0].delta
            if delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start  # 记录第一个内容 chunk 的到达时间
                # 不换行连续输出，形成打字机效果；flush 保证立即上屏
                print(delta.content, end="", flush=True)
    except Exception as exc:
        print(f"\n[错误] 流式调用失败（model={MODEL_NAME}）: {type(exc).__name__}: {exc}")
        sys.exit(1)

    total: float = time.perf_counter() - start
    print()  # 回答结束后换行
    print(f"TTFT（首 token 延迟）: {ttft:.2f} 秒" if ttft is not None else "TTFT: 未收到内容")
    print(f"总耗时: {total:.2f} 秒")


def main() -> None:
    if len(sys.argv) > 1:
        question: str = " ".join(sys.argv[1:]).strip()
    else:
        question = "请用中文简要介绍 RAG（检索增强生成）的工作原理。"
    print(f"提问: {question}\n")
    stream_chat(build_client(), question)


if __name__ == "__main__":
    main()
