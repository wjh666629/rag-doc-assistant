"""Day 2 B4：带会话记忆的命令行 chatbot（LCEL 版，Day 2 学习练习，不进 src/）。

记忆的本质：LLM API 本身无状态，每次请求都是独立的。
所谓"记忆"，就是**每次请求都把历史 messages 一起发给模型**——
RunnableWithMessageHistory 帮我们做的事只有两件：
1. 请求前：把该 session 的历史 messages 插进 prompt 的 MessagesPlaceholder；
2. 响应后：把本轮的用户消息和 AI 回复追加回历史。
历史越长 token 越多，工程上要做滑动窗口（只留最近 k 轮）或对早期历史做摘要。

运行：python scripts/chatbot_lcel.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
SYSTEM_PROMPT: str = "你是 LangChain 中文助教，回答简洁，适当给出代码示例"
SESSION_ID: str = "cli"  # 固定 session，命令行单用户场景足够


def build_chain() -> RunnableWithMessageHistory:
    """构建带记忆的 LCEL 链。

    管道：prompt | llm | StrOutputParser
    - prompt 里的 MessagesPlaceholder("history") 由包装器自动注入历史消息
    - StrOutputParser 把 AIMessage 转成纯字符串
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[错误] 请在 .env 中配置 DEEPSEEK_API_KEY，参考 .env.example")
        sys.exit(1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    llm = ChatOpenAI(model=MODEL_NAME, base_url=BASE_URL, api_key=api_key, temperature=0.3)
    chain = prompt | llm | StrOutputParser()

    # 会话历史存储：内存字典，key 为 session_id
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def main() -> None:
    with_history = build_chain()
    print("LangChain 中文助教已就绪。输入 exit 退出，输入 /history 查看历史条数。\n")

    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() == "exit":
            print("再见！")
            break
        if question == "/history":
            msgs = _peek_history(with_history)
            print(f"[当前会话消息数: {len(msgs)}]")
            for i, m in enumerate(msgs, start=1):
                kind = "用户" if m.type == "human" else "助手"
                print(f"  {i}. {kind}: {m.content[:80]}")
            continue

        print("助教: ", end="", flush=True)
        try:
            # 流式打印（LCEL 的 stream 能力由 Runnable 接口自动提供）
            for chunk in with_history.stream(
                {"question": question},
                config={"configurable": {"session_id": SESSION_ID}},
            ):
                print(chunk, end="", flush=True)
        except Exception as exc:
            print(f"\n[错误] 调用失败: {type(exc).__name__}: {exc}")
            continue
        print("\n")


def _peek_history(with_history: RunnableWithMessageHistory) -> list:
    """取出当前 session 的历史消息（用于 /history 命令）。"""
    hist = with_history.get_session_history(SESSION_ID)
    return list(hist.messages) if hist else []


if __name__ == "__main__":
    main()
