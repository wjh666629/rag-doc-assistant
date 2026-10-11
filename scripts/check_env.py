"""Day 1 验证脚本 1/3：检查环境变量与依赖是否正确配置。

直接运行：python scripts/check_env.py
"""
import os
import sys

from dotenv import load_dotenv


def check_env() -> None:
    """加载 .env 并检查关键环境变量是否存在。"""
    load_dotenv()  # 从项目根目录的 .env 读取环境变量

    # 检查 DeepSeek API key（只验证存在性，不打印明文）
    api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[错误] 未找到 DEEPSEEK_API_KEY。")
        print("       请复制 .env.example 为 .env，并填入你的 DeepSeek API key。")
        sys.exit(1)
    if api_key == "sk-xxxxxxxxxxxxxxxxxxxxxxxx":
        print("[错误] DEEPSEEK_API_KEY 仍是 .env.example 中的占位符，请替换为真实 key。")
        sys.exit(1)
    print(f"[OK] DEEPSEEK_API_KEY 已加载（长度 {len(api_key)}，已隐藏明文）")

    # HF_ENDPOINT 可选，未设置时给出提示
    hf_endpoint: str | None = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        print(f"[OK] HF_ENDPOINT = {hf_endpoint}")
    else:
        print("[提示] 未设置 HF_ENDPOINT，首次下载模型可能较慢；"
              "建议在 .env 中加 HF_ENDPOINT=https://hf-mirror.com")


def check_python_version() -> None:
    """确认 Python 版本满足 >= 3.10（脚本使用了 3.10+ 的类型注解语法）。"""
    major, minor = sys.version_info[:2]
    print(f"[OK] Python 版本: {major}.{minor}")
    if (major, minor) < (3, 10):
        print(f"[错误] 需要 Python >= 3.10，当前为 {major}.{minor}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        check_python_version()
        check_env()
        print("\n环境检查通过 ✔")
    except SystemExit:
        raise
    except Exception as exc:  # 兜底：打印带上下文的错误信息
        print(f"[错误] 环境检查过程中发生未预期异常: {type(exc).__name__}: {exc}")
        sys.exit(1)
