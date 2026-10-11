"""Day 1 验证脚本 3/3：验证 embedding 模型 BAAI/bge-small-zh-v1.5 可用。

模型优先从项目本地目录 models/bge-small-zh-v1.5 加载；
若本地不存在，则从 HF_ENDPOINT（默认 https://hf-mirror.com）下载。
直接运行：python scripts/test_embedding.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 必须在导入 sentence_transformers 之前设置镜像端点，
# 否则环境变量不会生效（HF 库在 import 时读取一次）。
load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 镜像站不支持 xet 加速协议（会下载到 0 字节空文件），必须禁用
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential  # noqa: E402

MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"
# 项目根目录下的本地模型目录（脚本无论从何处运行都能定位）
LOCAL_MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "bge-small-zh-v1.5"


@retry(
    stop=stop_after_attempt(3),  # 网络下载最多重试 3 次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def download_model() -> None:
    """从镜像下载模型到本地目录。"""
    snapshot_download(MODEL_NAME, local_dir=str(LOCAL_MODEL_DIR))


def load_model() -> SentenceTransformer:
    """加载模型：本地目录已存在则离线加载，否则先下载。"""
    if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "modules.json").exists():
        print(f"使用本地模型目录: {LOCAL_MODEL_DIR}")
        return SentenceTransformer(str(LOCAL_MODEL_DIR))
    print(f"本地模型不存在，开始下载: {MODEL_NAME} -> {LOCAL_MODEL_DIR}")
    download_model()
    return SentenceTransformer(str(LOCAL_MODEL_DIR))


def main() -> None:
    print(f"加载模型: {MODEL_NAME}（HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}）")
    try:
        model = load_model()
    except Exception as exc:
        print(f"[错误] 模型加载失败（model={MODEL_NAME}, local_dir={LOCAL_MODEL_DIR}, "
              f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}）: {type(exc).__name__}: {exc}")
        print("       请检查网络，或确认 HF_ENDPOINT=https://hf-mirror.com 已生效。")
        sys.exit(1)

    # 用一组中文技术文档风格的句子验证向量维度与相似度方向
    sentences: list[str] = [
        "LangGraph 是一个用于构建有状态多智能体应用的框架。",
        "RAG 通过检索相关文档来增强大模型的回答质量。",
        "今天天气不错，适合出门散步。",
    ]
    try:
        embeddings = model.encode(sentences, normalize_embeddings=True)
    except Exception as exc:
        print(f"[错误] 句子编码失败（共 {len(sentences)} 句）: {type(exc).__name__}: {exc}")
        sys.exit(1)

    dim: int = embeddings.shape[1]
    print(f"[OK] 向量维度: {dim}（bge-small-zh-v1.5 应为 512）")
    if dim != 512:
        print("[错误] 向量维度与预期不符，请确认模型名称是否正确。")
        sys.exit(1)

    # 余弦相似度（已归一化，直接点积）：技术相关句应明显高于无关句
    sim_tech: float = float(embeddings[0] @ embeddings[1])
    sim_unrelated: float = float(embeddings[0] @ embeddings[2])
    print(f"[OK] 技术句相似度: {sim_tech:.4f}  |  无关句相似度: {sim_unrelated:.4f}")
    if sim_tech <= sim_unrelated:
        print("[警告] 相似度方向异常，模型语义能力可能有问题，请人工确认。")

    print("\nEmbedding 模型验证通过 ✔")


if __name__ == "__main__":
    main()
