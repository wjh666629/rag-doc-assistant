"""Day 1 B5：本地中文 embedding 模型验证 + 余弦相似度矩阵。

embedding 把语义映射到向量空间，相似语义的句子在空间中距离更近——
这是 RAG 检索的全部基础：把文档块和用户问题都变成向量后，
"找相关内容"就等价于"在向量空间里找最近邻"。

直接运行：python scripts/hello_embedding.py
"""
import os

# 必须在 import sentence_transformers 之前设置镜像与禁用 xet，
# 否则 HF 库在导入时已完成初始化，之后再设无效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"  # 镜像不支持 xet，会下到 0 字节空文件

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

# 优先用 Day 1 已下载到项目内的本地模型，避免重复下载
LOCAL_MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "bge-small-zh-v1.5"
MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"

SENTENCES: list[str] = [
    "LangGraph 是一个 Agent 编排框架",
    "LangGraph 可以构建带循环的智能体工作流",
    "今天晚饭吃什么",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """用 numpy 手算余弦相似度：cos(a,b) = a·b / (|a|·|b|)。"""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    model_path: str = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_NAME
    print(f"加载模型: {model_path}")
    model = SentenceTransformer(model_path)

    embeddings: np.ndarray = np.asarray(model.encode(SENTENCES))
    print(f"向量维度: {embeddings.shape[1]}\n")

    # 计算并打印 3x3 两两余弦相似度矩阵
    n: int = len(SENTENCES)
    matrix: np.ndarray = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    print("余弦相似度矩阵（行/列均为句 1/2/3）:")
    print("          句1      句2      句3")
    for i in range(n):
        row = "  ".join(f"{matrix[i][j]:.4f}" for j in range(n))
        print(f"句{i + 1}    {row}")

    sim12: float = matrix[0][1]  # 两句 LangGraph 相关句
    sim13: float = matrix[0][2]  # 技术句 vs 闲聊句
    print(f"\nsim(句1,句2) = {sim12:.4f}  >  sim(句1,句3) = {sim13:.4f} ?", end=" ")
    if sim12 > sim13:
        print("成立 ✔  语义相关的句子在向量空间中确实更近")
    else:
        print("不成立 ✘  请检查模型加载是否正确")


if __name__ == "__main__":
    main()
