"""Day 6 B1：Ragas 接入（LLM-as-judge 生成层评估）。

三条指标：
- Faithfulness（忠实度）：回答是否忠于检索内容（防幻觉）
- AnswerRelevancy（答案相关性）：回答是否切题
- ContextRecall（上下文召回）：标准答案所需信息是否被检索回来

国内模型接入姿势：
- LLM → ChatOpenAI(base_url 指向中转/DeepSeek) 包成 ragas 的 LangchainLLMWrapper
- Embeddings → 本地 bge 包成 LangchainEmbeddingsWrapper（AnswerRelevancy 内部要用）

⚠️ 版本坑：ragas 0.4.x 的指标类位于 `ragas.metrics._Faithfulness` 等私有路径，
与 0.2/0.3 的公开 import 完全不同——本文件按**实际安装版本**（0.4.3）编写，
import 失败时给出明确报错而不是让用户猜。

运行（自检方向性）：python -m eval.ragas_eval --selftest
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.ingest.build_index import get_embeddings  # noqa: E402
from src.llm import get_chat_model  # noqa: E402

try:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import _AnswerRelevancy, _Faithfulness, _LLMContextRecall
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        f"ragas 导入失败（{exc}）。请先确认安装版本：pip show ragas；"
        "本项目按 ragas 0.4.3 的 API 编写（指标位于 ragas.metrics._Faithfulness 等路径）。"
    ) from exc


def build_judge(config: dict[str, Any]) -> tuple[Any, Any]:
    """构建裁判 LLM 与 embedding（temperature=0，保证评估可复现）。"""
    llm = get_chat_model(temperature=0.0, config=config)
    embeddings = get_embeddings(config)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def run_ragas(samples: list[dict[str, Any]], config: dict[str, Any] | None = None,
              show_progress: bool = True) -> dict[str, Any]:
    """跑 Ragas 三指标。

    Args:
        samples: 每条 {"question","answer","contexts"[str],"ground_truth"}
                 越界样本的 ground_truth 为 "__REFUSAL__"，不参与 ContextRecall
        config: 配置 dict

    Returns:
        {"faithfulness": float, "answer_relevancy": float, "context_recall": float,
         "refusal_rate": float, "per_sample": [...], "failures": [...]}
    """
    config = config or load_config()
    eval_cfg = config.get("eval", {})
    batch_size = int(eval_cfg.get("ragas_batch_size", 10))

    judge_llm, judge_emb = build_judge(config)
    metrics = [_Faithfulness(llm=judge_llm),
               _AnswerRelevancy(llm=judge_llm, embeddings=judge_emb),
               _LLMContextRecall(llm=judge_llm)]

    # 越界样本（无 ground_truth）单独走"拒答正确率"，不进 ContextRecall
    normal = [s for s in samples if s.get("ground_truth") not in (None, "", "__REFUSAL__")]
    oos = [s for s in samples if s.get("ground_truth") == "__REFUSAL__"]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def to_sample(s: dict[str, Any]) -> SingleTurnSample:
        return SingleTurnSample(
            user_input=s["question"],
            retrieved_contexts=list(s.get("contexts", [])),
            response=s.get("answer", ""),
            reference=s.get("ground_truth") if s.get("ground_truth") != "__REFUSAL__" else None,
        )

    start = time.perf_counter()
    for i in range(0, len(normal), batch_size):
        batch = normal[i:i + batch_size]
        dataset = EvaluationDataset(samples=[to_sample(s) for s in batch])
        try:
            result = evaluate(dataset=dataset, metrics=metrics,
                              llm=judge_llm, embeddings=judge_emb,
                              show_progress=show_progress, batch_size=batch_size)
        except Exception as exc:
            for s in batch:
                failures.append({"question": s["question"], "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[警告] 批次 {i // batch_size + 1} 评估失败: {type(exc).__name__}: {exc}")
            continue
        df = result.to_pandas() if hasattr(result, "to_pandas") else None
        for j, s in enumerate(batch):
            row = {"question": s["question"], "type": s.get("type", "")}
            if df is not None and len(df) > j:
                # 动态捕获所有指标列：ragas 各版本的列名不一致（如 llm_context_recall /
                # context_recall / answer_relevancy / response_relevancy），
                # 写死列名会在版本升级后静默取到 None
                for col in df.columns:
                    if col in ("user_input", "retrieved_contexts", "response", "reference"):
                        continue
                    val = df.iloc[j][col]
                    row[col] = float(val) if val == val else None  # NaN 记为 None
            rows.append(row)
        print(f"  批次 {i // batch_size + 1} 完成（{len(batch)} 条）")
    elapsed = time.perf_counter() - start

    def avg(*keys: str) -> float:
        """按候选列名依次尝试（兼容不同 ragas 版本的列名差异）。"""
        for key in keys:
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                return round(sum(vals) / len(vals), 4)
        return 0.0

    # 拒答正确率：越界样本的回答里是否包含拒答话术
    refusal_hits = sum(1 for s in oos if "无法回答" in (s.get("answer") or ""))
    refusal_rate = round(refusal_hits / len(oos), 4) if oos else 0.0

    failure_rate = len(failures) / max(len(samples), 1)
    if failure_rate > 0.1:
        print(f"[警告] 单样本失败率 {failure_rate:.1%} > 10%，指标可能偏高（失败样本被跳过）")

    out = {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy", "response_relevancy"),
        "context_recall": avg("llm_context_recall", "context_recall"),
        "refusal_rate": refusal_rate,
        "n": len(rows),
        "n_oos": len(oos),
        "failure_rate": round(failure_rate, 4),
        "elapsed": round(elapsed, 1),
        "per_sample": rows,
        "failures": failures,
    }
    return out


def save_detail(result: dict[str, Any], tag: str = "") -> Path:
    """保存 per_sample 明细，供人工抽查低分样本。"""
    reports = PROJECT_ROOT / "eval" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"ragas_detail_{tag or date.today().isoformat()}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def selftest() -> None:
    """用 3 条手工样本验证指标方向性：幻觉样本的 faithfulness 应明显更低。"""
    config = load_config()
    samples = [
        {"question": "LangGraph 的 state 可以怎么定义？",
         "answer": "可以用 TypedDict、Pydantic 模型或 dataclass 定义 state。",
         "contexts": ["State in LangGraph can be a TypedDict, Pydantic model, or dataclass."],
         "ground_truth": "TypedDict / Pydantic / dataclass"},
        {"question": "如何启用流式输出？",
         "answer": "可以在初始化模型时设置 streaming=True 来启用流式输出。",
         "contexts": ["Most models can stream their output content while it is being generated."],
         "ground_truth": "多数模型默认支持流式；可用 streaming=False 关闭"},
        # 明显幻觉：资料里完全没有价格信息
        {"question": "LangGraph 企业版多少钱一年？",
         "answer": "LangGraph 企业版每年收费 19999 元，包含 10 个席位。",
         "contexts": ["LangGraph is a low-level orchestration framework for building agents."],
         "ground_truth": "语料未涉及价格"},
    ]
    result = run_ragas(samples, config, show_progress=False)
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"},
                     ensure_ascii=False, indent=2))
    print("\n逐样本明细:")
    for r, s in zip(result["per_sample"], samples):
        print(f"  Q: {r['question'][:30]} | faith={r.get('faithfulness')} | "
              f"rel={r.get('answer_relevancy')} | recall={r.get('llm_context_recall')}")
    print("\n期望：第 3 条（幻觉）的 faithfulness 明显低于前两条")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ragas 生成层评估（Day 6）")
    parser.add_argument("--selftest", action="store_true", help="3 条样本验证指标方向性")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
