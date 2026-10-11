# 评估模块需求（Day 5）

## 目标

用 ≥40 条人工校验的 QA 评估集，量化系统的**检索质量**（Hit@5、MRR）与**拒答正确性**，
输出可复现的评估报告。

## 不做什么（范围边界）

- 今天不做生成质量的自动评分（那属于 Day 6 的 Ragas / LLM-as-judge）；
- 今天**不做任何系统优化**——只测量。测量工具本身先可信，明天的优化才有意义。

## 评估集规格

- 条数 ≥40，三类比例大致 18（简单事实）/ 14（综合）/ 8（越界）；
- 每条字段：`id / question / ground_truth / source[] / type / origin(auto|manual) / verified`；
- 越界样本的 `ground_truth` 固定为 `__REFUSAL__`，`source` 为空数组；
- 生成方式：LLM 出候选（auto）+ 人工书写综合题与全部越界题（manual），
  逐条原文核对后 `verified=true` 才进入最终集。

## 指标规格

- `Hit@k`：标准来源出现在前 k 位记 1，否则 0，全体取平均；
- `MRR`：第一个正确来源排第 r 位记 1/r，未命中记 0，全体取平均；
- 越界样本不进 hit/mrr 分母（没有"正确来源"可言），单独统计其 top1 分数分布；
- 指标函数自己实现（不调现成库），用 pytest 覆盖边界 case。

## 验收

- `pytest tests/test_metrics.py -v` 全绿；
- `eval/golden_qa.json` ≥40 条，每条有 source 标注；
- `eval/reports/retrieval_baseline_{date}.md` 生成，含未命中清单；
- 人工为 ≥5 个失败 case 填写归因。
