# SRBench — Structured Mathematical Reasoning

## Benchmark for Large Language Models

## Files

| File | Description |
|------|-------------|
| [`generate_mathrulebench_hard_1245.py`](generate_mathrulebench_hard_1245.py) | Defines and emits the **1,245-item** MathRuleBench Hard split: template-backed items, per-template quotas, and deterministic generation. |
| [`mathrulebench_hard_26_templates_report.md`](mathrulebench_hard_26_templates_report.md) | Lists **template** names, how many items each contributes, short descriptions, and **problem-type** counts. |
| [`benchmark_merged_extracted.csv`](benchmark_merged_extracted.csv) | Tabular benchmark: **`id`**, **`type`**, **`question`**, **`answer`** (reference), then **one column per evaluated model** with that model’s **extracted final answer** for grading. Empty cells mean no answer was available for that row and model. |
