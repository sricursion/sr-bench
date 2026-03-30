# SRBench — Structured Mathematical Reasoning

## Benchmark for Large Language Models

This repository (**sr-bench**) hosts **MathRuleBench Hard**: the public benchmark split (generator + documentation), **per-model LLM response logs** (JSONL), and a **merged grading table** with gold answers and rescued extracted model outputs.

## Files

| File | Description |
|------|-------------|
| [`generate_mathrulebench_hard_1245.py`](generate_mathrulebench_hard_1245.py) | Python script that **generates** the 1,245-row hard split: template logic, quotas, deterministic RNG seed, and export helpers. Run it to reproduce `mathrulebench_hard_1245.csv` / JSONL (not committed here in minimal form—only this generator). |
| [`mathrulebench_hard_26_templates_report.md`](mathrulebench_hard_26_templates_report.md) | Human-readable **report**: template names, per-template quotas, one-line descriptions, and high-level **type distribution** for the hard set. |
| [`benchmark_merged_extracted.csv`](benchmark_merged_extracted.csv) | **Wide grading table**: columns `id`, `type`, `question`, `answer` (gold), then **one column per evaluated model** with a **rescued final answer** (heuristic extraction from `parsed_prediction`, then `response_text`, then `reasoning_content`). Empty cells mean no numeric answer could be extracted. **Kimi K2.5 is omitted** from this merge by design. |
| [`benchmark_outputs/responses_<model_id>.jsonl`](benchmark_outputs/) | **One JSONL per DigitalOcean–hosted model** (OpenAI-compatible inference). Each line is one benchmark item: full `question`, `gold_answer`, raw `response_text`, optional `reasoning_content`, token usage, `parsed_prediction`, `correct` (vs gold at benchmark time), `error`, etc. **Excluded:** `responses_kimi-k2.5.jsonl` (not published). |
| [`benchmark_outputs_mistral/responses_<model_id>.jsonl`](benchmark_outputs_mistral/) | Same JSONL schema for **Mistral API** runs (`mistral-large-2512`, `mistral-small-2603`, `mistral-medium-2508`). |

## Models represented in JSONL (excluding Kimi)

DigitalOcean inference (`benchmark_outputs/`):

- `alibaba-qwen3-32b`
- `deepseek-r1-distill-llama-70b`
- `minimax-m2.5`
- `llama3.3-70b-instruct`
- `llama3-8b-instruct`
- `mistral-nemo-instruct-2407`
- `nvidia-nemotron-3-super-120b`
- `openai-gpt-oss-120b`
- `openai-gpt-oss-20b`
- `glm-5`

Mistral API (`benchmark_outputs_mistral/`):

- `mistral-large-2512`
- `mistral-small-2603`
- `mistral-medium-2508`

## License / usage

Use the generator and reports under your project’s terms. **Do not commit API keys**; inference was run locally with keys in environment only.
