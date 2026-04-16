"""
DigitalOcean serverless inference benchmark on MathRuleBench Hard (CSV).

Reads mathrulebench_hard_1245.csv, runs 11 models in parallel (MODEL_KEY1..MODEL_KEY11),
writes one JSONL per model under --output-dir, optional benchmark_summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, RateLimitError

BASE_URL = "https://inference.do-ai.run/v1/"

# Order matches MODEL_KEY1 .. MODEL_KEY11
MODELS: list[str] = [
    "alibaba-qwen3-32b",
    "deepseek-r1-distill-llama-70b",
    "minimax-m2.5",
    "kimi-k2.5",
    "llama3.3-70b-instruct",
    "llama3-8b-instruct",
    "mistral-nemo-instruct-2407",
    "nvidia-nemotron-3-super-120b",
    "openai-gpt-oss-120b",
    "openai-gpt-oss-20b",
    "glm-5",
]

REASONING_MODEL_IDS: frozenset[str] = frozenset(
    {
        "deepseek-r1-distill-llama-70b",
        "openai-gpt-oss-120b",
        "openai-gpt-oss-20b",
    }
)


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_")


def env_key_for_index(i: int) -> str:
    return f"MODEL_KEY{i + 1}"


def load_completed_ids(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    done: set[int] = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj["id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


_RATIONAL_RE = re.compile(r"-?\d+/\d+|-?\d+")


def parse_prediction(raw: str | None) -> str | None:
    """
    Heuristic: strip whitespace; prefer the last non-empty line; then extract
    a rational (integer or a/b) from that line or the whole string.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    candidate = lines[-1] if lines else s
    try:
        frac = Fraction(candidate)
        return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
    except (ValueError, ZeroDivisionError):
        pass
    matches = _RATIONAL_RE.findall(candidate)
    if matches:
        tok = matches[-1]
        try:
            frac = Fraction(tok)
            return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
        except (ValueError, ZeroDivisionError):
            return tok.strip()
    return candidate.strip() or None


def fraction_equal(a: str, b: str | None) -> bool:
    if b is None:
        return False
    a, b = a.strip(), b.strip()
    if not b:
        return False
    try:
        fa, fb = Fraction(a), Fraction(b)
        return fa == fb
    except (ValueError, ZeroDivisionError):
        return a.lower() == b.lower()


def extract_message_fields(choice: Any) -> tuple[str | None, str | None, str | None]:
    msg = choice.message
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text") or "")
            elif hasattr(p, "text"):
                parts.append(getattr(p, "text", "") or "")
        content = "".join(parts) if parts else None
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is None and hasattr(msg, "model_extra") and msg.model_extra:
        reasoning = msg.model_extra.get("reasoning_content")
    finish = getattr(choice, "finish_reason", None)
    return (content if content else None, reasoning, finish)


async def preflight_models(api_key: str, model_ids: list[str]) -> None:
    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    lst = await client.models.list()
    available = {m.id for m in lst.data}
    missing = [m for m in model_ids if m not in available]
    if missing:
        print(
            "Preflight: WARNING — these model ids were not returned by GET /v1/models "
            f"(first key): {missing}\nCatalog sample: {sorted(list(available))[:20]} ..."
        )
    else:
        print("Preflight: all 11 target model ids are listed in /v1/models.")


async def chat_completion_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    question: str,
    max_completion_tokens: int,
    max_completion_tokens_retry: int,
    use_reasoning_low: bool,
) -> dict[str, Any]:
    """Returns dict with response_text, reasoning_content, finish_reason, usage dict, error, latency_ms, retried_length."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "max_completion_tokens": max_completion_tokens,
    }
    if use_reasoning_low:
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    last_err: str | None = None
    retried_length = False
    attempt = 0
    max_attempts = 8

    while attempt < max_attempts:
        attempt += 1
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(**kwargs)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            choice = resp.choices[0]
            text, reasoning, finish = extract_message_fields(choice)
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            # Retry once with higher budget if truncated and content weak
            if finish == "length" and max_completion_tokens < max_completion_tokens_retry and not retried_length:
                retried_length = True
                kwargs["max_completion_tokens"] = max_completion_tokens_retry
                continue
            return {
                "response_text": text,
                "reasoning_content": reasoning,
                "finish_reason": finish,
                "usage": usage,
                "error": None,
                "latency_ms": latency_ms,
                "retried_length": retried_length,
            }
        except RateLimitError as e:
            last_err = str(e)
            wait = min(60.0, (2 ** min(attempt, 6)) + random.uniform(0, 1))
            await asyncio.sleep(wait)
        except APIStatusError as e:
            last_err = f"HTTP {e.status_code}: {e.message}"
            if e.status_code == 429:
                wait = min(60.0, (2 ** min(attempt, 6)) + random.uniform(0, 1))
                await asyncio.sleep(wait)
            elif 500 <= (e.status_code or 0) < 600:
                wait = min(30.0, 1.5**attempt + random.uniform(0, 0.5))
                await asyncio.sleep(wait)
            else:
                break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            wait = min(20.0, 1.5**attempt + random.uniform(0, 0.5))
            await asyncio.sleep(wait)

    return {
        "response_text": None,
        "reasoning_content": None,
        "finish_reason": None,
        "usage": {},
        "error": last_err or "unknown_error",
        "latency_ms": None,
        "retried_length": retried_length,
    }


@dataclass
class Row:
    id: int
    type: str
    question: str
    gold_answer: str


async def run_one_model(
    model_id: str,
    api_key: str,
    rows: list[Row],
    output_dir: Path,
    concurrency: int,
    max_completion_tokens: int,
    max_completion_tokens_retry: int,
) -> dict[str, Any]:
    out_path = output_dir / f"responses_{model_slug(model_id)}.jsonl"
    done = load_completed_ids(out_path)
    pending = [r for r in rows if r.id not in done]
    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    use_reasoning = model_id in REASONING_MODEL_IDS

    stats = {
        "correct": 0,
        "total": 0,
        "empty_response": 0,
        "truncated": 0,
        "errors": 0,
        "latency_sum_ms": 0,
        "latency_n": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    async def process(row: Row) -> None:
        async with sem:
            raw = await chat_completion_with_retry(
                client,
                model=model_id,
                question=row.question,
                max_completion_tokens=max_completion_tokens,
                max_completion_tokens_retry=max_completion_tokens_retry,
                use_reasoning_low=use_reasoning,
            )
        pred_raw = raw["response_text"]
        parsed = parse_prediction(pred_raw)
        correct = fraction_equal(row.gold_answer, parsed) if parsed else False

        if raw["error"]:
            stats["errors"] += 1
        elif not pred_raw or not str(pred_raw).strip():
            stats["empty_response"] += 1
        if raw.get("finish_reason") == "length":
            stats["truncated"] += 1
        if raw["latency_ms"] is not None:
            stats["latency_sum_ms"] += raw["latency_ms"]
            stats["latency_n"] += 1
        u = raw.get("usage") or {}
        if u.get("prompt_tokens"):
            stats["prompt_tokens"] += int(u["prompt_tokens"])
        if u.get("completion_tokens"):
            stats["completion_tokens"] += int(u["completion_tokens"])

        stats["total"] += 1
        if correct:
            stats["correct"] += 1

        record: dict[str, Any] = {
            "id": row.id,
            "type": row.type,
            "question": row.question,
            "gold_answer": row.gold_answer,
            "model": model_id,
            "response_text": pred_raw,
            "reasoning_content": raw["reasoning_content"],
            "finish_reason": raw["finish_reason"],
            "latency_ms": raw["latency_ms"],
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "total_tokens": u.get("total_tokens"),
            "error": raw["error"],
            "parsed_prediction": parsed,
            "correct": correct,
            "retried_higher_token_limit": raw.get("retried_length", False),
        }

        async with write_lock:

            def _write() -> None:
                append_jsonl(out_path, record)

            await asyncio.to_thread(_write)

    await asyncio.gather(*[process(r) for r in pending])

    n = stats["total"] or 1
    return {
        "model": model_id,
        "output_file": str(out_path),
        "rows_completed_this_session": stats["total"],
        "accuracy_this_session": stats["correct"] / n,
        "correct_this_session": stats["correct"],
        "empty_response_count_this_session": stats["empty_response"],
        "truncation_finish_count_this_session": stats["truncated"],
        "error_count_this_session": stats["errors"],
        "mean_latency_ms_this_session": (stats["latency_sum_ms"] / stats["latency_n"])
        if stats["latency_n"]
        else None,
        "sum_prompt_tokens_this_session": stats["prompt_tokens"],
        "sum_completion_tokens_this_session": stats["completion_tokens"],
    }


def summarize_from_disk(output_dir: Path, model_ids: list[str]) -> dict[str, Any]:
    per_model = []
    for mid in model_ids:
        p = output_dir / f"responses_{model_slug(mid)}.jsonl"
        if not p.exists():
            per_model.append({"model": mid, "error": "missing_file"})
            continue
        correct = total = empty = trunc = err = 0
        lat_sum = lat_n = 0
        pt = ct = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if o.get("correct"):
                    correct += 1
                rt = o.get("response_text")
                if not rt or not str(rt).strip():
                    if o.get("error"):
                        pass
                    else:
                        empty += 1
                if o.get("finish_reason") == "length":
                    trunc += 1
                if o.get("error"):
                    err += 1
                lm = o.get("latency_ms")
                if lm is not None:
                    lat_sum += int(lm)
                    lat_n += 1
                if o.get("prompt_tokens"):
                    pt += int(o["prompt_tokens"])
                if o.get("completion_tokens"):
                    ct += int(o["completion_tokens"])
        per_model.append(
            {
                "model": mid,
                "file": str(p),
                "accuracy": correct / total if total else 0.0,
                "correct": correct,
                "total_rows": total,
                "empty_response_count": empty,
                "truncation_finish_count": trunc,
                "error_count": err,
                "mean_latency_ms": lat_sum / lat_n if lat_n else None,
                "sum_prompt_tokens": pt,
                "sum_completion_tokens": ct,
            }
        )
    return {"per_model": per_model}


async def amain() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="MathRuleBench Hard — DO serverless inference benchmark")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("mathrulebench_hard_1245.csv"),
        help="Path to CSV with id,type,question,answer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs"),
        help="Directory for per-model JSONL and summary",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests per model")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8192,
        help="Initial max_completion_tokens per call",
    )
    parser.add_argument(
        "--max-completion-tokens-retry",
        type=int,
        default=16384,
        help="On length truncation, retry once with this max_completion_tokens",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Skip GET /v1/models validation")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only run preflight against MODEL_KEY1 and exit",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip writing benchmark_summary.json")
    args = parser.parse_args()

    keys: list[str] = []
    for i in range(len(MODELS)):
        k = os.getenv(env_key_for_index(i))
        if not k:
            raise SystemExit(f"Missing {env_key_for_index(i)} in environment (.env)")
        keys.append(k.strip())

    if not args.skip_preflight:
        await preflight_models(keys[0], MODELS)
    if args.preflight_only:
        return

    df = pd.read_csv(args.csv, dtype={"id": int})
    rows = [
        Row(id=int(r.id), type=str(r.type), question=str(r.question), gold_answer=str(r.answer))
        for r in df.itertuples(index=False)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        run_one_model(
            MODELS[i],
            keys[i],
            rows,
            args.output_dir,
            args.concurrency,
            args.max_completion_tokens,
            args.max_completion_tokens_retry,
        )
        for i in range(len(MODELS))
    ]
    print(f"Starting {len(MODELS)} model tracks, {len(rows)} questions each, concurrency={args.concurrency}")
    results = await asyncio.gather(*tasks)

    if not args.no_summary:
        summary_path = args.output_dir / "benchmark_summary.json"
        payload = {
            "config": {
                "csv": str(args.csv),
                "concurrency_per_model": args.concurrency,
                "max_completion_tokens": args.max_completion_tokens,
                "max_completion_tokens_retry": args.max_completion_tokens_retry,
                "reasoning_effort_low_models": sorted(REASONING_MODEL_IDS),
            },
            "per_model_from_disk": summarize_from_disk(args.output_dir, MODELS)["per_model"],
            "session_per_model": results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote {summary_path}")

    print("Done.")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
