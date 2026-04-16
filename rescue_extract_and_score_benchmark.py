"""
Recover final answers from benchmark JSONLs, then score and merge them.

Outputs:
- benchmark_merged_extracted.csv
- benchmark_extracted_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

RATIONAL_TOKEN_RE = re.compile(r"-?\d+(?:/\d+|\.\d+)?")
FINAL_ANSWER_RE = re.compile(r"final\s+answer\b", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")


def fmt_fraction(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def canonicalize_numeric(candidate: str | None) -> str | None:
    if candidate is None:
        return None
    s = str(candidate).strip()
    if not s:
        return None

    # Peel common wrappers repeatedly.
    changed = True
    while changed and s:
        changed = False
        new_s = s.strip()
        if new_s.startswith("$$") and new_s.endswith("$$") and len(new_s) >= 4:
            new_s = new_s[2:-2].strip()
            changed = True
        elif new_s.startswith("$") and new_s.endswith("$") and len(new_s) >= 2:
            new_s = new_s[1:-1].strip()
            changed = True
        elif new_s.startswith("\\(") and new_s.endswith("\\)"):
            new_s = new_s[2:-2].strip()
            changed = True
        elif new_s.startswith("\\[") and new_s.endswith("\\]"):
            new_s = new_s[2:-2].strip()
            changed = True
        elif new_s.startswith("**") and new_s.endswith("**") and len(new_s) >= 4:
            new_s = new_s[2:-2].strip()
            changed = True
        elif new_s.startswith("`") and new_s.endswith("`") and len(new_s) >= 2:
            new_s = new_s[1:-1].strip()
            changed = True
        s = new_s

    boxed = BOXED_RE.search(s)
    if boxed:
        inner = canonicalize_numeric(boxed.group(1))
        if inner is not None:
            return inner

    # Remove common lead-ins.
    s = re.sub(r"^(final\s+answer|answer)\s*[:\-]*\s*", "", s, flags=re.IGNORECASE).strip()
    s = s.replace(",", "")
    if not s:
        return None

    try:
        return fmt_fraction(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass

    matches = RATIONAL_TOKEN_RE.findall(s)
    if len(matches) == 1:
        try:
            return fmt_fraction(Fraction(matches[0]))
        except (ValueError, ZeroDivisionError):
            return matches[0]

    return None


def extract_from_text(text: str | None) -> str | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    # 1) Prefer boxed answers anywhere in the text.
    boxed_matches = BOXED_RE.findall(s)
    for item in reversed(boxed_matches):
        val = canonicalize_numeric(item)
        if val is not None:
            return val

    # 2) Prefer anything after the last 'Final Answer'.
    final_matches = list(FINAL_ANSWER_RE.finditer(s))
    for match in reversed(final_matches):
        tail = s[match.end():].strip()
        if tail:
            for line in [ln.strip() for ln in tail.splitlines() if ln.strip()]:
                val = canonicalize_numeric(line)
                if val is not None:
                    return val
            tail_tokens = RATIONAL_TOKEN_RE.findall(tail.replace(",", ""))
            if tail_tokens:
                try:
                    return fmt_fraction(Fraction(tail_tokens[-1]))
                except (ValueError, ZeroDivisionError):
                    return tail_tokens[-1]

    # 3) Scan lines bottom-up; answers are usually near the end.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    for line in reversed(lines):
        val = canonicalize_numeric(line)
        if val is not None:
            return val
        if "answer" in line.lower() or len(line) <= 120:
            tokens = RATIONAL_TOKEN_RE.findall(line.replace(",", ""))
            if tokens:
                try:
                    return fmt_fraction(Fraction(tokens[-1]))
                except (ValueError, ZeroDivisionError):
                    return tokens[-1]

    # 4) Last-number fallback across the whole text.
    tokens = RATIONAL_TOKEN_RE.findall(s.replace(",", ""))
    if tokens:
        try:
            return fmt_fraction(Fraction(tokens[-1]))
        except (ValueError, ZeroDivisionError):
            return tokens[-1]
    return None


def fraction_equal(a: str, b: str | None) -> bool:
    if b is None:
        return False
    try:
        return Fraction(a.strip()) == Fraction(b.strip())
    except (ValueError, ZeroDivisionError):
        return a.strip().lower() == b.strip().lower()


def collect_jsonl_paths(dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            print(f"Warning: not a directory, skipping: {d}", file=sys.stderr)
            continue
        for p in sorted(d.glob("responses_*.jsonl")):
            paths.append(p)
    return paths


def process_jsonl(path: Path) -> tuple[str, dict[int, str | None], dict[str, Any]]:
    model_id: str | None = None
    answers_by_id: dict[int, str | None] = {}
    source_counts: Counter[str] = Counter()
    dupes = 0
    seen: set[int] = set()
    total = extracted = correct = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if model_id is None and obj.get("model"):
                model_id = str(obj["model"])
            try:
                qid = int(obj["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if qid in seen:
                dupes += 1
            seen.add(qid)

            total += 1
            gold = str(obj.get("gold_answer", "")).strip()

            extracted_answer = canonicalize_numeric(obj.get("parsed_prediction"))
            source = "parsed_prediction"
            if extracted_answer is None:
                extracted_answer = extract_from_text(obj.get("response_text"))
                source = "response_text"
            if extracted_answer is None:
                extracted_answer = extract_from_text(obj.get("reasoning_content"))
                source = "reasoning_content"
            if extracted_answer is None:
                source = "none"
            else:
                extracted += 1
                if gold and fraction_equal(gold, extracted_answer):
                    correct += 1

            source_counts[source] += 1
            answers_by_id[qid] = extracted_answer

    if model_id is None:
        model_id = path.stem.removeprefix("responses_") or path.stem

    summary = {
        "model": model_id,
        "file": str(path),
        "total_rows": total,
        "extracted_count": extracted,
        "correct_count": correct,
        "accuracy": (correct / total) if total else 0.0,
        "extraction_rate": (extracted / total) if total else 0.0,
        "duplicate_id_lines": dupes,
        "source_counts": dict(source_counts),
    }
    return model_id, answers_by_id, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover final answers from benchmark JSONLs and score them"
    )
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=Path("mathrulebench_hard_1245.csv"),
        help="Canonical CSV with id,type,question,answer",
    )
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        action="append",
        dest="jsonl_dirs",
        default=[],
        help="Directory containing responses_*.jsonl (repeatable)",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=Path("benchmark_merged_extracted.csv"),
        help="Wide CSV with rescued extracted answers per model",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmark_extracted_summary.json"),
        help="JSON summary with per-model extraction/scoring stats",
    )
    args = parser.parse_args()

    if not args.jsonl_dirs:
        args.jsonl_dirs = [Path("benchmark_outputs"), Path("benchmark_outputs_mistral")]

    gold = pd.read_csv(args.gold_csv, dtype={"id": int})
    base_cols = ["id", "type", "question", "answer"]
    if list(gold.columns[:4]) != base_cols:
        raise SystemExit(
            f"Expected gold CSV columns id,type,question,answer; got {list(gold.columns)}"
        )

    jsonl_files = collect_jsonl_paths(args.jsonl_dirs)
    if not jsonl_files:
        raise SystemExit("No responses_*.jsonl files found under given --jsonl-dir paths")

    summaries: list[dict[str, Any]] = []
    for path in jsonl_files:
        model_id, answers_by_id, summary = process_jsonl(path)
        if summary["duplicate_id_lines"]:
            print(
                f"Warning: {summary['duplicate_id_lines']} duplicate id line(s) in {path.name}",
                file=sys.stderr,
            )
        gold[model_id] = [answers_by_id.get(i) for i in gold["id"]]
        summaries.append(summary)

    extra_cols = sorted([c for c in gold.columns if c not in base_cols])
    gold = gold[base_cols + extra_cols]
    gold.to_csv(args.merged_output, index=False, encoding="utf-8")

    payload = {
        "merged_output": str(args.merged_output),
        "summary_output": str(args.summary_output),
        "models": summaries,
    }
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        f"Wrote {args.merged_output} and {args.summary_output} for {len(extra_cols)} models.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()