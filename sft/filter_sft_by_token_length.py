#!/usr/bin/env python
"""Filter chat SFT JSONL rows by rendered token length."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, default=None)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--max-length", type=int, default=40960)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield line_no, row


def apply_chat_template(tokenizer, messages: list[dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def pct(lengths: list[int], q: float) -> int | None:
    if not lengths:
        return None
    idx = min(len(lengths) - 1, max(0, int(len(lengths) * q) - 1))
    return sorted(lengths)[idx]


def summarize_lengths(lengths: list[int]) -> dict[str, int | None]:
    if not lengths:
        return {
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "min": min(lengths),
        "p50": int(statistics.median(lengths)),
        "p90": pct(lengths, 0.90),
        "p95": pct(lengths, 0.95),
        "p99": pct(lengths, 0.99),
        "max": max(lengths),
    }


def process_batch(
    *,
    tokenizer,
    batch: list[tuple[int, dict[str, Any]]],
    raw_lines: list[str],
    output_handle,
    rejected_handle,
    max_length: int,
    stats: dict[str, Any],
) -> None:
    texts = []
    for line_no, row in batch:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"input line {line_no}: missing messages list")
        texts.append(apply_chat_template(tokenizer, messages))

    tokenized = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    for (line_no, row), raw_line, input_ids in zip(batch, raw_lines, tokenized["input_ids"]):
        length = len(input_ids)
        stats["seen"] += 1
        stats["lengths"].append(length)
        if length <= max_length:
            output_handle.write(raw_line + "\n")
            stats["kept"] += 1
            stats["kept_lengths"].append(length)
        else:
            stats["rejected"] += 1
            stats["rejected_lengths"].append(length)
            if rejected_handle is not None:
                rejected_row = dict(row)
                rejected_row["token_filter"] = {
                    "line_no": line_no,
                    "token_length": length,
                    "max_length": max_length,
                }
                rejected_handle.write(json.dumps(rejected_row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.rejected_output is not None:
        args.rejected_output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        use_fast=True,
    )

    stats: dict[str, Any] = {
        "seen": 0,
        "kept": 0,
        "rejected": 0,
        "lengths": [],
        "kept_lengths": [],
        "rejected_lengths": [],
    }
    batch: list[tuple[int, dict[str, Any]]] = []
    raw_lines: list[str] = []

    rejected_handle = (
        args.rejected_output.open("w", encoding="utf-8") if args.rejected_output is not None else None
    )
    try:
        with args.output.open("w", encoding="utf-8") as output_handle:
            for line_no, row in iter_jsonl(args.input):
                batch.append((line_no, row))
                raw_lines.append(json.dumps(row, ensure_ascii=False))
                if len(batch) >= args.batch_size:
                    process_batch(
                        tokenizer=tokenizer,
                        batch=batch,
                        raw_lines=raw_lines,
                        output_handle=output_handle,
                        rejected_handle=rejected_handle,
                        max_length=args.max_length,
                        stats=stats,
                    )
                    batch = []
                    raw_lines = []
            if batch:
                process_batch(
                    tokenizer=tokenizer,
                    batch=batch,
                    raw_lines=raw_lines,
                    output_handle=output_handle,
                    rejected_handle=rejected_handle,
                    max_length=args.max_length,
                    stats=stats,
                )
    finally:
        if rejected_handle is not None:
            rejected_handle.close()

    result = {
        "input": str(args.input),
        "output": str(args.output),
        "rejected_output": str(args.rejected_output) if args.rejected_output is not None else None,
        "tokenizer_name": args.tokenizer_name,
        "max_length": args.max_length,
        "seen": stats["seen"],
        "kept": stats["kept"],
        "rejected": stats["rejected"],
        "kept_rate": stats["kept"] / stats["seen"] if stats["seen"] else 0.0,
        "all_lengths": summarize_lengths(stats["lengths"]),
        "kept_lengths": summarize_lengths(stats["kept_lengths"]),
        "rejected_lengths": summarize_lengths(stats["rejected_lengths"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
