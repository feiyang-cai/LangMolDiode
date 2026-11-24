#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _render_prompt(tokenizer: Any, messages: Any, *, enable_thinking: bool) -> str:
    if isinstance(messages, list):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(messages)


def _sample_id(row: dict[str, Any], fallback: int) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("sample_id") is not None:
        return str(extra["sample_id"])
    return str(fallback)


def filter_parquet(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_json) if args.report_json else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    parquet = pq.ParquetFile(input_path)
    schema = parquet.schema_arrow
    writer: pq.ParquetWriter | None = None
    enable_thinking = not args.no_enable_thinking

    kept_rows = 0
    total_rows = 0
    lengths: list[int] = []
    dropped: list[dict[str, Any]] = []

    try:
        for batch in parquet.iter_batches(batch_size=args.batch_size):
            rows = batch.to_pylist()
            rendered = [_render_prompt(tokenizer, row["prompt"], enable_thinking=enable_thinking) for row in rows]
            encoded = tokenizer(rendered, add_special_tokens=False, return_attention_mask=False)

            keep: list[dict[str, Any]] = []
            for offset, ids in enumerate(encoded["input_ids"]):
                token_count = len(ids)
                row_index = total_rows + offset
                lengths.append(token_count)
                row = rows[offset]
                if token_count <= args.max_prompt_length:
                    keep.append(row)
                else:
                    dropped.append(
                        {
                            "row_index": row_index,
                            "sample_id": _sample_id(row, row_index),
                            "tokens": token_count,
                        }
                    )

            if keep:
                table = pa.Table.from_pylist(keep, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression=args.compression)
                writer.write_table(table)
                kept_rows += len(keep)

            total_rows += len(rows)
    finally:
        if writer is not None:
            writer.close()

    if kept_rows == 0:
        raise ValueError(f"No rows kept after filtering {input_path}")

    sorted_lengths = sorted(lengths)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "max_prompt_length": args.max_prompt_length,
        "enable_thinking": enable_thinking,
        "rows_in": total_rows,
        "rows_out": kept_rows,
        "rows_dropped": len(dropped),
        "dropped": dropped,
        "token_stats": {
            "min": sorted_lengths[0],
            "mean": sum(sorted_lengths) / len(sorted_lengths),
            "p50": _percentile(sorted_lengths, 50),
            "p90": _percentile(sorted_lengths, 90),
            "p95": _percentile(sorted_lengths, 95),
            "p99": _percentile(sorted_lengths, 99),
            "max": sorted_lengths[-1],
        },
    }
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter verl parquet rows by rendered prompt token count.")
    parser.add_argument("--model", required=True, help="Tokenizer/model path.")
    parser.add_argument("--input", required=True, help="Input parquet.")
    parser.add_argument("--output", required=True, help="Filtered output parquet.")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--no-enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = filter_parquet(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
