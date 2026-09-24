#!/usr/bin/env python3
"""Export the published LangMolDiode SFT corpus as chat-training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


REQUIRED_COLUMNS = {
    "id",
    "smiles",
    "system_prompt",
    "input_prompt",
    "sft_response",
    "generated_smiles",
    "generation_model",
    "reasoning_effort_effective",
    "matches_smiles",
}


def to_training_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_COLUMNS.difference(row)
    if missing:
        raise ValueError(f"Corpus row is missing columns: {sorted(missing)}")
    if row["matches_smiles"] is not True:
        raise ValueError(f"Corpus row {row['id']} is not a verified molecular match")
    for key in ("id", "smiles", "system_prompt", "input_prompt", "sft_response"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"Corpus row {row.get('id')} has an invalid {key}")

    return {
        "messages": [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["input_prompt"]},
            {"role": "assistant", "content": row["sft_response"]},
        ],
        "smiles": row["smiles"],
        "candidate_smiles": row["generated_smiles"],
        "expected_smiles": row["smiles"],
        "curator_model": row["generation_model"],
        "reasoning_effort": row["reasoning_effort_effective"],
        "sample_id": row["id"],
        "response_mode": "qwen_think",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ChemFM/LangMolDiode-SFT-Corpus")
    parser.add_argument(
        "--data-files",
        help="Optional local Parquet file or glob; bypasses the Hub dataset.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    source = args.dataset
    if args.data_files:
        dataset = load_dataset("parquet", data_files=args.data_files, split=args.split)
        source = args.data_files
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    total = min(len(dataset), args.limit) if args.limit else len(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(total):
            handle.write(json.dumps(to_training_row(dataset[index]), ensure_ascii=False) + "\n")
    print(json.dumps({"source": source, "split": args.split, "rows": total, "output": str(args.output)}))


if __name__ == "__main__":
    main()
