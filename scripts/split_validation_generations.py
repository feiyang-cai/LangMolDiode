#!/usr/bin/env python3
"""Split saved verl validation generations by their source dataset."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import pyarrow.parquet as pq


DATASETS = (
    "mollang_validated_success",
    "mollangbench_test",
    "mollangbench_extended",
)


def split_generations(
    input_jsonl: Path,
    validation_parquet: Path,
    output_dir: Path,
    generations_per_prompt: int,
    only: set[str],
) -> dict[str, int]:
    if generations_per_prompt < 1:
        raise ValueError("generations_per_prompt must be positive")

    prompts = pq.read_table(validation_parquet, columns=["data_source", "reward_model"]).to_pylist()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {dataset: 0 for dataset in DATASETS}

    with ExitStack() as stack:
        source = stack.enter_context(input_jsonl.open("r", encoding="utf-8"))
        destinations = {
            dataset: stack.enter_context((output_dir / f"{dataset}.jsonl").open("w", encoding="utf-8"))
            for dataset in only
        }
        for prompt_index, prompt in enumerate(prompts):
            dataset = prompt["data_source"]
            if dataset not in DATASETS:
                raise ValueError(f"Unknown data_source at prompt {prompt_index}: {dataset!r}")
            ground_truth = prompt["reward_model"]["ground_truth"]
            for generation_index in range(generations_per_prompt):
                line = source.readline()
                if not line:
                    raise ValueError(f"Missing generation for prompt {prompt_index}, sample {generation_index}")
                row = json.loads(line)
                if row.get("gts") != ground_truth:
                    raise ValueError(f"Ground truth mismatch at prompt {prompt_index}, sample {generation_index}")
                if dataset in destinations:
                    row["dataset"] = dataset
                    row["prompt_index"] = prompt_index
                    row["generation_index"] = generation_index
                    destinations[dataset].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[dataset] += 1
        if source.readline():
            raise ValueError("Generation file contains more rows than the validation parquet predicts")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--validation-parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generations-per-prompt", type=int, default=3)
    parser.add_argument("--only", choices=DATASETS, action="append")
    args = parser.parse_args()
    counts = split_generations(
        args.input,
        args.validation_parquet,
        args.output_dir,
        args.generations_per_prompt,
        set(args.only or DATASETS),
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
