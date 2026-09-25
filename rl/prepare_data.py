from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_TRAIN_JSONL = None
DEFAULT_VAL_JSONL = None
DEFAULT_BENCH_TEST_JSONL = None
DEFAULT_BENCH_EXTENDED_JSONL = None


DATA_SOURCE_BY_SPLIT = {
    "train_generated": "mollang_train_generated",
    "validated_finalpass": "mollang_validated_success",
    "bench_test": "mollangbench_test",
    "bench_extended": "mollangbench_extended",
}


VERL_SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        (
            "extra_info",
            pa.struct(
                [
                    ("sample_id", pa.string()),
                    ("split", pa.string()),
                    ("row_index", pa.int64()),
                    ("gt_smiles", pa.string()),
                    ("description", pa.string()),
                    ("final_pass", pa.bool_()),
                ]
            ),
        ),
    ]
)


@dataclass(frozen=True)
class ConversionStats:
    split: str
    input_path: str
    output_path: str
    rows: int
    bytes: int


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(value).__name__}")
            yield value


def _as_verl_row(row: dict[str, Any], *, row_index: int, split_name: str) -> dict[str, Any]:
    prompt = row.get("prompt")
    gt_smiles = row.get("gt_smiles")
    if not prompt:
        raise ValueError(f"Missing prompt at row_index={row_index}")
    if not gt_smiles:
        raise ValueError(f"Missing gt_smiles at row_index={row_index}")

    return {
        "data_source": DATA_SOURCE_BY_SPLIT.get(split_name, f"mollang_{split_name}"),
        "prompt": [{"role": "user", "content": str(prompt)}],
        "ability": "smiles_generation",
        "reward_model": {"style": "rule", "ground_truth": str(gt_smiles)},
        "extra_info": {
            "sample_id": str(row.get("sample_id", row_index)),
            "split": str(row.get("split", split_name)),
            "row_index": int(row_index),
            "gt_smiles": str(gt_smiles),
            "description": str(row.get("description", "")),
            "final_pass": row.get("final_pass") if isinstance(row.get("final_pass"), bool) else None,
        },
    }


def convert_jsonl_to_parquet(
    *,
    input_jsonl: str | Path,
    output_parquet: str | Path,
    split_name: str,
    max_rows: int | None = None,
    batch_size: int = 4096,
) -> ConversionStats:
    input_path = Path(input_jsonl)
    output_path = Path(output_parquet)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    rows_accepted = 0
    row_limit = max_rows if max_rows is not None and max_rows > 0 else None
    batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None

    try:
        for idx, row in enumerate(_iter_jsonl(input_path)):
            if row_limit is not None and rows_accepted >= row_limit:
                break
            batch.append(_as_verl_row(row, row_index=idx, split_name=split_name))
            rows_accepted += 1
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=VERL_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, VERL_SCHEMA, compression="zstd")
                writer.write_table(table)
                rows_written += len(batch)
                batch.clear()

        if batch:
            table = pa.Table.from_pylist(batch, schema=VERL_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(output_path, VERL_SCHEMA, compression="zstd")
            writer.write_table(table)
            rows_written += len(batch)
            batch.clear()
    finally:
        if writer is not None:
            writer.close()

    if rows_written == 0:
        raise ValueError(f"No rows written from {input_path}")

    return ConversionStats(
        split=split_name,
        input_path=str(input_path),
        output_path=str(output_path),
        rows=rows_written,
        bytes=output_path.stat().st_size,
    )


def combine_validation_parquets(
    *,
    inputs: list[str | Path],
    output_parquet: str | Path,
    split_name: str = "validation_combined",
) -> ConversionStats:
    input_paths = [Path(path) for path in inputs if Path(path).exists()]
    if not input_paths:
        raise FileNotFoundError("No validation parquet inputs were found for combined validation.")

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tables = [pq.read_table(path, schema=VERL_SCHEMA) for path in input_paths]
    combined = pa.concat_tables(tables, promote_options="none")
    pq.write_table(combined, output_path, compression="zstd")

    return ConversionStats(
        split=split_name,
        input_path=",".join(str(path) for path in input_paths),
        output_path=str(output_path),
        rows=combined.num_rows,
        bytes=output_path.stat().st_size,
    )


def write_manifest(path: Path, stats: list[ConversionStats]) -> None:
    payload = {"splits": [asdict(item) for item in stats]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MolLang prepared JSONL to verl parquet.")
    parser.add_argument("--train-jsonl", required=True, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", required=True, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--bench-test-jsonl", default=DEFAULT_BENCH_TEST_JSONL)
    parser.add_argument("--bench-extended-jsonl", default=DEFAULT_BENCH_EXTENDED_JSONL)
    parser.add_argument("--output-dir", default="data/verl_qwen35")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-bench-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--skip-bench", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    stats = [
        convert_jsonl_to_parquet(
            input_jsonl=args.train_jsonl,
            output_parquet=output_dir / "train.parquet",
            split_name="train_generated",
            max_rows=args.max_train_rows,
            batch_size=args.batch_size,
        ),
        convert_jsonl_to_parquet(
            input_jsonl=args.val_jsonl,
            output_parquet=output_dir / "val.parquet",
            split_name="validated_finalpass",
            max_rows=args.max_val_rows,
            batch_size=args.batch_size,
        ),
    ]
    if not args.skip_bench:
        for input_jsonl, split, name in [
            (args.bench_test_jsonl, "bench_test", "bench_test.parquet"),
            (args.bench_extended_jsonl, "bench_extended", "bench_extended.parquet"),
        ]:
            if input_jsonl and Path(input_jsonl).exists():
                stats.append(
                    convert_jsonl_to_parquet(
                        input_jsonl=input_jsonl,
                        output_parquet=output_dir / name,
                        split_name=split,
                        max_rows=args.max_bench_rows,
                        batch_size=args.batch_size,
                    )
                )
    validation_inputs = [output_dir / "val.parquet"]
    if not args.skip_bench:
        validation_inputs.extend(
            [
                output_dir / "bench_test.parquet",
                output_dir / "bench_extended.parquet",
            ]
        )
    stats.append(
        combine_validation_parquets(
            inputs=validation_inputs,
            output_parquet=output_dir / "val_mollangbench_combined.parquet",
        )
    )
    write_manifest(output_dir / "manifest.json", stats)
    print(json.dumps({"splits": [asdict(item) for item in stats]}, indent=2))


if __name__ == "__main__":
    main()
