from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_MOLLANGDATA_DATASET = "mollangdata/MolLangData"
DEFAULT_MOLLANGDATA_TRAIN_CONFIG = "generated_data"
DEFAULT_MOLLANGDATA_VAL_CONFIG = "validated_data"
DEFAULT_MOLLANGDATA_SPLIT = "data"
DEFAULT_MOLLANGBENCH_DATASET = "ChemFM/MolLangBench"
DEFAULT_MOLLANGBENCH_CONFIG = "generation"
DEFAULT_PROMPT_TEMPLATE = """**Instructions:**
Analyze the given molecule structure description and generate the corresponding molecular structure as SMILES representation.

- This description provides an unambiguous and detailed account sufficient to completely reconstruct the molecule.
- Interpret stereochemistry, charges, and isotopes if mentioned in the description.
- Return the resulting SMILES string enclosed within <smiles> and </smiles> tags.

**Molecule structure description:**
{description}"""


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


def _first_string(row: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def prepare_released_row(
    row: dict[str, Any],
    *,
    row_index: int,
    split_name: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    require_final_pass: bool = False,
) -> dict[str, Any] | None:
    final_pass = row.get("final_pass")
    if require_final_pass and final_pass is not True:
        return None

    description = _first_string(
        row,
        ("description", "structure_description", "text", "instruction"),
    )
    gt_smiles = _first_string(row, ("smiles", "SMILES", "target_smiles", "mol_smiles"))
    if description is None:
        raise ValueError(f"Released row {row_index} is missing a molecular description")
    if gt_smiles is None:
        raise ValueError(f"Released row {row_index} is missing a target SMILES")

    sample_id = next(
        (
            str(row[name])
            for name in ("sample_id", "id", "uid", "uuid", "cid")
            if row.get(name) is not None
        ),
        str(row_index),
    )
    try:
        prompt = prompt_template.format(
            description=description,
            structure_description=description,
        )
    except KeyError as exc:
        raise ValueError(
            "Prompt template must use {description} or {structure_description}"
        ) from exc
    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "description": description,
        "gt_smiles": gt_smiles,
        "split": split_name,
        "final_pass": final_pass if isinstance(final_pass, bool) else None,
    }


def iter_released_rows(
    rows: Iterable[dict[str, Any]],
    *,
    split_name: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    require_final_pass: bool = False,
) -> Iterable[dict[str, Any]]:
    for row_index, row in enumerate(rows):
        prepared = prepare_released_row(
            dict(row),
            row_index=row_index,
            split_name=split_name,
            prompt_template=prompt_template,
            require_final_pass=require_final_pass,
        )
        if prepared is not None:
            yield prepared


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


def convert_rows_to_parquet(
    *,
    rows: Iterable[dict[str, Any]],
    input_name: str,
    output_parquet: str | Path,
    split_name: str,
    max_rows: int | None = None,
    batch_size: int = 4096,
) -> ConversionStats:
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    row_limit = max_rows if max_rows is not None and max_rows > 0 else None
    batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None

    try:
        for row_index, row in enumerate(rows):
            if row_limit is not None and rows_written + len(batch) >= row_limit:
                break
            batch.append(_as_verl_row(row, row_index=row_index, split_name=split_name))
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
    finally:
        if writer is not None:
            writer.close()

    if rows_written == 0:
        raise ValueError(f"No rows written from {input_name}")

    return ConversionStats(
        split=split_name,
        input_path=input_name,
        output_path=str(output_path),
        rows=rows_written,
        bytes=output_path.stat().st_size,
    )


def convert_jsonl_to_parquet(
    *,
    input_jsonl: str | Path,
    output_parquet: str | Path,
    split_name: str,
    max_rows: int | None = None,
    batch_size: int = 4096,
) -> ConversionStats:
    input_path = Path(input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    return convert_rows_to_parquet(
        rows=_iter_jsonl(input_path),
        input_name=str(input_path),
        output_parquet=output_parquet,
        split_name=split_name,
        max_rows=max_rows,
        batch_size=batch_size,
    )


def combine_validation_parquets(
    *,
    inputs: list[str | Path],
    output_parquet: str | Path,
    split_name: str = "validation_combined",
) -> ConversionStats:
    input_paths = [Path(path) for path in inputs]
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Validation parquet inputs are missing: {missing}")

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


def _load_hub_rows(
    *,
    dataset_name: str,
    config_name: str,
    split: str,
    columns: tuple[str, ...],
    streaming: bool,
):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    missing = set(columns).difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"{dataset_name}/{config_name}/{split} is missing columns: {sorted(missing)}"
        )
    return dataset.select_columns(list(columns))


def prepare_hub_datasets(args: argparse.Namespace, prompt_template: str) -> list[ConversionStats]:
    output_dir = Path(args.output_dir)
    split_specs = [
        (
            args.mollangdata_dataset,
            args.mollangdata_train_config,
            args.mollangdata_split,
            ("cid", "description", "smiles", "final_pass"),
            "train_generated",
            "train.parquet",
            False,
            args.max_train_rows,
        ),
        (
            args.mollangdata_dataset,
            args.mollangdata_val_config,
            args.mollangdata_split,
            ("cid", "description", "smiles", "final_pass"),
            "validated_finalpass",
            "val.parquet",
            True,
            args.max_val_rows,
        ),
    ]
    if not args.skip_bench:
        split_specs.extend(
            [
                (
                    args.mollangbench_dataset,
                    args.mollangbench_config,
                    "test",
                    ("structure_description", "smiles"),
                    "bench_test",
                    "bench_test.parquet",
                    False,
                    args.max_bench_rows,
                ),
                (
                    args.mollangbench_dataset,
                    args.mollangbench_config,
                    "extended",
                    ("structure_description", "smiles"),
                    "bench_extended",
                    "bench_extended.parquet",
                    False,
                    args.max_bench_rows,
                ),
            ]
        )

    stats: list[ConversionStats] = []
    for dataset_name, config_name, split, columns, split_name, filename, final_only, limit in split_specs:
        source = f"hf://datasets/{dataset_name}/{config_name}/{split}"
        source_rows = _load_hub_rows(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            columns=columns,
            streaming=args.streaming,
        )
        prepared_rows = iter_released_rows(
            source_rows,
            split_name=split_name,
            prompt_template=prompt_template,
            require_final_pass=final_only,
        )
        stats.append(
            convert_rows_to_parquet(
                rows=prepared_rows,
                input_name=source,
                output_parquet=output_dir / filename,
                split_name=split_name,
                max_rows=limit,
                batch_size=args.batch_size,
            )
        )
    return stats


def prepare_local_jsonl(args: argparse.Namespace) -> list[ConversionStats]:
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
        for input_jsonl, split_name, filename in (
            (args.bench_test_jsonl, "bench_test", "bench_test.parquet"),
            (args.bench_extended_jsonl, "bench_extended", "bench_extended.parquet"),
        ):
            stats.append(
                convert_jsonl_to_parquet(
                    input_jsonl=input_jsonl,
                    output_parquet=output_dir / filename,
                    split_name=split_name,
                    max_rows=args.max_bench_rows,
                    batch_size=args.batch_size,
                )
            )
    return stats


def write_manifest(path: Path, stats: list[ConversionStats]) -> None:
    payload = {"splits": [asdict(item) for item in stats]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare released MolLangData and MolLangBench generation data for verl."
    )
    parser.add_argument("--train-jsonl")
    parser.add_argument("--val-jsonl")
    parser.add_argument("--bench-test-jsonl")
    parser.add_argument("--bench-extended-jsonl")
    parser.add_argument("--mollangdata-dataset", default=DEFAULT_MOLLANGDATA_DATASET)
    parser.add_argument("--mollangdata-train-config", default=DEFAULT_MOLLANGDATA_TRAIN_CONFIG)
    parser.add_argument("--mollangdata-val-config", default=DEFAULT_MOLLANGDATA_VAL_CONFIG)
    parser.add_argument("--mollangdata-split", default=DEFAULT_MOLLANGDATA_SPLIT)
    parser.add_argument("--mollangbench-dataset", default=DEFAULT_MOLLANGBENCH_DATASET)
    parser.add_argument("--mollangbench-config", default=DEFAULT_MOLLANGBENCH_CONFIG)
    parser.add_argument("--prompt-template-file", type=Path)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="data/verl_qwen35")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-bench-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    local_values = [args.train_jsonl, args.val_jsonl]
    if any(local_values) and not all(local_values):
        parser.error("--train-jsonl and --val-jsonl must be supplied together")
    if all(local_values) and not args.skip_bench:
        if not args.bench_test_jsonl or not args.bench_extended_jsonl:
            parser.error(
                "local JSONL mode requires both MolLangBench files unless --skip-bench is set"
            )
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prompt_template = (
        args.prompt_template_file.read_text(encoding="utf-8").strip()
        if args.prompt_template_file is not None
        else DEFAULT_PROMPT_TEMPLATE
    )
    if args.train_jsonl:
        stats = prepare_local_jsonl(args)
    else:
        stats = prepare_hub_datasets(args, prompt_template)

    validation_inputs = [output_dir / "val.parquet"]
    if not args.skip_bench:
        validation_inputs.extend(
            [output_dir / "bench_test.parquet", output_dir / "bench_extended.parquet"]
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
