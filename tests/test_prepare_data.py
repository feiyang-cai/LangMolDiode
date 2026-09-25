from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from rl.prepare_data import (
    DEFAULT_PROMPT_TEMPLATE,
    convert_jsonl_to_parquet,
    iter_released_rows,
    prepare_released_row,
)


def test_convert_jsonl_to_parquet(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps(
            {
                "sample_id": "a",
                "prompt": "Describe ethanol.",
                "description": "ethanol",
                "gt_smiles": "CCO",
                "split": "train",
                "final_pass": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.parquet"
    stats = convert_jsonl_to_parquet(input_jsonl=src, output_parquet=out, split_name="train")
    table = pq.read_table(out)
    row = table.to_pylist()[0]
    assert stats.rows == 1
    assert row["data_source"] == "mollang_train"
    assert row["prompt"][0]["role"] == "user"
    assert row["reward_model"]["ground_truth"] == "CCO"


def test_convert_jsonl_to_parquet_respects_max_rows(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": str(idx),
                    "prompt": f"Describe molecule {idx}.",
                    "gt_smiles": "CCO",
                }
            )
            + "\n"
            for idx in range(5)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.parquet"
    stats = convert_jsonl_to_parquet(
        input_jsonl=src,
        output_parquet=out,
        split_name="train",
        max_rows=2,
        batch_size=4096,
    )
    assert stats.rows == 2
    assert pq.read_table(out).num_rows == 2


def test_prepare_mollangdata_release_row() -> None:
    row = prepare_released_row(
        {
            "cid": "42",
            "description": "A molecule with two carbon atoms and one oxygen atom.",
            "smiles": "CCO",
            "final_pass": True,
        },
        row_index=0,
        split_name="validated_finalpass",
        require_final_pass=True,
    )
    assert row is not None
    assert row["sample_id"] == "42"
    assert row["gt_smiles"] == "CCO"
    assert row["prompt"] == DEFAULT_PROMPT_TEMPLATE.format(
        description="A molecule with two carbon atoms and one oxygen atom."
    )
    assert "Return your reasoning" not in row["prompt"]


def test_prepare_mollangbench_release_row() -> None:
    row = prepare_released_row(
        {"structure_description": "A benzene ring.", "smiles": "c1ccccc1"},
        row_index=7,
        split_name="bench_test",
    )
    assert row is not None
    assert row["sample_id"] == "7"
    assert row["description"] == "A benzene ring."


def test_iter_released_rows_filters_failed_validation() -> None:
    rows = list(
        iter_released_rows(
            [
                {"cid": "bad", "description": "bad", "smiles": "C", "final_pass": False},
                {"cid": "good", "description": "good", "smiles": "CC", "final_pass": True},
            ],
            split_name="validated_finalpass",
            require_final_pass=True,
        )
    )
    assert [row["sample_id"] for row in rows] == ["good"]
