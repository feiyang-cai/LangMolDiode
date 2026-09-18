from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from rl.prepare_data import convert_jsonl_to_parquet


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
