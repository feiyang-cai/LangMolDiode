#!/usr/bin/env python3
"""Build a portable, verified Parquet release from the exact SFT JSONL."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.error")
SMILES_TAG = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.IGNORECASE | re.DOTALL)
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("molecule_id", pa.string()),
        ("smiles", pa.string()),
        ("system_prompt", pa.string()),
        ("input_prompt", pa.string()),
        ("generated_response", pa.string()),
        ("generated_reasoning", pa.string()),
        ("generated_smiles", pa.string()),
        ("sft_response", pa.string()),
        ("generation_model", pa.string()),
        ("reasoning_effort_requested", pa.string()),
        ("reasoning_effort_effective", pa.string()),
        ("matches_smiles", pa.bool_()),
    ]
)


def canonical_smiles(smiles: str, row_id: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"{row_id}: invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def message_content(row: dict, role: str, row_id: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{row_id}: missing messages")
    matches = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == role]
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise ValueError(f"{row_id}: expected one nonempty {role} message")
    return matches[0]


def make_record(row: dict, index: int) -> dict:
    row_id = f"sft_{index:06d}"
    artifact_path = row.get("source_file")
    if not isinstance(artifact_path, str) or not Path(artifact_path).is_file():
        raise ValueError(f"{row_id}: source artifact missing")
    with Path(artifact_path).open(encoding="utf-8") as handle:
        artifact = json.load(handle)

    validation = artifact.get("validation") or {}
    model = artifact.get("model") or {}
    source = artifact.get("source") or {}
    selection = artifact.get("selection") or {}
    choices = (artifact.get("response") or {}).get("choices") or []
    message = choices[0].get("message") or {} if choices else {}
    raw_response = artifact.get("output_text") or message.get("content")
    raw_reasoning = message.get("reasoning") or ""
    generated_smiles = validation.get("candidate_smiles") or validation.get("extracted_smiles")
    target_smiles = row.get("smiles")
    model_name = model.get("model_name") or model.get("deployment")
    sft_response = message_content(row, "assistant", row_id)
    final_tags = SMILES_TAG.findall(sft_response)

    if validation.get("match") is not True:
        raise ValueError(f"{row_id}: source validation is not a match")
    if not all(isinstance(x, str) and x for x in (raw_response, generated_smiles, target_smiles, model_name)):
        raise ValueError(f"{row_id}: required generation metadata missing")
    if row.get("curator_model") != model_name:
        raise ValueError(f"{row_id}: model metadata disagrees with SFT row")
    if len(final_tags) != 1:
        raise ValueError(f"{row_id}: expected one final SMILES tag in SFT response")

    canonical_target = canonical_smiles(target_smiles, row_id)
    if canonical_smiles(generated_smiles, row_id) != canonical_target:
        raise ValueError(f"{row_id}: generated molecule does not match target")
    if canonical_smiles(final_tags[0], row_id) != canonical_target:
        raise ValueError(f"{row_id}: SFT response does not match target")
    if validation.get("expected_smiles") and canonical_smiles(validation["expected_smiles"], row_id) != canonical_target:
        raise ValueError(f"{row_id}: artifact target disagrees with SFT row")

    molecule_id = selection.get("cid") or source.get("cid")
    return {
        "id": row_id,
        "molecule_id": str(molecule_id) if molecule_id is not None else None,
        "smiles": target_smiles,
        "system_prompt": message_content(row, "system", row_id),
        "input_prompt": message_content(row, "user", row_id),
        "generated_response": raw_response,
        "generated_reasoning": raw_reasoning if isinstance(raw_reasoning, str) else "",
        "generated_smiles": generated_smiles,
        "sft_response": sft_response,
        "generation_model": model_name,
        "reasoning_effort_requested": model.get("requested_reasoning_effort"),
        "reasoning_effort_effective": model.get("effective_reasoning_effort"),
        "matches_smiles": True,
    }


def write_card(path: Path, row_count: int) -> None:
    card = f"""---
language:
  - en
task_categories:
  - text-generation
tags:
  - chemistry
  - text-to-smiles
  - supervised-fine-tuning
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
---

# LangMolDiode SFT Corpus

The private training corpus contains {row_count:,} examples from the filtered
Qwen3.5-4B SFT run (maximum training sequence length: 40,960 tokens).
Each row is one training example, in original training order.

`id` is a release-local row ID and `molecule_id` is the source molecule ID when
available. `smiles` is the target molecule. `system_prompt` and `input_prompt`
are the exact SFT messages. `generated_response` and `generated_reasoning` are
the original generator's answer and separate reasoning field; `generated_smiles`
is the SMILES extracted during curation. `sft_response` is the exact assistant
message used for SFT, including its reasoning and canonical target SMILES.
`generation_model` and both reasoning-effort fields come from the generator's
saved metadata. `matches_smiles` checks chemical equivalence, including
stereochemistry, between the generated and target SMILES; it does not mean
that their SMILES strings are identical.

Every released row has `matches_smiles=true`, independently checked with RDKit
against the saved curation decision. No local source-file paths are published.
Before making this dataset public, review source-data and generated-output
redistribution rights and the intended blind-review policy.
"""
    path.write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Exact filtered SFT JSONL")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, help="For local smoke tests only")
    args = parser.parse_args()
    if args.shard_size <= 0 or args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("sizes and limit must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be new or empty")

    with args.input.open(encoding="utf-8") as handle:
        total = sum(bool(line.strip()) for line in handle)
    total = min(total, args.limit) if args.limit else total
    if total == 0:
        parser.error("input is empty")
    shard_count = math.ceil(total / args.shard_size)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    counts = {"generation_model": Counter(), "reasoning_effort_effective": Counter()}
    writer = None
    batch = []
    written = 0

    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if written == total:
                break
            row = json.loads(line)
            record = make_record(row, written + 1)
            if written % args.shard_size == 0:
                if writer is not None:
                    if batch:
                        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                        batch.clear()
                    writer.close()
                shard_index = written // args.shard_size
                shard_path = data_dir / f"train-{shard_index:05d}-of-{shard_count:05d}.parquet"
                writer = pq.ParquetWriter(shard_path, SCHEMA, compression="zstd", compression_level=6)
            batch.append(record)
            written += 1
            counts["generation_model"][record["generation_model"]] += 1
            counts["reasoning_effort_effective"][record["reasoning_effort_effective"] or "unknown"] += 1
            if len(batch) >= args.batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                batch.clear()
            if written % 5000 == 0:
                print(f"validated {written}/{total}", flush=True)
    if writer is not None:
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        writer.close()
    if written != total:
        raise RuntimeError(f"expected {total} rows; wrote {written}")

    summary = {"rows": written, "shards": shard_count, "all_matches": True,
               "generation_model": dict(counts["generation_model"]),
               "reasoning_effort_effective": dict(counts["reasoning_effort_effective"])}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_card(args.output_dir / "README.md", written)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
