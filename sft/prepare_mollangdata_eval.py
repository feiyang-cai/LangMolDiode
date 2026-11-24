#!/usr/bin/env python
"""Prepare MolLangData exact-match generation eval JSONL from local parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_SYSTEM_PROMPT = (
    "You are a chemistry assistant specialized in molecular structure generation. "
    "Given an unambiguous molecular structure description, return the corresponding "
    "SMILES string enclosed in <smiles> and </smiles> tags."
)

DEFAULT_PROMPT_TEMPLATE = """**Instructions:**

Analyze the given molecule structure description and generate the corresponding molecular structure as SMILES representation.

- This description provides an unambiguous and detailed account sufficient to completely reconstruct the molecule.
- Interpret stereochemistry, charges, and isotopes if mentioned in the description.
- Return the resulting SMILES string enclosed within <smiles> and </smiles> tags.

**Molecule structure description:**

{description}
"""


def render_plain_text(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content'].strip()}<|im_end|>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-template-file", type=Path, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--final-pass-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    prompt_template = (
        args.prompt_template_file.read_text(encoding="utf-8")
        if args.prompt_template_file is not None
        else DEFAULT_PROMPT_TEMPLATE
    )

    table = pq.read_table(args.parquet)
    if args.final_pass_only and "final_pass" in table.column_names:
        table = table.filter(pc.fill_null(table["final_pass"], False))

    rows = table.to_pylist()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            description = row.get("description")
            smiles = row.get("smiles")
            if not isinstance(description, str) or not description.strip():
                continue
            if not isinstance(smiles, str) or not smiles.strip():
                continue

            user_prompt = prompt_template.format(description=description.strip())
            assistant = f"<smiles>{smiles.strip()}</smiles>"
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": user_prompt.strip()},
                {"role": "assistant", "content": assistant},
            ]
            output = {
                "messages": messages,
                "text": render_plain_text(messages),
                "smiles": smiles.strip(),
                "expected_smiles": smiles.strip(),
                "cid": row.get("cid"),
                "difficulty_level": row.get("difficulty_level"),
                "generation_model": row.get("generation_model"),
                "reasoning_effort": row.get("reasoning_effort"),
                "final_pass": row.get("final_pass"),
                "source_dataset": "mollangdata/MolLangData",
                "source_file": str(args.parquet),
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
            if args.limit is not None and written >= args.limit:
                break

    print(
        json.dumps(
            {
                "written": written,
                "parquet": str(args.parquet),
                "final_pass_only": args.final_pass_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
