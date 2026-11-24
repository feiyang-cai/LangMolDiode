from __future__ import annotations

import json
import sys
from pathlib import Path

import sft.curate_sft_corpus as curate
from sft.curate_sft_corpus import canonicalize_smiles, curate_row, extract_smiles


def test_extract_smiles_uses_last_tag() -> None:
    text = "Reason with <smiles>C</smiles>. Final: <smiles>CCO</smiles>"
    assert extract_smiles(text) == "CCO"


def test_canonicalize_smiles_preserves_molecular_equivalence() -> None:
    assert canonicalize_smiles("OCC") == canonicalize_smiles("CCO")


def test_curate_row_accepts_equivalent_smiles() -> None:
    def complete(_payload):
        return {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "This is a two-carbon alcohol.",
                        "content": "<smiles>OCC</smiles>",
                    }
                }
            ]
        }

    result = curate_row(
        {"id": "ethanol", "description": "A two-carbon alcohol.", "smiles": "CCO"},
        row_index=0,
        model="teacher-model",
        prompt_column="description",
        smiles_column="smiles",
        id_column="id",
        system_prompt="Return SMILES.",
        prompt_template="Molecule: {description}",
        temperature=0.6,
        max_tokens=1024,
        max_tokens_field="max_tokens",
        reasoning_effort=None,
        max_attempts=1,
        complete=complete,
    )

    assert result["match"] is True
    assert result["sample_id"] == "ethanol"
    assert result["candidate_smiles"] == "OCC"
    assert result["canonical_smiles"] == canonicalize_smiles("CCO")
    assert result["reasoning"] == "This is a two-carbon alcohol."


def test_cli_curates_local_jsonl(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "curated.jsonl"
    source.write_text(
        json.dumps({"id": "ethanol", "description": "A two-carbon alcohol.", "smiles": "CCO"})
        + "\n",
        encoding="utf-8",
    )

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "<smiles>OCC</smiles>"}}]}

    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(curate.requests, "post", post)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "curate_sft_corpus.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "teacher-model",
            "--workers",
            "1",
            "--max-attempts",
            "1",
        ],
    )

    curate.main()

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["match"] is True
    assert row["sample_id"] == "ethanol"
    assert calls[0][0] == "http://localhost:8000/v1/chat/completions"
