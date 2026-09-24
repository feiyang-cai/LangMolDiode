from __future__ import annotations

import pytest

from sft.export_sft_corpus import to_training_row


def corpus_row(**overrides):
    row = {
        "id": "sft_000001",
        "smiles": "CCO",
        "system_prompt": "Return SMILES.",
        "input_prompt": "A two-carbon alcohol.",
        "sft_response": "<think>Identify ethanol.</think>\n<smiles>CCO</smiles>",
        "generated_smiles": "OCC",
        "generation_model": "teacher-model",
        "reasoning_effort_effective": "high",
        "matches_smiles": True,
    }
    row.update(overrides)
    return row


def test_to_training_row_builds_chat_messages():
    output = to_training_row(corpus_row())

    assert [message["role"] for message in output["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert output["messages"][-1]["content"].endswith("<smiles>CCO</smiles>")
    assert output["smiles"] == "CCO"
    assert output["candidate_smiles"] == "OCC"
    assert output["curator_model"] == "teacher-model"
    assert output["reasoning_effort"] == "high"


def test_to_training_row_rejects_unverified_example():
    with pytest.raises(ValueError, match="not a verified molecular match"):
        to_training_row(corpus_row(matches_smiles=False))
