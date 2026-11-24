from __future__ import annotations

import pytest

from rl import reward


pytestmark = pytest.mark.skipif(reward.Chem is None, reason="RDKit is not installed")


def test_exact_smiles_scores_high() -> None:
    out = reward.compute_score(
        data_source="mollang_smiles",
        solution_str="<reasoning>short</reasoning><smiles>CCO</smiles>",
        ground_truth="CCO",
    )
    assert out["valid"] == 1.0
    assert out["exact"] == 1.0
    assert out["score"] == pytest.approx(1.5)


def test_uses_answer_after_thinking() -> None:
    out = reward.compute_score(
        data_source="mollang_smiles",
        solution_str="<think>wrong <smiles>CCC</smiles></think><smiles>CCO</smiles>",
        ground_truth="CCO",
    )
    assert out["exact"] == 1.0


def test_invalid_completion_penalty() -> None:
    out = reward.compute_score(
        data_source="mollang_smiles",
        solution_str="not a smiles answer",
        ground_truth="CCO",
    )
    assert out["valid"] == 0.0
    assert out["score"] == pytest.approx(-1.0)
