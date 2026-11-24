from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
except Exception:  # pragma: no cover - depends on the training image.
    Chem = None
    DataStructs = None
    GetMorganGenerator = None


SMILES_TAG_PATTERN = re.compile(r"<smiles\b[^>]*>(.*?)</smiles\s*>", re.IGNORECASE | re.DOTALL)
SMILES_OPEN_TAG_PATTERN = re.compile(r"<smiles\b[^>]*>", re.IGNORECASE)
SMILES_CLOSE_TAG_PATTERN = re.compile(r"</smiles\s*>", re.IGNORECASE)
REASONING_TAG_PATTERN = re.compile(r"<reasoning\b[^>]*>(.*?)</reasoning\s*>", re.IGNORECASE | re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
REASONING_MARKUP_PATTERN = re.compile(r"</?(?:think|reasoning)\b[^>]*>", re.IGNORECASE)
UNCLOSED_SMILES_LABEL_PATTERN = re.compile(r"^(?:smiles|answer|final answer)\s*[:=]\s*", re.IGNORECASE)
UNCLOSED_SMILES_CANDIDATE_PATTERN = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+$")
UNCLOSED_SMILES_REJECT_WORDS = {"and", "answer", "final", "smiles", "tag", "tags"}


class RewardConfig:
    DEFAULTS = {
        "invalid_penalty": -1.0,
        "tanimoto_weight": 1.0,
        "exact_weight": 0.4,
        "format_weight": 0.1,
        "reasoning_length_penalty_weight": 0.0,
        "reasoning_soft_limit_tokens": 4096,
    }

    __slots__ = tuple(DEFAULTS)

    def __init__(self, **kwargs: Any) -> None:
        values = dict(self.DEFAULTS)
        values.update({key: value for key, value in kwargs.items() if key in self.DEFAULTS})
        for key, value in values.items():
            setattr(self, key, value)


def _load_reward_config(extra_info: Any = None) -> RewardConfig:
    payload: dict[str, Any] = {}
    env_payload = os.environ.get("MOLLANG_REWARD_CONFIG_JSON")
    if env_payload:
        payload.update(json.loads(env_payload))
    if isinstance(extra_info, dict) and isinstance(extra_info.get("reward_config"), dict):
        payload.update(extra_info["reward_config"])
    allowed = set(RewardConfig.DEFAULTS)
    return RewardConfig(**{key: value for key, value in payload.items() if key in allowed})


def normalize_smiles_text(smiles: str) -> str:
    return "".join(str(smiles).split())


def smiles_to_mol(smiles: str):
    if Chem is None:
        raise RuntimeError("RDKit is required for MolLang reward scoring.")
    if smiles is None:
        return None
    text = normalize_smiles_text(smiles)
    if not text:
        return None
    try:
        return Chem.MolFromSmiles(text, sanitize=True)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> str | None:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


@lru_cache(maxsize=16)
def _morgan_generator(radius: int, nbits: int):
    if GetMorganGenerator is None:
        raise RuntimeError("RDKit is required for MolLang reward scoring.")
    return GetMorganGenerator(radius=int(radius), fpSize=int(nbits))


def tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2, nbits: int = 2048) -> float | None:
    if DataStructs is None:
        raise RuntimeError("RDKit is required for MolLang reward scoring.")
    mol_a = smiles_to_mol(smiles_a)
    mol_b = smiles_to_mol(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    gen = _morgan_generator(radius=radius, nbits=nbits)
    return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(mol_a), gen.GetFingerprint(mol_b)))


def _looks_like_smiles(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate or len(candidate) > 512:
        return False
    if any(ch.isspace() for ch in candidate) or "<" in candidate or ">" in candidate:
        return False
    allowed = set("#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]\\abcdefghijklmnopqrstuvwxyz")
    return all(ch in allowed for ch in candidate)


def extract_last_tag_content(text: str, pattern: re.Pattern[str]) -> str | None:
    if not text:
        return None
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return (matches[-1].group(1) or "").strip()


def extract_tag_content(text: str, pattern: re.Pattern[str]) -> str | None:
    if not text:
        return None
    match = pattern.search(text)
    if match is None:
        return None
    return (match.group(1) or "").strip()


def _remove_smiles_blocks(text: str) -> str:
    return SMILES_TAG_PATTERN.sub("", str(text or ""))


def _remove_smiles_markup(text: str) -> str:
    text = _remove_smiles_blocks(text)
    text = SMILES_OPEN_TAG_PATTERN.sub("", text)
    text = SMILES_CLOSE_TAG_PATTERN.sub("", text)
    return text


def _sanitize_reasoning_for_reward(text: str) -> str:
    text = _remove_smiles_markup(text)
    return REASONING_MARKUP_PATTERN.sub("", text).strip()


def _extract_unclosed_smiles_candidate(text: str) -> str:
    tail = UNCLOSED_SMILES_LABEL_PATTERN.sub("", str(text or "").strip())
    if not tail:
        return ""
    token = tail.split()[0].strip("`'\".,;")
    if not token or token.lower() in UNCLOSED_SMILES_REJECT_WORDS:
        return ""
    if UNCLOSED_SMILES_CANDIDATE_PATTERN.match(token):
        return token
    return ""


def _answer_for_reward(text: str) -> str:
    answer = str(text or "").strip()
    opens = list(SMILES_OPEN_TAG_PATTERN.finditer(answer))
    if not opens:
        return answer
    final_open = opens[-1]
    final_close = SMILES_CLOSE_TAG_PATTERN.search(answer, final_open.end())
    if final_close is not None:
        return answer[final_open.start() : final_close.end()].strip()
    candidate = _extract_unclosed_smiles_candidate(answer[final_open.end() :])
    if candidate:
        return f"<smiles>{candidate}</smiles>"
    return answer[final_open.start() :].strip()


def completion_for_reward(text: str) -> str:
    raw = str(text or "")
    lowered = raw.lower()
    close_positions = [lowered.rfind("</think>"), lowered.rfind("</reasoning>")]
    close_idx = max(close_positions)
    if close_idx >= 0:
        close_tag = "</think>" if lowered.startswith("</think>", close_idx) else "</reasoning>"
        answer_start = close_idx + len(close_tag)
        reasoning = _sanitize_reasoning_for_reward(raw[:answer_start])
        answer = _answer_for_reward(raw[answer_start:])
        if reasoning:
            return f"<reasoning>{reasoning}</reasoning>\n{answer}".strip()
        return answer
    if "<think" in lowered or "<reasoning" in lowered:
        return f"<reasoning>{_sanitize_reasoning_for_reward(raw)}</reasoning>"
    return _answer_for_reward(raw)


def extract_smiles(text: str) -> str | None:
    out = extract_last_tag_content(text, SMILES_TAG_PATTERN)
    if out:
        return out
    lowered = str(text).lower()
    think_close = lowered.rfind("</think>")
    if think_close != -1:
        trailing = str(text)[think_close + len("</think>") :].strip()
        if _looks_like_smiles(trailing):
            return trailing
    stripped = str(text).strip()
    return stripped if _looks_like_smiles(stripped) else None


def extract_reasoning(text: str) -> str | None:
    out = extract_tag_content(text, REASONING_TAG_PATTERN)
    if out:
        return out
    out = extract_tag_content(text, THINK_TAG_PATTERN)
    if out:
        return out
    lowered = str(text).lower()
    think_close = lowered.rfind("</think>")
    if think_close != -1:
        return str(text)[:think_close].strip() or None
    return None


def _reasoning_length_penalty(reasoning: str | None, limit_tokens: int) -> float:
    if not reasoning:
        return 0.0
    token_count = len(reasoning.split())
    if token_count <= limit_tokens:
        return 0.0
    return min(1.0, float(token_count - limit_tokens) / float(max(limit_tokens, 1)))


def score_completion(gt_smiles: str, completion: str, cfg: RewardConfig | None = None) -> dict[str, float]:
    cfg = cfg or RewardConfig()
    parsed_completion = completion_for_reward(completion)
    pred_smiles = extract_smiles(parsed_completion)
    pred_canon = canonicalize_smiles(pred_smiles) if pred_smiles else None
    gt_canon = canonicalize_smiles(gt_smiles)
    lowered = str(completion).lower()
    format_ok = 1.0 if ("<smiles>" in lowered and "</smiles>" in lowered) else 0.0
    reasoning_pen = _reasoning_length_penalty(extract_reasoning(parsed_completion), cfg.reasoning_soft_limit_tokens)

    if pred_canon is None:
        if pred_smiles:
            reward = -0.6 + 0.2 * format_ok - cfg.reasoning_length_penalty_weight * reasoning_pen
        else:
            reward = float(cfg.invalid_penalty)
        return {
            "score": float(reward),
            "valid": 0.0,
            "exact": 0.0,
            "tanimoto": 0.0,
            "format_ok": format_ok,
            "reasoning_length_penalty": reasoning_pen,
        }

    exact = 1.0 if (gt_canon is not None and pred_canon == gt_canon) else 0.0
    tanimoto = float(tanimoto_similarity(gt_smiles, pred_smiles or "") or 0.0)
    reward = (
        cfg.tanimoto_weight * tanimoto
        + cfg.exact_weight * exact
        + cfg.format_weight * format_ok
        - cfg.reasoning_length_penalty_weight * reasoning_pen
    )
    return {
        "score": float(reward),
        "valid": 1.0,
        "exact": exact,
        "tanimoto": tanimoto,
        "format_ok": format_ok,
        "reasoning_length_penalty": reasoning_pen,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    """verl custom reward entrypoint."""
    _ = data_source
    return score_completion(gt_smiles=str(ground_truth), completion=str(solution_str), cfg=_load_reward_config(extra_info))


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[dict[str, float]]:
    _ = data_sources, kwargs
    extra_infos = extra_infos or [{} for _ in solution_strs]
    return [
        score_completion(gt_smiles=str(gt), completion=str(sol), cfg=_load_reward_config(extra))
        for sol, gt, extra in zip(solution_strs, ground_truths, extra_infos, strict=True)
    ]
