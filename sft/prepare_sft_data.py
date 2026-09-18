#!/usr/bin/env python
"""Prepare LangMolDiode chat SFT JSONL from curation artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SYSTEM_PROMPT = (
    "You are a chemistry assistant specialized in molecular structure generation. "
    "Given an unambiguous molecular structure description, return the corresponding "
    "SMILES string enclosed in <smiles> and </smiles> tags."
)

SMILES_RE = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.IGNORECASE | re.DOTALL)
SMILES_TAG_RE = re.compile(r"</?smiles>", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|thought)>\s*(.*?)\s*</(?:think|thinking|thought)>",
    re.IGNORECASE | re.DOTALL,
)
REASONING_TAG_RE = re.compile(r"</?(?:think|thinking|thought)>", re.IGNORECASE)
INSTRUCTIONS_RE = re.compile(r"^\s*\*\*Instructions:\*\*\s*", re.IGNORECASE)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(row, dict):
                row.setdefault("_source_file", str(path))
                yield row


def extract_smiles(text: str | None) -> str | None:
    if not text:
        return None
    match = SMILES_RE.search(text)
    return match.group(1).strip() if match else None


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    prompt = INSTRUCTIONS_RE.sub("", prompt)
    return prompt.strip()


def get_nested(data: dict[str, Any], keys: tuple[str | int, ...]) -> Any:
    cur: Any = data
    for key in keys:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return None
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
    return cur


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def reasoning_from_artifact_path(path: str | None) -> str | None:
    if not path:
        return None
    data = load_json(Path(path))
    if not data:
        return None
    message = get_nested(data, ("response", "choices", 0, "message"))
    reasoning = message.get("reasoning") if isinstance(message, dict) else None
    return reasoning if isinstance(reasoning, str) and reasoning.strip() else None


def row_from_artifact(path: Path) -> dict[str, Any] | None:
    data = load_json(path)
    if not data:
        return None

    prompt = read_text(path.parent / "prompt.txt")
    output_text = data.get("output_text") or read_text(path.parent / "output.txt")
    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    message = get_nested(data, ("response", "choices", 0, "message"))
    reasoning = message.get("reasoning") if isinstance(message, dict) else None

    return {
        "prompt": prompt,
        "output_text": output_text,
        "reasoning": reasoning,
        "match": validation.get("match"),
        "candidate_smiles": validation.get("candidate_smiles") or extract_smiles(output_text),
        "canonical_smiles": validation.get("candidate_canonical_smiles")
        or validation.get("expected_canonical_smiles"),
        "expected_smiles": validation.get("expected_smiles") or source.get("expected_smiles"),
        "source_dataset": source.get("dataset"),
        "source_row_index": first_present(
            get_nested(data, ("selection", "source_row_index")),
            source.get("row_index"),
        ),
        "difficulty_level": get_nested(data, ("selection", "difficulty_level"))
        or source.get("difficulty_level"),
        "provider": get_nested(data, ("model", "provider")),
        "curator_model": get_nested(data, ("model", "model_name"))
        or get_nested(data, ("model", "deployment")),
        "source_file": str(path),
    }


def row_from_example_dir(path: Path) -> dict[str, Any] | None:
    prompt = read_text(path / "prompt.txt")
    output_text = read_text(path / "output.txt")
    if not prompt or not output_text:
        return None
    validation = load_json(path / "validation.json") or {}
    return {
        "prompt": prompt,
        "output_text": output_text,
        "reasoning": None,
        "match": validation.get("match"),
        "candidate_smiles": validation.get("candidate_smiles") or extract_smiles(output_text),
        "canonical_smiles": validation.get("candidate_canonical_smiles")
        or validation.get("expected_canonical_smiles"),
        "expected_smiles": validation.get("expected_smiles"),
        "source_dataset": None,
        "source_row_index": None,
        "difficulty_level": None,
        "provider": None,
        "curator_model": None,
        "source_file": str(path),
    }


def row_from_jsonl_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
    messages = obj.get("messages")
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    training_example = (
        obj.get("training_example")
        if isinstance(obj.get("training_example"), dict)
        else {}
    )
    if isinstance(messages, list):
        user_messages = [
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        assistant_messages = [
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        prompt = user_messages[-1] if user_messages else None
        output_text = assistant_messages[-1] if assistant_messages else None
    else:
        prompt = (
            obj.get("prompt")
            or obj.get("input")
            or obj.get("instruction")
            or training_example.get("prompt")
        )
        output_text = (
            obj.get("output_text")
            or obj.get("output")
            or obj.get("response")
            or training_example.get("response")
        )
    validation = obj.get("validation") if isinstance(obj.get("validation"), dict) else {}
    source = obj.get("source") if isinstance(obj.get("source"), dict) else {}
    dataset = obj.get("dataset") if isinstance(obj.get("dataset"), dict) else {}
    model = obj.get("model") if isinstance(obj.get("model"), dict) else {}
    selection = obj.get("selection") if isinstance(obj.get("selection"), dict) else {}
    run = obj.get("run") if isinstance(obj.get("run"), dict) else {}
    source_file = run.get("artifact_path") or metadata.get("artifact_path") or obj.get("_source_file")
    reasoning = (
        obj.get("reasoning")
        or metadata.get("reasoning")
        or reasoning_from_artifact_path(source_file)
    )
    if isinstance(output_text, dict):
        output_text = output_text.get("content")
    if not isinstance(prompt, str) or not isinstance(output_text, str):
        return None
    return {
        "prompt": prompt,
        "output_text": output_text,
        "reasoning": reasoning,
        "match": validation.get("match", obj.get("match")),
        "candidate_smiles": validation.get("candidate_smiles")
        or obj.get("candidate_smiles")
        or extract_smiles(output_text),
        "canonical_smiles": validation.get("candidate_canonical_smiles")
        or validation.get("expected_canonical_smiles")
        or obj.get("canonical_smiles"),
        "expected_smiles": validation.get("expected_smiles")
        or source.get("expected_smiles")
        or dataset.get("expected_smiles")
        or metadata.get("expected_smiles")
        or obj.get("expected_smiles"),
        "source_dataset": source.get("dataset")
        or dataset.get("name")
        or metadata.get("dataset_name")
        or obj.get("source_dataset"),
        "source_row_index": first_present(
            source.get("row_index"),
            dataset.get("row_index"),
            selection.get("source_row_index"),
            metadata.get("row_index"),
            obj.get("source_row_index"),
        ),
        "difficulty_level": source.get("difficulty_level")
        or dataset.get("difficulty_level")
        or selection.get("difficulty_level")
        or metadata.get("difficulty_level")
        or obj.get("difficulty_level"),
        "provider": model.get("provider") or obj.get("provider"),
        "curator_model": model.get("model_name")
        or model.get("deployment")
        or metadata.get("model_name")
        or metadata.get("deployment")
        or obj.get("curator_model")
        or obj.get("model_name"),
        "source_file": source_file,
    }


def collect_rows(input_roots: list[Path]) -> Iterable[dict[str, Any]]:
    seen_artifacts: set[Path] = set()
    for root in input_roots:
        for artifact in sorted(root.rglob("artifact.json")):
            seen_artifacts.add(artifact.resolve())
            row = row_from_artifact(artifact)
            if row:
                yield row
        for jsonl_path in sorted(root.rglob("*.jsonl")):
            for obj in iter_jsonl(jsonl_path):
                row = row_from_jsonl_obj(obj)
                if row:
                    yield row
        for prompt_path in sorted(root.rglob("prompt.txt")):
            if (prompt_path.parent / "artifact.json").resolve() in seen_artifacts:
                continue
            row = row_from_example_dir(prompt_path.parent)
            if row:
                yield row


def normalize_reasoning_text(text: str | None) -> str:
    if not text:
        return ""
    reasoning = SMILES_RE.sub("", str(text))
    reasoning = SMILES_TAG_RE.sub("", reasoning)
    reasoning = REASONING_TAG_RE.sub("", reasoning)
    return reasoning.strip()


def build_assistant(row: dict[str, Any], response_mode: str) -> str | None:
    smiles = row.get("canonical_smiles") or row.get("candidate_smiles") or row.get("expected_smiles")
    if not smiles:
        return None
    final = f"<smiles>{smiles}</smiles>"
    if response_mode == "final_smiles":
        return final
    if response_mode == "raw_output":
        return row.get("output_text") or final
    if response_mode == "qwen_think":
        reasoning = normalize_reasoning_text(row.get("reasoning"))
        if not reasoning:
            reasoning = normalize_reasoning_text(row.get("output_text"))
        return f"<think>\n{reasoning}\n</think>\n\n{final}" if reasoning else final
    raise ValueError(f"unknown response mode: {response_mode}")


def has_reasoning_trace(row: dict[str, Any]) -> bool:
    reasoning = normalize_reasoning_text(row.get("reasoning"))
    if reasoning:
        return True
    raw = normalize_reasoning_text(row.get("output_text"))
    if raw:
        return True
    return False


def render_plain_text(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", default=None, type=Path)
    parser.add_argument(
        "--input-jsonl",
        action="append",
        default=None,
        type=Path,
        help="Read specific JSONL file(s) instead of recursively scanning an input root.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--response-mode",
        choices=("final_smiles", "qwen_think", "raw_output"),
        default="final_smiles",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--require-match", action="store_true")
    parser.add_argument("--require-reasoning", action="store_true")
    parser.add_argument(
        "--dedupe-key",
        choices=("none", "canonical_smiles", "prompt"),
        default="canonical_smiles",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.input_root and not args.input_jsonl:
        parser.error("at least one --input-root or --input-jsonl is required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"seen": 0, "written": 0, "skipped": 0, "deduped": 0}
    dedupe_seen: set[str] = set()

    with args.output.open("w", encoding="utf-8") as handle:
        input_rows: Iterable[dict[str, Any]]
        if args.input_jsonl:
            input_rows = (
                row
                for jsonl_path in args.input_jsonl
                for obj in iter_jsonl(jsonl_path)
                for row in [row_from_jsonl_obj(obj)]
                if row
            )
        else:
            input_rows = collect_rows(args.input_root or [])

        for row in input_rows:
            stats["seen"] += 1
            if args.require_match and row.get("match") is not True:
                stats["skipped"] += 1
                continue
            if args.require_reasoning and not has_reasoning_trace(row):
                stats["skipped"] += 1
                continue
            prompt = row.get("prompt")
            assistant = build_assistant(row, args.response_mode)
            if not prompt or not assistant:
                stats["skipped"] += 1
                continue
            key_value = None
            if args.dedupe_key != "none":
                key_value = row.get(args.dedupe_key)
                if key_value in dedupe_seen:
                    stats["deduped"] += 1
                    continue
                if key_value:
                    dedupe_seen.add(str(key_value))

            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": normalize_prompt(prompt)},
                {"role": "assistant", "content": assistant},
            ]
            output = {
                "messages": messages,
                "text": render_plain_text(messages),
                "smiles": row.get("canonical_smiles")
                or row.get("candidate_smiles")
                or row.get("expected_smiles"),
                "candidate_smiles": row.get("candidate_smiles"),
                "expected_smiles": row.get("expected_smiles"),
                "source_dataset": row.get("source_dataset"),
                "source_row_index": row.get("source_row_index"),
                "difficulty_level": row.get("difficulty_level"),
                "provider": row.get("provider"),
                "curator_model": row.get("curator_model"),
                "source_file": row.get("source_file"),
                "response_mode": args.response_mode,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if args.limit and stats["written"] >= args.limit:
                break

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
