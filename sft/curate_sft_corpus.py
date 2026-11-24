#!/usr/bin/env python3
"""Curate verified reasoning-SMILES examples with a chat-completions API."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import requests
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.error")
SMILES_RE = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.IGNORECASE | re.DOTALL)
REASONING_RE = re.compile(
    r"<(?:think|thinking|reasoning)>\s*(.*?)\s*</(?:think|thinking|reasoning)>",
    re.IGNORECASE | re.DOTALL,
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a chemistry assistant specialized in molecular structure generation. "
    "Reason from the description, then return exactly one final SMILES string "
    "inside <smiles> and </smiles> tags."
)
DEFAULT_PROMPT_TEMPLATE = """Reconstruct the molecule described below as SMILES.

Molecular structure description:
{description}"""


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def extract_smiles(text: str | None) -> str | None:
    if not text:
        return None
    matches = SMILES_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_reasoning(text: str | None) -> str:
    if not text:
        return ""
    matches = REASONING_RE.findall(text)
    return matches[-1].strip() if matches else ""


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def parse_completion(payload: dict[str, Any]) -> tuple[str, str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completion response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("completion response has no assistant message")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    if not isinstance(content, str):
        content = ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = extract_reasoning(content)
    return content.strip(), reasoning.strip()


def request_completion(
    *,
    endpoint: str,
    api_key: str | None,
    request_payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        endpoint,
        headers=headers,
        json=request_payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("completion endpoint returned a non-object JSON response")
    return payload


def bounded_parallel_map(
    executor: concurrent.futures.Executor,
    function: Callable[[tuple[int, dict[str, Any]]], dict[str, Any]],
    rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    max_pending: int,
) -> Iterable[dict[str, Any]]:
    iterator = iter(rows)
    pending: set[concurrent.futures.Future[dict[str, Any]]] = set()
    for _ in range(max_pending):
        try:
            pending.add(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        done, pending = concurrent.futures.wait(
            pending,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
            yield future.result()
            try:
                pending.add(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def curate_row(
    row: dict[str, Any],
    *,
    row_index: int,
    model: str,
    prompt_column: str,
    smiles_column: str,
    id_column: str,
    system_prompt: str,
    prompt_template: str,
    temperature: float,
    max_tokens: int,
    max_tokens_field: str,
    reasoning_effort: str | None,
    max_attempts: int,
    complete: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    description = row.get(prompt_column)
    expected_smiles = row.get(smiles_column)
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"row {row_index}: missing {prompt_column}")
    if not isinstance(expected_smiles, str) or not expected_smiles.strip():
        raise ValueError(f"row {row_index}: missing {smiles_column}")
    expected_canonical = canonicalize_smiles(expected_smiles)
    if expected_canonical is None:
        raise ValueError(f"row {row_index}: invalid target SMILES")

    prompt = prompt_template.format(description=description.strip())
    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        max_tokens_field: max_tokens,
    }
    if reasoning_effort:
        request_payload["reasoning_effort"] = reasoning_effort

    last_content = ""
    last_reasoning = ""
    last_candidate = None
    last_canonical = None
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            last_content, last_reasoning = parse_completion(complete(request_payload))
            last_candidate = extract_smiles(last_content)
            last_canonical = canonicalize_smiles(last_candidate)
            last_error = None
            if last_canonical == expected_canonical:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    matched = last_canonical == expected_canonical
    return {
        "sample_id": str(row.get(id_column, row.get("id", row_index))),
        "prompt": prompt,
        "output_text": last_content,
        "reasoning": last_reasoning,
        "match": matched,
        "candidate_smiles": last_candidate,
        "canonical_smiles": last_canonical,
        "expected_smiles": expected_smiles.strip(),
        "source_dataset": row.get("source_dataset"),
        "source_row_index": row_index,
        "curator_model": model,
        "reasoning_effort": reasoning_effort,
        "attempts": attempt,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local JSONL source")
    parser.add_argument("--dataset", default="mollangdata/MolLangData")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="data")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--prompt-column", default="description")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default="cid")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt-template-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--max-tokens-field",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_tokens",
    )
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-failures", action="store_true")
    args = parser.parse_args()
    if args.max_attempts <= 0 or args.workers <= 0 or args.max_tokens <= 0:
        parser.error("max-attempts, workers, and max-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")

    prompt_template = (
        args.prompt_template_file.read_text(encoding="utf-8")
        if args.prompt_template_file
        else DEFAULT_PROMPT_TEMPLATE
    )
    if "{description}" not in prompt_template:
        parser.error("prompt template must contain {description}")

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    api_key = os.environ.get(args.api_key_env)
    if args.input:
        rows: Iterable[dict[str, Any]] = iter_jsonl(args.input)
        source_dataset = str(args.input)
    else:
        os.environ.setdefault(
            "HF_HOME",
            str(Path(__file__).resolve().parents[1] / ".runtime" / "hf_home"),
        )
        from datasets import load_dataset

        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            streaming=args.streaming,
        )
        rows = (dict(row) for row in dataset)
        source_dataset = args.dataset
    if args.limit is not None:
        rows = itertools.islice(rows, args.limit)

    def indexed_rows() -> Iterable[tuple[int, dict[str, Any]]]:
        for index, row in enumerate(rows):
            row.setdefault("source_dataset", source_dataset)
            yield index, row

    def process(index_and_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, row = index_and_row
        return curate_row(
            row,
            row_index=index,
            model=args.model,
            prompt_column=args.prompt_column,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
            system_prompt=args.system_prompt,
            prompt_template=prompt_template,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_tokens_field=args.max_tokens_field,
            reasoning_effort=args.reasoning_effort,
            max_attempts=args.max_attempts,
            complete=lambda payload: request_completion(
                endpoint=endpoint,
                api_key=api_key,
                request_payload=payload,
                timeout_seconds=args.timeout_seconds,
            ),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"seen": 0, "matched": 0, "failed": 0, "written": 0}
    with args.output.open("w", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in bounded_parallel_map(
                executor,
                process,
                indexed_rows(),
                max_pending=args.workers * 2,
            ):
                counts["seen"] += 1
                if result["match"]:
                    counts["matched"] += 1
                else:
                    counts["failed"] += 1
                    print(
                        f"row {result['source_row_index']} did not match after {result['attempts']} attempts",
                        file=sys.stderr,
                    )
                if result["match"] or args.keep_failures:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    counts["written"] += 1
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
