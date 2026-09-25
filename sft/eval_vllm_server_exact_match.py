#!/usr/bin/env python
"""Run exact-match molecular generation evaluation against one or more vLLM servers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


SMILES_RE = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.IGNORECASE | re.DOTALL)
SMILES_OPEN_RE = re.compile(r"<smiles>", re.IGNORECASE)
UNCLOSED_SMILES_LABEL_RE = re.compile(
    r"^(?:smiles|final smiles|answer|final answer)\s*:\s*", re.IGNORECASE
)
UNCLOSED_SMILES_CANDIDATE_RE = re.compile(r"[#%()+\-./0-9:=@A-Za-z\[\]\\]+")
UNCLOSED_SMILES_REJECT_WORDS = {"and", "answer", "final", "smiles", "tag", "tags", "string"}

try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.error")
except Exception:
    Chem = None
    DataStructs = None
    rdFingerprintGenerator = None


FINGERPRINT_GENERATOR = (
    rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    if rdFingerprintGenerator is not None
    else None
)
DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/qwen_eval/mollangdata_final_pass_final_smiles.jsonl"),
    )
    parser.add_argument(
        "--named-dataset",
        action="append",
        default=None,
        help=(
            "Named dataset in split_name=/path/to/file.jsonl format. "
            "Can be repeated; when provided, all splits are evaluated in one pooled queue."
        ),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vllm_server_exact_match"))
    parser.add_argument(
        "--split-output-root",
        type=Path,
        default=None,
        help="Also write per-split exact_match.json files under this root/<split_name>/<signature_id>/.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Rows to evaluate. Use <=0 for all rows.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument(
        "--api-mode",
        choices=("completions", "chat"),
        default="completions",
        help="Use /v1/completions by default so the locally rendered Qwen prompt is not chat-templated twice.",
    )
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-groundtruth-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow literal expected SMILES leakage in generation prompts. Default is to fail.",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-full-generations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--request-timeout-seconds", type=int, default=1800)
    parser.add_argument("--server-urls", nargs="+", default=None)
    parser.add_argument("--server-urls-file", type=Path, default=None)
    parser.add_argument("--client-concurrency-per-server", type=int, default=4)
    parser.add_argument("--dispatch-mode", choices=("fixed", "dynamic"), default="dynamic")
    parser.add_argument("--target-waiting-per-server", type=int, default=1)
    parser.add_argument("--max-outstanding-per-server", type=int, default=16)
    parser.add_argument("--scheduler-poll-seconds", type=float, default=1.0)
    parser.add_argument("--metrics-timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_smiles_text(smiles: str | None) -> str:
    return "".join(str(smiles or "").split())


def answer_after_final_reasoning(text: str | None) -> str:
    raw = str(text or "")
    lowered = raw.lower()
    close_match: tuple[int, str] | None = None
    for close_tag in ("</think>", "</reasoning>"):
        close_idx = lowered.rfind(close_tag)
        if close_idx != -1 and (close_match is None or close_idx > close_match[0]):
            close_match = (close_idx, close_tag)
    if close_match is not None:
        close_idx, close_tag = close_match
        return raw[close_idx + len(close_tag) :]

    # Do not score candidates inside unfinished thinking/reasoning traces.
    if "<think" in lowered or "<reasoning" in lowered:
        return ""
    return raw


def extract_unclosed_smiles_candidate(text: str | None) -> str | None:
    tail = str(text or "").strip()
    if not tail:
        return None
    first_line = re.split(r"[\r\n]", tail, maxsplit=1)[0].strip().strip("`'\"")
    first_line = UNCLOSED_SMILES_LABEL_RE.sub("", first_line).strip().strip("`'\"")
    if not first_line:
        return None
    match = UNCLOSED_SMILES_CANDIDATE_RE.search(first_line)
    if match is None:
        return None
    candidate = match.group(0).strip().strip("`'\"")
    if not candidate or candidate.lower() in UNCLOSED_SMILES_REJECT_WORDS:
        return None
    if not re.search(r"[A-Za-z\[]", candidate):
        return None
    return candidate


def extract_smiles_from_scope(text: str | None) -> str | None:
    if not text:
        return None
    matches = list(SMILES_RE.finditer(text))
    if matches:
        return (matches[-1].group(1) or "").strip() or None
    opens = list(SMILES_OPEN_RE.finditer(text))
    if opens:
        return extract_unclosed_smiles_candidate(text[opens[-1].end() :])
    return None


def extract_smiles(text: str | None) -> str | None:
    if not text:
        return None
    raw = str(text or "")
    lowered = raw.lower()
    has_reasoning_tag = any(tag in lowered for tag in ("<think", "</think>", "<reasoning", "</reasoning>"))
    if has_reasoning_tag:
        return extract_smiles_from_scope(answer_after_final_reasoning(raw))
    return extract_smiles_from_scope(raw)


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    smiles = normalize_smiles_text(smiles)
    if not smiles:
        return None
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def groundtruth_smiles_candidates(row: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for key in ("expected_smiles", "gt_smiles", "smiles"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.add(value.strip())
            canonical = canonicalize_smiles(value)
            if canonical:
                candidates.add(canonical)
    return candidates


def prompt_text_for_leak_check(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt

    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    prompt_messages = [message for message in messages if isinstance(message, dict)]
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages = prompt_messages[:-1]
    return "\n".join(str(message.get("content", "")) for message in prompt_messages)


def find_groundtruth_prompt_leak(row: dict[str, Any]) -> str | None:
    prompt_text = prompt_text_for_leak_check(row)
    if not prompt_text:
        return None
    for candidate in sorted(groundtruth_smiles_candidates(row), key=len, reverse=True):
        if not candidate:
            continue
        tagged = f"<smiles>{candidate}</smiles>"
        if tagged in prompt_text:
            return tagged
        # Avoid false positives for very short atom fragments such as C, N, Cl
        # that naturally occur in prose descriptions.
        if len(candidate) >= 12 and candidate in prompt_text:
            return candidate
    return None


def mol_from_smiles(smiles: str | None):
    if not smiles or Chem is None:
        return None
    try:
        normalized = normalize_smiles_text(smiles)
        return Chem.MolFromSmiles(normalized) if normalized else None
    except Exception:
        return None


def tanimoto_similarity(expected_smiles: str | None, predicted_smiles: str | None) -> float | None:
    if FINGERPRINT_GENERATOR is None or DataStructs is None:
        return None
    expected_mol = mol_from_smiles(expected_smiles)
    predicted_mol = mol_from_smiles(predicted_smiles)
    if expected_mol is None or predicted_mol is None:
        return None
    expected_fp = FINGERPRINT_GENERATOR.GetFingerprint(expected_mol)
    predicted_fp = FINGERPRINT_GENERATOR.GetFingerprint(predicted_mol)
    return float(DataStructs.TanimotoSimilarity(expected_fp, predicted_fp))


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    num_examples = len(results)
    correct = sum(int(row.get("match", False)) for row in results)
    valid = sum(int(bool(row.get("predicted_is_valid"))) for row in results)
    request_errors = sum(int(row.get("error") is not None) for row in results)
    generation_started = sum(int(int(row.get("generated_tokens") or 0) > 0) for row in results)
    smiles_extracted = sum(int(row.get("predicted_smiles_raw") is not None) for row in results)
    similarities_all = []
    similarities_valid = []
    invalid_no_smiles = 0
    invalid_parse = 0
    for row in results:
        similarity = row.get("tanimoto_similarity")
        if similarity is None:
            similarities_all.append(0.0)
        else:
            similarity = float(similarity)
            similarities_all.append(similarity)
            similarities_valid.append(similarity)
        predicted_raw = row.get("predicted_smiles_raw")
        predicted = row.get("predicted_smiles")
        if predicted is None:
            if predicted_raw is None:
                invalid_no_smiles += 1
            else:
                invalid_parse += 1
    return {
        "num_examples": num_examples,
        "correct": correct,
        "exact_match_accuracy": (correct / num_examples) if num_examples else 0.0,
        "request_errors": request_errors,
        "request_error_rate": (request_errors / num_examples) if num_examples else 0.0,
        "generation_started": generation_started,
        "generation_started_rate": (generation_started / num_examples) if num_examples else 0.0,
        "smiles_extracted": smiles_extracted,
        "smiles_extraction_rate": (smiles_extracted / num_examples) if num_examples else 0.0,
        "exact_match_given_extracted": (correct / smiles_extracted) if smiles_extracted else 0.0,
        "valid_smiles": valid,
        "validity_rate": (valid / num_examples) if num_examples else 0.0,
        "validity_given_extracted": (valid / smiles_extracted) if smiles_extracted else 0.0,
        "mean_tanimoto_all": (sum(similarities_all) / num_examples) if num_examples else 0.0,
        "mean_tanimoto_valid": (
            sum(similarities_valid) / len(similarities_valid)
            if similarities_valid
            else 0.0
        ),
        "invalid_no_smiles": invalid_no_smiles,
        "invalid_parse": invalid_parse,
    }


def local_snapshot_for_model(model_name: str) -> Path | None:
    maybe_path = Path(model_name)
    if maybe_path.exists():
        return maybe_path.resolve()
    if "/" not in model_name:
        return None

    repo_dir_name = "models--" + model_name.replace("/", "--")
    hub_roots: list[Path] = []
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache:
        hub_roots.append(Path(transformers_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hub_roots.append(Path(hf_home) / "hub")
    for hub_root in hub_roots:
        repo_dir = hub_root / repo_dir_name
        ref_path = repo_dir / "refs" / "main"
        snapshots_dir = repo_dir / "snapshots"
        if ref_path.exists():
            snapshot = snapshots_dir / ref_path.read_text(encoding="utf-8").strip()
            if snapshot.exists():
                return snapshot.resolve()
        if snapshots_dir.exists():
            snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
            if snapshots:
                return snapshots[-1].resolve()
    return None


def resolve_model_load_name(model_name: str, local_files_only: bool) -> str:
    maybe_path = Path(model_name)
    if maybe_path.exists():
        return str(maybe_path.resolve())
    if local_files_only:
        snapshot = local_snapshot_for_model(model_name)
        if snapshot is not None:
            return str(snapshot)
    return model_name


def load_tokenizer(tokenizer_name: str, local_files_only: bool):
    errors = []
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
                use_fast=use_fast,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            errors.append(f"use_fast={use_fast}: {exc}")
            if use_fast:
                print(
                    f"[{now()}] Fast tokenizer load failed; retrying with use_fast=False",
                    flush=True,
                )
                continue
            raise RuntimeError(
                "Tokenizer load failed for both fast and slow paths:\n"
                + "\n".join(errors)
            ) from exc


def load_rows(path: Path, limit: int | None, offset: int) -> list[dict[str, Any]]:
    max_rows = None if limit is None or limit <= 0 else limit
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if source_index < offset:
                continue
            row = json.loads(line)
            original_source_index = row.get("original_source_index")
            if original_source_index is None:
                original_source_index = row.get("_source_index")
            row["_eval_line_index"] = source_index
            row["_source_index"] = source_index if original_source_index is None else original_source_index
            normalize_eval_row(row)
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def parse_named_dataset_specs(args: argparse.Namespace) -> tuple[list[tuple[str, Path]], bool]:
    if not args.named_dataset:
        return [("dataset", args.dataset.resolve())], False

    specs: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    for item in args.named_dataset:
        if "=" not in item:
            raise ValueError(f"--named-dataset must be split_name=/path form, got: {item}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name or DATASET_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                f"Invalid split name {name!r}; use only letters, numbers, underscore, dot, or hyphen."
            )
        if name in seen_names:
            raise ValueError(f"Duplicate split name: {name}")
        seen_names.add(name)
        specs.append((name, Path(raw_path).expanduser().resolve()))
    return specs, True


def load_eval_rows(
    dataset_specs: list[tuple[str, Path]],
    limit: int | None,
    offset: int,
    allow_groundtruth_in_prompt: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    for dataset_name, dataset_path in dataset_specs:
        split_rows = load_rows(dataset_path, limit=limit, offset=offset)
        if not split_rows:
            raise ValueError(f"No rows selected from {dataset_path} for split {dataset_name}")
        dataset_counts[dataset_name] = len(split_rows)
        for local_index, row in enumerate(split_rows):
            row["_dataset_name"] = dataset_name
            row["_dataset_path"] = str(dataset_path)
            row["_dataset_local_index"] = local_index
            row["_global_index"] = len(rows)
            leaked = find_groundtruth_prompt_leak(row)
            if leaked and not allow_groundtruth_in_prompt:
                raise ValueError(
                    "Ground-truth SMILES appears in generation prompt: "
                    f"split={dataset_name} source_index={row.get('_source_index')} "
                    f"leaked={leaked!r}"
                )
            rows.append(row)
    return rows, dataset_counts


def row_result_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": row.get("_dataset_name"),
        "dataset_path": row.get("_dataset_path"),
        "dataset_local_index": row.get("_dataset_local_index"),
        "eval_line_index": row.get("_eval_line_index"),
        "global_index": row.get("_global_index"),
    }


def normalize_eval_row(row: dict[str, Any]) -> None:
    """Accept both chat SFT rows and llm_rl processed eval rows."""
    if "expected_smiles" not in row:
        expected = row.get("smiles") or row.get("gt_smiles")
        if isinstance(expected, str) and expected.strip():
            row["expected_smiles"] = expected.strip()

    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return

    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return

    messages = [{"role": "user", "content": prompt.strip()}]
    expected = row.get("expected_smiles")
    if isinstance(expected, str) and expected.strip():
        messages.append({"role": "assistant", "content": f"<smiles>{expected.strip()}</smiles>"})
    row["messages"] = messages


def build_prompt(tokenizer, row: dict[str, Any], enable_thinking: bool, final_only: bool) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        prompt_text = prompt.strip()
        if final_only:
            prompt_text = (
                prompt_text.rstrip()
                + "\n\nReturn only the final answer as <smiles>...</smiles>. "
                + "Do not include reasoning, markdown, or code fences."
            )
        prompt_messages = [{"role": "user", "content": prompt_text}]
    else:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Row {row.get('_source_index')} has no prompt or messages list")

        prompt_messages = [dict(message) for message in messages]
        if prompt_messages and prompt_messages[-1].get("role") == "assistant":
            prompt_messages = prompt_messages[:-1]

        if final_only:
            for message in reversed(prompt_messages):
                if message.get("role") == "user":
                    message["content"] = (
                        str(message.get("content", "")).rstrip()
                        + "\n\nReturn only the final answer as <smiles>...</smiles>. "
                        + "Do not include reasoning, markdown, or code fences."
                    )
                    break

    kwargs = {
        "conversation": prompt_messages,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(**kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(**kwargs)


def path_fingerprint(path: Path | None) -> Any:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    interesting = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "generation_config.json",
    ]
    files = []
    for name in interesting:
        candidate = resolved / name
        if candidate.exists():
            stat = candidate.stat()
            files.append({"name": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"path": str(resolved), "files": files}


def make_signature(
    args: argparse.Namespace,
    num_rows: int,
    server_urls: list[str],
    dataset_specs: list[tuple[str, Path]],
    dataset_counts: dict[str, int],
    named_dataset_mode: bool,
) -> tuple[str, dict[str, Any]]:
    if named_dataset_mode:
        dataset_signature: Any = [
            {
                "name": dataset_name,
                "rows": dataset_counts.get(dataset_name, 0),
                "dataset": path_fingerprint(dataset_path),
            }
            for dataset_name, dataset_path in dataset_specs
        ]
    else:
        dataset_signature = path_fingerprint(args.dataset)
    payload = {
        "dataset": dataset_signature,
        "dataset_mode": "named" if named_dataset_mode else "single",
        "model_name": args.model_name,
        "served_model_name": args.served_model_name or args.model_name,
        "tokenizer_name": args.tokenizer_name or args.model_name,
        "lora": path_fingerprint(args.lora_path),
        "limit": args.limit,
        "offset": args.offset,
        "num_rows": num_rows,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "enable_thinking": args.enable_thinking,
        "final_only": args.final_only,
        "api_mode": args.api_mode,
        "stop": args.stop,
        "server_urls": server_urls,
        "client_concurrency_per_server": args.client_concurrency_per_server,
        "dispatch_mode": args.dispatch_mode,
        "target_waiting_per_server": args.target_waiting_per_server,
        "max_outstanding_per_server": args.max_outstanding_per_server,
    }
    signature_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return signature_id, payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def resolve_server_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if args.server_urls:
        urls.extend(args.server_urls)
    if args.server_urls_file is not None and args.server_urls_file.exists():
        for line in args.server_urls_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                urls.append(line)
    if not urls:
        raise ValueError("Provide --server-urls or --server-urls-file")
    normalized = []
    for url in urls:
        normalized.append(url.rstrip("/"))
    return normalized


def write_progress_snapshot(
    *,
    path: Path,
    completed: bool,
    signature_id: str,
    output_path: Path,
    processed: int,
    num_examples: int,
    correct: int,
    total_generated_tokens: int,
    total_errors: int,
    inflight: int,
    wall_start: float,
    per_server: dict[str, dict[str, Any]],
    state: str,
) -> None:
    wall_seconds = time.time() - wall_start
    payload = {
        "completed": completed,
        "state": state,
        "signature_id": signature_id,
        "output": str(output_path),
        "processed": processed,
        "num_examples": num_examples,
        "exact_match_accuracy": (correct / processed) if processed else 0.0,
        "correct": correct,
        "errors": total_errors,
        "inflight": inflight,
        "total_generated_tokens": total_generated_tokens,
        "wall_seconds": wall_seconds,
        "wall_tokens_per_second": (total_generated_tokens / wall_seconds) if wall_seconds > 0 else 0.0,
        "updated_at": time.time(),
        "per_server": per_server,
    }
    atomic_write_json(path, payload)


class EvalState:
    def __init__(self, rows: list[dict[str, Any]], partial_path: Path, save_full_generations: bool):
        self.rows = rows
        self.partial_path = partial_path
        self.save_full_generations = save_full_generations
        self.lock = threading.Lock()
        self.queue: queue.Queue[tuple[int, dict[str, Any], str]] = queue.Queue()
        self.processed = 0
        self.correct = 0
        self.errors = 0
        self.inflight = 0
        self.total_generated_tokens = 0
        self.results: list[dict[str, Any]] = []
        self.per_server: dict[str, dict[str, Any]] = {}

    def set_per_server(self, server_urls: list[str]) -> None:
        with self.lock:
            self.per_server = {
                url: {
                    "completed": 0,
                    "errors": 0,
                    "inflight": 0,
                    "last_source_index": None,
                    "metrics_running": 0,
                    "metrics_waiting": 0,
                    "metrics_updated_at": None,
                }
                for url in server_urls
            }

    def claim(self, server_url: str) -> tuple[int, dict[str, Any], str] | None:
        try:
            item = self.queue.get_nowait()
        except queue.Empty:
            return None
        with self.lock:
            self.inflight += 1
            self.per_server[server_url]["inflight"] += 1
            self.per_server[server_url]["last_source_index"] = item[1].get("_source_index")
        return item

    def finish(
        self,
        *,
        server_url: str,
        result: dict[str, Any],
        generated_tokens: int,
        match: bool,
        is_error: bool,
    ) -> None:
        with self.lock:
            self.processed += 1
            self.correct += int(match)
            self.errors += int(is_error)
            self.inflight -= 1
            self.total_generated_tokens += generated_tokens
            self.per_server[server_url]["completed"] += 1
            self.per_server[server_url]["errors"] += int(is_error)
            self.per_server[server_url]["inflight"] -= 1
            self.results.append(result)
            with self.partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            self.queue.task_done()

    def update_server_metrics(self, server_url: str, running: int, waiting: int) -> None:
        with self.lock:
            if server_url not in self.per_server:
                return
            self.per_server[server_url]["metrics_running"] = running
            self.per_server[server_url]["metrics_waiting"] = waiting
            self.per_server[server_url]["metrics_updated_at"] = time.time()


def parse_prometheus_metric(body: str, metric_name: str) -> int:
    total = 0.0
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith(metric_name):
            continue
        try:
            total += float(line.rsplit(None, 1)[-1])
        except Exception:
            continue
    return int(total)


def fetch_server_scheduler_metrics(
    session: requests.Session,
    server_url: str,
    timeout_seconds: float,
) -> tuple[int, int]:
    response = session.get(f"{server_url}/metrics", timeout=timeout_seconds)
    response.raise_for_status()
    body = response.text
    running = parse_prometheus_metric(body, "vllm:num_requests_running")
    waiting = parse_prometheus_metric(body, "vllm:num_requests_waiting")
    return running, waiting


def request_completion(
    *,
    session: requests.Session,
    server_url: str,
    served_model_name: str,
    prompt: str,
    args: argparse.Namespace,
) -> tuple[str, list[int], str | None, dict[str, Any], dict[str, Any]]:
    common_body = {
        "model": served_model_name,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.stop:
        common_body["stop"] = args.stop
    if args.api_mode == "chat":
        endpoint = "chat/completions"
        body = {
            **common_body,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
    else:
        endpoint = "completions"
        body = {
            **common_body,
            "prompt": prompt,
        }
    response = session.post(
        f"{server_url}/v1/{endpoint}",
        json=body,
        timeout=args.request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    text = choice.get("text") if args.api_mode == "completions" else message.get("content", "")
    if text is None:
        text = ""
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )
    elif not isinstance(text, str):
        text = str(text)
    finish_reason = choice.get("finish_reason")
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    token_ids = [0] * int(completion_tokens) if completion_tokens is not None else []
    return text, token_ids, finish_reason, payload, choice


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    if args.split_output_root is not None:
        args.split_output_root = args.split_output_root.resolve()
    if args.lora_path is not None:
        args.lora_path = args.lora_path.resolve()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    server_urls = resolve_server_urls(args)
    served_model_name = args.served_model_name or args.model_name
    model_load_name = resolve_model_load_name(args.model_name, args.local_files_only)
    tokenizer_load_name = resolve_model_load_name(
        args.tokenizer_name or args.model_name,
        args.local_files_only,
    )

    dataset_specs, named_dataset_mode = parse_named_dataset_specs(args)
    rows, dataset_counts = load_eval_rows(
        dataset_specs,
        limit=args.limit,
        offset=args.offset,
        allow_groundtruth_in_prompt=args.allow_groundtruth_in_prompt,
    )
    if not rows:
        raise ValueError("No rows selected from evaluation datasets")

    signature_id, signature = make_signature(
        args,
        len(rows),
        server_urls,
        dataset_specs,
        dataset_counts,
        named_dataset_mode,
    )
    if args.output is None:
        output_dir = args.output_dir / signature_id
        output_path = output_dir / "exact_match.json"
    else:
        output_path = args.output
        output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.jsonl")
    progress_path = output_path.with_suffix(".progress.json")
    signature_path = output_path.with_suffix(".signature.json")

    if output_path.exists() and not args.force:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if existing.get("completed") and existing.get("signature_id") == signature_id:
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "signature_id": signature_id,
                        "output": str(output_path),
                        "exact_match_accuracy": existing.get("exact_match_accuracy"),
                        "num_examples": existing.get("num_examples"),
                    },
                    indent=2,
                ),
                flush=True,
            )
            return

    partial_path.write_text("", encoding="utf-8")
    atomic_write_json(signature_path, {"signature_id": signature_id, "signature": signature})

    print(
        f"[{now()}] Loading tokenizer={tokenizer_load_name} "
        f"rows={len(rows)} splits={dataset_counts} servers={len(server_urls)} "
        f"client_concurrency_per_server={args.client_concurrency_per_server} "
        f"dispatch_mode={args.dispatch_mode} "
        f"api_mode={args.api_mode} enable_thinking={args.enable_thinking} "
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    tokenizer = load_tokenizer(tokenizer_load_name, args.local_files_only)
    prompts = [build_prompt(tokenizer, row, args.enable_thinking, args.final_only) for row in rows]
    prompt_token_counts = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]
    if args.dry_run:
        first_prompt_tokens = len(tokenizer.encode(prompts[0], add_special_tokens=False))
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "signature_id": signature_id,
                    "rows": len(rows),
                    "splits": dataset_counts,
                    "server_urls": server_urls,
                    "first_source_index": rows[0].get("_source_index"),
                    "first_expected_smiles": rows[0].get("expected_smiles") or rows[0].get("smiles"),
                    "first_prompt_chars": len(prompts[0]),
                    "first_prompt_tokens": first_prompt_tokens,
                    "output": str(output_path),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    state = EvalState(rows, partial_path, args.save_full_generations)
    state.set_per_server(server_urls)
    for index, (row, prompt) in enumerate(zip(rows, prompts)):
        row["_prompt_chars"] = len(prompt)
        row["_prompt_tokens"] = prompt_token_counts[index]
        state.queue.put((index, row, prompt))

    wall_start = time.time()
    stop_heartbeat = threading.Event()

    def emit_progress(state_name: str) -> None:
        with state.lock:
            per_server = json.loads(json.dumps(state.per_server))
            write_progress_snapshot(
                path=progress_path,
                completed=False,
                signature_id=signature_id,
                output_path=output_path,
                processed=state.processed,
                num_examples=len(rows),
                correct=state.correct,
                total_generated_tokens=state.total_generated_tokens,
                total_errors=state.errors,
                inflight=state.inflight,
                wall_start=wall_start,
                per_server=per_server,
                state=state_name,
            )

    def heartbeat() -> None:
        while not stop_heartbeat.wait(args.heartbeat_seconds):
            emit_progress("running")
            with state.lock:
                accuracy = state.correct / state.processed if state.processed else 0.0
                print(
                    f"[{now()}] heartbeat processed={state.processed}/{len(rows)} "
                    f"inflight={state.inflight} exact_match={accuracy:.4f} "
                    f"errors={state.errors} queue_remaining={state.queue.qsize()}",
                    flush=True,
                )

    heartbeat_thread = threading.Thread(target=heartbeat, name="vllm_server_eval_heartbeat", daemon=True)
    heartbeat_thread.start()
    emit_progress("starting")

    def run_one_request(server_url: str, worker_index: int) -> None:
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        claimed = state.claim(server_url)
        if claimed is None:
            return
        _, row, prompt = claimed
        try:
            generated, token_ids, finish_reason, response_payload, response_choice = request_completion(
                session=session,
                server_url=server_url,
                served_model_name=served_model_name,
                prompt=prompt,
                args=args,
            )
            response_message = response_choice.get("message") or {}
            reasoning = response_message.get("reasoning")
            if reasoning is None:
                reasoning = ""
            elif not isinstance(reasoning, str):
                reasoning = str(reasoning)
            stop_reason = response_choice.get("stop_reason")
            predicted_raw = extract_smiles(generated)
            predicted = canonicalize_smiles(predicted_raw)
            expected = canonicalize_smiles(row.get("expected_smiles") or row.get("smiles"))
            predicted_is_valid = predicted is not None
            similarity = tanimoto_similarity(expected, predicted_raw)
            match = predicted is not None and expected is not None and predicted == expected
            raw_generation_with_reasoning = generated
            if reasoning:
                raw_generation_with_reasoning = f"<think>{reasoning}</think>\n{generated}"
            result = {
                "source_index": row.get("_source_index"),
                **row_result_metadata(row),
                "cid": row.get("cid"),
                "difficulty_level": row.get("difficulty_level"),
                "source_file": row.get("source_file"),
                "server_url": server_url,
                "worker_index": worker_index,
                "prompt_chars": row.get("_prompt_chars"),
                "prompt_tokens": row.get("_prompt_tokens"),
                "expected_smiles": expected,
                "predicted_smiles": predicted,
                "predicted_smiles_raw": predicted_raw,
                "predicted_is_valid": predicted_is_valid,
                "tanimoto_similarity": similarity,
                "match": match,
                "thinking_detected": bool(reasoning) or "<think>" in generated.lower(),
                "has_smiles_open": "<smiles>" in generated.lower(),
                "has_smiles_close": "</smiles>" in generated.lower() or stop_reason == "</smiles>",
                "generated_tokens": len(token_ids),
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "reasoning": reasoning,
                "response_id": response_payload.get("id"),
                "response_model": response_payload.get("model"),
                "response_created": response_payload.get("created"),
                "response_usage": response_payload.get("usage"),
                "response_choice": response_choice,
                "error": None,
                "raw_generation_preview": generated[:4096],
                "raw_generation": generated,
                "raw_generation_with_reasoning": raw_generation_with_reasoning,
            }
            state.finish(
                server_url=server_url,
                result=result,
                generated_tokens=len(token_ids),
                match=match,
                is_error=False,
            )
        except Exception as exc:
            result = {
                "source_index": row.get("_source_index"),
                **row_result_metadata(row),
                "cid": row.get("cid"),
                "difficulty_level": row.get("difficulty_level"),
                "source_file": row.get("source_file"),
                "server_url": server_url,
                "worker_index": worker_index,
                "prompt_chars": row.get("_prompt_chars"),
                "prompt_tokens": row.get("_prompt_tokens"),
                "expected_smiles": canonicalize_smiles(row.get("expected_smiles") or row.get("smiles")),
                "predicted_smiles": None,
                "predicted_smiles_raw": None,
                "predicted_is_valid": False,
                "tanimoto_similarity": None,
                "match": False,
                "thinking_detected": False,
                "has_smiles_open": False,
                "has_smiles_close": False,
                "generated_tokens": 0,
                "finish_reason": None,
                "stop_reason": None,
                "reasoning": "",
                "response_id": None,
                "response_model": None,
                "response_created": None,
                "response_usage": None,
                "response_choice": None,
                "error": str(exc),
                "raw_generation_preview": "",
                "raw_generation": "",
                "raw_generation_with_reasoning": "",
            }
            state.finish(
                server_url=server_url,
                result=result,
                generated_tokens=0,
                match=False,
                is_error=True,
            )

    def fixed_worker(server_url: str, worker_index: int) -> None:
        while True:
            with state.lock:
                queue_remaining = state.queue.qsize()
            if queue_remaining <= 0:
                return
            claimed = state.claim(server_url)
            if claimed is None:
                return
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            _, row, prompt = claimed
            try:
                generated, token_ids, finish_reason, response_payload, response_choice = request_completion(
                    session=session,
                    server_url=server_url,
                    served_model_name=served_model_name,
                    prompt=prompt,
                    args=args,
                )
                response_message = response_choice.get("message") or {}
                reasoning = response_message.get("reasoning")
                if reasoning is None:
                    reasoning = ""
                elif not isinstance(reasoning, str):
                    reasoning = str(reasoning)
                stop_reason = response_choice.get("stop_reason")
                predicted_raw = extract_smiles(generated)
                predicted = canonicalize_smiles(predicted_raw)
                expected = canonicalize_smiles(row.get("expected_smiles") or row.get("smiles"))
                predicted_is_valid = predicted is not None
                similarity = tanimoto_similarity(expected, predicted_raw)
                match = predicted is not None and expected is not None and predicted == expected
                raw_generation_with_reasoning = generated
                if reasoning:
                    raw_generation_with_reasoning = f"<think>{reasoning}</think>\n{generated}"
                result = {
                    "source_index": row.get("_source_index"),
                    **row_result_metadata(row),
                    "cid": row.get("cid"),
                    "difficulty_level": row.get("difficulty_level"),
                    "source_file": row.get("source_file"),
                    "server_url": server_url,
                    "worker_index": worker_index,
                    "prompt_chars": row.get("_prompt_chars"),
                    "prompt_tokens": row.get("_prompt_tokens"),
                    "expected_smiles": expected,
                    "predicted_smiles": predicted,
                    "predicted_smiles_raw": predicted_raw,
                    "predicted_is_valid": predicted_is_valid,
                    "tanimoto_similarity": similarity,
                    "match": match,
                    "thinking_detected": bool(reasoning) or "<think>" in generated.lower(),
                    "has_smiles_open": "<smiles>" in generated.lower(),
                    "has_smiles_close": "</smiles>" in generated.lower() or stop_reason == "</smiles>",
                    "generated_tokens": len(token_ids),
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                    "reasoning": reasoning,
                    "response_id": response_payload.get("id"),
                    "response_model": response_payload.get("model"),
                    "response_created": response_payload.get("created"),
                    "response_usage": response_payload.get("usage"),
                    "response_choice": response_choice,
                    "error": None,
                    "raw_generation_preview": generated[:4096],
                    "raw_generation": generated,
                    "raw_generation_with_reasoning": raw_generation_with_reasoning,
                }
                state.finish(
                    server_url=server_url,
                    result=result,
                    generated_tokens=len(token_ids),
                    match=match,
                    is_error=False,
                )
            except Exception as exc:
                result = {
                    "source_index": row.get("_source_index"),
                    **row_result_metadata(row),
                    "cid": row.get("cid"),
                    "difficulty_level": row.get("difficulty_level"),
                    "source_file": row.get("source_file"),
                    "server_url": server_url,
                    "worker_index": worker_index,
                    "prompt_chars": row.get("_prompt_chars"),
                    "prompt_tokens": row.get("_prompt_tokens"),
                    "expected_smiles": canonicalize_smiles(row.get("expected_smiles") or row.get("smiles")),
                    "predicted_smiles": None,
                    "predicted_smiles_raw": None,
                    "predicted_is_valid": False,
                    "tanimoto_similarity": None,
                    "match": False,
                    "thinking_detected": False,
                    "has_smiles_open": False,
                    "has_smiles_close": False,
                    "generated_tokens": 0,
                    "finish_reason": None,
                    "stop_reason": None,
                    "reasoning": "",
                    "response_id": None,
                    "response_model": None,
                    "response_created": None,
                    "response_usage": None,
                    "response_choice": None,
                    "error": str(exc),
                    "raw_generation_preview": "",
                    "raw_generation": "",
                    "raw_generation_with_reasoning": "",
                }
                state.finish(
                    server_url=server_url,
                    result=result,
                    generated_tokens=0,
                    match=False,
                    is_error=True,
                )

    max_workers = len(server_urls) * max(1, args.max_outstanding_per_server)
    if args.dispatch_mode == "fixed":
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for server_url in server_urls:
                for worker_index in range(args.client_concurrency_per_server):
                    futures.append(executor.submit(fixed_worker, server_url, worker_index))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        metrics_sessions = {server_url: requests.Session() for server_url in server_urls}
        for session in metrics_sessions.values():
            session.headers.update({"Content-Type": "application/json"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            active_futures: dict[concurrent.futures.Future[None], tuple[str, int]] = {}
            worker_sequence = {server_url: 0 for server_url in server_urls}

            def maybe_submit(server_url: str, desired: int) -> None:
                for _ in range(max(0, desired)):
                    with state.lock:
                        queue_remaining = state.queue.qsize()
                        local_inflight = state.per_server[server_url]["inflight"]
                    if queue_remaining <= 0 or local_inflight >= args.max_outstanding_per_server:
                        return
                    worker_id = worker_sequence[server_url]
                    worker_sequence[server_url] += 1
                    future = executor.submit(run_one_request, server_url, worker_id)
                    active_futures[future] = (server_url, worker_id)

            while True:
                completed = [future for future in active_futures if future.done()]
                for future in completed:
                    future.result()
                    active_futures.pop(future, None)

                with state.lock:
                    queue_remaining = state.queue.qsize()
                    inflight = state.inflight

                if queue_remaining == 0 and inflight == 0 and not active_futures:
                    break

                for server_url in server_urls:
                    try:
                        running, waiting = fetch_server_scheduler_metrics(
                            metrics_sessions[server_url],
                            server_url,
                            args.metrics_timeout_seconds,
                        )
                        state.update_server_metrics(server_url, running, waiting)
                    except Exception:
                        waiting = 0

                    with state.lock:
                        queue_remaining = state.queue.qsize()
                        local_inflight = state.per_server[server_url]["inflight"]
                    if queue_remaining <= 0 or local_inflight >= args.max_outstanding_per_server:
                        continue

                    desired = max(0, args.target_waiting_per_server - waiting)
                    if waiting <= 0 and local_inflight == 0:
                        desired = max(desired, 1)
                    desired = min(desired, args.max_outstanding_per_server - local_inflight)
                    maybe_submit(server_url, desired)

                time.sleep(args.scheduler_poll_seconds)

    stop_heartbeat.set()
    heartbeat_thread.join(timeout=1.0)

    wall_seconds = time.time() - wall_start
    with state.lock:
        ordered_results = sorted(
            state.results,
            key=lambda row: int(
                row.get("global_index")
                if row.get("global_index") is not None
                else row.get("source_index") or 0
            ),
        )
        summary = summarize_results(ordered_results)
        split_summaries: dict[str, dict[str, Any]] = {}
        split_outputs: dict[str, str] = {}
        split_payloads: list[tuple[Path, dict[str, Any]]] = []
        if named_dataset_mode:
            for split_name, _split_path in dataset_specs:
                split_results = [
                    result for result in ordered_results if result.get("dataset_name") == split_name
                ]
                split_summary = summarize_results(split_results)
                split_total_generated_tokens = sum(
                    int(result.get("generated_tokens") or 0) for result in split_results
                )
                split_summaries[split_name] = {
                    **split_summary,
                    "total_generated_tokens": split_total_generated_tokens,
                }
                if args.split_output_root is not None:
                    split_output_path = (
                        args.split_output_root / split_name / signature_id / "exact_match.json"
                    )
                    split_outputs[split_name] = str(split_output_path)
                    split_payloads.append(
                        (
                            split_output_path,
                            {
                                "completed": True,
                                "signature_id": signature_id,
                                "signature": signature,
                                "split_name": split_name,
                                "model_name": args.model_name,
                                "served_model_name": served_model_name,
                                "tokenizer_name": args.tokenizer_name or args.model_name,
                                "server_urls": server_urls,
                                "num_examples": split_summary["num_examples"],
                                "correct": split_summary["correct"],
                                "errors": split_summary["request_errors"],
                                "request_errors": split_summary["request_errors"],
                                "request_error_rate": split_summary["request_error_rate"],
                                "generation_started": split_summary["generation_started"],
                                "generation_started_rate": split_summary["generation_started_rate"],
                                "smiles_extracted": split_summary["smiles_extracted"],
                                "smiles_extraction_rate": split_summary["smiles_extraction_rate"],
                                "exact_match_accuracy": split_summary["exact_match_accuracy"],
                                "exact_match_given_extracted": split_summary[
                                    "exact_match_given_extracted"
                                ],
                                "valid_smiles": split_summary["valid_smiles"],
                                "validity_rate": split_summary["validity_rate"],
                                "validity_given_extracted": split_summary[
                                    "validity_given_extracted"
                                ],
                                "mean_tanimoto_all": split_summary["mean_tanimoto_all"],
                                "mean_tanimoto_valid": split_summary["mean_tanimoto_valid"],
                                "invalid_no_smiles": split_summary["invalid_no_smiles"],
                                "invalid_parse": split_summary["invalid_parse"],
                                "total_generated_tokens": split_total_generated_tokens,
                                "wall_seconds": wall_seconds,
                                "wall_tokens_per_second": (
                                    split_total_generated_tokens / wall_seconds
                                    if wall_seconds > 0
                                    else 0.0
                                ),
                                "combined_output": str(output_path),
                                "results": split_results,
                            },
                        )
                    )
        final_payload = {
            "completed": True,
            "signature_id": signature_id,
            "signature": signature,
            "model_name": args.model_name,
            "served_model_name": served_model_name,
            "tokenizer_name": args.tokenizer_name or args.model_name,
            "server_urls": server_urls,
            "num_examples": summary["num_examples"],
            "correct": summary["correct"],
            "errors": state.errors,
            "request_errors": summary["request_errors"],
            "request_error_rate": summary["request_error_rate"],
            "generation_started": summary["generation_started"],
            "generation_started_rate": summary["generation_started_rate"],
            "smiles_extracted": summary["smiles_extracted"],
            "smiles_extraction_rate": summary["smiles_extraction_rate"],
            "exact_match_accuracy": summary["exact_match_accuracy"],
            "exact_match_given_extracted": summary["exact_match_given_extracted"],
            "valid_smiles": summary["valid_smiles"],
            "validity_rate": summary["validity_rate"],
            "validity_given_extracted": summary["validity_given_extracted"],
            "mean_tanimoto_all": summary["mean_tanimoto_all"],
            "mean_tanimoto_valid": summary["mean_tanimoto_valid"],
            "invalid_no_smiles": summary["invalid_no_smiles"],
            "invalid_parse": summary["invalid_parse"],
            "total_generated_tokens": state.total_generated_tokens,
            "wall_seconds": wall_seconds,
            "wall_tokens_per_second": state.total_generated_tokens / wall_seconds if wall_seconds > 0 else 0.0,
            "partial_results": str(partial_path),
            "progress": str(progress_path),
            "split_summaries": split_summaries,
            "split_outputs": split_outputs,
            "results": ordered_results,
        }
        atomic_write_json(output_path, final_payload)
        for split_output_path, split_payload in split_payloads:
            split_output_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(split_output_path, split_payload)
        atomic_write_json(
            progress_path,
            final_payload | {"results": None, "state": "complete", "per_server": state.per_server},
        )
        for split_name, split_summary in split_summaries.items():
            print(
                f"[{now()}] split={split_name} examples={split_summary['num_examples']} "
                f"exact_match={split_summary['exact_match_accuracy']:.4f} "
                f"validity={split_summary['validity_rate']:.4f} "
                f"tanimoto_all={split_summary['mean_tanimoto_all']:.4f} "
                f"errors={split_summary['request_errors']}",
                flush=True,
            )
        print(
            f"[{now()}] complete examples={len(state.results)} "
            f"exact_match={final_payload['exact_match_accuracy']:.4f} "
            f"validity={final_payload['validity_rate']:.4f} "
            f"tanimoto_all={final_payload['mean_tanimoto_all']:.4f} "
            f"errors={state.errors} "
            f"wall_tok/s={final_payload['wall_tokens_per_second']:.2f} "
            f"saved={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
