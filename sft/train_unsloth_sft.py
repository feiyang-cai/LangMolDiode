#!/usr/bin/env python
"""Train a Qwen LoRA adapter with Unsloth SFT."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainerCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--eval-dataset", type=Path, default=None)
    parser.add_argument("--generation-eval-dataset", type=Path, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="outputs/qwen35_4b_lora")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-template-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-groundtruth-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow literal target SMILES in system/user prompt text. Default is to fail.",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--generation-eval-steps", type=int, default=200)
    parser.add_argument("--generation-eval-subset-size", type=int, default=128)
    parser.add_argument("--generation-eval-initial-subset-size", type=int, default=8)
    parser.add_argument("--generation-eval-max-new-tokens", type=int, default=32768)
    parser.add_argument("--generation-eval-max-time-per-sample", type=float, default=1800.0)
    parser.add_argument("--generation-eval-progress-interval", type=int, default=1)
    parser.add_argument("--generation-eval-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-eval-final-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", default="unsloth")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--save-merged-16bit", action="store_true")
    return parser.parse_args()


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))


def world_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_world_process_zero() -> bool:
    return world_rank() == 0


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


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
    if Path(model_name).exists():
        return str(Path(model_name).resolve())
    if local_files_only:
        snapshot = local_snapshot_for_model(model_name)
        if snapshot is not None:
            return str(snapshot)
    return model_name


def resolve_torch_dtype(dtype: str):
    if dtype == "auto":
        return None
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def token_exists(tokenizer, token: str | None) -> bool:
    if not token:
        return False
    if token == "<EOS_TOKEN>":
        return False
    try:
        vocab = tokenizer.get_vocab()
        if token not in vocab:
            return False
    except Exception:
        pass
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return False
    if token_id is None:
        return False
    unk_id = getattr(tokenizer, "unk_token_id", None)
    return token_id != unk_id


def choose_eos_token(tokenizer) -> str | None:
    candidates = [
        getattr(tokenizer, "eos_token", None),
        "<|im_end|>",
        "<|endoftext|>",
    ]
    for token in candidates:
        if token_exists(tokenizer, token):
            return token
    return None


SMILES_RE = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.IGNORECASE | re.DOTALL)


def extract_smiles(text: str | None) -> str | None:
    if not text:
        return None
    match = SMILES_RE.search(text)
    return match.group(1).strip() if match else None


try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
except Exception:
    Chem = None


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    smiles = smiles.strip()
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def groundtruth_smiles_candidates(row: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for key in ("smiles", "expected_smiles", "candidate_smiles"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.add(value.strip())
            canonical = canonicalize_smiles(value)
            if canonical:
                candidates.add(canonical)
    return candidates


def prompt_text_for_leak_check(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    prompt_parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            continue
        prompt_parts.append(str(message.get("content", "")))
    return "\n".join(prompt_parts)


def find_groundtruth_prompt_leak(row: dict[str, Any]) -> str | None:
    prompt_text = prompt_text_for_leak_check(row.get("messages"))
    if not prompt_text:
        return None
    for candidate in sorted(groundtruth_smiles_candidates(row), key=len, reverse=True):
        if not candidate:
            continue
        tagged = f"<smiles>{candidate}</smiles>"
        if tagged in prompt_text:
            return tagged
        # Very short SMILES fragments, e.g. C, N, Cl, can occur naturally in
        # prose descriptions. Require a longer literal match unless it is tagged.
        if len(candidate) >= 12 and candidate in prompt_text:
            return candidate
    return None


def validate_no_groundtruth_in_prompts(dataset, dataset_name: str, allow: bool) -> None:
    if allow or "messages" not in dataset.column_names:
        return
    leaks = []
    for index, row in enumerate(dataset):
        leaked = find_groundtruth_prompt_leak(dict(row))
        if leaked:
            leaks.append(
                {
                    "index": index,
                    "leaked": leaked,
                    "source_file": row.get("source_file"),
                    "source_row_index": row.get("source_row_index"),
                }
            )
            if len(leaks) >= 5:
                break
    if leaks:
        raise ValueError(
            f"Ground-truth SMILES detected in {dataset_name} prompt text: "
            + json.dumps(leaks, ensure_ascii=False)
        )


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        features = [dict(feature) for feature in features]
        labels = []
        for feature in features:
            labels.append(feature.pop("labels", list(feature["input_ids"])))
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_len - len(label)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, stop_sequence_ids: list[int]):
        self.stop_sequence_ids = stop_sequence_ids

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.stop_sequence_ids:
            return False
        if input_ids.shape[-1] < len(self.stop_sequence_ids):
            return False
        suffix = input_ids[0, -len(self.stop_sequence_ids):].tolist()
        return suffix == self.stop_sequence_ids


class JsonlMetricsCallback(TrainerCallback):
    """Write rank-0 Trainer logs plus lightweight throughput/memory estimates."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / "training_metrics.jsonl"
        self.start_time: float | None = None
        self.last_log_time: float | None = None
        self.last_log_step = 0
        self.effective_train_batch_size = 0

    def _memory_metrics(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        denom = 1024**3
        return {
            "cuda_device": device,
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(device) / denom, 4),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(device) / denom, 4),
            "gpu_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(device) / denom, 4),
            "gpu_max_memory_reserved_gb": round(torch.cuda.max_memory_reserved(device) / denom, 4),
        }

    def _append(self, record: dict[str, Any]) -> None:
        if not is_world_process_zero():
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": socket.gethostname(),
            **record,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")

    def _base_record(self, event: str, state) -> dict[str, Any]:
        now = time.time()
        elapsed = 0.0 if self.start_time is None else now - self.start_time
        step = int(getattr(state, "global_step", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        record = {
            "event": event,
            "global_step": step,
            "max_steps": max_steps,
            "epoch": getattr(state, "epoch", None),
            "progress": (step / max_steps) if max_steps > 0 else None,
            "elapsed_seconds": round(elapsed, 3),
            "world_size": world_size(),
            "effective_train_batch_size": self.effective_train_batch_size,
            "estimated_samples_seen": step * self.effective_train_batch_size,
        }
        record.update(self._memory_metrics())
        return record

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = int(getattr(state, "global_step", 0) or 0)
        self.effective_train_batch_size = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * world_size()
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._append(
            {
                **self._base_record("train_begin", state),
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "optim": str(args.optim),
            }
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.time()
        step = int(getattr(state, "global_step", 0) or 0)
        seconds_since_last_log = None
        steps_since_last_log = step - self.last_log_step
        if self.last_log_time is not None:
            seconds_since_last_log = now - self.last_log_time
        record = self._base_record("log", state)
        if seconds_since_last_log and seconds_since_last_log > 0 and steps_since_last_log > 0:
            record.update(
                {
                    "seconds_since_last_log": round(seconds_since_last_log, 3),
                    "optimizer_steps_per_second": round(steps_since_last_log / seconds_since_last_log, 6),
                    "estimated_samples_per_second": round(
                        steps_since_last_log
                        * self.effective_train_batch_size
                        / seconds_since_last_log,
                        6,
                    ),
                }
            )
        if logs:
            record["trainer_logs"] = logs
        self._append(record)
        self.last_log_time = now
        self.last_log_step = step
        return control

    def on_save(self, args, state, control, **kwargs):
        self._append(self._base_record("save", state))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._append(self._base_record("train_end", state))
        return control


class GenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        raw_dataset,
        output_dir: str,
        subset_size: int,
        initial_subset_size: int,
        max_new_tokens: int,
        max_time_per_sample: float,
        interval_steps: int,
        progress_interval: int,
        gradient_checkpointing: str,
        enable_thinking: bool,
        final_only: bool,
    ):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir) / "generation_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.max_time_per_sample = max_time_per_sample
        self.interval_steps = interval_steps
        self.progress_interval = max(1, progress_interval)
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_thinking = enable_thinking
        self.final_only = final_only
        stop_sequence_ids = tokenizer.encode("</smiles>", add_special_tokens=False)
        self.stopping_criteria = (
            StoppingCriteriaList([StopOnTokenSequence(stop_sequence_ids)])
            if stop_sequence_ids
            else None
        )
        count = min(subset_size, len(raw_dataset))
        self.examples = []
        for i in range(count):
            row = dict(raw_dataset[i])
            row["_eval_index"] = i
            self.examples.append(row)
        initial_count = (
            len(self.examples)
            if initial_subset_size < 0
            else min(initial_subset_size, len(self.examples))
        )
        self.initial_examples = self.examples[:initial_count]
        self.baseline_state_path = self.output_dir / "baseline_state.json"
        self.baseline_signature = {
            "num_examples": len(self.initial_examples),
            "full_subset_examples": len(self.examples),
            "max_new_tokens": self.max_new_tokens,
            "max_time_per_sample": self.max_time_per_sample,
            "interval_steps": self.interval_steps,
            "progress_interval": self.progress_interval,
            "enable_thinking": self.enable_thinking,
            "final_only": self.final_only,
            "source_file": self.examples[0].get("source_file") if self.examples else None,
            "first_expected_smiles": self.examples[0].get("expected_smiles") if self.examples else None,
        }

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _is_world_process_zero(self) -> bool:
        return not self._dist_ready() or dist.get_rank() == 0

    def _rank(self) -> int:
        return dist.get_rank() if self._dist_ready() else 0

    def _world_size(self) -> int:
        return dist.get_world_size() if self._dist_ready() else 1

    def _step_dir(self, step: int) -> Path:
        return self.output_dir / f"step_{step:06d}_shards"

    def _progress_path(self, step: int, rank: int) -> Path:
        return self._step_dir(step) / f"rank_{rank:04d}_progress.json"

    def _results_path(self, step: int, rank: int) -> Path:
        return self._step_dir(step) / f"rank_{rank:04d}_results.json"

    def _done_path(self, step: int, rank: int) -> Path:
        return self._step_dir(step) / f"rank_{rank:04d}.done"

    def _merged_done_path(self, step: int) -> Path:
        return self._step_dir(step) / "merged.done"

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_progress(
        self,
        step: int,
        rank: int,
        processed: int,
        correct: int,
        total_local: int,
        generated_tokens: int = 0,
        generation_seconds: float = 0.0,
    ) -> None:
        tokens_per_second = generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
        self._write_json(
            self._progress_path(step, rank),
            {
                "rank": rank,
                "processed": processed,
                "correct": correct,
                "total_local": total_local,
                "generated_tokens": generated_tokens,
                "generation_seconds": generation_seconds,
                "tokens_per_second": tokens_per_second,
                "updated_at": time.time(),
            },
        )

    def _generation_prompt(self, prompt_messages: list[dict]) -> torch.Tensor:
        messages = [dict(message) for message in prompt_messages]
        if self.final_only:
            for message in reversed(messages):
                if message.get("role") == "user":
                    message["content"] = (
                        message.get("content", "").rstrip()
                        + "\n\nReturn only the final answer as <smiles>...</smiles>. "
                        + "Do not include reasoning, markdown, or code fences."
                    )
                    break
        kwargs = {
            "conversation": messages,
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(**kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(**kwargs)

    def _read_global_progress(self, step: int, world_size: int) -> tuple[int, int, int, float]:
        total_processed = 0
        total_correct = 0
        total_generated_tokens = 0
        total_generation_seconds = 0.0
        for rank in range(world_size):
            path = self._progress_path(step, rank)
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            total_processed += int(payload.get("processed", 0))
            total_correct += int(payload.get("correct", 0))
            total_generated_tokens += int(payload.get("generated_tokens", 0))
            total_generation_seconds += float(payload.get("generation_seconds", 0.0))
        return total_processed, total_correct, total_generated_tokens, total_generation_seconds

    def _wait_for_path(self, path: Path, label: str, poll_interval: float = 2.0) -> None:
        last_log = 0.0
        while not path.exists():
            now = time.time()
            if self._is_world_process_zero() and now - last_log >= 60.0:
                print(f"Waiting for {label}: {path}", flush=True)
                last_log = now
            time.sleep(poll_interval)

    def _baseline_done(self) -> bool:
        try:
            saved = json.loads(self.baseline_state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except json.JSONDecodeError:
            return False
        if not isinstance(saved, dict):
            return False
        return saved == self.baseline_signature

    def _mark_baseline_done(self) -> None:
        self.baseline_state_path.write_text(
            json.dumps(self.baseline_signature, indent=2),
            encoding="utf-8",
        )

    def _run_eval(self, state, model, examples: list[dict] | None = None, label: str = "generation"):
        examples = self.examples if examples is None else examples
        if not examples:
            return

        model_for_eval = model.module if hasattr(model, "module") else model
        was_training = model_for_eval.training
        old_use_cache = getattr(getattr(model_for_eval, "config", None), "use_cache", None)
        if hasattr(model_for_eval, "for_inference"):
            model_for_eval.for_inference()
        else:
            model_for_eval.eval()
        if hasattr(model_for_eval, "config"):
            model_for_eval.config.use_cache = True
        device = next(model_for_eval.parameters()).device

        world_size = self._world_size()
        rank = self._rank()
        local_examples = examples[rank::world_size]
        local_processed = 0
        local_correct = 0
        local_generated_tokens = 0
        local_generation_seconds = 0.0
        local_results = []
        eval_wall_start = time.time()
        step_dir = self._step_dir(state.global_step)
        step_dir.mkdir(parents=True, exist_ok=True)
        progress_path = self._progress_path(state.global_step, rank)
        results_path = self._results_path(state.global_step, rank)
        done_path = self._done_path(state.global_step, rank)
        merged_done_path = self._merged_done_path(state.global_step)
        for path in (progress_path, results_path, done_path):
            if path.exists():
                path.unlink()
        if self._is_world_process_zero() and merged_done_path.exists():
            merged_done_path.unlink()
        self._write_progress(state.global_step, rank, 0, 0, len(local_examples))

        progress_bar = None
        last_global_processed = 0
        if self._is_world_process_zero():
            print(
                f"Generation eval start: step={state.global_step} "
                f"label={label} examples={len(examples)} world_size={world_size} "
                f"max_new_tokens={self.max_new_tokens} "
                f"max_time_per_sample={self.max_time_per_sample} "
                f"progress_interval={self.progress_interval}",
                flush=True,
            )
        if self._is_world_process_zero() and tqdm is not None:
            progress_bar = tqdm(
                total=len(examples),
                desc=f"Generation eval step={state.global_step} {label}",
                dynamic_ncols=True,
            )

        for example in local_examples:
            sample_start = time.time()
            prompt_messages = example["messages"][:-1]
            expected = canonicalize_smiles(example.get("expected_smiles") or example.get("smiles"))
            inputs = self._generation_prompt(prompt_messages).to(device)
            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "stopping_criteria": self.stopping_criteria,
                }
                if self.max_time_per_sample and self.max_time_per_sample > 0:
                    generation_kwargs["max_time"] = self.max_time_per_sample
                outputs = model_for_eval.generate(inputs, **generation_kwargs)
            generated = self.tokenizer.decode(
                outputs[0, inputs.shape[-1]:],
                skip_special_tokens=False,
            )
            predicted = canonicalize_smiles(extract_smiles(generated))
            match = predicted is not None and expected is not None and predicted == expected
            elapsed = time.time() - sample_start
            generated_tokens = int(outputs.shape[-1] - inputs.shape[-1])
            sample_tokens_per_second = generated_tokens / elapsed if elapsed > 0 else 0.0
            has_smiles_close = "</smiles>" in generated.lower()
            timed_out = (
                self.max_time_per_sample is not None
                and self.max_time_per_sample > 0
                and elapsed >= (0.95 * self.max_time_per_sample)
                and not has_smiles_close
            )
            local_processed += 1
            local_correct += int(match)
            local_generated_tokens += generated_tokens
            local_generation_seconds += elapsed
            local_results.append(
                {
                    "eval_index": example["_eval_index"],
                    "source_file": example.get("source_file"),
                    "expected_smiles": expected,
                    "predicted_smiles": predicted,
                    "match": match,
                    "has_smiles_close": has_smiles_close,
                    "timed_out": timed_out,
                    "generated_tokens": generated_tokens,
                    "seconds": elapsed,
                    "tokens_per_second": sample_tokens_per_second,
                    "raw_generation_preview": generated[:4096],
                }
            )
            self._write_json(
                results_path,
                {"rank": rank, "partial": True, "results": local_results},
            )
            self._write_progress(
                state.global_step,
                rank,
                local_processed,
                local_correct,
                len(local_examples),
                generated_tokens=local_generated_tokens,
                generation_seconds=local_generation_seconds,
            )
            if self._is_world_process_zero():
                (
                    global_processed,
                    global_correct,
                    global_generated_tokens,
                    global_generation_seconds,
                ) = self._read_global_progress(state.global_step, world_size)
                accuracy = (global_correct / global_processed) if global_processed else 0.0
                gpu_tokens_per_second = (
                    global_generated_tokens / global_generation_seconds
                    if global_generation_seconds > 0
                    else 0.0
                )
                if progress_bar is not None:
                    if global_processed > last_global_processed:
                        progress_bar.update(global_processed - last_global_processed)
                        last_global_processed = global_processed
                    progress_bar.set_postfix(
                        exact_match=f"{accuracy:.4f}",
                        processed=f"{global_processed}/{len(examples)}",
                        tok_s=f"{gpu_tokens_per_second:.2f}",
                    )
                else:
                    print(
                        f"Generation eval progress: step={state.global_step} "
                        f"processed={global_processed}/{len(examples)} "
                        f"exact_match={accuracy:.4f} "
                        f"tok/s={gpu_tokens_per_second:.2f}",
                        flush=True,
                    )

        if progress_bar is not None:
            progress_bar.close()

        self._write_json(results_path, {"rank": rank, "results": local_results})
        done_path.write_text("done\n", encoding="utf-8")

        if self._is_world_process_zero():
            for wait_rank in range(world_size):
                self._wait_for_path(
                    self._done_path(state.global_step, wait_rank),
                    f"rank {wait_rank} generation eval completion",
                )
            results = []
            for read_rank in range(world_size):
                shard_payload = json.loads(
                    self._results_path(state.global_step, read_rank).read_text(encoding="utf-8")
                )
                results.extend(shard_payload.get("results", []))
            results.sort(key=lambda row: row["eval_index"])
            correct = sum(int(row["match"]) for row in results)
            accuracy = correct / len(results)
            total_generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in results)
            total_generation_seconds = sum(float(row.get("seconds", 0.0)) for row in results)
            gpu_tokens_per_second = (
                total_generated_tokens / total_generation_seconds
                if total_generation_seconds > 0
                else 0.0
            )
            wall_seconds = time.time() - eval_wall_start
            wall_tokens_per_second = (
                total_generated_tokens / wall_seconds
                if wall_seconds > 0
                else 0.0
            )
            payload = {
                "global_step": state.global_step,
                "label": label,
                "num_examples": len(results),
                "exact_match_accuracy": accuracy,
                "total_generated_tokens": total_generated_tokens,
                "total_generation_seconds": total_generation_seconds,
                "gpu_tokens_per_second": gpu_tokens_per_second,
                "wall_seconds": wall_seconds,
                "wall_tokens_per_second": wall_tokens_per_second,
                "results": results,
            }
            out_path = self.output_dir / f"step_{state.global_step:06d}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"Generation eval: step={state.global_step} label={label} "
                f"examples={len(results)} exact_match={accuracy:.4f} "
                f"gpu_tok/s={gpu_tokens_per_second:.2f} "
                f"wall_tok/s={wall_tokens_per_second:.2f} "
                f"saved={out_path}",
                flush=True,
            )
            merged_done_path.write_text("done\n", encoding="utf-8")
        elif self._dist_ready():
            self._wait_for_path(merged_done_path, "merged generation eval results")

        if hasattr(model_for_eval, "config") and old_use_cache is not None:
            model_for_eval.config.use_cache = old_use_cache
        if was_training:
            if hasattr(model_for_eval, "for_training"):
                model_for_eval.for_training(use_gradient_checkpointing=self.gradient_checkpointing)
            else:
                model_for_eval.train()

    def on_step_end(self, args, state, control, **kwargs):
        if self.interval_steps > 0 and state.global_step > 0 and state.global_step % self.interval_steps == 0:
            self._run_eval(state, kwargs["model"], examples=self.examples, label="periodic")
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        should_run_baseline = not self._baseline_done()

        if not should_run_baseline:
            if self._is_world_process_zero():
                print(
                    f"Generation eval baseline already exists: {self.baseline_state_path}",
                    flush=True,
                )
            return control
        self._run_eval(state, kwargs["model"], examples=self.initial_examples, label="baseline")
        if self._is_world_process_zero():
            self._mark_baseline_done()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._run_eval(state, kwargs["model"], examples=self.examples, label="train_end")
        return control


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_load_name = resolve_model_load_name(args.model_name, args.local_files_only)
    tokenizer_load_name = resolve_model_load_name(
        args.tokenizer_name or args.model_name,
        args.local_files_only,
    )

    if local_rank() == 0:
        print(f"Loading model {model_load_name}", flush=True)
    model, _processing_class = FastLanguageModel.from_pretrained(
        model_name=model_load_name,
        max_seq_length=args.max_seq_length,
        dtype=resolve_torch_dtype(args.dtype),
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_files_only,
        device_map={"": local_rank()} if "LOCAL_RANK" in os.environ else None,
    )
    if local_rank() == 0:
        print(f"Loading tokenizer {tokenizer_load_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_load_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    eos_token = choose_eos_token(tokenizer)
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    if getattr(tokenizer, "pad_token", None) is None and eos_token is not None:
        tokenizer.pad_token = eos_token
    if local_rank() == 0:
        print(
            f"Using tokenizer={type(tokenizer).__name__} "
            f"eos_token={tokenizer.eos_token!r} pad_token={tokenizer.pad_token!r}",
            flush=True,
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.gradient_checkpointing,
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    if hasattr(model, "for_training"):
        model.for_training(use_gradient_checkpointing=args.gradient_checkpointing)
    else:
        FastLanguageModel.for_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    model.config.use_cache = False

    if local_rank() == 0:
        print(f"Loading training dataset {args.dataset}", flush=True)
    dataset = load_dataset("json", data_files=str(args.dataset), split="train")

    if args.use_chat_template and "messages" in dataset.column_names:
        validate_no_groundtruth_in_prompts(
            dataset,
            "training dataset",
            args.allow_groundtruth_in_prompt,
        )
        if local_rank() == 0:
            print("Applying chat template to training dataset", flush=True)

        def apply_qwen_chat_template(
            messages: list[dict],
            *,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            kwargs = {
                "conversation": messages,
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if add_generation_prompt:
                kwargs["enable_thinking"] = enable_thinking
            try:
                return tokenizer.apply_chat_template(**kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return tokenizer.apply_chat_template(**kwargs)

        def apply_template(batch):
            texts = []
            prompt_texts = []
            for messages in batch["messages"]:
                full_text = apply_qwen_chat_template(
                    messages,
                    add_generation_prompt=False,
                    enable_thinking=args.train_template_enable_thinking,
                )
                prompt_messages = messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages
                prompt_text = apply_qwen_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    enable_thinking=args.train_template_enable_thinking,
                )
                if not full_text.startswith(prompt_text):
                    fallback_prompt_text = apply_qwen_chat_template(
                        prompt_messages,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    if full_text.startswith(fallback_prompt_text):
                        prompt_text = fallback_prompt_text
                    else:
                        raise ValueError(
                            "Qwen chat template prompt is not a prefix of the full "
                            "training sample; label masking would be misaligned. "
                            f"full_prefix={full_text[:200]!r} prompt_prefix={prompt_text[:200]!r}"
                        )
                texts.append(full_text)
                prompt_texts.append(prompt_text)
            return {"text": texts, "prompt_text": prompt_texts}

        dataset = dataset.map(
            apply_template,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Applying tokenizer chat template",
        )
        text_field = "text"
    else:
        text_field = args.text_field

    if args.packing:
        raise ValueError("`--packing` is not supported in the current Trainer-based path.")

    eval_dataset = None
    raw_generation_eval_dataset = None
    if args.eval_dataset is not None:
        if local_rank() == 0:
            print(f"Loading loss eval dataset {args.eval_dataset}", flush=True)
        eval_dataset = load_dataset("json", data_files=str(args.eval_dataset), split="train")
        if args.use_chat_template and "messages" in eval_dataset.column_names:
            validate_no_groundtruth_in_prompts(
                eval_dataset,
                "loss eval dataset",
                args.allow_groundtruth_in_prompt,
            )
            if local_rank() == 0:
                print("Applying chat template to loss eval dataset", flush=True)
            eval_dataset = eval_dataset.map(
                apply_template,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Applying tokenizer chat template to loss eval",
            )
    elif args.validation_split and args.validation_split > 0:
        split = dataset.train_test_split(
            test_size=args.validation_split,
            seed=args.seed,
            shuffle=True,
        )
        dataset = split["train"]
        eval_dataset = split["test"]

    if args.generation_eval_dataset is not None:
        if local_rank() == 0:
            print(f"Loading generation eval dataset {args.generation_eval_dataset}", flush=True)
        raw_generation_eval_dataset = load_dataset(
            "json",
            data_files=str(args.generation_eval_dataset),
            split="train",
        )
        validate_no_groundtruth_in_prompts(
            raw_generation_eval_dataset,
            "generation eval dataset",
            args.allow_groundtruth_in_prompt,
        )

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch[text_field],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=not args.use_chat_template,
        )
        if "prompt_text" not in batch:
            tokenized["labels"] = [list(input_ids) for input_ids in tokenized["input_ids"]]
            return tokenized

        prompt_tokenized = tokenizer(
            batch["prompt_text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        labels = []
        for input_ids, prompt_ids in zip(tokenized["input_ids"], prompt_tokenized["input_ids"]):
            row_labels = list(input_ids)
            mask_len = min(len(prompt_ids), len(row_labels))
            row_labels[:mask_len] = [-100] * mask_len
            labels.append(row_labels)
        tokenized["labels"] = labels
        return tokenized

    if local_rank() == 0:
        print("Tokenizing training dataset", flush=True)
    train_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing train split",
    )
    if eval_dataset is not None:
        if local_rank() == 0:
            print("Tokenizing loss eval dataset", flush=True)
        eval_dataset = eval_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval split",
        )

    if local_rank() == 0 and len(train_dataset) > 0:
        lengths = train_dataset["input_ids"]
        lengths = [len(x) for x in lengths]
        lengths_sorted = sorted(lengths)

        def pct(p: float) -> int:
            idx = min(len(lengths_sorted) - 1, max(0, int(p * (len(lengths_sorted) - 1))))
            return lengths_sorted[idx]

        print(
            "Train token lengths: "
            f"min={lengths_sorted[0]} "
            f"p50={pct(0.50)} "
            f"p90={pct(0.90)} "
            f"p95={pct(0.95)} "
            f"p99={pct(0.99)} "
            f"max={lengths_sorted[-1]}",
            flush=True,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        optim=args.optim,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        prediction_loss_only=True,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        seed=args.seed,
        do_train=True,
        do_eval=eval_dataset is not None,
        report_to=args.report_to,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    if local_rank() == 0:
        print(
            f"Train samples={len(train_dataset)} "
            f"Eval samples={len(eval_dataset) if eval_dataset is not None else 0}",
            flush=True,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "training_config.json"
        config = {
            "args": vars(args),
            "resolved_model_load_name": model_load_name,
            "resolved_tokenizer_load_name": tokenizer_load_name,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "world_size": world_size(),
            "effective_train_batch_size": (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * world_size()
            ),
            "hostname": socket.gethostname(),
        }
        config_path.write_text(
            json.dumps(to_jsonable(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved training config: {config_path}", flush=True)

    callbacks = [JsonlMetricsCallback(args.output_dir)]
    if raw_generation_eval_dataset is not None:
        if local_rank() == 0:
            print(
                "Configuring generation eval: "
                f"rows={len(raw_generation_eval_dataset)} "
                f"subset_size={min(args.generation_eval_subset_size, len(raw_generation_eval_dataset))} "
                f"initial_subset_size={args.generation_eval_initial_subset_size} "
                f"interval_steps={args.generation_eval_steps} "
                f"max_new_tokens={args.generation_eval_max_new_tokens} "
                f"max_time_per_sample={args.generation_eval_max_time_per_sample} "
                f"progress_interval={args.generation_eval_progress_interval} "
                f"enable_thinking={args.generation_eval_enable_thinking} "
                f"final_only={args.generation_eval_final_only}",
                flush=True,
            )
        callbacks.append(
            GenerationEvalCallback(
                tokenizer=tokenizer,
                raw_dataset=raw_generation_eval_dataset,
                output_dir=args.output_dir,
                subset_size=args.generation_eval_subset_size,
                initial_subset_size=args.generation_eval_initial_subset_size,
                max_new_tokens=args.generation_eval_max_new_tokens,
                max_time_per_sample=args.generation_eval_max_time_per_sample,
                interval_steps=args.generation_eval_steps,
                progress_interval=args.generation_eval_progress_interval,
                gradient_checkpointing=args.gradient_checkpointing,
                enable_thinking=args.generation_eval_enable_thinking,
                final_only=args.generation_eval_final_only,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=CausalLMCollator(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if local_rank() == 0:
        print("Starting trainer.train()", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged_16bit:
        merged_dir = str(Path(args.output_dir) / "merged_16bit")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    main()
