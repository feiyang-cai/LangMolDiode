#!/usr/bin/env python3
"""Publish selected inference adapters to existing private ChemFM model repos."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def private_repo(api: HfApi, repo_id: str) -> None:
    info = api.model_info(repo_id)
    if not info.private:
        raise RuntimeError(f"Refusing to upload to a non-private repository: {repo_id}")


def config_operation(path: Path, base_model: str, destination: str) -> CommitOperationAdd:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = base_model
    contents = json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    return CommitOperationAdd(path_in_repo=destination, path_or_fileobj=io.BytesIO(contents))


def adapter_operations(adapter_dir: Path, base_model: str, prefix: str = "") -> list[CommitOperationAdd]:
    weights = adapter_dir / "adapter_model.safetensors"
    config = adapter_dir / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError(f"Missing adapter files in {adapter_dir}")
    return [
        config_operation(config, base_model, prefix + "adapter_config.json"),
        CommitOperationAdd(path_in_repo=prefix + "adapter_model.safetensors", path_or_fileobj=weights),
    ]


def tokenizer_operations(folder: Path) -> list[CommitOperationAdd]:
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json")
    return [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=folder / name)
        for name in names
        if (folder / name).is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument("--rl-run", type=Path, required=True)
    parser.add_argument("--sft-repo", default="ChemFM/LangMolDiode-Qwen3.5-4B-SFT-LoRA")
    parser.add_argument("--rl-repo", default="ChemFM/LangMolDiode-Qwen3.5-4B-RL-LoRA")
    parser.add_argument("--rl-base-repo", default="ChemFM/LangMolDiode-Qwen3.5-4B-SFT")
    parser.add_argument("--only", choices=("sft", "rl_final", "rl_best"), action="append")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN before publishing")
    api = HfApi(token=token)
    root = Path(__file__).resolve().parents[1]
    selection = json.loads((root / "artifacts/selection.json").read_text(encoding="utf-8"))
    requested = set(args.only or ("sft", "rl_final", "rl_best"))

    private_repo(api, args.rl_base_repo)
    if "sft" in requested:
        private_repo(api, args.sft_repo)
        ops = adapter_operations(args.sft_checkpoint, "Qwen/Qwen3.5-4B")
        ops += tokenizer_operations(args.sft_checkpoint)
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=root / "docs/hf_sft_lora_card.md"))
        print(api.create_commit(repo_id=args.sft_repo, operations=ops, commit_message="Upload SFT checkpoint 14190 LoRA"), flush=True)

    if "rl_final" in requested:
        private_repo(api, args.rl_repo)
        step = selection["final_rl_step"]
        folder = args.rl_run / f"global_step_{step}" / "actor/huggingface"
        ops = adapter_operations(folder / "adapter", args.rl_base_repo)
        ops += tokenizer_operations(folder)
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=root / "docs/hf_rl_lora_card.md"))
        print(api.create_commit(repo_id=args.rl_repo, operations=ops, commit_message=f"Upload final RL step {step} LoRA"), flush=True)

    if "rl_best" in requested:
        private_repo(api, args.rl_repo)
        for dataset, info in selection["best"].items():
            step = info["rollout_step"]
            folder = args.rl_run / f"global_step_{step}" / "actor/huggingface/adapter"
            prefix = {
                "mollang_validated_success": "best_validated",
                "mollangbench_test": "best_test",
                "mollangbench_extended": "best_extended",
            }[dataset] + f"_step{step}/"
            ops = adapter_operations(folder, args.rl_base_repo, prefix)
            print(api.create_commit(repo_id=args.rl_repo, operations=ops, commit_message=f"Upload {dataset} best step {step} LoRA"), flush=True)


if __name__ == "__main__":
    main()
