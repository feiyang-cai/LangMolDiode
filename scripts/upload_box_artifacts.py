#!/usr/bin/env python3
"""Archive selected private training artifacts using an authenticated Box CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def box(cli: Path, home: Path, *args: str):
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(ROOT / ".runtime/xdg_cache")
    result = subprocess.run(
        [str(cli), *args, "--json"], check=True, text=True, capture_output=True, env=env
    )
    return json.loads(result.stdout)


def entries(result):
    if isinstance(result, list):
        return result
    if "entries" in result:
        return result["entries"]
    return [result]


def children(cli: Path, home: Path, parent_id: str):
    return entries(
        box(cli, home, "folders:items", parent_id, "--max-items", "1000", "--fields", "id,name,type,size,sha1")
    )


def ensure_folder(cli: Path, home: Path, parent_id: str, name: str):
    existing = [item for item in children(cli, home, parent_id) if item.get("name") == name]
    if existing:
        if existing[0].get("type") != "folder":
            raise RuntimeError(f"Box item {name!r} exists but is not a folder")
        return str(existing[0]["id"])
    return str(box(cli, home, "folders:create", parent_id, name)["id"])


def sha1(path: Path):
    digest = hashlib.sha1()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload(cli: Path, home: Path, folder_id: str, source: Path):
    existing = [item for item in children(cli, home, folder_id) if item.get("name") == source.name]
    local_sha1 = sha1(source)
    if existing and existing[0].get("sha1") == local_sha1:
        print(f"already uploaded: {source}", flush=True)
        return
    arguments = ["files:upload", str(source), "--parent-id", folder_id]
    if existing:
        arguments.append("--overwrite")
    result = entries(box(cli, home, *arguments))[0]
    uploaded = box(cli, home, "files:get", str(result["id"]), "--fields", "sha1,name,size")
    if uploaded.get("sha1") != local_sha1:
        raise RuntimeError(f"Box checksum mismatch for {source}")
    print(f"uploaded: {source} -> Box file {result['id']}", flush=True)


def portable_source(parts: tuple[str, ...], source: Path):
    if source.name != "adapter_config.json":
        return source
    config = json.loads(source.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = (
        "Qwen/Qwen3.5-4B" if parts[0] == "SFT" else "ChemFM/LangMolDiode-Qwen3.5-4B-SFT"
    )
    staged = ROOT / ".runtime/box_staging" / Path(*parts) / source.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return staged


def files(args, selection):
    yield ("SFT", "corpus"), args.sft_corpus
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        yield ("SFT", "adapter"), args.sft_adapter / name

    step = selection["final_rl_step"]
    folder = args.rl_run / f"global_step_{step}" / "actor/huggingface/adapter"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        yield ("RL", f"step{step}"), folder / name

    names = {
        "mollang_validated_success": "best_validated",
        "mollangbench_test": "best_test",
        "mollangbench_extended": "best_extended",
    }
    for dataset, info in selection["best"].items():
        step = info["rollout_step"]
        folder = args.rl_run / f"global_step_{step}" / "actor/huggingface/adapter"
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            yield ("RL", f"{names[dataset]}_step{step}"), folder / name

    for folder in sorted(args.responses.iterdir()):
        if folder.is_dir():
            for response in sorted(folder.glob("*.jsonl")):
                yield ("responses", folder.name), response
    yield (), ROOT / "artifacts/selection.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-corpus", required=True, type=Path)
    parser.add_argument("--sft-adapter", required=True, type=Path)
    parser.add_argument("--rl-run", required=True, type=Path)
    parser.add_argument("--responses", type=Path, default=ROOT / "artifacts/staged/responses")
    parser.add_argument("--box-parent", default="419341962405")
    parser.add_argument("--box-cli", type=Path, default=ROOT / ".runtime/box-cli/node_modules/.bin/box")
    parser.add_argument("--box-home", type=Path, default=ROOT / ".runtime/box_home")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selection = json.loads((ROOT / "artifacts/selection.json").read_text())
    manifest = list(files(args, selection))
    for parts, source in manifest:
        if not source.is_file():
            raise FileNotFoundError(source)
        print(f"LangMolDiode/{'/'.join(parts + (source.name,))} <- {source}", flush=True)
    if args.dry_run:
        return

    if not args.box_cli.is_file():
        raise FileNotFoundError(args.box_cli)
    args.box_home.mkdir(parents=True, exist_ok=True)
    destination = box(args.box_cli, args.box_home, "folders:get", args.box_parent, "--fields", "id,name")
    if destination.get("type") != "folder":
        raise RuntimeError(f"Box destination {args.box_parent} is not a folder")
    folders = {(): args.box_parent}
    for parts, source in manifest:
        for length in range(1, len(parts) + 1):
            prefix = parts[:length]
            if prefix not in folders:
                folders[prefix] = ensure_folder(
                    args.box_cli, args.box_home, folders[prefix[:-1]], prefix[-1]
                )
        upload(args.box_cli, args.box_home, folders[parts], portable_source(parts, source))


if __name__ == "__main__":
    main()
