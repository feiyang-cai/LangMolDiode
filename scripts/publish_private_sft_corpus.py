#!/usr/bin/env python3
"""Publish a prepared SFT corpus to one private Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    summary = json.loads((args.folder / "summary.json").read_text(encoding="utf-8"))
    shards = sorted((args.folder / "data").glob("train-*.parquet"))
    if not shards or len(shards) != summary["shards"] or not summary["all_matches"]:
        raise ValueError("prepared corpus summary is incomplete")
    rows = sum(pq.ParquetFile(shard).metadata.num_rows for shard in shards)
    if rows != summary["rows"]:
        raise ValueError(f"Parquet row count {rows} differs from summary {summary['rows']}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id=args.repo_id, repo_type="dataset").private:
        raise RuntimeError(f"Refusing to upload to public dataset: {args.repo_id}")
    paths = [args.folder / "README.md", args.folder / "summary.json", *shards]
    operations = [CommitOperationAdd(path_in_repo=path.relative_to(args.folder).as_posix(), path_or_fileobj=path)
                  for path in paths]
    result = api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=operations,
                               commit_message=f"Add verified SFT corpus ({rows} rows)")
    info = api.repo_info(repo_id=args.repo_id, repo_type="dataset")
    if not info.private:
        raise RuntimeError("Dataset became public unexpectedly")
    uploaded = {s.rfilename for s in info.siblings}
    if not {path.relative_to(args.folder).as_posix() for path in paths}.issubset(uploaded):
        raise RuntimeError("Dataset upload is incomplete")
    print(f"{args.repo_id}: private={info.private}, rows={rows}, files={len(paths)}, commit={result.oid}")


if __name__ == "__main__":
    main()
