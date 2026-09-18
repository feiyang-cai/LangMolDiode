# LangMolDiode

LangMolDiode trains a Qwen3.5-4B text-to-SMILES model in two stages: Unsloth
LoRA supervised fine-tuning (SFT), followed by verl DAPO reinforcement learning
with Megatron-Bridge training and vLLM rollout. This repository contains the
maintained source, configuration, and evaluation code. Training data, model
weights, generated responses, container images, and runtime caches are external
artifacts and are never committed.

## Source layout

- `sft/`: corpus preparation, token-length filtering, Unsloth training, and
  vLLM generation evaluation.
- `rl/`: verl parquet conversion, prompt filtering, and the SMILES reward.
- `scripts/run_sft.sh`: one-node SFT launch from prepared JSONL.
- `scripts/run_rl.sh`: one-node verl DAPO launch; also used by the two-node
  Slurm allocation launcher.
- `scripts/run_rl_two_nodes_existing_jobs.sh`: launch over two already-owned
  eight-GPU Slurm allocations without releasing either allocation.
- `sitecustomize.py`: opt-in W&B history and GPU-memory monitoring hooks for
  the pinned verl code. Upstream source trees remain unmodified.

## Reproduce the environment

The tested RL stack uses `verl` commit
`bcb638649a50e58494a8ddd92085ad1174f674b8` and `verl-recipe` commit
`ba246418f4de12b845a09bba975f1a5242adc898`. Run
`scripts/bootstrap_upstream.sh` to populate ignored `third_party/` checkouts.
The tested Apptainer image is `verlai/verl:vllm023.dev1`; set `VERL_SIF` to an
existing image or run `scripts/pull_verl_apptainer.sh`. The tested container has
Transformers 5.3.0, vLLM 0.23.1.dev0, Megatron-Core 0.16.1,
Megatron-Bridge 0.4.0rc0 (commit `6259ae83c735c4412796fc5cfb4c9607b949ae29`),
FLA 0.5.1, and RDKit 2026.3.3. Install the project-local overlay packages with
`scripts/install_container_rdkit.sh`, `scripts/install_container_qwen35_fastpath.sh`,
and `scripts/install_megatron_bridge.sh` if the image lacks them.

All launchers root caches under this directory's ignored `.runtime/` or
`containers/`; they do not use `~/.cache`. Set `HF_HOME`, `VERL_SIF`,
`MOLLANG_PYTHON_DEPS`, and `MOLLANG_MEGATRON_PYTHON_DEPS` to existing external
locations when reusing a prepared node environment.

## SFT

Prepare the corpus from curation artifacts using
`python sft/prepare_sft_data.py --input-root INPUT --output TRAIN.jsonl --response-mode qwen_think --require-match`.
Then filter at 40,960 rendered tokens with
`python sft/filter_sft_by_token_length.py --input TRAIN.jsonl --output TRAIN_40960.jsonl --tokenizer-name Qwen/Qwen3.5-4B`.
For the final three-epoch run:

```bash
TRAIN_JSONL=/path/to/filtered_sft.jsonl \
EVAL_JSONL=/path/to/loss_eval.jsonl \
OUTPUT_DIR=/path/to/sft_run \
bash scripts/run_sft.sh
```

The SFT seed used for the RL run was `checkpoint-14190` from the June 18
Qwen3.5-4B run. `scripts/run_sft.sh` accepts environment overrides for batch
size, accumulation, learning rate, and checkpoint frequency.

The exact filtered training corpus is also available as the private
`ChemFM/LangMolDiode-SFT-Corpus` dataset. It includes the raw generator output,
the exact SFT assistant response, model and reasoning-effort metadata, and an
independently checked molecule-match flag. To rebuild it from the filtered
JSONL and its linked curation artifacts, run
`python scripts/build_sft_corpus_dataset.py --input TRAIN_40960.jsonl --output-dir .runtime/sft_corpus_release`.
`scripts/publish_private_sft_corpus.py` only uploads to private dataset repos.

## RL

Convert prepared prompt JSONL files with
`python -m rl.prepare_data --train-jsonl TRAIN.jsonl --val-jsonl VAL.jsonl --bench-test-jsonl TEST.jsonl --bench-extended-jsonl EXTENDED.jsonl --output-dir /path/to/verl_data`.
Use `python rl/filter_prompt_tokens.py --help` to filter the training parquet
to the 2,048-token prompt cap. The production launch requires the merged SFT
base and explicit data paths:

```bash
MODEL_PATH=/path/to/merged_sft_model \
TRAIN_FILE=/path/to/train_maxprompt2048.parquet \
VAL_FILE=/path/to/val_mollangbench_combined.parquet \
bash scripts/run_in_apptainer.sh scripts/run_rl.sh
```

The final two-node configuration uses 640 generated prompt groups, eight
completions each, native DAPO filtering to 512 groups, and 16 actor optimizer
mini-batches per rollout. It uses fixed actor micro-batches of one, LoRA rank
64, temperature 1, top-p 1, top-k -1, 32,768 response tokens, sequence-level
rollout correction, token-mean loss, no KL loss, 1e-6 learning rate, validation
every five rollouts, and a checkpoint every rollout. Validation generates
three answers for each of 1,972 validated, 200 test, and 200 extended prompts.
The exact launch arguments live in `scripts/run_rl.sh`; the two-node resource
setup lives in `scripts/run_rl_two_nodes_existing_jobs.sh`.

## Artifact map

`artifacts/selection.json` records the SFT seed, final RL step, and three
per-dataset best RL checkpoints. Best is defined by `exact@1` among full
three-generation validation runs; ties use the earliest step. The raw
step-0/selected/final generations can be split by dataset with
`scripts/split_validation_generations.py`. This manifest is a provenance
record, not model data. The private Hugging Face model repositories and Box
archive are listed in `docs/artifacts.md`.
