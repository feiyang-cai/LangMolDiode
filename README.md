# LangMolDiode

<a id="readme-top"></a>

[![Anonymous dataset](https://img.shields.io/badge/Anonymous%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/mollangdata/MolLangData)
[![Anonymous checkpoints](https://img.shields.io/badge/Anonymous%20Hugging%20Face-Checkpoints-yellow.svg)](https://huggingface.co/collections/mollangdata/langmoldiode-checkpoints-6aad8ee76122dd2c28fde274)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](environment.yml)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE)

**LangMolDiode** is a language-conditional molecule generator that produces
SMILES from detailed molecular structure descriptions. It is built on
Qwen3.5-4B and trained with reinforcement learning using the curated
[anonymous MolLangData dataset](https://huggingface.co/datasets/mollangdata/MolLangData).

## Anonymous Resources

The following project resources are hosted anonymously for double-blind review.

| Resource | Links | Description |
| --- | --- | --- |
| MolLangData | [Anonymous Hugging Face dataset](https://huggingface.co/datasets/mollangdata/MolLangData) | Source molecule-description dataset |
| MolLangBench | [External Hugging Face dataset](https://huggingface.co/datasets/ChemFM/MolLangBench) | External core and extended generation evaluation sets |
| SFT training corpus | [Anonymous Hugging Face dataset](https://huggingface.co/datasets/mollangdata/LangMolDiode-SFT-Corpus) | 75,666 verified reasoning traces and final answers |
| Model checkpoints | [Anonymous Hugging Face collection](https://huggingface.co/collections/mollangdata/langmoldiode-checkpoints-6aad8ee76122dd2c28fde274) | Post-SFT model and SFT/RL LoRA adapters |

## Installation

### Conda

The Conda environment supports corpus preparation, SFT, and evaluation. After
cloning this repository, create it from the repository root:

```bash
conda env create -f environment.yml
conda activate langmoldiode
```

### Apptainer

The Apptainer environment is recommended for RL training. One command downloads
the upstream training code, pulls the verl image, and installs the Qwen3.5 and
Megatron dependencies into project-local directories:

```bash
bash scripts/setup_rl_environment.sh
```

Run any command inside the prepared environment with:

```bash
bash scripts/run_in_apptainer.sh python -c "import verl; print('verl ready')"
```

## Curating the SFT Corpus

We use descriptions from the MolLangData training set as prompts, collect
reasoning traces and SMILES answers from stronger teacher models, retain only
answers that reconstruct the target molecule, deduplicate by target molecule,
and filter examples whose rendered Qwen sequence exceeds 40,960 tokens.

To curate responses directly from MolLangData, use any OpenAI-compatible
chat-completions endpoint:

```bash
OPENAI_API_KEY=your_api_key \
python sft/curate_sft_corpus.py \
  --dataset mollangdata/MolLangData \
  --dataset-config generated_data \
  --split data \
  --output data/teacher_generations.jsonl \
  --base-url https://your-provider.example/v1 \
  --model your-teacher-model \
  --workers 8
```

The curator extracts the final tagged SMILES, verifies molecular equivalence
with RDKit, retries unsuccessful generations, and retains only verified matches
by default. Local endpoints can be used without an API key, and a local JSONL
file with `description` and `smiles` fields can be supplied through `--input`.
Keep the evaluation molecules out of the source used for training.

Convert the verified generations into chat-training JSONL and apply the token
length filter:

```bash
python sft/prepare_sft_data.py \
  --input-jsonl data/teacher_generations.jsonl \
  --output data/sft_train.jsonl \
  --response-mode qwen_think \
  --require-match \
  --require-reasoning \
  --dedupe-key canonical_smiles

python sft/filter_sft_by_token_length.py \
  --input data/sft_train.jsonl \
  --output data/sft_train_max40960.jsonl \
  --rejected-output data/sft_train_over40960.jsonl \
  --tokenizer-name Qwen/Qwen3.5-4B \
  --max-length 40960 \
  --no-local-files-only
```

The verified corpus is provided in an
[anonymous Hugging Face dataset](https://huggingface.co/datasets/mollangdata/LangMolDiode-SFT-Corpus).
To recreate the training JSONL directly from the release:

```bash
python sft/export_sft_corpus.py \
  --dataset mollangdata/LangMolDiode-SFT-Corpus \
  --output data/sft_train_max40960.jsonl
```

Every released row passed an independent RDKit molecular-equivalence check.

## SFT Training

Launch the three-epoch, eight-GPU LoRA training:

```bash
TRAIN_JSONL=data/sft_train_max40960.jsonl \
EVAL_JSONL=/path/to/eval.jsonl \
OUTPUT_DIR=outputs/langmoldiode_sft \
NPROC_PER_NODE=8 \
bash scripts/run_sft.sh
```

The launcher accepts environment overrides for sequence length, batch size,
gradient accumulation, learning rate, evaluation, logging, and checkpoint
frequency. Its defaults reproduce the final SFT setup: Qwen3.5-4B, three
epochs, rank-64 LoRA with alpha 128, a 40,960-token limit, batch size 1 per GPU,
two gradient-accumulation steps, eight GPUs, and a learning rate of 2e-4.

## RL Training

Download the released MolLangData training and held-out sets together with the
external MolLangBench generation benchmark, convert them to verl Parquet, and
apply the same prompt-token filter used for training:

```bash
python -m rl.prepare_data \
  --output-dir data/verl

python rl/filter_prompt_tokens.py \
  --model /path/to/merged_sft_model \
  --input data/verl/train.parquet \
  --output data/verl/train_maxprompt2048.parquet \
  --max-prompt-length 2048
```

This produces 161,111 MolLangData training prompts and a combined validation
file containing 1,972 accepted MolLangData examples plus the 200-example core
and 200-example extended MolLangBench generation sets. Local prepared JSONL can
still be supplied with `--train-jsonl`, `--val-jsonl`,
`--bench-test-jsonl`, and `--bench-extended-jsonl`.

Launch DAPO from the head of a two-node Ray cluster with eight GPUs per node:

```bash
MODEL_PATH=/path/to/merged_sft_model \
TRAIN_FILE=data/verl/train_maxprompt2048.parquet \
VAL_FILE=data/verl/val_mollangbench_combined.parquet \
RUN_NAME=langmoldiode_dapo \
bash scripts/run_in_apptainer.sh scripts/run_rl.sh \
  +ray_kwargs.ray_init.address=auto \
  '~ray_kwargs.ray_init.num_cpus'
```

The defaults reproduce the final RL setup: 1,500 rollout steps, eight
generations per prompt, 512 prompt groups per update, a learning rate of 1e-6,
validation every five steps with three generations, and a checkpoint after
every step. The reward uses weights 1.0/0.4/0.1 for Tanimoto similarity, exact
match, and answer format; its overlength penalty ramps from 2,048 to 32,768
response tokens with a maximum penalty of 0.4. Settings remain configurable
with environment variables in `scripts/run_rl.sh`.

## License

LangMolDiode is available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). It may be used for
noncommercial research, education, and other purposes permitted by the license.
Commercial use requires separate permission from the licensors.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
