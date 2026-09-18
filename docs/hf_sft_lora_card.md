---
library_name: peft
base_model: Qwen/Qwen3.5-4B
datasets:
  - ChemFM/LangMolDiode-SFT-Corpus
tags:
  - chemistry
  - text-to-smiles
  - langmoldiode
  - lora
---

# LangMolDiode SFT LoRA

This private repository holds the Unsloth SFT adapter from
`checkpoint-14190`, trained on the filtered June 18 reasoning corpus.
The verified training examples are in the private
`ChemFM/LangMolDiode-SFT-Corpus` dataset.
Use it with `Qwen/Qwen3.5-4B`. LoRA rank is 64 and alpha is 128.
The adapter weights are the saved training output; only the
`base_model_name_or_path` field was changed from a node-local cache path to
the public base model identifier for portability.

The full merged SFT model used as the RL base is stored separately in
`ChemFM/LangMolDiode-Qwen3.5-4B-SFT`.
