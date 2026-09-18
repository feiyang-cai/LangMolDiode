---
library_name: transformers
base_model: Qwen/Qwen3.5-4B
tags:
  - chemistry
  - text-to-smiles
  - langmoldiode
---

# LangMolDiode Qwen3.5-4B SFT base

This private model is the merged full-precision SFT checkpoint used as the
base for LangMolDiode DAPO training. The SFT seed is the Unsloth LoRA
`checkpoint-14190` from a three-epoch, 40,960-token Qwen3.5-4B run.
The model was prepared without the Qwen3.5 MTP head for the tested verl/vLLM
training path. It accepts a molecular structure description and generates a
SMILES answer inside `<smiles>...</smiles>` tags, usually with a reasoning trace.

Load with `AutoModelForCausalLM.from_pretrained` or the compatible Qwen3.5
model class. Text-only evaluation was used. The model is private while its
artifacts and training details are being reviewed.

The base checkpoint is `Qwen/Qwen3.5-4B`; inspect that model's license and
terms before any future public release.
