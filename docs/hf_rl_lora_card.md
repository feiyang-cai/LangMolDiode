---
library_name: peft
base_model: ChemFM/LangMolDiode-Qwen3.5-4B-SFT
tags:
  - chemistry
  - text-to-smiles
  - langmoldiode
  - lora
  - dapo
---

# LangMolDiode RL LoRA

This private repository holds adapters from a Qwen3.5-4B DAPO run with
Megatron-Bridge actor training and vLLM rollout. The root adapter is the final
rollout step 1500. Additional adapters live under:

| Subfolder | Selection set | Rollout step | exact@1 |
| --- | --- | ---: | ---: |
| `best_validated_step1260/` | 1,972 validated prompts | 1260 | 0.8519 |
| `best_test_step1315/` | 200 MolLangBench test prompts | 1315 | 0.7150 |
| `best_extended_step1370/` | 200 MolLangBench extended prompts | 1370 | 0.7750 |

Each validation prompt was sampled three times. `exact@1` is the first sample's
canonical-SMILES exact match. Best means highest observed `exact@1` among the
full saved validation runs, with the earliest step winning a tie. Selection on
the test and extended sets means their reported values should be treated as
selection results, not untouched holdout estimates.

The adapters require the private merged SFT base at
`ChemFM/LangMolDiode-Qwen3.5-4B-SFT`. LoRA rank is 64 and alpha is 128.
The saved weights are unchanged; adapter configurations replace only the
node-local `base_model_name_or_path` with that repository ID.
