# Private artifact layout

The source repository excludes model weights, corpora, and full generated
responses. The Hugging Face repositories and Box archive below have been
uploaded and verified private to the authenticated accounts.

| Destination | Contents |
| --- | --- |
| `ChemFM/LangMolDiode-Qwen3.5-4B-SFT` | merged SFT base model |
| `ChemFM/LangMolDiode-Qwen3.5-4B-SFT-LoRA` | SFT adapter, checkpoint 14190 |
| `ChemFM/LangMolDiode-Qwen3.5-4B-RL-LoRA` | final step 1500 adapter at root; steps 1260, 1315, 1370 in named subfolders |
| [Box folder `419341962405`](https://clemson.app.box.com/folder/419341962405) | filtered SFT corpus, SFT and RL adapters, and per-dataset generated responses |

For Box, the staged response files are under ignored
`artifacts/staged/responses/`. Step 0 and step 1500 each have three dataset
files. The best checkpoints contain only the corresponding dataset file.
The local selection metadata is in `artifacts/selection.json`.
After authenticating the project-local Box CLI, run
`python scripts/upload_box_artifacts.py --sft-corpus CORPUS.jsonl --sft-adapter CHECKPOINT-14190 --rl-run RUN_DIR`.
Add `--dry-run` to inspect source and destination paths without contacting Box.
The upload skips files whose Box SHA-1 already matches the local file.

Full Megatron optimizer checkpoints are retained in the original training run
directory. The Hugging Face repositories contain inference artifacts rather
than the terabyte-scale optimizer state from every rollout.
