from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rl_defaults_match_final_training_run() -> None:
    launcher = (ROOT / "scripts" / "run_rl.sh").read_text(encoding="utf-8")
    expected_defaults = (
        '\\"tanimoto_weight\\":1.0',
        '${OVERLONG_EXPECTED_TOKENS:-2048}',
        '${OVERLONG_PENALTY_FACTOR:-0.4}',
        '${ACTOR_PARAM_OFFLOAD:-True}',
        '${VLLM_GPU_MEMORY_UTILIZATION:-0.875}',
        '${VAL_GENERATIONS:-3}',
        '${NNODES:-2}',
        '${GPUS_PER_NODE:-8}',
        '${TEST_FREQ:-5}',
        '${SAVE_FREQ:-1}',
        '${TOTAL_EPOCHS:-999}',
        '${TOTAL_TRAINING_STEPS:-1500}',
    )
    for value in expected_defaults:
        assert value in launcher


def test_sft_defaults_match_final_training_run() -> None:
    launcher = (ROOT / "scripts" / "run_sft.sh").read_text(encoding="utf-8")
    expected_defaults = (
        '${MODEL_NAME:-Qwen/Qwen3.5-4B}',
        '${MAX_SEQ_LENGTH:-40960}',
        '${NUM_TRAIN_EPOCHS:-3}',
        '${PER_DEVICE_TRAIN_BATCH_SIZE:-1}',
        '${GRADIENT_ACCUMULATION_STEPS:-2}',
        '${LEARNING_RATE:-2e-4}',
        '${WARMUP_STEPS:-20}',
        '${NPROC_PER_NODE:-8}',
    )
    for value in expected_defaults:
        assert value in launcher
