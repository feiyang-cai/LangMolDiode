"""Opt-in verl monitoring hooks for LangMolDiode training.

The pinned upstream checkout remains unchanged. Hooks are inert unless the
corresponding LANGMOLDIODE_* or legacy MOLLANG_* flag is set to 1.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import os
import sys
from functools import wraps


MONITOR_ALIASES = {
    "actor/loss": "monitor/actor_loss",
    "actor/pg_loss": "monitor/policy_loss",
    "actor/ppo_kl": "monitor/policy_kl",
    "actor/entropy": "monitor/entropy",
    "actor/grad_norm": "monitor/grad_norm",
    "actor/lr": "monitor/lr",
    "critic/rewards/mean": "monitor/reward_mean",
    "response_length/mean": "monitor/response_len_mean",
    "response_length/clip_ratio": "monitor/response_clip_ratio",
    "timing_s/gen": "monitor/rollout_s",
    "timing_s/old_log_prob": "monitor/old_log_prob_s",
    "timing_s/update_actor": "monitor/update_actor_s",
    "timing_s/step": "monitor/step_s",
    "mollang_perf/rollout/response_tokens_per_s": "monitor/rollout_response_tokens_per_s",
    "mollang_perf/rollout/kept_prompt_group_frac": "monitor/dapo_keep_frac",
    "mollang_memory/actor_update_after_cuda_used_gib": "monitor/actor_update_cuda_used_gib",
}


def _scalar(value):
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.detach().cpu().item()
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _patch_tracking(module):
    tracking = getattr(module, "Tracking", None)
    if tracking is None or getattr(tracking, "_langmoldiode_history_patch", False):
        return

    def log(self, data, step, backend=None):
        for name, logger in self.logger.items():
            if backend is not None and name not in backend:
                continue
            if name != "wandb":
                logger.log(data=data, step=step)
                continue

            # Patched for LangMolDiode: write scalars to the live W&B run.
            metrics = {key: val for key, raw in data.items() if (val := _scalar(raw)) is not None}
            for source, alias in MONITOR_ALIASES.items():
                if source in metrics:
                    metrics[alias] = metrics[source]
            if "training/global_step" in metrics:
                metrics["monitor/global_step"] = metrics["training/global_step"]
            run = getattr(logger, "run", None)
            if run is not None:
                run.log(metrics, step=step, commit=True)
            else:
                logger.log(metrics, step=step, commit=True)

    tracking.log = log
    tracking._langmoldiode_history_patch = True


def _snapshot(stage):
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        gib = 1024**3
        return {
            "stage": stage,
            "cuda_used_gib": (total - free) / gib,
            "free_gib": free / gib,
            "allocated_gib": torch.cuda.memory_allocated(device) / gib,
            "reserved_gib": torch.cuda.memory_reserved(device) / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        }
    except Exception as exc:
        print(f"[langmoldiode-memory] {stage} telemetry failed: {exc!r}", flush=True)
        return None


def _patch_actor_memory(module):
    worker = getattr(module, "ActorRolloutRefWorker", None)
    if worker is None or getattr(module, "_langmoldiode_memory_patch", False):
        return

    def preserve_rpc(original, wrapped):
        try:
            from verl.single_controller.base.decorator import MAGIC_ATTR

            if hasattr(original, MAGIC_ATTR):
                setattr(wrapped, MAGIC_ATTR, getattr(original, MAGIC_ATTR))
        except Exception:
            pass
        return wrapped

    def measure(method_name):
        original = getattr(worker, method_name, None)
        if original is None:
            return

        @wraps(original)
        def wrapped(self, *args, **kwargs):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            except Exception:
                pass
            before = _snapshot(f"{method_name}_before")
            if before:
                print(f"[langmoldiode-memory] {before}", flush=True)
            try:
                result = original(self, *args, **kwargs)
            except Exception:
                print(f"[langmoldiode-memory] {_snapshot(f'{method_name}_exception')}", flush=True)
                raise
            after = _snapshot(f"{method_name}_after")
            if after:
                print(f"[langmoldiode-memory] {after}", flush=True)
            if method_name == "update_actor" and result is not None and before and after:
                try:
                    from verl.utils import tensordict_utils as tu

                    metrics = tu.get(result, "metrics")
                    if metrics is not None:
                        for label, values in (("before", before), ("after", after)):
                            for key, value in values.items():
                                if key != "stage":
                                    metrics[f"mollang_memory/actor_update_{label}_{key}"] = value
                except Exception as exc:
                    print(f"[langmoldiode-memory] metric attachment failed: {exc!r}", flush=True)
            return result

        setattr(worker, method_name, preserve_rpc(original, wrapped))

    for name in ("update_actor", "compute_log_prob", "compute_ref_log_prob"):
        measure(name)
    module._langmoldiode_memory_patch = True


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, loader, patch):
        self.loader = loader
        self.patch = patch

    def create_module(self, spec):
        if hasattr(self.loader, "create_module"):
            return self.loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.patch(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch):
        self.target = target
        self.patch = patch

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchLoader(spec.loader, self.patch)
        return spec


def _enabled(primary, legacy):
    return os.environ.get(primary, os.environ.get(legacy)) == "1"


def _install(target, patch):
    if target in sys.modules:
        patch(sys.modules[target])
    else:
        sys.meta_path.insert(0, _PatchFinder(target, patch))


if _enabled("LANGMOLDIODE_WANDB_HISTORY", "MOLLANG_VERL_PATCH_WANDB_HISTORY"):
    _install("verl.utils.tracking", _patch_tracking)

if _enabled("LANGMOLDIODE_MEMORY_METRICS", "MOLLANG_VERL_PATCH_MEMORY_INSTRUMENTATION"):
    _install("verl.workers.engine_workers", _patch_actor_memory)
