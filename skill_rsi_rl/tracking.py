"""Optional SwanLab metric sink for the primary training rank."""

import logging
import os
import sys
from contextlib import contextmanager
from importlib import import_module
from inspect import signature
from pathlib import Path


TRACKING_MODES = ("off", "local", "cloud")
_LOGGER = logging.getLogger(__name__)


def training_metrics(row):
    metrics = {f"train/{key}": value for key, value in row.items() if key not in {"step", "rank_memory"}}
    metrics.update({f"gpu/rank_{memory['rank']}/{kind}_gib": memory[f"{kind}_bytes"] / 2**30
                    for memory in row.get("rank_memory", ()) for kind in ("allocated", "reserved")})
    return metrics


class _Tracking:
    def __init__(self, run=None):
        self._run = run

    def define(self, key, **options):
        define_metric = getattr(self._run, "define_metric", None)
        if callable(define_metric):
            try:
                define_metric(key, **options)
            except Exception as exc:
                _LOGGER.warning("SwanLab chart definition failed for %s (%s).", key, type(exc).__name__)

    def log(self, metrics, *, step):
        if self._run is None:
            return False
        try:
            self._run.log(metrics, step=step)
        except Exception as exc:
            self._run = None
            _LOGGER.warning("SwanLab metric logging stopped (%s). The training JSONL remains available.",
                            type(exc).__name__)
            return False
        return True


@contextmanager
def open_tracking(mode, output, config, project="skill_rsi", workspace="hzh", name=None):
    """Start tracking when enabled and keep tracking failures isolated from training."""
    if mode not in TRACKING_MODES:
        raise ValueError(f"Unknown tracking mode: {mode}")
    if mode == "off":
        yield _Tracking()
        return

    try:
        swanlab = import_module("swanlab")
    except ImportError:
        raise RuntimeError("SwanLab is unavailable. Install swanlab==0.7.20 in the training environment.") from None

    output = Path(output)
    # In SwanLab 0.7.20, SDK local mode requires the optional swanboard UI.
    modern_sdk = "on_online_error" in signature(swanlab.init).parameters
    run_name = name or output.name
    log_dir = str(output / "swanlog")
    options = ({"name": run_name, "log_dir": log_dir, "on_online_error": "raise"}
               if modern_sdk else {"experiment_name": run_name, "logdir": log_dir})
    sdk_mode = "offline" if mode == "local" else ("online" if modern_sdk else "cloud")
    try:
        # SwanLab 0.7.20 prompts on a TTY if a cloud run has no saved or environment credentials.
        with open(os.devnull, encoding="utf-8") as no_input:
            original_input, sys.stdin = sys.stdin, no_input
            try:
                settings = (swanlab.Settings(interactive=False) if modern_sdk
                            else swanlab.Settings(log_proxy_type="none"))
                run = swanlab.init(project=project, workspace=workspace, mode=sdk_mode,
                                   config=config, settings=settings, **options)
            finally:
                sys.stdin = original_input
    except Exception as exc:
        remedy = ("Run swanlab login or set SWANLAB_API_KEY and check workspace access."
                  if mode == "cloud" else "Check the SDK version and that the output directory is writable.")
        raise RuntimeError(f"SwanLab {mode} initialization failed for {workspace}/{project} "
                           f"({type(exc).__name__}). {remedy}") from None

    failed = None
    try:
        yield _Tracking(run)
    except BaseException as exc:
        failed = type(exc).__name__
        raise
    finally:
        try:
            if failed:
                state = "crashed" if modern_sdk else swanlab.State.CRASHED
                run.finish(state=state, error=failed)
            else:
                run.finish()
        except Exception as exc:
            _LOGGER.warning("SwanLab finalization failed (%s). The training JSONL remains available.",
                            type(exc).__name__)
