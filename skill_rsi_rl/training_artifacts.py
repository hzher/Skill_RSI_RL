"""Own the filesystem boundary for a training run and its durable checkpoints."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


@dataclass(frozen=True)
class TrainingProgress:
    next_epoch: int = 0
    step: int = 0
    optimizer: dict | None = None
    loss_ema: float | None = None
    training_seconds: float = 0.0


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def lock_output(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".train.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"training output is already locked: {output}") from error
        yield


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
    finally:
        os.close(descriptor)


def _write_run_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".run-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _portable_settings(settings):
    """Old checkpoints included machine paths alongside the actual content identities."""
    if not isinstance(settings, dict):
        return settings
    result = settings.copy()
    if result.get("base_checkpoint_id"):
        result.pop("model", None)
    backend = result.get("backend")
    if isinstance(backend, dict) and backend.get("commit") and backend.get("sources_sha256"):
        result["backend"] = {k: v for k, v in backend.items() if k != "checkout"}
    return result


def initialize_run(output, run_data, progress, epochs, *, before_commit=None):
    for path in output.glob("checkpoint-*"):
        number = path.name.removeprefix("checkpoint-")
        if number.isascii() and number.isdecimal():
            epoch = int(number)
            if (path.name == f"checkpoint-{epoch:04d}" and epoch > progress.next_epoch
                    and (epochs is None or epoch <= epochs)):
                raise ValueError("output contains a checkpoint for an epoch scheduled to train")
    run_json = output / "run.json"
    metrics = output / "metrics.jsonl"
    existing = run_json.exists()
    if existing:
        saved = json.loads(run_json.read_text(encoding="utf-8"))
        if not progress.next_epoch or _portable_settings(saved.get("settings")) != _portable_settings(run_data["settings"]):
            raise ValueError(f"{run_json} belongs to a different run")
        metrics = output / f"metrics-resume-{progress.step:08d}-{time.time_ns()}.jsonl"
    if before_commit is not None:
        before_commit()
    if not existing:
        _write_run_json(run_json, run_data)
    print(json.dumps({"metrics_path": str(metrics), "resumed_step": progress.step}), flush=True)
    return metrics


def append_record(path, value):
    line = json.dumps(value, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_training_state(path, settings, epochs):
    if path is None:
        return TrainingProgress()
    saved = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or _portable_settings(saved.get("settings")) != _portable_settings(settings):
        raise ValueError("resume checkpoint settings or replay identity differ")
    next_epoch, step, optimizer = (saved.get(field) for field in ("next_epoch", "step", "optimizer"))
    if (type(next_epoch) is not int or type(step) is not int or not 1 <= next_epoch <= step
            or not isinstance(optimizer, dict)):
        raise ValueError("invalid epoch, step, or optimizer in resume checkpoint")
    if epochs is not None and next_epoch >= epochs:
        raise ValueError("--epochs must exceed the completed epochs in the checkpoint")
    expected_hashes = saved.get("adapter_sha256")
    if expected_hashes is not None and expected_hashes != _adapter_hashes(path):
        raise ValueError("resume adapter files do not match training_state.pt")
    if expected_hashes is None:
        for name in ADAPTER_FILES:
            if not (path / name).is_file():
                raise FileNotFoundError(f"resume checkpoint is missing {name}: {path}")
    loss_ema = saved.get("loss_ema")
    if loss_ema is not None and (type(loss_ema) not in (int, float) or not math.isfinite(loss_ema)):
        raise ValueError("invalid loss moving average in resume checkpoint")
    training_seconds = saved.get("training_seconds", 0.0)
    if (type(training_seconds) not in (int, float) or not math.isfinite(training_seconds)
            or training_seconds < 0):
        raise ValueError("invalid elapsed training time in resume checkpoint")
    return TrainingProgress(next_epoch, step, optimizer, loss_ema, training_seconds)


def _adapter_hashes(path):
    return {name: file_sha256(path / name) for name in ADAPTER_FILES}


def save_checkpoint(path, model, optimizer, settings, next_epoch, step, *, loss_ema=None, training_seconds=0.0):
    if os.path.lexists(path):
        raise FileExistsError(f"checkpoint already exists: {path}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    try:
        model.save_pretrained(temporary)
        torch.save({"settings": settings, "optimizer": optimizer.state_dict(),
                    "next_epoch": next_epoch, "step": step, "loss_ema": loss_ema,
                    "training_seconds": training_seconds,
                    "adapter_sha256": _adapter_hashes(temporary)}, temporary / "training_state.pt")
        for entry in temporary.iterdir():
            if entry.is_file():
                with entry.open("rb") as handle:
                    os.fsync(handle.fileno())
        _sync_directory(temporary)
        temporary.rename(path)
        _sync_directory(path.parent)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
