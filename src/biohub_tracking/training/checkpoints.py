"""Reproducible run manifests and resumable Torch checkpoint payloads."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def _sha256(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"artifact not found: {file_path}")
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _commit_sha(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class RunManifest:
    """Immutable provenance attached to every cloud training run."""

    candidate: str
    created_at: str
    commit_sha: str
    config_path: str
    config_sha256: str
    split_path: str
    split_sha256: str
    fold: str
    seed: int
    max_runtime_hours: float

    def write_json(self, path: str | Path) -> None:
        """Write a deterministic JSON representation of the manifest."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def build_run_manifest(
    *,
    candidate: str,
    config_path: str | Path,
    split_path: str | Path,
    fold: str,
    seed: int,
    max_runtime_hours: float,
    repo_root: str | Path = ".",
) -> RunManifest:
    """Build a manifest with hashes for the exact config and split inputs."""
    if not candidate.strip() or Path(candidate).name != candidate or candidate in {".", ".."}:
        raise ValueError("candidate must be a non-empty directory name, not a path")
    if seed < 0 or max_runtime_hours <= 0:
        raise ValueError("seed must be non-negative and max_runtime_hours must be positive")
    config = Path(config_path).resolve()
    split = Path(split_path).resolve()
    return RunManifest(
        candidate=candidate,
        created_at=datetime.now(timezone.utc).isoformat(),
        commit_sha=_commit_sha(repo_root),
        config_path=str(config),
        config_sha256=_sha256(config),
        split_path=str(split),
        split_sha256=_sha256(split),
        fold=str(fold),
        seed=int(seed),
        max_runtime_hours=float(max_runtime_hours),
    )


def validate_run_manifest(
    manifest: RunManifest,
    *,
    config_path: str | Path,
    split_path: str | Path,
    candidate: str | None = None,
) -> None:
    """Reject continuation when immutable campaign inputs have changed."""
    problems = []
    if manifest.config_sha256 != _sha256(config_path):
        problems.append("config hash")
    if manifest.split_sha256 != _sha256(split_path):
        problems.append("split hash")
    if candidate is not None and manifest.candidate != candidate:
        problems.append("candidate")
    if problems:
        raise ValueError(
            "run manifest does not match continuation inputs: " + ", ".join(problems)
        )


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Validated state restored from a full training checkpoint."""

    epoch: int
    best_official_score: float
    model_state: Mapping[str, object]
    optimizer_state: Mapping[str, object]
    scheduler_state: Mapping[str, object] | None
    scaler_state: Mapping[str, object] | None
    rng_state: Mapping[str, object]
    manifest: Mapping[str, object]


def _require_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Checkpoint serialization requires the optional torch extra: "
            "install with `pip install -e '.[torch]'`."
        ) from exc
    return torch


def save_training_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    best_official_score: float,
    model_state: Mapping[str, object],
    optimizer_state: Mapping[str, object],
    scheduler_state: Mapping[str, object] | None,
    scaler_state: Mapping[str, object] | None,
    rng_state: Mapping[str, object],
    manifest: RunManifest | Mapping[str, object],
) -> None:
    """Atomically save all state needed to resume an interrupted run."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    torch = _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "best_official_score": float(best_official_score),
        "model_state": dict(model_state),
        "optimizer_state": dict(optimizer_state),
        "scheduler_state": None if scheduler_state is None else dict(scheduler_state),
        "scaler_state": None if scaler_state is None else dict(scaler_state),
        "rng_state": dict(rng_state),
        "manifest": asdict(manifest) if isinstance(manifest, RunManifest) else dict(manifest),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_inference_checkpoint(
    path: str | Path,
    model_state: Mapping[str, object],
) -> None:
    """Atomically save a predictor-compatible state dict.

    Training can use ``DataParallel`` on dual T4s, while the official predictor
    expects the underlying UNet keys. Only that known prefix is normalized.
    """
    torch = _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        key.replace("unet.module.", "unet.", 1): value
        for key, value in model_state.items()
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(normalized, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(path: str | Path, *, map_location: str = "cpu") -> TrainingCheckpoint:
    """Load and validate a full-state checkpoint created by this package."""
    torch = _require_torch()
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"training checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    required = {
        "format_version",
        "epoch",
        "best_official_score",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "manifest",
    }
    if not isinstance(payload, dict) or required - set(payload):
        missing = sorted(required - set(payload)) if isinstance(payload, dict) else sorted(required)
        raise ValueError(f"invalid training checkpoint; missing fields: {missing}")
    if payload["format_version"] != 1:
        raise ValueError(f"unsupported checkpoint format_version {payload['format_version']!r}")
    return TrainingCheckpoint(
        epoch=int(payload["epoch"]),
        best_official_score=float(payload["best_official_score"]),
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        scaler_state=payload["scaler_state"],
        rng_state=payload["rng_state"],
        manifest=payload["manifest"],
    )
