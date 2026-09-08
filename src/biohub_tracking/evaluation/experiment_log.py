"""Append-only experiment logging for local validation runs."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExperimentLogEntry:
    """One local-evaluation experiment row."""

    candidate: str
    score: float
    adjusted_edge_jaccard: float
    division_jaccard: float
    split: str
    split_provenance: str
    scorer_provenance: str
    commit_sha: str
    config_path: str = ""
    config_sha256: str = ""
    public_lb: float | None = None
    private_lb: float | None = None
    notes: str = ""
    created_at: str = ""


def current_commit_sha(repo_root: str | Path = ".") -> str:
    """Return the current git commit SHA, or `unknown` outside git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_sha256(path: str | Path | None) -> str:
    """Hash a config/artifact file for reproducibility metadata."""
    if path is None:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def append_experiment_log(entry: ExperimentLogEntry, path: str | Path) -> pd.DataFrame:
    """Append `entry` to `path`, preserving existing rows."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(entry)
    if not payload["created_at"]:
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
    new_row = pd.DataFrame([payload])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        frame = pd.concat([existing, new_row], ignore_index=True)
    else:
        frame = new_row
    frame.to_csv(log_path, index=False)
    return frame
