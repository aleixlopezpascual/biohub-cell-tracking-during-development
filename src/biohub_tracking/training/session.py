"""State machine for bounded, resumable competitive training sessions."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class StopDecision:
    """Whether a training session should stop and why."""

    stop: bool
    reason: str


@dataclass
class TrainingSession:
    """Track checkpoint cadence, early stopping, and a wall-clock deadline.

    Model-specific loops call :meth:`record_evaluation` after an official
    graph evaluation and :meth:`stop_decision` before beginning another epoch.
    The state is intentionally small enough to include in checkpoint RNG/meta
    payloads when a job resumes in a later Kaggle session.
    """

    max_runtime_hours: float
    early_stopping_patience: int
    checkpoint_every_epochs: int
    start_epoch: int = 0
    best_official_score: float = float("-inf")
    best_epoch: int = -1
    last_evaluation_epoch: int = -1
    last_official_score: float | None = None
    evaluations_without_improvement: int = 0
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if self.max_runtime_hours <= 0:
            raise ValueError("max_runtime_hours must be positive")
        if self.early_stopping_patience < 1 or self.checkpoint_every_epochs < 1:
            raise ValueError("patience and checkpoint cadence must be positive")
        if self.start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")
        if self.started_at == 0.0:
            self.started_at = monotonic()

    def record_evaluation(self, epoch: int, official_score: float) -> bool:
        """Record an official score and return whether it is a new best."""
        if epoch < self.start_epoch or not 0 <= official_score <= 1.1:
            raise ValueError("invalid epoch or official score")
        if epoch < self.last_evaluation_epoch:
            raise ValueError("official evaluations must be recorded in epoch order")
        if epoch == self.last_evaluation_epoch:
            if official_score != self.last_official_score:
                raise ValueError("official score changed for an already recorded epoch")
            return False
        self.last_evaluation_epoch = int(epoch)
        self.last_official_score = float(official_score)
        if official_score > self.best_official_score:
            self.best_official_score = float(official_score)
            self.best_epoch = int(epoch)
            self.evaluations_without_improvement = 0
            return True
        self.evaluations_without_improvement += 1
        return False

    def checkpoint_due(self, epoch: int) -> bool:
        """Return whether the configured periodic checkpoint is due."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        return (epoch + 1) % self.checkpoint_every_epochs == 0

    def stop_decision(self, *, now: float | None = None) -> StopDecision:
        """Stop before quota loss or after repeated official-score stagnation."""
        current = monotonic() if now is None else float(now)
        elapsed_hours = (current - self.started_at) / 3600
        if elapsed_hours >= self.max_runtime_hours:
            return StopDecision(True, "runtime budget exhausted")
        if self.evaluations_without_improvement >= self.early_stopping_patience:
            return StopDecision(True, "official-score early stopping")
        return StopDecision(False, "continue")

    def state_dict(self) -> dict[str, int | float | None]:
        """Return serializable state for a full training checkpoint."""
        return {
            "start_epoch": self.start_epoch,
            "best_official_score": self.best_official_score,
            "best_epoch": self.best_epoch,
            "last_evaluation_epoch": self.last_evaluation_epoch,
            "last_official_score": self.last_official_score,
            "evaluations_without_improvement": self.evaluations_without_improvement,
        }

    def load_state_dict(self, state: dict[str, int | float | None]) -> None:
        """Restore progress counters while starting a fresh wall-clock budget."""
        required = {
            "start_epoch",
            "best_official_score",
            "best_epoch",
            "last_evaluation_epoch",
            "last_official_score",
            "evaluations_without_improvement",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"training session state is missing: {sorted(missing)}")
        self.start_epoch = int(state["start_epoch"])
        self.best_official_score = float(state["best_official_score"])
        self.best_epoch = int(state["best_epoch"])
        self.last_evaluation_epoch = int(state["last_evaluation_epoch"])
        last_score = state["last_official_score"]
        self.last_official_score = None if last_score is None else float(last_score)
        self.evaluations_without_improvement = int(state["evaluations_without_improvement"])
        self.started_at = monotonic()
