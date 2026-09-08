"""Deterministic seeding utilities.

Kept dependency-light: seeds :mod:`random` and :mod:`numpy` unconditionally,
and optionally seeds :mod:`torch` (and enables its deterministic algorithms)
only if torch is installed, so the core package never requires it.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed all known sources of randomness for reproducible runs.

    Parameters
    ----------
    seed:
        Non-negative integer seed applied to ``random``, ``numpy`` and,
        if available, ``torch``.
    deterministic_torch:
        When torch is installed, also request deterministic (non-benchmark)
        cuDNN algorithms. Ignored if torch is not installed.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
