"""Official Biohub tracking metrics (see metrics.md in the competition repo).

Implements:

- 7 um, time-aware, sparse-tolerant node matching (:mod:`.matching`).
- Sparse-aware edge accounting and the adjusted edge Jaccard (:mod:`.edge_jaccard`).
- Local-window division Jaccard with connected-component evidence (:mod:`.division_jaccard`).
- Micro-averaged aggregation across samples into the final combined score (:mod:`.score`).
"""

from __future__ import annotations

from biohub_tracking.metrics.division_jaccard import DivisionJaccardResult, compute_division_jaccard
from biohub_tracking.metrics.edge_jaccard import EdgeJaccardResult, compute_edge_jaccard
from biohub_tracking.metrics.matching import match_nodes
from biohub_tracking.metrics.score import SampleResult, aggregate_scores, evaluate_sample

__all__ = [
    "match_nodes",
    "EdgeJaccardResult",
    "compute_edge_jaccard",
    "DivisionJaccardResult",
    "compute_division_jaccard",
    "SampleResult",
    "evaluate_sample",
    "aggregate_scores",
]
