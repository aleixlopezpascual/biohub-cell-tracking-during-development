"""7 um, time-aware, sparse-tolerant node matching (metrics.md, "Edge Jaccard").

Predicted nodes are paired with ground-truth nodes by centroid distance, up
to a maximum distance (7 um by default), using an optimal bipartite
assignment. Because a track's identity is only meaningful within a single
timepoint, matching is always restricted to nodes sharing the same frame
("time-aware"): the assignment problem is solved independently per frame and
the results are merged.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from biohub_tracking.tracking.graph import Detection, NodeId

_UNREACHABLE_COST = 1e12


def _match_single_frame(
    pred_nodes: list[Detection], gt_nodes: list[Detection], max_distance: float
) -> dict[NodeId, NodeId]:
    if not pred_nodes or not gt_nodes:
        return {}
    pred_pos = np.array([n.position for n in pred_nodes], dtype=float)
    gt_pos = np.array([n.position for n in gt_nodes], dtype=float)
    dist = cdist(pred_pos, gt_pos)
    cost = np.where(dist <= max_distance, dist, _UNREACHABLE_COST)
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: dict[NodeId, NodeId] = {}
    for r, c in zip(row_ind, col_ind):
        if dist[r, c] <= max_distance:
            matches[pred_nodes[r].id] = gt_nodes[c].id
    return matches


def match_nodes(
    pred_nodes: Iterable[Detection],
    gt_nodes: Iterable[Detection],
    max_distance: float = 7.0,
) -> dict[NodeId, NodeId]:
    """Match predicted nodes to ground-truth nodes.

    Parameters
    ----------
    pred_nodes, gt_nodes:
        Detections (with physical-unit centroids) to match.
    max_distance:
        Maximum centroid distance (microns) for a valid match; the official
        metric uses 7 um.

    Returns
    -------
    dict
        Mapping from predicted node id to its matched ground-truth node id.
        Each predicted node maps to at most one ground-truth node and vice
        versa (an injective mapping), since matching solves an optimal
        bipartite assignment independently per frame.
    """
    pred_by_frame: dict[int, list[Detection]] = {}
    for node in pred_nodes:
        pred_by_frame.setdefault(node.frame, []).append(node)
    gt_by_frame: dict[int, list[Detection]] = {}
    for node in gt_nodes:
        gt_by_frame.setdefault(node.frame, []).append(node)

    matches: dict[NodeId, NodeId] = {}
    for frame, preds in pred_by_frame.items():
        gts = gt_by_frame.get(frame, [])
        matches.update(_match_single_frame(preds, gts, max_distance))
    return matches
