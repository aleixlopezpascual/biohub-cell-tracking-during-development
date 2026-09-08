"""Local-window division Jaccard (metrics.md, "Division Jaccard").

This is the most structurally involved of the official metrics. Ground-truth
divisions are evaluated through a local window
(``grandparent -> parent -> children -> grandchildren``) rather than
graph-wide reachability, so a predicted fork one timepoint early or late can
still recover a division. The implementation below follows metrics.md
section by section:

* local parent anchoring (parent or its immediate predecessor),
* two distinct daughter branches recovered via bipartite matching,
* directed local topology (daughters strictly downstream of the fork),
* connected-component evidence consistency for each branch, and
* rejection of merged/shared branches,

before pairing candidate forks to GT divisions with a maximum-cardinality
bipartite matching (one fork recovers at most one division and vice versa).

Note on scope: node matching is shared with the edge-Jaccard metric (a
single time-aware, 7 um optimal assignment across the whole graph) rather
than re-solved independently inside every local window; this is a
documented, pragmatic simplification of the very small window-local
re-matching described in metrics.md, kept because it is deterministic,
avoids double-counting, and coincides with the windowed assignment whenever
a window's frames don't overlap another division's window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from biohub_tracking.metrics.matching import match_nodes
from biohub_tracking.tracking.graph import NodeId, TrackingGraph


@dataclass(frozen=True)
class DivisionJaccardResult:
    """TP/FP/FN division counts and the resulting Jaccard index."""

    tp: int
    fp: int
    fn: int

    @property
    def jaccard(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 1.0


@dataclass
class _BranchEvidence:
    """What a predicted division branch's nearest GT match tells us."""

    evidence_gt_id: NodeId | None = None
    component: int | None = None
    merged: bool = False


def _branch_evidence(
    branch_child: NodeId,
    pred_graph: TrackingGraph,
    match: dict[NodeId, NodeId],
    gt_components: dict[NodeId, int],
) -> _BranchEvidence:
    """Determine the GT evidence a predicted daughter branch supplies.

    A matched direct child supplies its own component. If unmatched, an
    unambiguous matched grandchild may supply it instead; if grandchildren
    disagree on component, the branch supplies no evidence.
    """
    merged = len(pred_graph.predecessors(branch_child)) > 1
    direct_match = match.get(branch_child)
    if direct_match is not None:
        return _BranchEvidence(evidence_gt_id=direct_match, component=gt_components.get(direct_match), merged=merged)

    grandchildren = pred_graph.successors(branch_child)
    matched_gcs = {gc: match[gc] for gc in grandchildren if gc in match}
    if not matched_gcs:
        return _BranchEvidence(merged=merged)

    components = {gt_components.get(gid) for gid in matched_gcs.values()}
    if len(components) != 1:
        # Ambiguous fallback evidence: branch supplies no component evidence.
        return _BranchEvidence(merged=merged)

    any_gc, any_gid = next(iter(matched_gcs.items()))
    branch_merged = merged or len(pred_graph.predecessors(any_gc)) > 1
    return _BranchEvidence(evidence_gt_id=any_gid, component=components.pop(), merged=branch_merged)


def _max_bipartite_matching_size(left_to_right: dict[int, set[int]], num_left: int, num_right: int) -> int:
    """Maximum-cardinality matching size for a small bipartite graph."""
    if num_left == 0 or num_right == 0:
        return 0
    data, rows, cols = [], [], []
    for left, rights in left_to_right.items():
        for right in rights:
            rows.append(left)
            cols.append(right)
            data.append(1)
    if not data:
        return 0
    graph = csr_matrix((data, (rows, cols)), shape=(num_left, num_right))
    matching = maximum_bipartite_matching(graph, perm_type="column")
    return int(np.sum(matching != -1))


def compute_division_jaccard(
    pred_graph: TrackingGraph,
    gt_graph: TrackingGraph,
    max_distance: float = 7.0,
    match: dict[NodeId, NodeId] | None = None,
) -> DivisionJaccardResult:
    """Compute the local-window division-Jaccard TP/FP/FN counts.

    Parameters
    ----------
    pred_graph, gt_graph:
        Predicted and ground-truth tracking graphs.
    max_distance:
        Node-matching radius in microns (7 um officially).
    match:
        Optional precomputed ``pred_id -> gt_id`` mapping, shared with the
        edge metric to avoid recomputing the assignment problem.
    """
    if match is None:
        match = match_nodes(pred_graph.nodes.values(), gt_graph.nodes.values(), max_distance)
    gt_to_pred: dict[NodeId, NodeId] = {gid: pid for pid, gid in match.items()}

    gt_components = gt_graph.weakly_connected_components()
    gt_out_degree = {n: gt_graph.out_degree(n) for n in gt_graph.nodes}
    gt_divisions = [n for n, deg in gt_out_degree.items() if deg >= 2]
    pred_forks = {n for n in pred_graph.nodes if pred_graph.out_degree(n) >= 2}

    # Precompute every predicted fork's per-branch evidence once.
    fork_branch_evidence: dict[NodeId, dict[NodeId, _BranchEvidence]] = {
        fork: {
            branch: _branch_evidence(branch, pred_graph, match, gt_components)
            for branch in pred_graph.successors(fork)
        }
        for fork in pred_forks
    }

    def cross_component_fp(fork: NodeId) -> bool:
        components = {
            ev.component for ev in fork_branch_evidence[fork].values() if ev.component is not None
        }
        return len(components) > 1

    def any_branch_merged(fork: NodeId) -> bool:
        return any(ev.merged for ev in fork_branch_evidence[fork].values())

    # For each GT division, find predicted forks that could plausibly recover it.
    division_candidates: dict[NodeId, set[NodeId]] = {}
    all_candidate_forks: set[NodeId] = set()

    for parent in gt_divisions:
        children = gt_graph.successors(parent)
        daughters = [(child, gt_graph.successors(child)) for child in children]
        division_component = gt_components.get(parent)

        grandparents = gt_graph.predecessors(parent)
        anchor_gt = [parent, *grandparents]
        anchor_pred = {gt_to_pred[g] for g in anchor_gt if g in gt_to_pred}

        candidates = set(anchor_pred)
        for a in anchor_pred:
            candidates.update(pred_graph.successors(a))
        candidates &= pred_forks

        valid_forks: set[NodeId] = set()
        for fork in candidates:
            # Directed local topology: fork must be an anchor node or an
            # immediate successor of one (guaranteed by construction above).
            branch_evidence = fork_branch_evidence[fork]
            branches = list(pred_graph.successors(fork))

            # daughter index -> set of compatible branch indices
            compat: dict[int, set[int]] = {}
            for di, (child, grandchildren) in enumerate(daughters):
                window_ids = {child, *grandchildren}
                compat[di] = {
                    bi
                    for bi, branch in enumerate(branches)
                    if not branch_evidence[branch].merged
                    and branch_evidence[branch].component == division_component
                    and branch_evidence[branch].evidence_gt_id in window_ids
                }

            matched_size = _max_bipartite_matching_size(compat, len(daughters), len(branches))
            if matched_size == len(daughters) and len(daughters) >= 2:
                valid_forks.add(fork)

        if valid_forks:
            division_candidates[parent] = valid_forks
            all_candidate_forks |= valid_forks

    # Pair GT divisions to predicted forks with maximum-cardinality bipartite matching.
    divisions = list(division_candidates)
    division_index = {p: i for i, p in enumerate(divisions)}
    forks = list(all_candidate_forks)
    fork_index = {f: i for i, f in enumerate(forks)}
    left_to_right = {
        division_index[p]: {fork_index[f] for f in fork_set} for p, fork_set in division_candidates.items()
    }
    tp = _max_bipartite_matching_size(left_to_right, len(divisions), len(forks))
    fn = len(gt_divisions) - tp

    # Determine which forks actually end up in the optimal pairing to know "leftovers".
    matched_forks: set[NodeId] = set()
    if divisions and forks:
        data, rows, cols = [], [], []
        for li, ris in left_to_right.items():
            for ri in ris:
                rows.append(li)
                cols.append(ri)
                data.append(1)
        if data:
            graph = csr_matrix((data, (rows, cols)), shape=(len(divisions), len(forks)))
            perm = maximum_bipartite_matching(graph, perm_type="column")
            matched_forks = {forks[j] for j in perm if j != -1}

    fp = 0
    for fork in pred_forks:
        if fork in matched_forks:
            continue
        is_fp = False
        matched_gt = match.get(fork)
        if matched_gt is not None and gt_out_degree.get(matched_gt, 0) >= 1:
            is_fp = True
        if not is_fp and fork in all_candidate_forks:
            is_fp = True  # local candidate left over after bipartite pairing
        if not is_fp and cross_component_fp(fork):
            is_fp = True
        if not is_fp and any_branch_merged(fork):
            is_fp = True
        if is_fp:
            fp += 1

    return DivisionJaccardResult(tp=tp, fp=fp, fn=fn)
