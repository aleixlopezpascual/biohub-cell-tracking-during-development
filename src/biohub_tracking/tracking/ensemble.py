"""Graph-level bipartite consensus ensembling for cell tracking graphs.

Merges predictions from multiple models or post-processing configurations by:
1. Matching detections across models within a spatial radius (default 3.5 µm).
2. Fusing canonical centroid coordinates across agreeing detections.
3. Voting on temporal edges and resolving contested links using spatial proximity
   and graph degree constraints (<= 2 daughters, <= 1 parent).
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Hashable

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_tracking.tracking.graph import Detection, TrackingGraph


def _euclidean_dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def ensemble_tracking_graphs(
    graphs: list[TrackingGraph],
    *,
    matching_radius_um: float = 3.5,
    min_node_votes: int = 1,
    min_edge_votes: int = 1,
    resolve_contested: bool = True,
) -> TrackingGraph:
    """Combine multiple tracking graphs into a consensus TrackingGraph.

    Parameters
    ----------
    graphs:
        List of input TrackingGraphs (must non-empty).
    matching_radius_um:
        Maximum distance in microns to cluster detections from different models
        at the same timepoint.
    min_node_votes:
        Minimum number of models that must detect a cell for it to be kept.
    min_edge_votes:
        Minimum number of model votes required for an edge to be included.
    resolve_contested:
        If True, enforces graph invariants (max 2 outgoing, max 1 incoming) by
        selecting highest-voted and lowest-distance edges.

    Returns
    -------
    TrackingGraph
        The ensembled consensus graph.
    """
    if not graphs:
        raise ValueError("graphs list cannot be empty")
    if len(graphs) == 1:
        return graphs[0]

    # Collect frames present across all graphs
    all_frames = sorted(
        set().union(*(set(d.frame for d in g.nodes.values()) for g in graphs))
    )

    canonical_graph = TrackingGraph()
    # Mapping from (graph_idx, original_node_id) -> canonical_node_id
    node_to_canonical: dict[tuple[int, Hashable], Hashable] = {}

    next_canonical_id = 1

    # Step 1: Match nodes across models per frame
    for t in all_frames:
        # Collect detections in frame t from each graph
        frame_nodes: list[list[Detection]] = [
            [d for d in g.nodes.values() if d.frame == t] for g in graphs
        ]

        # Use graph 0 as reference, then match subsequent graphs
        # If graph 0 has no detections, take the first non-empty graph
        clusters: list[list[tuple[int, Detection]]] = []

        # Initialize clusters with detections from the first graph that has detections
        assigned_in_graph: list[set[Hashable]] = [set() for _ in graphs]

        for g_idx, detections in enumerate(frame_nodes):
            for det in detections:
                if det.id in assigned_in_graph[g_idx]:
                    continue
                # Try to match with existing cluster
                matched_cluster_idx = -1
                best_dist = matching_radius_um

                for c_idx, cluster in enumerate(clusters):
                    # Check if this graph is already represented in cluster
                    if any(c_g_idx == g_idx for c_g_idx, _ in cluster):
                        continue
                    # Check distance to cluster centroid
                    avg_pos = (
                        sum(d.z for _, d in cluster) / len(cluster),
                        sum(d.y for _, d in cluster) / len(cluster),
                        sum(d.x for _, d in cluster) / len(cluster),
                    )
                    dist = _euclidean_dist(det.position, avg_pos)
                    if dist <= best_dist:
                        best_dist = dist
                        matched_cluster_idx = c_idx

                if matched_cluster_idx >= 0:
                    clusters[matched_cluster_idx].append((g_idx, det))
                    assigned_in_graph[g_idx].add(det.id)
                else:
                    # New cluster
                    clusters.append([(g_idx, det)])
                    assigned_in_graph[g_idx].add(det.id)

        # Filter clusters by min_node_votes and add canonical node
        for cluster in clusters:
            if len(cluster) < min_node_votes:
                continue

            # Consensus position is mean of detections in cluster
            mean_z = sum(d.z for _, d in cluster) / len(cluster)
            mean_y = sum(d.y for _, d in cluster) / len(cluster)
            mean_x = sum(d.x for _, d in cluster) / len(cluster)

            c_id = next_canonical_id
            next_canonical_id += 1

            canonical_graph.add_node(
                Detection(id=c_id, frame=t, z=mean_z, y=mean_y, x=mean_x)
            )

            for g_idx, det in cluster:
                node_to_canonical[(g_idx, det.id)] = c_id

    # Step 2: Accumulate edge votes and max observed out/in degrees per node
    edge_votes: dict[tuple[Hashable, Hashable], int] = defaultdict(int)
    max_out_degree: dict[Hashable, int] = defaultdict(lambda: 1)
    max_in_degree: dict[Hashable, int] = defaultdict(lambda: 1)

    for g_idx, g in enumerate(graphs):
        # Count observed degrees in this graph
        g_out: dict[Hashable, int] = defaultdict(int)
        g_in: dict[Hashable, int] = defaultdict(int)
        for u, v in g.edges:
            g_out[u] += 1
            g_in[v] += 1

        for orig_u, deg in g_out.items():
            can_u = node_to_canonical.get((g_idx, orig_u))
            if can_u is not None and deg > max_out_degree[can_u]:
                max_out_degree[can_u] = min(deg, 2)

        for orig_v, deg in g_in.items():
            can_v = node_to_canonical.get((g_idx, orig_v))
            if can_v is not None and deg > max_in_degree[can_v]:
                max_in_degree[can_v] = min(deg, 1)

        for u, v in g.edges:
            can_u = node_to_canonical.get((g_idx, u))
            can_v = node_to_canonical.get((g_idx, v))
            if can_u is not None and can_v is not None and can_u != can_v:
                # Ensure directed forward edge
                if canonical_graph.nodes[can_v].frame > canonical_graph.nodes[can_u].frame:
                    edge_votes[(can_u, can_v)] += 1

    # Filter edges by min_edge_votes
    admitted_edges = [
        (u, v, votes)
        for (u, v), votes in edge_votes.items()
        if votes >= min_edge_votes
    ]

    if not resolve_contested:
        for u, v, _ in admitted_edges:
            canonical_graph.add_edge(u, v)
        return canonical_graph

    # Step 3: Resolve conflicts (degree constraints: out_degree <= max_out, in_degree <= 1)
    # Sort candidate edges: highest votes first, then shortest 3D distance
    def edge_key(item: tuple[Hashable, Hashable, int]) -> tuple[int, float]:
        u, v, votes = item
        nu = canonical_graph.nodes[u]
        nv = canonical_graph.nodes[v]
        dist = _euclidean_dist(nu.position, nv.position)
        return (-votes, dist)

    admitted_edges.sort(key=edge_key)

    in_degree: dict[Hashable, int] = defaultdict(int)
    out_degree: dict[Hashable, int] = defaultdict(int)

    for u, v, _ in admitted_edges:
        if out_degree[u] < max_out_degree[u] and in_degree[v] < max_in_degree[v]:
            canonical_graph.add_edge(u, v)
            out_degree[u] += 1
            in_degree[v] += 1

    return canonical_graph


def ensemble_submission_files(
    submission_paths: list[str | Path],
    output_path: str | Path,
    *,
    matching_radius_um: float = 3.5,
    min_node_votes: int = 1,
    min_edge_votes: int = 1,
    resolve_contested: bool = True,
) -> pd.DataFrame:
    """Ensemble multiple submission CSV files into a single consensus submission CSV.

    Parameters
    ----------
    submission_paths:
        List of paths to valid submission CSV files.
    output_path:
        Target output path for the exported consensus CSV.
    matching_radius_um:
        Maximum distance in microns for bipartite node clustering.
    min_node_votes:
        Minimum number of submissions that must include a detection.
    min_edge_votes:
        Minimum number of submissions that must include an edge.
    resolve_contested:
        Whether to enforce out_degree <= 2 and in_degree <= 1 constraints.

    Returns
    -------
    pandas.DataFrame
        Exported consensus submission DataFrame.
    """
    from pathlib import Path
    import pandas as pd
    from biohub_tracking.evaluation.local_score import load_submission_graphs
    from biohub_tracking.submission.export import export_submission

    if len(submission_paths) < 1:
        raise ValueError("Must provide at least one submission path")

    # Load all submission graphs: list of dict[dataset_name -> TrackingGraph]
    submission_graphs = [
        load_submission_graphs(Path(p), require_consecutive_edges=False)
        for p in submission_paths
    ]

    # Find union of all datasets
    all_datasets = sorted(set().union(*(set(g.keys()) for g in submission_graphs)))

    ensembled_graphs: dict[str, TrackingGraph] = {}
    for ds in all_datasets:
        graphs_for_ds = [sg[ds] for sg in submission_graphs if ds in sg]
        if graphs_for_ds:
            ensembled_graphs[ds] = ensemble_tracking_graphs(
                graphs_for_ds,
                matching_radius_um=matching_radius_um,
                min_node_votes=min_node_votes,
                min_edge_votes=min_edge_votes,
                resolve_contested=resolve_contested,
            )

    return export_submission(ensembled_graphs, output_path)

