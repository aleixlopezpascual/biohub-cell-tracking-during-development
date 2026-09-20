"""Hungarian (optimal bipartite) frame-to-frame tracker with division branching.

The tracker works in two passes per consecutive frame pair:

1. **Primary linking.** An optimal one-to-one assignment (Hungarian
   algorithm via :func:`scipy.optimize.linear_sum_assignment`) links each
   source detection to at most one target detection, gated by
   ``max_link_distance_um`` so implausibly distant pairs are never linked.
2. **Division branching.** Targets left unmatched after the primary pass are
   greedily attached as a second (or later) daughter to the nearest already
   linked source that is still below ``max_daughters`` and within
   ``division_search_radius_um``, producing the two (or more) outgoing
   edges that mark a predicted cell division.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from biohub_tracking.tracking.graph import Detection, NodeId, TrackingGraph

_UNREACHABLE_COST = 1e12


@dataclass(frozen=True)
class TrackerConfig:
    """Gating parameters for :class:`HungarianTracker`."""

    max_link_distance_um: float = 15.0
    division_search_radius_um: float = 20.0
    max_daughters: int = 2
    gating_time_window: int = 1
    use_sister_symmetry_gate: bool = False
    sister_symmetry_tau: float = 0.6
    use_ema_velocity_projection: bool = False
    ema_velocity_alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.max_link_distance_um <= 0:
            raise ValueError("max_link_distance_um must be positive")
        if self.division_search_radius_um <= 0:
            raise ValueError("division_search_radius_um must be positive")
        if self.max_daughters < 2:
            raise ValueError("max_daughters must be >= 2 to represent divisions")
        if self.sister_symmetry_tau <= 0:
            raise ValueError("sister_symmetry_tau must be positive")
        if not 0.0 < self.ema_velocity_alpha <= 1.0:
            raise ValueError("ema_velocity_alpha must be in (0, 1]")


class HungarianTracker:
    """Links detections across frames into a :class:`TrackingGraph`."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()

    def link_frames(
        self,
        sources: list[Detection],
        targets: list[Detection],
        projected_positions: dict[NodeId, tuple[float, float, float]] | None = None,
    ) -> list[tuple[NodeId, NodeId]]:
        """Link one frame of ``sources`` to the next frame's ``targets``.

        Returns a list of ``(source_id, target_id)`` edges. A source may
        appear in more than one edge when division branching attaches a
        second daughter to it.
        """
        if not sources or not targets:
            return []

        cfg = self.config
        src_pos = np.array(
            [
                projected_positions.get(s.id, s.position)
                if projected_positions is not None
                else s.position
                for s in sources
            ],
            dtype=float,
        )
        tgt_pos = np.array([t.position for t in targets], dtype=float)
        dist = cdist(src_pos, tgt_pos)

        cost = np.where(dist <= cfg.max_link_distance_um, dist, _UNREACHABLE_COST)
        row_ind, col_ind = linear_sum_assignment(cost)

        edges: list[tuple[NodeId, NodeId]] = []
        matched_target_cols: set[int] = set()
        out_count: dict[int, int] = defaultdict(int)
        linked_cols: dict[int, list[int]] = defaultdict(list)

        for r, c in zip(row_ind, col_ind):
            if dist[r, c] <= cfg.max_link_distance_um:
                edges.append((sources[r].id, targets[c].id))
                matched_target_cols.add(c)
                out_count[r] += 1
                linked_cols[r].append(c)

        # Division branching: attach unmatched targets as extra daughters.
        unmatched_cols = [c for c in range(len(targets)) if c not in matched_target_cols]
        for c in unmatched_cols:
            candidate_rows = []
            for r in range(len(sources)):
                if (
                    out_count[r] >= 1
                    and out_count[r] < cfg.max_daughters
                    and dist[r, c] <= cfg.division_search_radius_um
                ):
                    # Sister Symmetry Gate check if enabled
                    if cfg.use_sister_symmetry_gate and r in linked_cols:
                        d1_col = linked_cols[r][0]
                        d1_dist = dist[r, d1_col]
                        d2_dist = dist[r, c]
                        mean_dist = (d1_dist + d2_dist) / 2.0
                        if mean_dist > 0:
                            symmetry_ratio = abs(d1_dist - d2_dist) / mean_dist
                            if symmetry_ratio > cfg.sister_symmetry_tau:
                                continue  # Violates symmetry, filter out this candidate

                    candidate_rows.append(r)

            if not candidate_rows:
                continue

            best_r = min(candidate_rows, key=lambda r: dist[r, c])
            edges.append((sources[best_r].id, targets[c].id))
            out_count[best_r] += 1
            linked_cols[best_r].append(c)

        return edges

    def track(self, graph: TrackingGraph) -> TrackingGraph:
        """Link every consecutive frame pair of a node-only graph.

        Any pre-existing edges in ``graph`` are discarded; this method is
        meant to (re)build the edge set from scratch given only detections.
        """
        by_frame = graph.nodes_by_frame()
        result = TrackingGraph(nodes=dict(graph.nodes))
        frames = sorted(by_frame)
        
        # Track active velocity vectors per trajectory: last_node_id -> velocity_vector_um
        active_tracks_velocity: dict[NodeId, tuple[float, float, float]] = {}
        cfg = self.config

        for t0, t1 in zip(frames, frames[1:]):
            sources = by_frame[t0]
            targets = by_frame[t1]
            dt = t1 - t0

            # Compute projected positions for active trajectories
            projected_positions: dict[NodeId, tuple[float, float, float]] = {}
            if cfg.use_ema_velocity_projection:
                for s in sources:
                    if s.id in active_tracks_velocity:
                        v = active_tracks_velocity[s.id]
                        projected_positions[s.id] = (
                            s.z + v[0] * dt,
                            s.y + v[1] * dt,
                            s.x + v[2] * dt,
                        )

            # Link frames using projections
            links = self.link_frames(sources, targets, projected_positions=projected_positions if cfg.use_ema_velocity_projection else None)
            
            # Map targets and sources by ID for fast lookup
            targets_dict = {t.id: t for t in targets}
            sources_dict = {s.id: s for s in sources}

            # Propagate and update EMA velocities for successfully matched edges
            new_velocities: dict[NodeId, tuple[float, float, float]] = {}
            for s_id, t_id in links:
                s_node = sources_dict[s_id]
                t_node = targets_dict[t_id]
                
                v_current = (
                    (t_node.z - s_node.z) / dt,
                    (t_node.y - s_node.y) / dt,
                    (t_node.x - s_node.x) / dt,
                )
                
                if s_id in active_tracks_velocity:
                    v_prev = active_tracks_velocity[s_id]
                    alpha = cfg.ema_velocity_alpha
                    v_new = (
                        alpha * v_current[0] + (1 - alpha) * v_prev[0],
                        alpha * v_current[1] + (1 - alpha) * v_prev[1],
                        alpha * v_current[2] + (1 - alpha) * v_prev[2],
                    )
                else:
                    v_new = v_current
                    
                new_velocities[t_id] = v_new
                result.add_edge(s_id, t_id)
            
            # Update active velocities for the next frame
            active_tracks_velocity = new_velocities
            
        return result
