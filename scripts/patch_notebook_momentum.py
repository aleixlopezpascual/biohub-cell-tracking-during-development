#!/usr/bin/env python3
"""Programmatically inject EMA Spatiotemporal Velocity Projection into the 0.948 notebook."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("scripts/kaggle_kernels/biohub_0_948_momentum_deepcenter_tta/biohub-0-948-momentum-deepcenter-tta.ipynb")

with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

cell_5_source = "".join(nb["cells"][5]["source"])

# 1. Verification of the target original block
target_original = """    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)"""

target_enhanced = """    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    track_velocity_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            vel = track_velocity_um.get(source_id)
            if vel is None:
                prev_pos = predecessor_position_um.get(source_id)
                vel = (source_pos - prev_pos) if prev_pos is not None else np.zeros_like(source_pos)
            predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * vel"""

# 2. Verification of velocity accumulation block
target_velocity_update_original = """        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]"""

target_velocity_update_enhanced = """        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            instant_vel = position_um[target_id] - position_um[source_id]
            prior_vel = track_velocity_um.get(source_id)
            if prior_vel is None:
                track_velocity_um[target_id] = instant_vel
            else:
                track_velocity_um[target_id] = 0.6 * instant_vel + 0.4 * prior_vel
            predecessor_position_um[target_id] = position_um[source_id]"""

if target_original not in cell_5_source:
    raise RuntimeError("target_original block not found in Cell 05")
if target_velocity_update_original not in cell_5_source:
    raise RuntimeError("target_velocity_update_original block not found in Cell 05")

cell_5_source = cell_5_source.replace(target_original, target_enhanced)
cell_5_source = cell_5_source.replace(target_velocity_update_original, target_velocity_update_enhanced)

nb["cells"][5]["source"] = [cell_5_source]

with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully injected EMA Velocity Projection into the 0.948 notebook!")
