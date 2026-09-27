"""CPU-only characterization tests for the staged notebook's EMA relinker.

These synthetic cases exercise the helper implementation only. They do not measure
tracking quality, establish a score gain, or satisfy the notebook promotion gate.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from biohub_tracking.tracking.graph import Detection
from biohub_tracking.tracking.motion_relink import MotionRelinkConfig, motion_relink

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "kaggle_kernels"
    / "biohub_0_948_momentum_deepcenter_tta"
    / "biohub-0-948-momentum-deepcenter-tta.ipynb"
)
_CONFIG_NAMES = {
    "OUTPUT_MOTION_RELINK",
    "MOTION_RELINK_TIGHT_UM",
    "MOTION_RELINK_RELAXED_UM",
    "MOTION_RELINK_VELOCITY_WEIGHT",
    "MOTION_RELINK_LEARNED_BONUS",
    "MOTION_RELINK_MAX_FRAME_NODES",
}


def _cell_source(notebook: dict[str, Any], index: int) -> str:
    source = notebook["cells"][index]["source"]
    return "".join(source) if isinstance(source, list) else str(source)


def _load_motion_relinker() -> tuple[Any, tuple[float, float, float]]:
    """Compile only the target helpers and their literal/default config, never a cell."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    config_tree = ast.parse(_cell_source(notebook, 2))
    config_assignments: list[ast.stmt] = []
    for node in config_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if names & _CONFIG_NAMES:
            config_assignments.append(node)

    found_config_names = {
        target.id
        for node in config_assignments
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert found_config_names == _CONFIG_NAMES

    cell_five = _cell_source(notebook, 5)
    cell_five_tree = ast.parse(cell_five)
    target_functions = {
        node.name: node
        for node in cell_five_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_position_um", "motion_relink_edges"}
    }
    assert set(target_functions) == {"_position_um", "motion_relink_edges"}

    voxel_scale_assignment = next(
        node
        for node in cell_five_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "VOXEL_SCALE_UM"
            for target in node.targets
        )
    )
    voxel_scale = ast.literal_eval(voxel_scale_assignment.value)

    namespace: dict[str, Any] = {
        "VOXEL_SCALE_UM": voxel_scale,
        "math": math,
        "np": np,
        "linear_sum_assignment": linear_sum_assignment,
        # Empty environment keeps the notebook's documented defaults deterministic.
        "os": SimpleNamespace(environ={}),
    }
    config_module = ast.fix_missing_locations(
        ast.Module(body=config_assignments, type_ignores=[])
    )
    exec(compile(config_module, str(NOTEBOOK_PATH), "exec"), namespace)
    helper_module = ast.fix_missing_locations(
        ast.Module(
            body=[target_functions["_position_um"], target_functions["motion_relink_edges"]],
            type_ignores=[],
        )
    )
    exec(compile(helper_module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace["motion_relink_edges"], voxel_scale


def _node_at_um(frame: int, x_um: float, scale_x_um: float) -> dict[str, float | int]:
    return {"t": frame, "z": 0.0, "y": 0.0, "x": x_um / scale_x_um}


def _empty_motion_stats() -> dict[str, int]:
    return {
        "motion_relink_frames": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
    }


def test_motion_relink_uses_velocity_prediction_over_nearest_raw_node() -> None:
    """Characterize helper selection only; this is not score or promotion evidence."""
    motion_relink_edges, voxel_scale = _load_motion_relinker()
    scale_x_um = voxel_scale[2]
    nodes = {
        1: _node_at_um(0, 0.0, scale_x_um),
        2: _node_at_um(1, 10.0, scale_x_um),
        3: _node_at_um(2, 11.0, scale_x_um),
        4: _node_at_um(2, 14.0, scale_x_um),
    }

    edges = motion_relink_edges(nodes, _empty_motion_stats())

    assert [(edge["source_id"], edge["target_id"]) for edge in edges] == [
        (1, 2),
        (2, 4),
    ]


def test_motion_relink_ema_update_influences_next_frame_assignment() -> None:
    """Characterize EMA velocity update only; this does not validate tracking quality."""
    motion_relink_edges, voxel_scale = _load_motion_relinker()
    scale_x_um = voxel_scale[2]
    nodes = {
        1: _node_at_um(0, 0.0, scale_x_um),
        2: _node_at_um(1, 10.0, scale_x_um),
        3: _node_at_um(2, 15.0, scale_x_um),
        4: _node_at_um(3, 17.0, scale_x_um),
        5: _node_at_um(3, 19.5, scale_x_um),
    }

    edges = motion_relink_edges(nodes, _empty_motion_stats())

    assert [(edge["source_id"], edge["target_id"]) for edge in edges] == [
        (1, 2),
        (2, 3),
        (3, 5),
    ]


def test_core_motion_relink_matches_notebook_helper() -> None:
    notebook_relink, voxel_scale = _load_motion_relinker()
    nodes = {
        1: _node_at_um(0, 0.0, voxel_scale[2]),
        2: _node_at_um(1, 10.0, voxel_scale[2]),
        3: _node_at_um(2, 11.0, voxel_scale[2]),
        4: _node_at_um(2, 14.0, voxel_scale[2]),
    }
    learned_edge_probs = {(1, 2): 0.2, (2, 3): 0.9, (2, 4): 0.0}
    notebook_stats = _empty_motion_stats()
    expected = notebook_relink(nodes, notebook_stats, learned_edge_probs)
    detections = {
        node_id: Detection(
            id=node_id,
            frame=int(node["t"]),
            z=float(node["z"]) * voxel_scale[0],
            y=float(node["y"]) * voxel_scale[1],
            x=float(node["x"]) * voxel_scale[2],
        )
        for node_id, node in nodes.items()
    }

    actual = motion_relink(
        detections,
        learned_edge_probs,
        MotionRelinkConfig(learned_edge_bonus=0.75),
    )

    expected_structure = [
        (int(edge["source_id"]), int(edge["target_id"]), str(edge["motion_pass"]))
        for edge in expected
    ]
    assert [(edge.source_id, edge.target_id, edge.pass_name) for edge in actual.edges] == (
        expected_structure
    )
    expected_values = [
        (float(edge["edge_prob"]), float(edge["distance_um"]), float(edge["motion_distance_um"]))
        for edge in expected
    ]
    actual_values = [
        (edge.edge_probability, edge.distance_um, edge.motion_distance_um)
        for edge in actual.edges
    ]
    for actual_row, expected_row in zip(actual_values, expected_values):
        assert actual_row == pytest.approx(expected_row)
    assert actual.stats.frames_processed == notebook_stats["motion_relink_frames"]
    assert actual.stats.tight_edges == notebook_stats["motion_relink_tight_edges"]
    assert actual.stats.relaxed_edges == notebook_stats["motion_relink_relaxed_edges"]
