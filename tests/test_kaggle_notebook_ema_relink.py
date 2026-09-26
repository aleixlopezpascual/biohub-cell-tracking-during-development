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
from scipy.optimize import linear_sum_assignment

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
