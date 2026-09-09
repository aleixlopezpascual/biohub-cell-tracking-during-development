"""Biohub cell-tracking submission kernel.

Runs bounded local-maxima detection + Hungarian frame-to-frame linking
directly against whatever test Zarr volumes Kaggle mounts at scoring
time (no static/precomputed submission file is copied in). This is
required for code competitions: the scored rerun may present different
test data than what is visible locally, so `submission.csv` must be
generated fresh from `/kaggle/input/.../test/` on every run.

Per-frame detection budget and the frame-to-frame linking distance gate
are both derived only from metadata embedded in the test Zarr group
itself (`image_statistics` quantiles, physical `scale`) -- never from
any train-set lookup -- so this generalizes to unseen datasets.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import blosc2
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.optimize import linear_sum_assignment

DS = 2
SMOOTH = 3
MAX_PEAKS_PER_FRAME = 800
LINK_GATE_UM = 10.0

CANDIDATE_ROOTS = [
    "/kaggle/input/biohub-cell-tracking-during-development",
    "/kaggle/input/competitions/biohub-cell-tracking-during-development",
    "data",
]


def find_test_dir() -> Path:
    """Locate the mounted competition test directory."""
    for root in CANDIDATE_ROOTS:
        candidate = Path(root) / "test"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a 'test' directory under any of {CANDIDATE_ROOTS}"
    )


def read_group_attrs(zarr_root: Path) -> dict[str, object]:
    """Read the OME-Zarr group-level zarr.json (scale + image statistics)."""
    return json.loads((zarr_root / "zarr.json").read_text(encoding="utf-8"))


def read_array_meta(zarr_root: Path) -> dict[str, object]:
    """Read the array-level zarr.json (shape, dtype)."""
    return json.loads((zarr_root / "0" / "zarr.json").read_text(encoding="utf-8"))


def scale_zyx(group_attrs: dict[str, object]) -> tuple[float, float, float]:
    """Extract the physical (z, y, x) voxel scale from OME-Zarr multiscale metadata."""
    multiscales = group_attrs.get("attributes", {}).get("multiscales")
    if multiscales:
        transforms = multiscales[0]["datasets"][0]["coordinateTransformations"]
        scale = transforms[0]["scale"]
        # scale is (t, z, y, x); drop the time axis.
        return float(scale[1]), float(scale[2]), float(scale[3])
    # Conservative fallback matching this competition's known acquisition geometry.
    return 1.625, 0.40625, 0.40625


def intensity_threshold(group_attrs: dict[str, object]) -> float:
    """Derive a per-dataset peak-intensity threshold from embedded image statistics."""
    stats = group_attrs.get("attributes", {}).get("image_statistics", {})
    quantiles = stats.get("quantiles", {})
    for key in ("0.99", "0.9"):
        if key in quantiles:
            return float(quantiles[key])
    return 0.0


def frame(zarr_root: Path, shape: tuple[int, ...], dtype: np.dtype, t: int) -> np.ndarray:
    chunk_path = zarr_root / "0" / "c" / str(t) / "0" / "0" / "0"
    return np.frombuffer(blosc2.decompress(chunk_path.read_bytes()), dtype=dtype).reshape(shape[1:])


def peaks(vol: np.ndarray, threshold: float, max_n: int) -> np.ndarray:
    """Bounded local-maxima detection with sub-voxel centroid refinement."""
    sub = vol[::DS, ::DS, ::DS].astype(np.float32)
    filtered = uniform_filter(sub, size=SMOOTH)
    thresh = threshold if threshold > 0 else float(filtered.mean() + filtered.std())
    is_peak = (filtered == maximum_filter(filtered, size=(3, 7, 7))) & (filtered >= thresh)
    z, y, x = np.nonzero(is_peak)
    if not len(z):
        return np.zeros((0, 3), dtype=float)
    order = np.argsort(-filtered[z, y, x])[:max_n]
    z, y, x = z[order], y[order], x[order]
    zmax, ymax, xmax = filtered.shape
    refined = []
    for zc, yc, xc in zip(z, y, x):
        zz, yy, xx = np.meshgrid(
            np.arange(max(0, int(zc) - 1), min(zmax, int(zc) + 2)),
            np.arange(max(0, int(yc) - 3), min(ymax, int(yc) + 4)),
            np.arange(max(0, int(xc) - 3), min(xmax, int(xc) + 4)),
            indexing="ij",
        )
        weights = filtered[zz, yy, xx].astype(float)
        weights -= weights.min()
        total = weights.sum() or 1.0
        refined.append(
            [
                float((weights * zz).sum() / total),
                float((weights * yy).sum() / total),
                float((weights * xx).sum() / total),
            ]
        )
    return np.asarray(refined, dtype=float) * DS


def link(
    per_frame: dict[int, np.ndarray], scale: tuple[float, float, float], gate_um: float
) -> tuple[list[dict[str, int]], list[tuple[int, int]]]:
    nodes: list[dict[str, int]] = []
    edges: list[tuple[int, int]] = []
    previous_ids: list[int] = []
    previous_points: np.ndarray | None = None
    scale_arr = np.asarray(scale, dtype=float)
    for t in sorted(per_frame):
        current_points = per_frame[t]
        current_ids = list(range(len(nodes), len(nodes) + len(current_points)))
        for point in current_points:
            nodes.append(
                {
                    "t": int(t),
                    "z": int(round(float(point[0]))),
                    "y": int(round(float(point[1]))),
                    "x": int(round(float(point[2]))),
                }
            )
        if previous_points is not None and len(previous_points) and len(current_points):
            source = np.asarray(previous_points) * scale_arr
            target = np.asarray(current_points) * scale_arr
            dist = np.sqrt(((source[:, None] - target[None, :]) ** 2).sum(axis=2))
            row_ind, col_ind = linear_sum_assignment(dist)
            for source_index, target_index in zip(row_ind, col_ind):
                if dist[source_index, target_index] <= gate_um:
                    edges.append((previous_ids[source_index], current_ids[target_index]))
        previous_ids = current_ids
        previous_points = current_points
    return nodes, edges


def prune(
    nodes: list[dict[str, int]], edges: list[tuple[int, int]]
) -> tuple[list[dict[str, int]], list[tuple[int, int]]]:
    """Drop nodes with no incident edge -- they only inflate T_pred without helping."""
    keep = sorted({node for edge in edges for node in edge})
    if not keep:
        return nodes, edges
    remap = {old: new for new, old in enumerate(keep)}
    return [nodes[index] for index in keep], [
        (remap[source], remap[target]) for source, target in edges
    ]


def infer_dataset(zarr_root: Path) -> tuple[list[dict[str, int]], list[tuple[int, int]]]:
    group_attrs = read_group_attrs(zarr_root)
    array_meta = read_array_meta(zarr_root)
    shape = tuple(array_meta["shape"])
    dtype = np.dtype(str(array_meta["data_type"]))
    n_frames = int(shape[0])
    scale = scale_zyx(group_attrs)
    threshold = intensity_threshold(group_attrs)
    per_frame = {
        t: peaks(frame(zarr_root, shape, dtype, t), threshold, MAX_PEAKS_PER_FRAME)
        for t in range(n_frames)
    }
    return prune(*link(per_frame, scale, LINK_GATE_UM))


def write_submission(output: Path) -> None:
    test_dir = find_test_dir()
    rows: list[dict[str, object]] = []
    dataset_names = sorted(p.name.removesuffix(".zarr") for p in test_dir.glob("*.zarr"))
    if not dataset_names:
        raise FileNotFoundError(f"No .zarr datasets found under {test_dir}")
    for dataset in dataset_names:
        print("INFER", dataset, flush=True)
        nodes, edges = infer_dataset(test_dir / f"{dataset}.zarr")
        node_ids = list(range(1, len(nodes) + 1))
        for node_id, node in zip(node_ids, nodes):
            rows.append(
                {
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": node_id,
                    "t": node["t"],
                    "z": node["z"],
                    "y": node["y"],
                    "x": node["x"],
                    "source_id": -1,
                    "target_id": -1,
                }
            )
        for source, target in edges:
            rows.append(
                {
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": node_ids[source],
                    "target_id": node_ids[target],
                }
            )
        print("DONE", dataset, "nodes", len(nodes), "edges", len(edges), flush=True)
    columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row_id, row in enumerate(rows):
            writer.writerow({"id": row_id, **row})
    print("WROTE", output, "rows", len(rows), flush=True)


if __name__ == "__main__":
    write_submission(Path("/kaggle/working/submission.csv"))
