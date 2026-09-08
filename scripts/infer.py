"""Inference entry point: detect + track an OME-Zarr volume into a submission CSV.

Usage::

    python scripts/infer.py --config configs/inference.yaml --checkpoint outputs/train/unet3d.pt \\
        --dataset testA --output submission.csv

For each frame, runs the trained :class:`~biohub_tracking.models.UNet3D` to
produce a heatmap, extracts local maxima above a threshold as detections
(using :func:`scipy.ndimage.maximum_filter`), links detections across frames
with :class:`~biohub_tracking.tracking.HungarianTracker` (including division
branching), and writes the result with
:func:`~biohub_tracking.submission.export_submission`.

This reference pipeline processes one full frame at a time; for volumes too
large to fit a whole frame through the network, tile with
:class:`~biohub_tracking.data.PatchGrid3D` and stitch heatmaps before peak
extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from biohub_tracking.data import OMEZarrVolume
from biohub_tracking.models import UNet3D
from biohub_tracking.submission import export_submission
from biohub_tracking.tracking import Detection, HungarianTracker, TrackerConfig, TrackingGraph
from biohub_tracking.utils import get_logger, load_config

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML PipelineConfig.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained UNet3D checkpoint (.pt).")
    parser.add_argument("--dataset", required=True, help="Dataset name for the submission CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output submission CSV path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap peak-detection threshold.")
    parser.add_argument("--min-peak-distance", type=int, default=3, help="Non-max suppression window (voxels).")
    return parser.parse_args(argv)


def _extract_peaks(heatmap: np.ndarray, threshold: float, min_distance: int) -> list[tuple[float, float, float]]:
    """Non-maximum suppression peak detection over a 3D heatmap."""
    local_max = maximum_filter(heatmap, size=min_distance) == heatmap
    above_threshold = heatmap >= threshold
    peaks = np.argwhere(local_max & above_threshold)
    return [(float(z), float(y), float(x)) for z, y, x in peaks]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)

    logger.info("Loading model checkpoint from %s", args.checkpoint)
    model = UNet3D.load(str(args.checkpoint))

    logger.info("Opening OME-Zarr volume at %s", config.data.zarr_path)
    volume = OMEZarrVolume.open(
        config.data.zarr_path,
        resolution_level=config.data.resolution_level,
        voxel_size_um=config.data.voxel_size_um,
    )

    graph = TrackingGraph()
    node_id = 0
    detections_by_frame: dict[int, list[Detection]] = {}
    voxel_size = np.array(config.data.voxel_size_um)

    for t, frame in volume.iter_frames(channels=config.data.channels):
        heatmap = model.predict(frame)[0]
        for z, y, x in _extract_peaks(heatmap, args.threshold, args.min_peak_distance):
            zu, yu, xu = np.array([z, y, x]) * voxel_size
            detection = Detection(id=node_id, frame=t, z=float(zu), y=float(yu), x=float(xu))
            graph.add_node(detection)
            detections_by_frame.setdefault(t, []).append(detection)
            node_id += 1
        logger.info("frame=%d detections=%d", t, len(detections_by_frame.get(t, [])))

    tracker = HungarianTracker(
        TrackerConfig(
            max_link_distance_um=config.tracking.max_link_distance_um,
            division_search_radius_um=config.tracking.division_search_radius_um,
            max_daughters=config.tracking.max_daughters,
        )
    )
    tracked_graph = tracker.track(graph)
    logger.info("Tracked graph: %d nodes, %d edges", len(tracked_graph.nodes), len(tracked_graph.edges))

    export_submission({args.dataset: tracked_graph}, args.output)
    logger.info("Wrote submission to %s", args.output)


if __name__ == "__main__":
    main()
