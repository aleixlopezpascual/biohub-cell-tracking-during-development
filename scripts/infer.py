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
import json
from pathlib import Path

import numpy as np

from biohub_tracking.data import OMEZarrVolume
from biohub_tracking.detection import LocalMaximaDetector, LocalMaximaDetectorConfig
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
    parser.add_argument(
        "--min-peak-distance-um",
        type=float,
        default=3.0,
        help="Peak separation suppression radius in physical microns.",
    )
    parser.add_argument(
        "--use-subpixel-refinement",
        action="store_true",
        help="Enable center-of-mass coordinate refinement.",
    )
    parser.add_argument(
        "--use-dog",
        action="store_true",
        help="Enable Multi-Scale Difference of Gaussians (DoG) peak filtering.",
    )
    parser.add_argument(
        "--dog-sigmas",
        type=float,
        nargs="+",
        help="Optional physical sigmas in microns for multi-scale DoG.",
    )
    parser.add_argument(
        "--dog-ratio",
        type=float,
        default=1.6,
        help="DoG large-to-small sigma ratio.",
    )
    return parser.parse_args(argv)


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

    detector = LocalMaximaDetector(
        LocalMaximaDetectorConfig(
            threshold=args.threshold,
            min_distance=args.min_peak_distance_um,
            voxel_size_um=config.data.voxel_size_um,
            use_subpixel_refinement=args.use_subpixel_refinement,
            use_dog=args.use_dog,
            dog_sigmas=args.dog_sigmas,
            dog_ratio=args.dog_ratio,
        )
    )

    graph = TrackingGraph()
    node_id = 0
    detections_by_frame: dict[int, list[Detection]] = {}
    voxel_size = np.array(config.data.voxel_size_um)

    for t, frame in volume.iter_frames(channels=config.data.channels):
        heatmap = model.predict(frame)[0]
        for z, y, x in detector.detect(heatmap):
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
