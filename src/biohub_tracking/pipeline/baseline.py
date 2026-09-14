"""Composable model-free baseline: detect blobs, link frames, export a graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub_tracking.data import OMEZarrVolume
from biohub_tracking.detection import LocalMaximaDetector, LocalMaximaDetectorConfig
from biohub_tracking.tracking import Detection, HungarianTracker, TrackerConfig, TrackingGraph


@dataclass(frozen=True)
class BaselinePipelineConfig:
    """Detector and tracker settings for :func:`infer_volume`."""

    threshold: float = 0.0
    min_distance: float = 3.0
    channel: int = 0
    voxel_size_um: tuple[float, float, float] | None = None
    tracker: TrackerConfig = TrackerConfig()
    use_dog: bool = False
    dog_sigmas: list[float] | None = None
    dog_ratio: float = 1.6


def infer_volume(volume: OMEZarrVolume, config: BaselinePipelineConfig | None = None) -> TrackingGraph:
    """Detect and track cells in an OME-Zarr-compatible volume.

    The requested channel is read one frame at a time. Detected voxel
    coordinates are converted to microns before constructing graph nodes, as
    required by the physical-distance tracker and submission format.
    """
    cfg = config or BaselinePipelineConfig()
    if not 0 <= cfg.channel < volume.num_channels:
        raise IndexError(f"channel {cfg.channel} out of range [0, {volume.num_channels})")

    voxel_size = cfg.voxel_size_um or volume.voxel_size_um
    detector = LocalMaximaDetector(
        LocalMaximaDetectorConfig(
            threshold=cfg.threshold,
            min_distance=cfg.min_distance,
            voxel_size_um=voxel_size,
            use_dog=cfg.use_dog,
            dog_sigmas=cfg.dog_sigmas,
            dog_ratio=cfg.dog_ratio,
        )
    )
    graph = TrackingGraph()
    node_id = 0
    for frame, image in volume.iter_frames(channels=[cfg.channel]):
        for z, y, x in detector.detect(image[0]):
            position_um = np.asarray((z, y, x)) * np.asarray(voxel_size)
            graph.add_node(
                Detection(
                    id=node_id,
                    frame=frame,
                    z=float(position_um[0]),
                    y=float(position_um[1]),
                    x=float(position_um[2]),
                )
            )
            node_id += 1
    return HungarianTracker(cfg.tracker).track(graph)
