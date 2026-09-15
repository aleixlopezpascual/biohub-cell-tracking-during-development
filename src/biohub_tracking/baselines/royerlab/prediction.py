# src/biohub_tracking/baselines/royerlab/prediction.py
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from biohub_tracking.baselines.royerlab.tta import XY_D4_TRANSFORMS, apply_spatial_transform, invert_spatial_transform

class NativePredictor:
    def __init__(self, detector_weights: list[Path], edge_weights: list[Path]) -> None:
        self.detector_weights = detector_weights
        self.edge_weights = edge_weights
        self.detectors: list[Any] = []
        self.edge_scorers: list[Any] = []

    def predict_heatmaps_and_features(self, frames: np.ndarray, use_tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if not self.detectors:
            raise ValueError("No detectors loaded")
        model = self.detectors[0]
        if not use_tta:
            return model.predict_heatmaps_and_features(frames)
            
        heatmaps = []
        features = []
        for transform in XY_D4_TRANSFORMS:
            # Apply transform to spatial axes (last two axes Y, X)
            aug_frames = apply_spatial_transform(frames, transform)
            h, f = model.predict_heatmaps_and_features(aug_frames)
            # Invert transform
            heatmaps.append(invert_spatial_transform(h, transform))
            features.append(invert_spatial_transform(f, transform))
            
        return np.mean(heatmaps, axis=0), np.mean(features, axis=0)
