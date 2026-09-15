# src/biohub_tracking/baselines/royerlab/prediction.py
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from biohub_tracking.baselines.royerlab.tta import XY_D4_TRANSFORMS, apply_spatial_transform, invert_spatial_transform
from biohub_tracking.baselines.royerlab.linking import fuse_bidirectional_probabilities

class NativePredictor:
    def __init__(self, detector_weights: list[Path], edge_weights: list[Path]) -> None:
        self.detector_weights = detector_weights
        self.edge_weights = edge_weights
        self.detectors: list[Any] = []
        self.edge_scorers: list[Any] = []

    def predict_heatmaps_and_features(self, frames: np.ndarray, use_tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if not self.detectors:
            raise ValueError("No detectors loaded")
            
        all_heatmaps = []
        all_features = []
        
        for model in self.detectors:
            if not use_tta:
                h, f = model.predict_heatmaps_and_features(frames)
                all_heatmaps.append(h)
                all_features.append(f)
                continue
                
            for transform in XY_D4_TRANSFORMS:
                # Apply transform to spatial axes (last two axes Y, X)
                aug_frames = apply_spatial_transform(frames, transform)
                h, f = model.predict_heatmaps_and_features(aug_frames)
                # Invert transform
                all_heatmaps.append(invert_spatial_transform(h, transform))
                all_features.append(invert_spatial_transform(f, transform))
                
        return np.mean(all_heatmaps, axis=0), np.mean(all_features, axis=0)

    def predict_bidirectional_edges(self, src_features: np.ndarray, tgt_features: np.ndarray) -> np.ndarray:
        if not self.edge_scorers:
            raise ValueError("No edge scorers loaded")
        model = self.edge_scorers[0]
        
        # Predict forward probabilities (u -> v)
        p_forward = model.predict_edge_logits(src_features, tgt_features)
        
        # Predict reverse probabilities (v -> u)
        p_reverse = model.predict_edge_logits(tgt_features, src_features)
        p_reverse_aligned = p_reverse.T
        
        # Fuse using existing harmonic block
        return fuse_bidirectional_probabilities(p_forward, p_reverse_aligned, mode="harmonic")

