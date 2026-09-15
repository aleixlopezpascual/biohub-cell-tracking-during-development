import numpy as np
import pytest
from biohub_tracking.baselines.royerlab.prediction import NativePredictor

def test_tta_canonical_averaging_is_invariant() -> None:
    # A fake mock model that returns the input as features
    class MockModel:
        def predict_heatmaps_and_features(self, frames):
            # returns (heatmap, features)
            return frames, frames

    predictor = NativePredictor(detector_weights=[], edge_weights=[])
    predictor.detectors = [MockModel()]
    
    # Create a simple asymmetric 3D volume (B=1, C=1, Z=4, Y=8, X=8)
    image = np.zeros((1, 1, 4, 8, 8), dtype=float)
    image[0, 0, 2, 4, 5] = 10.0  # single bright peak
    
    mean_heatmap, mean_features = predictor.predict_heatmaps_and_features(image, use_tta=True)
    
    # Verify peak coordinates are correctly localized at (2, 4, 5) after TTA inversion
    peak_idx = np.unravel_index(np.argmax(mean_heatmap), mean_heatmap.shape)
    assert peak_idx == (0, 0, 2, 4, 5)
