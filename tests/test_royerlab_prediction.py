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

def test_bidirectional_harmonic_veto() -> None:
    class MockEdgeModel:
        def __init__(self, forward_prob, reverse_prob):
            self.forward_prob = forward_prob
            self.reverse_prob = reverse_prob
            
        def predict_edge_logits(self, src_feats, tgt_feats):
            # Dummy logic returning pre-defined prob as logits for testing
            # (Assuming model returns probabilities as values)
            if len(src_feats) == 2: # Forward
                return np.array([[self.forward_prob]])
            else: # Reverse
                return np.array([[self.reverse_prob]])

    # Case A: Forward high, backward low (should be vetoed)
    predictor = NativePredictor(detector_weights=[], edge_weights=[])
    predictor.edge_scorers = [MockEdgeModel(0.99, 0.01)]
    
    # Dummy mock feature vectors
    src_feats, tgt_feats = np.zeros((2, 10)), np.zeros((1, 10))
    probs = predictor.predict_bidirectional_edges(src_feats, tgt_feats)
    
    # Harmonic mean of 0.99 and 0.01 is highly penalized (~0.0198)
    assert probs[0, 0] < 0.05
    
    # Case B: Both high (should pass)
    predictor.edge_scorers = [MockEdgeModel(0.95, 0.95)]
    probs_pass = predictor.predict_bidirectional_edges(src_feats, tgt_feats)
    assert probs_pass[0, 0] > 0.90


def test_ensemble_logits_averaging() -> None:
    class MockModelA:
        def predict_heatmaps_and_features(self, frames):
            return frames * 2.0, frames
            
    class MockModelB:
        def predict_heatmaps_and_features(self, frames):
            return frames * 4.0, frames

    predictor = NativePredictor(detector_weights=[], edge_weights=[])
    predictor.detectors = [MockModelA(), MockModelB()]
    
    image = np.ones((1, 1, 4, 8, 8), dtype=float)
    # Average across models A (weight 2.0) and B (weight 4.0) should be 3.0
    mean_heatmap, _ = predictor.predict_heatmaps_and_features(image, use_tta=False)
    assert np.allclose(mean_heatmap, 3.0)


