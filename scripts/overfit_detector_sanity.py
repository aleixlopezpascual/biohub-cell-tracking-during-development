#!/usr/bin/env python3
"""Overfit and verify 3D U-Net center detection on a small synthetic batch.

This serves as a diagnostic/sanity test to verify target generation, Gaussian
PU loss, peak extraction, and coordinate transformation alignment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure our local packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biohub_tracking.detection import LocalMaximaDetector, LocalMaximaDetectorConfig
from biohub_tracking.models.unet3d import UNet3D, UNet3DConfig
from biohub_tracking.training.targets import make_torch_gaussian_detection_loss
from biohub_tracking.utils import get_logger

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150, help="Number of overfitting epochs.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args(argv)


def run_sanity_check(epochs: int = 150, lr: float = 0.005, seed: int = 42) -> bool:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Setup anisotropic voxel size and sigmas
    voxel_size_um = (1.625, 0.40625, 0.40625)
    sigma_um = (2.0, 1.0, 1.0)

    # Batch of 2 samples, 1 channel, shape: (B, C, Z, Y, X)
    batch_size = 2
    shape = (32, 64, 64)  # (Z, Y, X)
    
    # 2. Define fixed ground-truth centroids (in voxel coordinates)
    # Sample 0 has 2 cells, Sample 1 has 1 cell
    gt_coords_list = [
        [(8.0, 16.0, 20.0), (20.0, 45.0, 30.0)],
        [(15.0, 32.0, 32.0)]
    ]
    
    # Pack into tensors
    max_cells = max(len(c) for c in gt_coords_list)
    gt_coords = torch.zeros((batch_size, max_cells, 3), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_cells), dtype=torch.bool)
    
    for i, coords in enumerate(gt_coords_list):
        for j, coord in enumerate(coords):
            gt_coords[i, j] = torch.as_tensor(coord)
            mask[i, j] = True

    # 3. Render synthetic input volumes
    # Put bright Gaussian profiles with noise at coordinate positions
    inputs = torch.zeros((batch_size, 1, *shape), dtype=torch.float32)
    for b in range(batch_size):
        for j in range(int(mask[b].sum().item())):
            coord = gt_coords[b, j].cpu().numpy()
            for z in range(shape[0]):
                for y in range(shape[1]):
                    for x in range(shape[2]):
                        dist_sq = (
                            ((z - coord[0]) * voxel_size_um[0])**2 +
                            ((y - coord[1]) * voxel_size_um[1])**2 +
                            ((x - coord[2]) * voxel_size_um[2])**2
                        )
                        inputs[b, 0, z, y, x] += np.exp(-dist_sq / 2.0) * 10.0
                        
    # Add a bit of background noise
    inputs += torch.randn_like(inputs) * 0.1

    # 4. Instantiate Model
    config = UNet3DConfig(in_channels=1, out_channels=1, base_features=16, depth=3, norm="instance")
    model_wrapper = UNet3D(config)
    model = model_wrapper.build()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_torch_gaussian_detection_loss(voxel_size_um=voxel_size_um, sigma_um=sigma_um)

    # 5. Overfit loop
    logger.info("Starting detector overfitting sanity check for %d epochs...", epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, gt_coords, mask, neg_weight=0.01)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("epoch=%03d loss=%.6f", epoch + 1, loss.item())

    # 6. Extraction & Validation
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        # Apply sigmoid to get heatmaps (probabilities)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Extract peaks from the overfitted heatmaps using LocalMaximaDetector
    detector_config = LocalMaximaDetectorConfig(
        threshold=0.5,
        min_distance=2.0,
        voxel_size_um=voxel_size_um,
        use_subpixel_refinement=False,
    )
    detector = LocalMaximaDetector(detector_config)

    all_success = True
    for b in range(batch_size):
        heatmap = probs[b, 0]
        detected = detector.detect(heatmap)
        expected = gt_coords_list[b]
        
        logger.info("Sample %d: expected %d peaks, detected %d peaks", b, len(expected), len(detected))
        logger.info("Expected: %s", expected)
        logger.info("Detected: %s", detected)
        
        # Verify that each expected coordinate has a corresponding detected coordinate within 1.5 voxels
        sample_success = True
        for exp in expected:
            found = False
            for det in detected:
                # Euclidean distance in voxel coordinates
                dist = np.linalg.norm(np.asarray(exp) - np.asarray(det))
                if dist <= 1.5:
                    found = True
                    break
            if not found:
                logger.error("Failed to recover peak near expected coord %s", exp)
                sample_success = False
                all_success = False
        
        if sample_success:
            logger.info("Sample %d successfully recovered all peaks!", b)
            
    return all_success


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        import torch
    except ImportError:
        logger.warning("Torch is not installed. Sanity check skipped.")
        sys.exit(0)
        
    success = run_sanity_check(epochs=args.epochs, lr=args.lr, seed=args.seed)
    if success:
        logger.info("DETECTOR SANITY CHECK: PASSED (Near-complete peak recovery achieved)")
        sys.exit(0)
    else:
        logger.error("DETECTOR SANITY CHECK: FAILED (Peaks not fully recovered)")
        sys.exit(1)


if __name__ == "__main__":
    main()
