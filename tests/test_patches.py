"""Tests for 3D patch grids and frame-pair iteration."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.data.ome_zarr import OMEZarrVolume
from biohub_tracking.data.patches import FramePairDataset, PatchGrid3D


def test_patch_grid_non_overlapping_exact_multiple() -> None:
    grid = PatchGrid3D(spatial_shape=(8, 8, 8), patch_size=(4, 4, 4))
    offsets = grid.offsets()
    assert len(offsets) == 8  # 2 * 2 * 2
    assert (0, 0, 0) in offsets
    assert (4, 4, 4) in offsets


def test_patch_grid_clamps_last_patch_to_stay_in_bounds() -> None:
    grid = PatchGrid3D(spatial_shape=(10, 10, 10), patch_size=(4, 4, 4), stride=(4, 4, 4))
    for z0, y0, x0 in grid.offsets():
        assert z0 + 4 <= 10 and y0 + 4 <= 10 and x0 + 4 <= 10
    # last start along each axis must reach exactly dim - patch_len (no gap at the boundary)
    zs = sorted({z for z, _, _ in grid.offsets()})
    assert zs[-1] == 6


def test_patch_grid_rejects_patch_larger_than_volume() -> None:
    with pytest.raises(ValueError):
        PatchGrid3D(spatial_shape=(4, 4, 4), patch_size=(8, 4, 4))


def test_frame_pair_dataset_length_and_sample_shapes() -> None:
    arr = np.arange(3 * 1 * 8 * 8 * 8).reshape(3, 1, 8, 8, 8).astype(np.float32)
    volume = OMEZarrVolume(arr)
    dataset = FramePairDataset(volume, patch_size=(4, 4, 4))
    # 2 frame pairs (0,1) and (1,2), 8 patches each => 16 samples
    assert len(dataset) == 16

    sample = dataset[0]
    assert sample["frame_a"].shape == (1, 4, 4, 4)
    assert sample["frame_b"].shape == (1, 4, 4, 4)
    assert sample["t0"] == 0 and sample["t1"] == 1


def test_frame_pair_dataset_requires_enough_frames() -> None:
    arr = np.zeros((1, 1, 8, 8, 8))
    volume = OMEZarrVolume(arr)
    with pytest.raises(ValueError):
        FramePairDataset(volume, patch_size=(4, 4, 4), frame_stride=1)


def test_frame_pair_dataset_spec_at_matches_getitem_offsets() -> None:
    arr = np.zeros((2, 1, 8, 8, 8))
    volume = OMEZarrVolume(arr)
    dataset = FramePairDataset(volume, patch_size=(4, 4, 4))
    spec_a, spec_b = dataset.spec_at(0)
    sample = dataset[0]
    assert sample["z0"] == spec_a.z0 and sample["y0"] == spec_a.y0 and sample["x0"] == spec_a.x0
    assert spec_a.t == 0 and spec_b.t == 1
