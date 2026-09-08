"""Tests for the lazy OME-Zarr volume reader."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.data.ome_zarr import OMEZarrVolume


def test_from_array_expands_3d_to_5d() -> None:
    arr = np.zeros((4, 8, 8), dtype=np.uint16)
    volume = OMEZarrVolume.from_array(arr)
    assert volume.shape == (1, 1, 4, 8, 8)
    assert volume.num_frames == 1
    assert volume.num_channels == 1
    assert volume.spatial_shape == (4, 8, 8)


def test_from_array_expands_4d_to_5d() -> None:
    arr = np.zeros((3, 4, 8, 8), dtype=np.float32)
    volume = OMEZarrVolume.from_array(arr)
    assert volume.shape == (3, 1, 4, 8, 8)
    assert volume.num_frames == 3


def test_get_frame_and_get_patch_return_expected_shapes() -> None:
    arr = np.arange(2 * 2 * 6 * 10 * 10).reshape(2, 2, 6, 10, 10).astype(np.float32)
    volume = OMEZarrVolume.from_array(arr, voxel_size_um=(2.0, 0.5, 0.5))
    frame = volume.get_frame(0)
    assert frame.shape == (2, 6, 10, 10)

    patch = volume.get_patch(1, z=(0, 4), y=(2, 5), x=(1, 3), channels=[0])
    assert patch.shape == (1, 4, 5, 3)
    np.testing.assert_array_equal(patch[0], arr[1, 0, 0:4, 2:7, 1:4])


def test_get_frame_out_of_range_raises() -> None:
    volume = OMEZarrVolume.from_array(np.zeros((2, 4, 4, 4)))
    with pytest.raises(IndexError):
        volume.get_frame(5)


def test_iter_frames_is_lazy_and_covers_all_frames() -> None:
    arr = np.zeros((3, 1, 2, 2, 2))
    volume = OMEZarrVolume.from_array(arr)
    frames = list(volume.iter_frames())
    assert [t for t, _ in frames] == [0, 1, 2]


def test_rejects_wrong_rank_backing_array() -> None:
    with pytest.raises(ValueError):
        OMEZarrVolume(np.zeros((2, 2, 2)))  # 3D array passed directly, not via from_array


def test_open_without_zarr_extra_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "zarr":
            raise ImportError("no zarr")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="zarr"):
        OMEZarrVolume.open("nonexistent.zarr")
