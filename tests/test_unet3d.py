"""Tests for the UNet3D model interface (torch-optional)."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.models.unet3d import UNet3D, UNet3DConfig

torch = pytest.importorskip("torch")


def test_unet3d_forward_pass_preserves_spatial_shape() -> None:
    config = UNet3DConfig(in_channels=1, out_channels=1, base_features=4, depth=2, norm="none")
    model = UNet3D(config)
    patch = np.random.rand(1, 16, 16, 16).astype(np.float32)
    output = model.predict(patch)
    assert output.shape == (1, 16, 16, 16)


def test_unet3d_config_rejects_invalid_norm() -> None:
    with pytest.raises(ValueError):
        UNet3DConfig(norm="bogus")


def test_unet3d_save_and_load_roundtrip(tmp_path) -> None:
    config = UNet3DConfig(in_channels=1, out_channels=1, base_features=4, depth=2, norm="none")
    model = UNet3D(config)
    model.build()
    path = tmp_path / "model.pt"
    model.save(str(path))

    loaded = UNet3D.load(str(path))
    patch = np.random.rand(1, 16, 16, 16).astype(np.float32)
    out1 = model.predict(patch)
    out2 = loaded.predict(patch)
    np.testing.assert_allclose(out1, out2, atol=1e-5)
