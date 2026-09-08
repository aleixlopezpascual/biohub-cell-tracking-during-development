"""Baseline 3D U-Net model.

Defines a lightweight, torch-optional interface. ``UNet3DConfig`` and the
class itself can always be imported (so type hints, configs, and non-torch
code paths work without torch installed); building or running the actual
network requires the optional ``torch`` extra and raises a clear
``ImportError`` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    import torch.nn as nn


@dataclass(frozen=True)
class UNet3DConfig:
    """Hyperparameters for the baseline 3D U-Net."""

    in_channels: int = 1
    out_channels: int = 1
    base_features: int = 32
    depth: int = 4
    norm: str = "instance"

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.norm not in ("batch", "instance", "none"):
            raise ValueError(f"unknown norm {self.norm!r}")


def _require_torch() -> Any:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "UNet3D requires the optional 'torch' extra: "
            "install with `pip install biohub-cell-tracking[torch]`."
        ) from exc
    return torch


class UNet3D:
    """Baseline encoder-decoder 3D U-Net for voxel-wise prediction.

    Produces per-voxel outputs (e.g. cell-center heatmaps and/or semantic
    masks) consumed by the tracking stage. This is intentionally a thin,
    dependency-isolated wrapper: the actual ``torch.nn.Module`` is built lazily
    inside :meth:`build` so importing this module never requires torch.
    """

    def __init__(self, config: UNet3DConfig) -> None:
        self.config = config
        self._module: "nn.Module | None" = None

    def build(self) -> "nn.Module":
        """Construct (or return the cached) underlying ``torch.nn.Module``."""
        if self._module is not None:
            return self._module
        torch = _require_torch()
        import torch.nn as nn

        cfg = self.config

        def norm_layer(channels: int) -> "nn.Module":
            if cfg.norm == "batch":
                return nn.BatchNorm3d(channels)
            if cfg.norm == "instance":
                return nn.InstanceNorm3d(channels, affine=True)
            return nn.Identity()

        def conv_block(in_ch: int, out_ch: int) -> "nn.Module":
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                norm_layer(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                norm_layer(out_ch),
                nn.ReLU(inplace=True),
            )

        class _UNet3DModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                feats = [cfg.base_features * (2**i) for i in range(cfg.depth)]
                self.encoders = nn.ModuleList()
                self.pools = nn.ModuleList()
                prev_ch = cfg.in_channels
                for f in feats:
                    self.encoders.append(conv_block(prev_ch, f))
                    self.pools.append(nn.MaxPool3d(2))
                    prev_ch = f
                self.bottleneck = conv_block(prev_ch, prev_ch * 2)
                self.upsamples = nn.ModuleList()
                self.decoders = nn.ModuleList()
                dec_in = prev_ch * 2
                for f in reversed(feats):
                    self.upsamples.append(nn.ConvTranspose3d(dec_in, f, kernel_size=2, stride=2))
                    self.decoders.append(conv_block(f * 2, f))
                    dec_in = f
                self.head = nn.Conv3d(feats[0], cfg.out_channels, kernel_size=1)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                skips = []
                h = x
                for enc, pool in zip(self.encoders, self.pools):
                    h = enc(h)
                    skips.append(h)
                    h = pool(h)
                h = self.bottleneck(h)
                for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
                    h = up(h)
                    h = torch.cat([h, skip], dim=1)
                    h = dec(h)
                return self.head(h)

        self._module = _UNet3DModule()
        return self._module

    def predict(self, patch: np.ndarray) -> np.ndarray:
        """Run inference on a single ``(C, Z, Y, X)`` NumPy patch.

        Returns an ``(out_channels, Z, Y, X)`` NumPy array. Adds/removes the
        batch dimension internally and runs under ``torch.no_grad()``.
        """
        torch = _require_torch()
        module = self.build()
        module.eval()
        tensor = torch.as_tensor(patch, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = module(tensor)
        return output.squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        """Persist model weights (state dict) + config to ``path``."""
        torch = _require_torch()
        module = self.build()
        torch.save({"state_dict": module.state_dict(), "config": self.config.__dict__}, path)

    @classmethod
    def load(cls, path: str) -> "UNet3D":
        """Load a model previously saved with :meth:`save`."""
        torch = _require_torch()
        checkpoint = torch.load(path, map_location="cpu")
        config = UNet3DConfig(**checkpoint["config"])
        model = cls(config)
        module = model.build()
        module.load_state_dict(checkpoint["state_dict"])
        return model
