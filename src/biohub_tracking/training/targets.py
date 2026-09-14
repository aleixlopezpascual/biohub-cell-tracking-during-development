"""Physical-unit center targets suitable for sparse cell annotations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _positive_triplet(values: tuple[float, float, float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{name} must contain three positive finite values")
    return array


def render_gaussian_center_target(
    shape: tuple[int, int, int],
    centers_zyx: np.ndarray,
    *,
    voxel_size_um: tuple[float, float, float],
    sigma_um: tuple[float, float, float] = (2.0, 1.0, 1.0),
    truncate: float = 3.0,
) -> np.ndarray:
    """Render max-composed anisotropic Gaussian centers in voxel space.

    ``centers_zyx`` may contain continuous voxel coordinates. Gaussian widths
    are specified in microns so the supervision remains physically consistent
    across Z/XY anisotropy and spatial downsampling.
    """
    if len(shape) != 3 or any(int(size) != size or size < 1 for size in shape):
        raise ValueError("shape must contain three positive integers")
    if not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("truncate must be positive and finite")
    scale = _positive_triplet(voxel_size_um, "voxel_size_um")
    sigma_voxel = _positive_triplet(sigma_um, "sigma_um") / scale
    centers = np.asarray(centers_zyx, dtype=float)
    if centers.size == 0:
        centers = centers.reshape(0, 3)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_zyx must have shape (N, 3)")
    if not np.isfinite(centers).all():
        raise ValueError("centers_zyx must contain finite coordinates")

    target = np.zeros(tuple(int(size) for size in shape), dtype=np.float32)
    radii = np.ceil(truncate * sigma_voxel).astype(int)
    for center in centers:
        lower = np.maximum(0, np.floor(center - radii).astype(int))
        upper = np.minimum(np.asarray(shape), np.ceil(center + radii).astype(int) + 1)
        if np.any(lower >= upper):
            continue
        axes = [np.arange(lo, hi, dtype=float) for lo, hi in zip(lower, upper)]
        zz, yy, xx = np.meshgrid(*axes, indexing="ij")
        squared = (
            ((zz - center[0]) / sigma_voxel[0]) ** 2
            + ((yy - center[1]) / sigma_voxel[1]) ** 2
            + ((xx - center[2]) / sigma_voxel[2]) ** 2
        )
        patch = np.exp(-0.5 * squared).astype(np.float32)
        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        np.maximum(target[slices], patch, out=target[slices])
    return target


def positive_unlabeled_weights(
    target: np.ndarray,
    *,
    negative_weight: float = 0.01,
    positive_threshold: float = 1e-6,
) -> np.ndarray:
    """Return loss weights that avoid treating all unlabeled voxels as hard negatives."""
    values = np.asarray(target, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("target values must be finite and in [0, 1]")
    if not 0 <= negative_weight <= 1:
        raise ValueError("negative_weight must be in [0, 1]")
    if not 0 <= positive_threshold <= 1:
        raise ValueError("positive_threshold must be in [0, 1]")
    return np.where(values > positive_threshold, 1.0, negative_weight).astype(np.float32)


def make_torch_gaussian_detection_loss(
    *,
    voxel_size_um: tuple[float, float, float],
    sigma_um: tuple[float, float, float],
    truncate: float = 3.0,
) -> Callable[[Any, Any, Any, float], Any]:
    """Build an official-trainer-compatible Gaussian PU detection loss.

    Torch is imported only when this factory is used inside the optional
    Kaggle training runtime. Coordinates are expected in the downsampled input
    grid and ``voxel_size_um`` must therefore describe that grid.
    """
    physical_scale = _positive_triplet(voxel_size_um, "voxel_size_um")
    sigma_voxel_np = _positive_triplet(sigma_um, "sigma_um") / physical_scale
    if truncate <= 0 or not np.isfinite(truncate):
        raise ValueError("truncate must be positive and finite")
    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:
        raise ImportError(
            "Gaussian tensor loss requires the optional torch extra: "
            "install with `pip install -e '.[torch]'`."
        ) from exc

    def loss_fn(det_logits: Any, coords: Any, mask: Any, neg_weight: float = 0.01) -> Any:
        if not 0 <= neg_weight <= 1:
            raise ValueError("neg_weight must be in [0, 1]")
        if det_logits.ndim != 5 or det_logits.shape[1] != 1:
            raise ValueError("det_logits must have shape (B, 1, Z, Y, X)")
        batch_size = det_logits.shape[0]
        spatial = tuple(int(value) for value in det_logits.shape[2:])
        target = torch.zeros_like(det_logits[:, 0])
        sigma_voxel = torch.as_tensor(
            sigma_voxel_np, dtype=det_logits.dtype, device=det_logits.device
        )
        radii = torch.ceil(float(truncate) * sigma_voxel).to(dtype=torch.long)
        for batch_index in range(batch_size):
            count = int(mask[batch_index].sum().item())
            for center in coords[batch_index, :count]:
                lower = torch.maximum(
                    torch.zeros(3, dtype=torch.long, device=center.device),
                    torch.floor(center - radii).to(dtype=torch.long),
                )
                upper = torch.minimum(
                    torch.as_tensor(spatial, dtype=torch.long, device=center.device),
                    torch.ceil(center + radii).to(dtype=torch.long) + 1,
                )
                if bool(torch.any(lower >= upper)):
                    continue
                axes = [
                    torch.arange(
                        lower[axis],
                        upper[axis],
                        device=center.device,
                        dtype=det_logits.dtype,
                    )
                    for axis in range(3)
                ]
                zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
                squared = (
                    ((zz - center[0]) / sigma_voxel[0]) ** 2
                    + ((yy - center[1]) / sigma_voxel[1]) ** 2
                    + ((xx - center[2]) / sigma_voxel[2]) ** 2
                )
                gaussian = torch.exp(-0.5 * squared)
                slices = tuple(
                    slice(int(lower[axis].item()), int(upper[axis].item())) for axis in range(3)
                )
                target[batch_index][slices] = torch.maximum(
                    target[batch_index][slices], gaussian
                )
        losses = torch_functional.binary_cross_entropy_with_logits(
            det_logits[:, 0], target, reduction="none"
        )
        positive = target > 1e-6
        positive_count = positive.reshape(batch_size, -1).sum(dim=1).clamp(min=1)
        negative_count = (~positive).reshape(batch_size, -1).sum(dim=1).clamp(min=1)
        shape = (batch_size,) + (1,) * len(spatial)
        weights = torch.where(
            positive,
            (1.0 / positive_count).reshape(shape),
            (float(neg_weight) / negative_count).reshape(shape),
        )
        return (losses * weights).reshape(batch_size, -1).sum(dim=1).mean()

    return loss_fn
