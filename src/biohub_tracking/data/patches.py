"""Deterministic 3D patch grids and frame-pair iteration for training/inference.

Both :class:`PatchGrid3D` and :class:`FramePairDataset` only compute integer
offsets up front; actual voxel data is pulled lazily from an
:class:`~biohub_tracking.data.ome_zarr.OMEZarrVolume` only when a patch is
materialized, keeping memory usage independent of dataset size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from biohub_tracking.data.ome_zarr import OMEZarrVolume


@dataclass(frozen=True)
class PatchSpec:
    """A single 3D patch location within one frame."""

    t: int
    z0: int
    y0: int
    x0: int
    size: tuple[int, int, int]

    @property
    def z_slice(self) -> slice:
        return slice(self.z0, self.z0 + self.size[0])

    @property
    def y_slice(self) -> slice:
        return slice(self.y0, self.y0 + self.size[1])

    @property
    def x_slice(self) -> slice:
        return slice(self.x0, self.x0 + self.size[2])


@dataclass
class PatchGrid3D:
    """Deterministic grid of 3D patch offsets covering a spatial shape.

    Patches that would run past the volume boundary are shifted back
    (clamped) so every patch has exactly ``patch_size``, matching common
    3D-U-Net training setups without needing padding.
    """

    spatial_shape: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    stride: tuple[int, int, int] | None = None
    _offsets: list[tuple[int, int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for dim, size in zip(self.spatial_shape, self.patch_size):
            if size > dim:
                raise ValueError(f"patch_size {self.patch_size} exceeds spatial_shape {self.spatial_shape}")
        stride = self.stride or self.patch_size
        self._offsets = []
        starts_per_axis = []
        for dim_size, patch_len, step in zip(self.spatial_shape, self.patch_size, stride):
            if step <= 0:
                raise ValueError("stride must be positive")
            starts = list(range(0, dim_size - patch_len + 1, step))
            last_start = dim_size - patch_len
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            starts_per_axis.append(starts)
        for z0 in starts_per_axis[0]:
            for y0 in starts_per_axis[1]:
                for x0 in starts_per_axis[2]:
                    self._offsets.append((z0, y0, x0))

    def __len__(self) -> int:
        return len(self._offsets)

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        return iter(self._offsets)

    def offsets(self) -> list[tuple[int, int, int]]:
        """Return all (z0, y0, x0) patch origins, in deterministic order."""
        return list(self._offsets)


class FramePairDataset(Sequence):
    """Deterministic, memory-efficient index over 3D patch/frame-pair samples.

    Iterates consecutive (or strided) frame pairs ``(t, t + frame_stride)``
    of a volume, and further tiles each pair into 3D patches according to a
    :class:`PatchGrid3D`. Implements :class:`collections.abc.Sequence`
    (``__len__``/``__getitem__``) so it can be wrapped by any DataLoader
    (including torch's) without requiring torch as a dependency here.
    """

    def __init__(
        self,
        volume: OMEZarrVolume,
        patch_size: tuple[int, int, int],
        patch_stride: tuple[int, int, int] | None = None,
        frame_stride: int = 1,
        channels: Sequence[int] | None = None,
    ) -> None:
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if volume.num_frames < frame_stride + 1:
            raise ValueError(
                f"volume has {volume.num_frames} frames, need at least {frame_stride + 1} "
                f"for a frame_stride={frame_stride} pair"
            )
        self.volume = volume
        self.channels = list(channels) if channels is not None else list(range(volume.num_channels))
        self.frame_stride = frame_stride
        self.grid = PatchGrid3D(volume.spatial_shape, patch_size, patch_stride)

        self._frame_pairs = [(t, t + frame_stride) for t in range(volume.num_frames - frame_stride)]
        self._index: list[tuple[int, int, tuple[int, int, int]]] = [
            (t0, t1, offset)
            for (t0, t1) in self._frame_pairs
            for offset in self.grid.offsets()
        ]

    def __len__(self) -> int:
        return len(self._index)

    def spec_at(self, index: int) -> tuple[PatchSpec, PatchSpec]:
        """Return the (frame_a, frame_b) :class:`PatchSpec` pair for ``index``, without reading data."""
        t0, t1, (z0, y0, x0) = self._index[index]
        size = self.grid.patch_size
        return (
            PatchSpec(t=t0, z0=z0, y0=y0, x0=x0, size=size),
            PatchSpec(t=t1, z0=z0, y0=y0, x0=x0, size=size),
        )

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        """Materialize one patch/frame-pair sample: reads only this patch's voxels."""
        spec_a, spec_b = self.spec_at(index)
        patch_a = self.volume.get_patch(spec_a.t, spec_a.z_slice, spec_a.y_slice, spec_a.x_slice, self.channels)
        patch_b = self.volume.get_patch(spec_b.t, spec_b.z_slice, spec_b.y_slice, spec_b.x_slice, self.channels)
        return {
            "frame_a": patch_a,
            "frame_b": patch_b,
            "t0": spec_a.t,
            "t1": spec_b.t,
            "z0": spec_a.z0,
            "y0": spec_a.y0,
            "x0": spec_a.x0,
        }
