"""Lazy OME-Zarr volume reader.

The reader never materializes more than the requested slice in memory: all
indexing operations forward directly to the backing chunked array (a real
``zarr`` array when available, or any NumPy-like array for testing), so
callers can stream arbitrarily large 5D (T, C, Z, Y, X) volumes patch by
patch or frame by frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel


@runtime_checkable
class ArrayLike(Protocol):
    """Minimal structural type satisfied by numpy, zarr, and dask arrays."""

    shape: tuple[int, ...]
    dtype: Any

    def __getitem__(self, key: Any) -> Any: ...


class VolumeMetadata(BaseModel):
    """Serializable description of an opened volume."""

    shape: tuple[int, int, int, int, int]
    voxel_size_um: tuple[float, float, float]
    dtype: str
    axes: str = "tczyx"
    channel_names: list[str] | None = None


def _to_slice(spec: slice | tuple[int, int] | int) -> slice:
    """Normalize a (start, size) tuple, bare int, or slice into a ``slice``."""
    if isinstance(spec, slice):
        return spec
    if isinstance(spec, int):
        return slice(spec, spec + 1)
    start, size = spec
    return slice(start, start + size)


def _validate_index(index: int, limit: int, label: str) -> None:
    if not (0 <= index < limit):
        raise IndexError(f"{label} {index} out of range [0, {limit})")


def _resolve_ome_zarr_level(store: Any, resolution_level: int) -> tuple[Any, tuple[float, float, float] | None]:
    """Best-effort extraction of an array + voxel size from an OME-Zarr group.

    Falls back gracefully (returning ``None`` voxel size) when multiscale
    metadata is missing or in an unexpected shape, since OME-Zarr producers
    vary in strictness.
    """
    attrs = dict(getattr(store, "attrs", {}) or {})
    multiscales = attrs.get("multiscales")
    voxel_size_um: tuple[float, float, float] | None = None
    if multiscales:
        try:
            datasets = multiscales[0]["datasets"]
            path = datasets[resolution_level]["path"]
            array = store[path]
            axes = multiscales[0].get("axes", [])
            transforms = datasets[resolution_level].get("coordinateTransformations", [])
            scale = next((t["scale"] for t in transforms if t.get("type") == "scale"), None)
            if scale is not None and axes:
                axis_names = [a.get("name", "") for a in axes]
                spatial = {name: value for name, value in zip(axis_names, scale) if name in ("z", "y", "x")}
                if all(k in spatial for k in ("z", "y", "x")):
                    voxel_size_um = (spatial["z"], spatial["y"], spatial["x"])
            return array, voxel_size_um
        except (KeyError, IndexError, TypeError):
            pass
    # Fall back to a plain integer-keyed group (e.g. "0", "1", ...).
    array = store[str(resolution_level)]
    return array, voxel_size_um


class OMEZarrVolume:
    """Lazy reader over a 5D ``(T, C, Z, Y, X)`` OME-Zarr-compatible array.

    All read methods slice the backing array directly rather than loading
    the whole volume, so memory usage stays proportional to the requested
    region and not to the dataset size.
    """

    def __init__(
        self,
        array: ArrayLike,
        voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
        channel_names: list[str] | None = None,
    ) -> None:
        if len(array.shape) != 5:
            raise ValueError(
                "OMEZarrVolume expects a 5D (T, C, Z, Y, X) array; "
                f"got shape={tuple(array.shape)}. Use `from_array` to auto-expand lower-rank arrays."
            )
        self._array = array
        self.voxel_size_um = voxel_size_um
        self.channel_names = channel_names

    @classmethod
    def open(
        cls,
        path: str | Path,
        resolution_level: int = 0,
        voxel_size_um: tuple[float, float, float] | None = None,
        channel_names: list[str] | None = None,
    ) -> "OMEZarrVolume":
        """Open a real OME-Zarr store on disk or object storage.

        Requires the optional ``zarr`` extra (``pip install
        biohub-cell-tracking[zarr]``); this import is deferred so the core
        package has no hard dependency on it.
        """
        try:
            import zarr
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "Reading OME-Zarr stores requires the optional 'zarr' extra: "
                "install with `pip install biohub-cell-tracking[zarr]`."
            ) from exc

        store = zarr.open(str(path), mode="r")
        array, resolved_voxel_size = _resolve_ome_zarr_level(store, resolution_level)
        resolved = voxel_size_um or resolved_voxel_size or (1.0, 1.0, 1.0)
        return cls(array, voxel_size_um=resolved, channel_names=channel_names)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
        channel_names: list[str] | None = None,
    ) -> "OMEZarrVolume":
        """Build a volume from an in-memory array, auto-expanding to 5D.

        Accepts 3D ``(Z, Y, X)``, 4D ``(T, Z, Y, X)``, or 5D ``(T, C, Z, Y, X)``
        arrays. Primarily used for synthetic tests and small in-memory data.
        """
        arr = np.asarray(array)
        if arr.ndim == 3:
            arr = arr[np.newaxis, np.newaxis]
        elif arr.ndim == 4:
            arr = arr[:, np.newaxis]
        elif arr.ndim != 5:
            raise ValueError(f"expected a 3D, 4D, or 5D array, got ndim={arr.ndim}")
        return cls(arr, voxel_size_um=voxel_size_um, channel_names=channel_names)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return tuple(self._array.shape)  # type: ignore[return-value]

    @property
    def num_frames(self) -> int:
        return int(self._array.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self._array.shape[1])

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return tuple(self._array.shape[2:])  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._array.dtype)

    def metadata(self) -> VolumeMetadata:
        return VolumeMetadata(
            shape=self.shape,
            voxel_size_um=self.voxel_size_um,
            dtype=str(self.dtype),
            channel_names=self.channel_names,
        )

    def get_frame(self, t: int, channels: Sequence[int] | None = None) -> np.ndarray:
        """Read one full spatial frame ``(len(channels), Z, Y, X)`` at time ``t``."""
        _validate_index(t, self.num_frames, "frame index")
        chans = list(channels) if channels is not None else list(range(self.num_channels))
        return np.asarray(self._array[t, chans])

    def get_patch(
        self,
        t: int,
        z: slice | tuple[int, int],
        y: slice | tuple[int, int],
        x: slice | tuple[int, int],
        channels: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Read a single 3D patch ``(len(channels), pz, py, px)`` at time ``t``.

        ``z``, ``y``, ``x`` may be ``slice`` objects or ``(start, size)`` tuples.
        Only the requested sub-region is read from the backing store.
        """
        _validate_index(t, self.num_frames, "frame index")
        chans = list(channels) if channels is not None else list(range(self.num_channels))
        z_sl, y_sl, x_sl = _to_slice(z), _to_slice(y), _to_slice(x)
        return np.asarray(self._array[t, chans, z_sl, y_sl, x_sl])

    def iter_frames(self, channels: Sequence[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(t, frame)`` lazily, one time step at a time."""
        for t in range(self.num_frames):
            yield t, self.get_frame(t, channels=channels)

    def to_numpy(self) -> np.ndarray:
        """Materialize the *entire* volume into memory. Use only for small data."""
        return np.asarray(self._array[...])
