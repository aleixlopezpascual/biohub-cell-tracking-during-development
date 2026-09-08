"""Biohub Cell Tracking During Development.

A modular, typed pipeline for the Kaggle "Biohub - Cell Tracking During
Development" competition: OME-Zarr data loading, 3D patch/frame-pair
iteration, a baseline 3D U-Net interface, Hungarian frame-to-frame tracking
with division branching, the official Biohub tracking metrics, and a
deterministic submission exporter.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
