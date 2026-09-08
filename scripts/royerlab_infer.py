"""Preflight entry point for the optional Royerlab competitive baseline.

This script intentionally validates the external Kaggle support-pack artifacts
before inference.  The portable NumPy stages live in
``biohub_tracking.baselines.royerlab``; TemporalUNet3D, SimpleNodeTransformer,
tracksdata, and GEFF remain external because they are not core dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biohub_tracking.baselines.royerlab.adapters import (
    SimpleNodeTransformerAdapter,
    TemporalUNet3DAdapter,
    TracksdataGEFFAdapter,
)
from biohub_tracking.utils import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse artifact locations needed by a concrete support-pack integration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Royerlab-enabled PipelineConfig YAML.")
    parser.add_argument("--temporal-checkpoint", required=True, type=Path)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--require-ilp", action="store_true", help="Also validate tracksdata and GEFF.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Validate configuration and optional artifacts, then explain integration status."""
    args = parse_args(argv)
    config = load_config(args.config)
    if not config.royerlab.enabled:
        raise ValueError("config.royerlab.enabled must be true for royerlab_infer.py")
    TemporalUNet3DAdapter().load(args.temporal_checkpoint)
    SimpleNodeTransformerAdapter().load(args.transformer_checkpoint)
    if args.require_ilp:
        TracksdataGEFFAdapter().require_runtime()


if __name__ == "__main__":
    main()
