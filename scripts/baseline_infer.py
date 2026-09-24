"""Run model-free 3D blob detection and tracking on an OME-Zarr volume.

Example::

    python scripts/baseline_infer.py --input data/test/testA.zarr --dataset testA \
        --output submission.csv --threshold 100 --min-distance 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biohub_tracking.data import OMEZarrVolume
from biohub_tracking.pipeline import BaselinePipelineConfig, infer_volume
from biohub_tracking.submission import export_submission
from biohub_tracking.tracking import TrackerConfig
from biohub_tracking.utils import get_logger

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the reproducible baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="OME-Zarr store path.")
    parser.add_argument("--dataset", required=True, help="Dataset value in the submission CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Destination submission CSV.")
    parser.add_argument("--resolution-level", type=int, default=0, help="OME-Zarr pyramid level.")
    parser.add_argument("--channel", type=int, default=0, help="Image channel to detect.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum blob intensity.")
    parser.add_argument("--min-distance", type=float, default=3.0, help="Blob separation in microns.")
    parser.add_argument(
        "--voxel-size-um",
        type=float,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Override OME-Zarr voxel size in microns.",
    )
    parser.add_argument("--max-link-distance-um", type=float, default=15.0)
    parser.add_argument("--division-search-radius-um", type=float, default=20.0)
    parser.add_argument("--max-daughters", type=int, default=2)
    parser.add_argument(
        "--use-subpixel-refinement",
        action="store_true",
        help="Enable center-of-mass centroid refinement.",
    )
    parser.add_argument(
        "--use-dog",
        action="store_true",
        help="Enable Difference-of-Gaussians (DoG) peak detection.",
    )
    parser.add_argument(
        "--dog-sigmas",
        type=float,
        nargs="+",
        help="Optional physical sigmas in microns for multi-scale DoG.",
    )
    parser.add_argument(
        "--dog-ratio",
        type=float,
        default=1.6,
        help="Sigma ratio for DoG calculation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load a volume, infer a graph, and write the strict submission CSV."""
    args = parse_args(argv)
    voxel_size = tuple(args.voxel_size_um) if args.voxel_size_um is not None else None
    volume = OMEZarrVolume.open(
        args.input, resolution_level=args.resolution_level, voxel_size_um=voxel_size
    )
    graph = infer_volume(
        volume,
        BaselinePipelineConfig(
            threshold=args.threshold,
            min_distance=args.min_distance,
            channel=args.channel,
            voxel_size_um=voxel_size,
            tracker=TrackerConfig(
                max_link_distance_um=args.max_link_distance_um,
                division_search_radius_um=args.division_search_radius_um,
                max_daughters=args.max_daughters,
            ),
            use_dog=args.use_dog,
            dog_sigmas=args.dog_sigmas,
            dog_ratio=args.dog_ratio,
            use_subpixel_refinement=args.use_subpixel_refinement,
        ),
    )
    export_submission({args.dataset: graph}, args.output)
    logger.info("Wrote %d nodes and %d edges to %s", len(graph.nodes), len(graph.edges), args.output)


if __name__ == "__main__":
    main()
