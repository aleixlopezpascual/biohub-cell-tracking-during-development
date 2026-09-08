"""Training entry point for the baseline 3D U-Net.

Usage::

    python scripts/train.py --config configs/train.yaml

Loads a :class:`~biohub_tracking.utils.config.PipelineConfig`, seeds all RNGs,
builds a :class:`~biohub_tracking.data.FramePairDataset` from an OME-Zarr
volume, builds the :class:`~biohub_tracking.models.UNet3D`, and runs a
minimal supervised training loop. Requires the optional ``torch`` extra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biohub_tracking.data import FramePairDataset, OMEZarrVolume
from biohub_tracking.models import UNet3D, UNet3DConfig
from biohub_tracking.utils import get_logger, load_config, set_global_seed

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML PipelineConfig.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    set_global_seed(config.train.seed)

    logger.info("Opening OME-Zarr volume at %s", config.data.zarr_path)
    volume = OMEZarrVolume.open(
        config.data.zarr_path,
        resolution_level=config.data.resolution_level,
        voxel_size_um=config.data.voxel_size_um,
    )
    dataset = FramePairDataset(
        volume,
        patch_size=config.data.patch_size,
        patch_stride=config.data.patch_stride,
        frame_stride=config.data.frame_pair_stride,
        channels=config.data.channels,
    )
    logger.info("Dataset has %d frame-pair patches", len(dataset))

    model = UNet3D(
        UNet3DConfig(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            base_features=config.model.base_features,
            depth=config.model.depth,
            norm=config.model.norm,
        )
    )
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Training requires the optional 'torch' extra: install with `pip install -e '.[torch]'`."
        ) from exc

    module = model.build()
    optimizer = torch.optim.Adam(module.parameters(), lr=config.train.learning_rate)
    loss_fn = torch.nn.MSELoss()

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module.train()
    for epoch in range(config.train.epochs):
        epoch_loss = 0.0
        for i in range(len(dataset)):
            sample = dataset[i]
            frame_a = torch.as_tensor(sample["frame_a"], dtype=torch.float32).unsqueeze(0)
            frame_b = torch.as_tensor(sample["frame_b"], dtype=torch.float32).unsqueeze(0)
            optimizer.zero_grad()
            prediction = module(frame_a)
            loss = loss_fn(prediction, frame_b[:, : prediction.shape[1]])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        mean_loss = epoch_loss / max(len(dataset), 1)
        logger.info("epoch=%d mean_loss=%.6f", epoch, mean_loss)

    checkpoint_path = output_dir / "unet3d.pt"
    model.save(str(checkpoint_path))
    logger.info("Saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
