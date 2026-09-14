#!/usr/bin/env python3
"""Train or continue a Royerlab model to one gated evaluation epoch.

Unlike the official convenience loop, this runner saves model, optimizer,
scheduler, RNG, and session state. It deliberately stops at one requested
evaluation epoch; OOF prediction and the official graph metric decide whether
another Kaggle GPU session is authorized.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

from biohub_tracking.training import (
    RunManifest,
    TrainingSession,
    load_gold_training_config,
    load_official_trainer,
    load_training_checkpoint,
    make_torch_gaussian_detection_loss,
    save_inference_checkpoint,
    save_training_checkpoint,
    validate_run_manifest,
)


class _SeededAugmentedDataset:
    """Apply upstream augmentations with a deterministic epoch/sample RNG."""

    def __init__(self, base: Any, augmentations: list[Any], seed: int) -> None:
        from multiprocessing import Value

        self.base = base
        self.augmentations = augmentations
        self.seed = seed
        self._epoch = Value("q", 0)

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        """Publish the epoch to persistent DataLoader workers."""
        self._epoch.value = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = self.base[index]
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(self._epoch.value), int(index)])
        )
        images, coordinates, masks = batch["imgs"], batch["coords"], batch["masks"]
        for augmentation in self.augmentations:
            images, coordinates, masks = augmentation(
                images, coordinates, masks, rng=rng
            )
        return {**batch, "imgs": images, "coords": coordinates, "masks": masks}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--fold-index", required=True, type=int)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--target-epoch", required=True, type=int)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--preflight-output",
        type=Path,
        help=(
            "Run one synthetic empty-attention backward pass, one real training batch, "
            "and a checkpoint round-trip; write a JSON result instead of training an epoch."
        ),
    )
    parser.add_argument(
        "--previous-official-score",
        type=float,
        help="Official OOF score measured for the checkpoint being resumed.",
    )
    parser.add_argument(
        "--previous-official-epoch",
        type=int,
        help="Evaluation epoch associated with --previous-official-score.",
    )
    return parser.parse_args(argv)


def _load_video_data(backend: Any, paths: list[Path], config: Any) -> list[Any]:
    output = []
    for path in paths:
        output.append(
            backend.load_dataset_windows(
                path,
                window_size=config.window_size,
                downsample=config.downsample,
            )
        )
    return output


def _rng_state(torch: Any, session: TrainingSession) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "session": session.state_dict(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(torch: Any, state: dict[str, object]) -> None:
    def cpu_byte_tensor(value: object) -> Any:
        if torch.is_tensor(value):
            return value.detach().to(device="cpu", dtype=torch.uint8)
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        # Loading the checkpoint with map_location="cuda" also moves the CPU
        # generator state. PyTorch requires a CPU ByteTensor here.
        torch.set_rng_state(cpu_byte_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [cpu_byte_tensor(value) for value in state["torch_cuda"]]
        )


def _adapt_model_state_for_runtime(
    state: dict[str, object], *, data_parallel: bool
) -> dict[str, object]:
    """Translate only the UNet DataParallel prefix across Kaggle GPU types."""
    if data_parallel:
        return {
            key.replace("unet.", "unet.module.", 1)
            if key.startswith("unet.") and not key.startswith("unet.module.")
            else key: value
            for key, value in state.items()
        }
    return {key.replace("unet.module.", "unet.", 1): value for key, value in state.items()}


def _install_empty_attention_guard(model: Any) -> int:
    """Keep cross-attention finite when a sample has no detected key nodes.

    The pinned support-pack transformer turns its real-node mask into a
    ``MultiheadAttention`` padding mask. PyTorch attention returns NaNs when
    every key is masked for one batch row. Detection-driven training can
    legitimately produce such rows, especially early in detector learning.
    Give those rows one zero-valued dummy key; their edge region is excluded
    from the upstream loss by the unchanged real-node masks.

    Returns the number of guarded transformer blocks.
    """
    blocks = getattr(getattr(model, "transformer", None), "blocks", ())
    guarded = 0
    for block in blocks:
        if getattr(block, "_biohub_empty_attention_guard", False):
            continue
        original_forward = block.forward

        def safe_forward(
            self: Any,
            q: Any,
            kv: Any,
            kv_mask: Any | None = None,
            *,
            _original_forward: Any = original_forward,
        ) -> Any:
            if kv_mask is not None:
                empty_rows = ~kv_mask.any(dim=1)
                if bool(empty_rows.any().item()):
                    kv = kv.clone()
                    kv_mask = kv_mask.clone()
                    kv[empty_rows, 0] = 0
                    kv_mask[empty_rows, 0] = True
            return _original_forward(q, kv, kv_mask)

        block.forward = MethodType(safe_forward, block)
        block._biohub_empty_attention_guard = True
        guarded += 1
    if guarded == 0 and not blocks:
        raise ValueError("attached edge transformer exposes no attention blocks to guard")
    return guarded


def _manifest_from_json(path: Path) -> RunManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunManifest(**payload)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_gold_training_config(args.config)
    if args.target_epoch not in config.learning_curve.evaluation_epochs:
        raise ValueError("target epoch must be listed in learning_curve.evaluation_epochs")
    if args.resume_checkpoint is not None and args.bootstrap_checkpoint is not None:
        raise ValueError("supply either a resume checkpoint or a bootstrap checkpoint, not both")
    if (args.previous_official_score is None) != (args.previous_official_epoch is None):
        raise ValueError("previous official score and epoch must be supplied together")
    first_gate = config.learning_curve.evaluation_epochs[0]
    if (
        args.resume_checkpoint is None
        and args.bootstrap_checkpoint is None
        and args.target_epoch != first_gate
    ):
        raise ValueError("a fresh campaign must stop at the first evaluation epoch")
    if config.mixed_precision:
        raise ValueError(
            "the current official train_epoch has no GradScaler hook; leave mixed_precision false "
            "until the AMP benchmark is integrated safely"
        )
    folds = json.loads(args.splits.read_text(encoding="utf-8"))
    if args.fold_index < 0:
        raise ValueError("fold index must be non-negative")
    try:
        fold = folds[args.fold_index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"fold index {args.fold_index} is absent from {args.splits}") from exc
    if not isinstance(fold, dict) or not {"name", "train", "test"} <= set(fold):
        raise ValueError("selected fold must contain name, train, and test fields")
    manifest = _manifest_from_json(Path(config.output_dir) / "run_manifest.json")
    validate_run_manifest(
        manifest,
        config_path=args.config,
        split_path=args.splits,
        candidate=args.candidate,
    )
    if manifest.fold != fold.get("name"):
        raise ValueError(
            f"manifest fold {manifest.fold!r} does not match selected fold {fold.get('name')!r}"
        )

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("competitive continuation requires the optional torch extra") from exc

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    backend = load_official_trainer(Path(config.official_source_dir).resolve())
    downsampled_scale = tuple(
        scale * downsample for scale, downsample in zip(config.voxel_size_um, config.downsample)
    )
    backend.compute_detection_loss = make_torch_gaussian_detection_loss(
        voxel_size_um=downsampled_scale,
        sigma_um=config.gaussian_sigma_um,
    )
    data_dir = Path(config.data_dir)
    train_paths = [data_dir / name for name in fold["train"]]
    validation_paths = [data_dir / name for name in fold["test"]]
    train_video_data = _load_video_data(backend, train_paths, config)
    validation_video_data = _load_video_data(backend, validation_paths, config)
    windows = [
        window
        for _meta, group in train_video_data + validation_video_data
        for window in group
    ]
    if not windows:
        raise ValueError("selected fold contains no trainable frame windows")
    max_nodes = max(max(window.node_counts) for window in windows)
    base_train_dataset = backend.FrameWindowDataset(
        train_video_data,
        max_nodes=max_nodes,
    )
    train_dataset = _SeededAugmentedDataset(
        base_train_dataset,
        list(backend.DEFAULT_AUGMENTATIONS),
        config.seed,
    )
    validation_dataset = backend.FrameWindowDataset(
        validation_video_data,
        max_nodes=max_nodes,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "prefetch_factor": 2 if config.num_workers > 0 else None,
        "persistent_workers": config.num_workers > 0,
        "pin_memory": False,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_kwargs: dict[str, object] = {
        "in_channels": 1,
        "out_channels": config.unet_out_channels,
        "layers": list(config.unet_layers),
    }
    supports_gradient_checkpointing = (
        "gradient_checkpointing" in inspect.signature(backend.TemporalUNet3D).parameters
    )
    if config.gradient_checkpointing and not supports_gradient_checkpointing:
        raise ValueError("attached TemporalUNet3D does not support gradient checkpointing")
    if supports_gradient_checkpointing:
        unet_kwargs["gradient_checkpointing"] = config.gradient_checkpointing
    unet = backend.TemporalUNet3D(**unet_kwargs)
    model = backend.UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config.unet_out_channels,
        pos_feat_dim=4 * backend._POS_EMBED_DIM,
        hidden_dim=config.transformer_hidden_dim,
        n_heads=config.transformer_heads,
        n_blocks=config.transformer_blocks,
        dropout=config.transformer_dropout,
    ).to(device)
    guarded_attention_blocks = _install_empty_attention_guard(model)
    print(
        f"installed empty-detection attention guard on {guarded_attention_blocks} blocks",
        flush=True,
    )
    if torch.cuda.device_count() > 1:
        model.unet = nn.DataParallel(model.unet)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.learning_curve.evaluation_epochs)
        )
    elif config.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    else:
        scheduler = None
    skipped_gradient_steps = 0
    clip_was_nonfinite = False
    original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def checked_clip_grad_norm(*clip_args: Any, **clip_kwargs: Any) -> Any:
        nonlocal clip_was_nonfinite
        total_norm = original_clip_grad_norm(*clip_args, **clip_kwargs)
        clip_was_nonfinite = not bool(torch.isfinite(total_norm).item())
        return total_norm

    original_optimizer_step = optimizer.step

    def finite_optimizer_step(*step_args: Any, **step_kwargs: Any) -> Any:
        nonlocal clip_was_nonfinite, skipped_gradient_steps
        if clip_was_nonfinite:
            skipped_gradient_steps += 1
            optimizer.zero_grad(set_to_none=True)
            clip_was_nonfinite = False
            return None
        return original_optimizer_step(*step_args, **step_kwargs)

    # The pinned upstream loop does not reject non-finite attention gradients.
    # Keep a rare empty/masked batch from permanently corrupting the model.
    torch.nn.utils.clip_grad_norm_ = checked_clip_grad_norm
    optimizer.step = finite_optimizer_step

    output_dir = Path(config.output_dir) / "weights" / args.candidate / f"split_{args.fold_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    session = TrainingSession(
        max_runtime_hours=config.max_runtime_hours,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_every_epochs=config.checkpoint_every_epochs,
    )
    start_epoch = 0
    if args.resume_checkpoint is not None:
        restored = load_training_checkpoint(args.resume_checkpoint, map_location=str(device))
        if dict(restored.manifest) != manifest.__dict__:
            raise ValueError("resume checkpoint manifest differs from the campaign manifest")
        model.load_state_dict(
            _adapt_model_state_for_runtime(
                dict(restored.model_state), data_parallel=isinstance(model.unet, nn.DataParallel)
            )
        )
        optimizer.load_state_dict(restored.optimizer_state)
        if scheduler is not None and restored.scheduler_state is not None:
            scheduler.load_state_dict(restored.scheduler_state)
        start_epoch = restored.epoch + 1
        session.load_state_dict(dict(restored.rng_state.get("session", {})))
        _restore_rng(torch, dict(restored.rng_state))
    elif args.bootstrap_checkpoint is not None:
        state = torch.load(args.bootstrap_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(
            _adapt_model_state_for_runtime(
                dict(state), data_parallel=isinstance(model.unet, nn.DataParallel)
            )
        )
        start_epoch = config.learning_curve.evaluation_epochs[0]
    if args.previous_official_score is not None:
        if args.previous_official_epoch not in config.learning_curve.evaluation_epochs:
            raise ValueError("previous official epoch is not an evaluation epoch")
        if args.previous_official_epoch > start_epoch:
            raise ValueError("previous official epoch is later than the resumed training state")
        session.record_evaluation(args.previous_official_epoch - 1, args.previous_official_score)

    def save(path: Path, epoch: int) -> None:
        save_training_checkpoint(
            path,
            epoch=epoch,
            best_official_score=session.best_official_score,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None if scheduler is None else scheduler.state_dict(),
            scaler_state=None,
            rng_state=_rng_state(torch, session),
            manifest=manifest,
        )
        save_inference_checkpoint(
            path.with_name(f"{path.stem}_model.pth"), model.state_dict()
        )

    if args.preflight_output is not None:
        if args.resume_checkpoint is None:
            raise ValueError("preflight requires --resume-checkpoint")
        model.zero_grad(set_to_none=True)
        hidden_dim = config.transformer_hidden_dim
        q = torch.randn(2, 2, hidden_dim, device=device, requires_grad=True)
        kv = torch.randn(2, 3, hidden_dim, device=device, requires_grad=True)
        kv_mask = torch.tensor(
            [[False, False, False], [True, True, False]],
            dtype=torch.bool,
            device=device,
        )
        attention_output = model.transformer.blocks[0](q, kv, kv_mask=kv_mask)
        attention_output.square().mean().backward()
        attention_finite = bool(torch.isfinite(attention_output).all().item())
        gradient_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.transformer.blocks[0].parameters()
        )
        if not attention_finite or not gradient_finite:
            raise FloatingPointError("empty-detection attention preflight produced non-finite values")
        model.zero_grad(set_to_none=True)

        skipped_before_batch = skipped_gradient_steps
        edge_loss, detection_loss = backend.train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config.detection_loss_weight,
            config.unlabeled_negative_weight,
            max_iters=1,
            pool_kernel_um=config.pool_kernel_um,
        )
        skipped_batches = skipped_gradient_steps - skipped_before_batch
        if not np.isfinite(edge_loss) or not np.isfinite(detection_loss):
            raise FloatingPointError("real-batch preflight returned a non-finite loss")
        if skipped_batches:
            raise FloatingPointError("real-batch preflight skipped a non-finite optimizer step")

        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        roundtrip_path = args.preflight_output.with_suffix(".pt")
        save(roundtrip_path, start_epoch - 1)
        roundtrip = load_training_checkpoint(roundtrip_path, map_location=str(device))
        _restore_rng(torch, dict(roundtrip.rng_state))
        if roundtrip.epoch != start_epoch - 1:
            raise ValueError("preflight checkpoint round-trip changed the epoch")
        if set(roundtrip.model_state) != set(model.state_dict()):
            raise ValueError("preflight checkpoint round-trip changed model state keys")
        summary = {
            "status": "passed",
            "resume_epoch": start_epoch - 1,
            "attention_blocks_guarded": guarded_attention_blocks,
            "empty_attention_finite": attention_finite,
            "empty_attention_gradients_finite": gradient_finite,
            "real_batch_edge_loss": float(edge_loss),
            "real_batch_detection_loss": float(detection_loss),
            "real_batch_skipped_steps": skipped_batches,
            "checkpoint_roundtrip": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        args.preflight_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    if start_epoch >= args.target_epoch:
        raise ValueError(
            f"checkpoint already reached epoch {start_epoch}; target is {args.target_epoch}"
        )
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.target_epoch):
        decision = session.stop_decision()
        if decision.stop:
            save(output_dir / "latest.pt", final_epoch)
            print(f"stopped before epoch {epoch}: {decision.reason}", flush=True)
            return
        train_dataset.set_epoch(epoch)
        skipped_before_epoch = skipped_gradient_steps
        edge_loss, detection_loss = backend.train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config.detection_loss_weight,
            config.unlabeled_negative_weight,
            pool_kernel_um=config.pool_kernel_um,
        )
        skipped_this_epoch = skipped_gradient_steps - skipped_before_epoch
        if skipped_this_epoch > config.max_nonfinite_gradient_batches_per_epoch:
            raise FloatingPointError(
                "non-finite gradient batches exceeded the configured limit: "
                f"{skipped_this_epoch} > "
                f"{config.max_nonfinite_gradient_batches_per_epoch}"
            )
        validation_loss, validation_accuracy, validation_recall = backend.evaluate(
            model,
            validation_loader,
            device,
            pool_kernel_um=config.pool_kernel_um,
        )
        if scheduler is not None:
            if config.lr_scheduler == "plateau":
                scheduler.step(validation_loss)
            else:
                scheduler.step()
        final_epoch = epoch
        print(
            f"epoch={epoch} edge={edge_loss:.6f} detection={detection_loss:.6f} "
            f"proxy_loss={validation_loss:.6f} proxy_accuracy={validation_accuracy:.6f} "
            f"proxy_recall={validation_recall:.6f}",
            f"skipped_nonfinite={skipped_this_epoch}",
            flush=True,
        )
        if session.checkpoint_due(epoch):
            save(output_dir / "latest.pt", epoch)

    checkpoint = output_dir / f"epoch_{args.target_epoch:03d}.pt"
    save(checkpoint, final_epoch)
    model_config = {
        "unet_out_channels": config.unet_out_channels,
        "unet_layers": list(config.unet_layers),
        "transformer_hidden_dim": config.transformer_hidden_dim,
        "transformer_heads": config.transformer_heads,
        "transformer_blocks": config.transformer_blocks,
        "transformer_dropout": config.transformer_dropout,
        "downsample": list(config.downsample),
        "window_size": config.window_size,
        "pool_kernel_um": config.pool_kernel_um,
    }
    (output_dir / "config.json").write_text(
        json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved gated checkpoint {checkpoint}", flush=True)
    print("run OOF prediction and official graph scoring before continuing", flush=True)


if __name__ == "__main__":
    main()
