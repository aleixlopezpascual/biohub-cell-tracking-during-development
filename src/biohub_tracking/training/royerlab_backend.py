"""Lazy loading of the optional upstream Royerlab training and inference scripts."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Collection, Iterator


@contextmanager
def _upstream_import_context(source_dir: Path) -> Iterator[None]:
    """Temporarily expose an upstream package that shares our package name.

    Older competition support packs publish their model and metric helpers as
    ``biohub_tracking`` too. The loaded script retains direct references to
    those helpers, while this repository's already imported modules are put
    back immediately after script execution.
    """
    original_path = list(sys.path)
    local_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "biohub_tracking" or name.startswith("biohub_tracking.")
    }
    for name in local_modules:
        sys.modules.pop(name, None)
    for candidate in (source_dir / "scripts", source_dir / "src"):
        if candidate.is_dir():
            sys.path.insert(0, str(candidate.resolve()))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "biohub_tracking" or name.startswith("biohub_tracking."):
                sys.modules.pop(name, None)
        sys.modules.update(local_modules)
        sys.path[:] = original_path


@contextmanager
def official_import_context(source_dir: str | Path) -> Iterator[None]:
    """Keep the upstream package visible while calling lazily importing APIs.

    Some pinned metric functions import sibling modules only when evaluation
    starts. Loading the script is therefore not enough; the upstream package
    namespace must also be active for the duration of the call.
    """
    with _upstream_import_context(Path(source_dir)):
        yield


def load_official_script(
    source_dir: str | Path,
    script_name: str,
    *,
    module_name: str,
    required: Collection[str] = (),
) -> ModuleType:
    """Load one upstream script and validate the interfaces used by this repository."""
    script_dir = Path(source_dir) / "scripts"
    script = script_dir / script_name
    if not script.is_file():
        raise FileNotFoundError(f"official Royerlab script not found: {script}")
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load official script from {script}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    with _upstream_import_context(Path(source_dir)):
        specification.loader.exec_module(module)
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise ImportError(f"official script {script_name} lacks expected interfaces: {missing}")
    return module


def load_official_trainer(source_dir: str | Path) -> ModuleType:
    """Load the upstream TemporalUNet3D/node-transformer training backend."""
    return load_official_script(
        source_dir,
        "train_unet_transformer.py",
        module_name="biohub_official_training",
        required={
            "DEFAULT_AUGMENTATIONS",
            "FrameWindowDataset",
            "TemporalUNet3D",
            "UNetNodeTransformer",
            "_POS_EMBED_DIM",
            "compute_detection_loss",
            "evaluate",
            "load_dataset_windows",
            "train_epoch",
        },
    )
