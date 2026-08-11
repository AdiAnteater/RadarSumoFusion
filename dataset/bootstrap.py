"""Shared import bootstrap for Scripts/DatasetCreation."""

from __future__ import annotations

import sys
from pathlib import Path

from dataset_paths import SUBPACKAGE_NAMES, ensure_import_paths


def prime_imports(root: Path | None = None) -> Path:
    """Register the DatasetCreation root (+ venv site-packages) on sys.path."""
    return ensure_import_paths(root=root)


def prime_and_prepare(caller_file: str | Path) -> Path:
    """Register import paths for a script at the package root or in a subfolder."""
    path = Path(caller_file).resolve()
    root = path.parent.parent if path.parent.name in SUBPACKAGE_NAMES else path.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return prime_imports(root)


# Backward-compatible alias used during the reorganization.
install = prime_and_prepare
