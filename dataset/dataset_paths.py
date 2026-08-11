"""Shared paths for the Scripts/DatasetCreation tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DATA_OUTPUT_DIRNAME = "Data"
CONFIG_DIRNAME = "config"

SUBPACKAGE_NAMES = frozenset(
    {"setup", "capture", "world", "testing", "tools", "DataExploration"}
)


def dataset_root() -> Path:
    return Path(__file__).resolve().parent


def data_output_dir(root: Path | None = None) -> Path:
    return (root or dataset_root()) / DATA_OUTPUT_DIRNAME


def config_dir(root: Path | None = None) -> Path:
    return (root or dataset_root()) / CONFIG_DIRNAME


def capture_dir(root: Path | None = None) -> Path:
    return (root or dataset_root()) / "capture"


def testing_dir(root: Path | None = None) -> Path:
    return (root or dataset_root()) / "testing"


def world_dir(root: Path | None = None) -> Path:
    return (root or dataset_root()) / "world"


def setup_dir(root: Path | None = None) -> Path:
    return (root or dataset_root()) / "setup"


def tools_dir(root: Path | None = None) -> Path:
    return (root or dataset_root()) / "tools"


def ensure_import_paths(*_extra: str, root: Path | None = None) -> Path:
    """Put the package root (+ project venv site-packages) on sys.path."""
    root = root or dataset_root()
    venv_sp = venv_site_packages(root)
    if venv_sp is not None:
        sp = str(venv_sp)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def venv_site_packages(root: Path | None = None) -> Path | None:
    """Project .venv site-packages (CARLA is usually installed here)."""
    project_root = (root or dataset_root()).parents[1]
    sp = project_root / ".venv" / "Lib" / "site-packages"
    return sp if sp.is_dir() else None


def pythonpath_env(root: Path | None = None) -> str:
    root = root or dataset_root()
    parts = [str(root)]
    venv_sp = venv_site_packages(root)
    if venv_sp is not None:
        parts.insert(0, str(venv_sp))
    existing = os.environ.get("PYTHONPATH", "").strip()
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)
