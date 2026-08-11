"""Load via importlib from subfolder scripts (see bootstrap snippet in each script)."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap(caller_file: str | Path) -> Path:
    """Add script dir + package root to sys.path, then register venv imports."""
    caller = Path(caller_file).resolve()
    root = caller.parent.parent
    for path in (caller.parent, root):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from dataset_paths import SUBPACKAGE_NAMES
    from bootstrap import prime_imports

    if caller.parent.name not in SUBPACKAGE_NAMES:
        root = caller.parent

    return prime_imports(root)
