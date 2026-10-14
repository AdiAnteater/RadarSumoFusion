"""Find SUMO the same way on Windows, Linux and macOS.

SUMO can be installed several ways and they put things in different places:
  Windows installer      SUMO_HOME set, binaries in SUMO_HOME/bin/*.exe
  Linux apt / ppa        binaries in /usr/bin, SUMO_HOME=/usr/share/sumo (often unset)
  pip install eclipse-sumo   binaries on PATH, SUMO_HOME inside the sumo package
  source build           SUMO_HOME set, binaries in SUMO_HOME/bin

Use sumo_binary("sumo"), sumo_binary("sumo-gui"), sumo_binary("netconvert")
instead of hard-coding "sumo.exe". Call ensure_traci_importable() before
"import traci" so a SUMO install without the pip traci package still works.
"""

from __future__ import annotations

import os
import shutil
import sys

IS_WINDOWS = sys.platform == "win32"


def sumo_home() -> str:
    """SUMO_HOME if set, else a best guess from the install (may be "")."""
    home = os.environ.get("SUMO_HOME", "").strip()
    if home:
        return home
    try:
        import sumo  # pip install eclipse-sumo
        home = getattr(sumo, "SUMO_HOME", "") or ""
        if home:
            return str(home)
    except ImportError:
        pass
    for cand in ("/usr/share/sumo", "/usr/local/share/sumo", "/opt/homebrew/share/sumo"):
        if os.path.isdir(os.path.join(cand, "tools")):
            return cand
    return ""


def sumo_binary(name: str = "sumo") -> str:
    """Absolute path of a SUMO executable, or the bare name as a last resort."""
    home = sumo_home()
    if home:
        exts = (".exe", "") if IS_WINDOWS else ("", ".exe")
        for ext in exts:
            cand = os.path.join(home, "bin", name + ext)
            if os.path.isfile(cand):
                return cand
    found = shutil.which(name)
    return found or name


def ensure_traci_importable() -> None:
    """Add SUMO_HOME/tools to sys.path when traci is not pip-installed."""
    try:
        import traci  # noqa: F401
        return
    except ImportError:
        pass
    home = sumo_home()
    tools = os.path.join(home, "tools") if home else ""
    if tools and os.path.isdir(tools) and tools not in sys.path:
        sys.path.append(tools)
