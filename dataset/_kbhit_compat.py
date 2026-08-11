"""Cross-platform 'has Enter been pressed?' polling.

Replaces msvcrt.kbhit()/getwch() so the dataset scripts run on Linux too.
- Windows: thin wrapper around msvcrt.
- POSIX:   non-blocking select on stdin in cbreak mode; restored at exit.
- Non-TTY (e.g. stdin redirected by a parent like Start.py subprocess.Popen):
  returns False forever so the caller's main loop runs uninterrupted until
  SIGTERM/SIGINT.
"""
from __future__ import annotations

import sys

try:
    import msvcrt  # type: ignore[import-not-found]
    _WINDOWS = True
except ImportError:
    _WINDOWS = False

if not _WINDOWS:
    import atexit
    import select
    import termios
    import tty

    _saved_attrs = None
    _cbreak_ready = False

    def _setup_cbreak() -> None:
        global _saved_attrs, _cbreak_ready
        if _cbreak_ready or not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        try:
            _saved_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            _cbreak_ready = True
            atexit.register(_restore)
        except (termios.error, OSError):
            _saved_attrs = None

    def _restore() -> None:
        global _saved_attrs
        if _saved_attrs is None:
            return
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _saved_attrs)
        except (termios.error, OSError):
            pass
        _saved_attrs = None


def enter_pressed() -> bool:
    """Non-blocking: True iff Enter (CR/LF) is buffered on stdin."""
    if _WINDOWS:
        if not msvcrt.kbhit():
            return False
        try:
            ch = msvcrt.getwch()
        except Exception:
            return False
        return ch in ("\r", "\n")
    if not sys.stdin.isatty():
        return False
    _setup_cbreak()
    try:
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return False
    if not rlist:
        return False
    try:
        ch = sys.stdin.read(1)
    except (OSError, ValueError):
        return False
    return ch in ("\r", "\n")
