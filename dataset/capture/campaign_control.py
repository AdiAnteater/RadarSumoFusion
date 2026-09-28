"""Campaign mode for CaptureRadarCameraData.py.

A campaign is ONE capture process that records several traffic scenarios back
to back. The capture keeps owning the CARLA tick for the whole campaign; the
orchestrator (fusion/campaign.py) starts and stops one SUMO runner per scenario
underneath it and tells the capture which frames to record.

Protocol (files in DATASET_CAMPAIGN_CONTROL_DIR):

  control.json   orchestrator -> capture, rewritten atomically. Each command has
                 an increasing "seq"; the capture applies every seq once.
                   {"seq": n, "command": "window", "segment": {...},
                    "start_frame": F0, "end_frame": F1}
                       record frames F0..F1 (inclusive) as segment["segment_id"]
                   {"seq": n, "command": "close", "segment_id": k}
                       end segment k now (runner failed / campaign aborted)
                   {"seq": n, "command": "stop"}
                       stop recording and finish the capture (label + post)
  status.json    capture -> orchestrator, rewritten ~4x per second:
                   {"pid", "run_dir", "latest_frame", "state", "segments": [...]}

Outside a recording window the capture keeps ticking (the SUMO runner needs
the clock) but writes nothing: radar returns, camera frames and actor frames are
dropped at the listen callbacks. So cleanup between scenarios and each
scenario's warm-up never reach the dataset.

The capture writes <run_dir>/segments.json with the ACTUAL recorded frame range
of every segment; the labeler uses it to add segment_id / scenario_id columns.

Everything here is file-based and stdlib-only so it works the same on Windows
and Linux and needs no extra process-to-process plumbing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

CONTROL_FILENAME = "control.json"
STATUS_FILENAME = "status.json"
SEGMENTS_FILENAME = "segments.json"


def campaign_control_dir_from_env() -> Path | None:
    raw = os.environ.get("DATASET_CAMPAIGN_CONTROL_DIR", "").strip()
    return Path(raw) if raw else None


def write_json_atomic(path: Path, data) -> None:
    """Write JSON via a temp file + os.replace so readers never see half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows: the reader has the file open for a moment. Retry.
            time.sleep(0.02)
    os.replace(tmp, path)


def read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_segments(capture_dir) -> list:
    """Segments of a capture folder, sorted by start frame ([] if not a campaign)."""
    data = read_json(Path(capture_dir) / SEGMENTS_FILENAME)
    if not isinstance(data, dict):
        return []
    segs = [s for s in data.get("segments", []) if s.get("start_frame") is not None
            and s.get("end_frame") is not None and s["end_frame"] >= s["start_frame"]]
    segs.sort(key=lambda s: s["start_frame"])
    return segs


class SegmentLookup:
    """frame -> segment dict, O(log n). Frames outside every segment -> None."""

    def __init__(self, segments: list):
        import bisect
        self._bisect = bisect
        self._segs = sorted(segments, key=lambda s: s["start_frame"])
        self._starts = [s["start_frame"] for s in self._segs]

    def __bool__(self):
        return bool(self._segs)

    def get(self, frame: int):
        i = self._bisect.bisect_right(self._starts, frame) - 1
        if i < 0:
            return None
        s = self._segs[i]
        return s if frame <= s["end_frame"] else None


class CampaignGate:
    """Decides, per CARLA frame, whether the capture records it.

    Thread-safety: ``segment_for_frame`` is called from CARLA's sensor threads;
    ``poll`` and ``on_tick`` from the capture main loop. The window list is
    replaced wholesale under a lock and read as an immutable tuple, so readers
    never block the listen callbacks for more than a tuple lookup.
    """

    def __init__(self, control_dir: Path, run_dir: str):
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._lock = threading.Lock()
        # Each window: dict(segment_id, start_frame, end_frame, planned_end_frame,
        # status, meta). Stored as a tuple of dicts, replaced on change.
        self._windows: tuple = ()
        self._last_seq = -1
        self._last_mtime = 0.0
        self._latest_frame = 0
        self._stop = False
        self._state = "running"
        self._last_status_write = 0.0

    # -- per-frame gate (sensor threads) ------------------------------------
    def segment_for_frame(self, frame: int):
        for w in self._windows:
            if w["start_frame"] <= frame <= w["end_frame"]:
                return w["segment_id"]
        return None

    def is_recorded(self, frame: int) -> bool:
        return self.segment_for_frame(frame) is not None

    # -- main loop ----------------------------------------------------------
    @property
    def stop_requested(self) -> bool:
        return self._stop

    def on_tick(self, frame: int | None) -> None:
        if frame is not None and frame > self._latest_frame:
            self._latest_frame = int(frame)
        self.poll()
        now = time.monotonic()
        if now - self._last_status_write >= 0.25:
            self.write_status()

    def poll(self) -> None:
        path = self.control_dir / CONTROL_FILENAME
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        cmd = read_json(path)
        if not isinstance(cmd, dict):
            self._last_mtime = 0.0   # partial read; try again next tick
            return
        seq = int(cmd.get("seq", -1))
        if seq <= self._last_seq:
            return
        self._last_seq = seq
        self._apply(cmd)

    def _apply(self, cmd: dict) -> None:
        kind = cmd.get("command")
        if kind == "window":
            seg = dict(cmd.get("segment") or {})
            sid = int(seg.get("segment_id", len(self._windows)))
            start = int(cmd["start_frame"])
            end = int(cmd["end_frame"])
            if start <= self._latest_frame:
                # Asked to start in the past (zero warm-up, slow poll): start
                # on the next tick instead and keep the requested length.
                shift = self._latest_frame + 1 - start
                start += shift
                end += shift
            w = {
                "segment_id": sid,
                "start_frame": start,
                "end_frame": end,
                "planned_end_frame": end,
                "status": "recording",
                "meta": seg,
            }
            with self._lock:
                self._windows = tuple(x for x in self._windows if x["segment_id"] != sid) + (w,)
            print(f"[capture] campaign: segment {sid} ({seg.get('scenario_name', '')}) "
                  f"will record frames {start}..{end}", flush=True)
        elif kind == "close":
            sid = cmd.get("segment_id")
            reason = cmd.get("reason", "closed")
            self._close(lambda w: sid is None or w["segment_id"] == sid, reason)
        elif kind == "stop":
            self._close(lambda w: True, cmd.get("reason", "campaign stopped"))
            self._stop = True
            print("[capture] campaign: stop requested by the orchestrator.", flush=True)
        self.write_segments()
        self.write_status()

    def _close(self, match, reason: str) -> None:
        with self._lock:
            out = []
            for w in self._windows:
                if match(w) and w["end_frame"] > self._latest_frame:
                    w = dict(w)
                    w["end_frame"] = max(w["start_frame"] - 1, self._latest_frame)
                    w["status"] = "truncated" if w["end_frame"] >= w["start_frame"] else "empty"
                    w["close_reason"] = reason
                out.append(w)
            self._windows = tuple(out)

    def _segments_view(self) -> list:
        segs = []
        for w in self._windows:
            status = w["status"]
            if status == "recording" and self._latest_frame >= w["end_frame"]:
                status = "complete"
            rec_end = min(w["end_frame"], self._latest_frame)
            n = max(0, rec_end - w["start_frame"] + 1)
            seg = dict(w["meta"])
            seg.update({
                "segment_id": w["segment_id"],
                "start_frame": w["start_frame"],
                "end_frame": w["end_frame"],
                "planned_end_frame": w["planned_end_frame"],
                "recorded_frames": n,
                "status": status,
            })
            if "close_reason" in w:
                seg["close_reason"] = w["close_reason"]
            segs.append(seg)
        return segs

    def write_segments(self) -> None:
        segs = [s for s in self._segments_view() if s["status"] != "empty"]
        try:
            write_json_atomic(Path(self.run_dir) / SEGMENTS_FILENAME,
                              {"format": 1, "segments": segs})
        except OSError as exc:
            print(f"[capture] campaign: could not write segments.json: {exc}", flush=True)

    def set_state(self, state: str) -> None:
        self._state = state
        self.write_status()

    def write_status(self) -> None:
        self._last_status_write = time.monotonic()
        try:
            write_json_atomic(self.control_dir / STATUS_FILENAME, {
                "pid": os.getpid(),
                "run_dir": self.run_dir,
                "latest_frame": self._latest_frame,
                "state": self._state,
                "segments": self._segments_view(),
                "time": time.time(),
            })
        except OSError:
            pass

    def finalize(self) -> None:
        """Close open windows at the last ticked frame and write segments.json."""
        self._close(lambda w: True, "capture ended")
        self.write_segments()
