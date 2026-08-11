"""Per-simulation-frame actor geometry log for offline radar labeling."""

from __future__ import annotations

import json
import queue
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import carla


def _vec3(loc) -> dict:
    return {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)}


def _rot(rot) -> dict:
    return {"pitch": float(rot.pitch), "yaw": float(rot.yaw), "roll": float(rot.roll)}


def enrich_snapshot_with_bbox(world, snapshot: dict) -> dict:
    """Ensure actor transform + OBB fields exist (no-op when snapshot is already enriched).

    Snapshots from `get_radar_target_snapshots` already include rotation + bbox to avoid
    redundant RPCs in the radar hot path. This fallback runs ONLY if those fields are
    missing — keeps backward compatibility with older snapshot producers.
    """
    has_rotation = "rotation" in snapshot
    has_bbox = "bbox" in snapshot
    if has_rotation and has_bbox:
        return snapshot

    try:
        actor = world.get_actor(int(snapshot["id"]))
    except RuntimeError:
        return snapshot

    actor_tf = actor.get_transform()
    bbox = actor.bounding_box
    out = dict(snapshot)
    out["location"] = _vec3(actor_tf.location)
    if not has_rotation:
        out["rotation"] = _rot(actor_tf.rotation)
    if not has_bbox:
        out["bbox"] = {
            "location": _vec3(bbox.location),
            "extent": _vec3(bbox.extent),
            "rotation": _rot(bbox.rotation) if bbox.rotation else None,
        }
    return out


def snapshot_location(snapshot: dict) -> carla.Location:
    loc = snapshot["location"]
    if isinstance(loc, carla.Location):
        return loc
    return carla.Location(float(loc["x"]), float(loc["y"]), float(loc["z"]))


class ActorFrameLogger:
    """Append one JSON line per simulation frame (deduped).

    ``log_frame`` is called from CARLA's streaming thread (via
    ``TickActorSnapshotter._on_tick``). In asynchronous mode the server free-runs
    and silently drops the next ``on_tick`` if the previous callback hasn't
    returned, so the callback MUST be cheap. Therefore ``log_frame`` only enqueues
    the (frame_id, snapshots) tuple; a dedicated background **writer thread** does
    the JSON serialization + disk write + flush off the hot path. This stops a slow
    disk or a long serialization from stalling the callback and dropping actor
    frames (the ~21% bursty gaps that turned matched returns into all-clutter).

    ``close()`` is called from the main thread at shutdown: it enqueues a sentinel,
    joins the writer (draining everything already queued), then closes the handle.
    ``_write_lock`` keeps the final flush/close atomic against the writer so an
    in-flight write can never hit a freshly closed handle (the
    ``I/O operation on closed file`` error seen in capture 231410).
    """

    ACTOR_FRAMES_FILENAME = "actor_frames.jsonl"

    def __init__(self, capture_dir: str | Path, world) -> None:
        self._world = world
        self._path = Path(capture_dir) / self.ACTOR_FRAMES_FILENAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_frames: set[int] = set()  # writer-thread-only; no lock needed
        self._write_lock = threading.Lock()
        self._closed = False
        self._handle = open(self._path, "a", encoding="utf-8")
        # Unbounded queue: never block the on_tick callback. The writer thread is
        # the sole consumer and the only thread that touches _seen_frames/_handle.
        self._queue: "queue.Queue" = queue.Queue()
        self._write_errors = 0
        self._writer = threading.Thread(
            target=self._writer_loop, name="actor-frame-writer", daemon=True
        )
        self._writer.start()

    def _writer_loop(self) -> None:
        """Drain the queue: serialize + write each frame, flushing once caught up."""
        while True:
            item = self._queue.get()
            try:
                if item is None:  # sentinel: all queued frames drained -> stop
                    return
                frame_id, actor_snapshots = item
                try:
                    self._write_record(frame_id, actor_snapshots)
                except Exception as exc:  # noqa: BLE001 - never kill the writer
                    self._write_errors += 1
                    if self._write_errors <= 3:
                        print(
                            f"[capture] actor-frame writer error: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
            finally:
                self._queue.task_done()
            # Batch writes during drop-burst-prone periods; flush when caught up so
            # the file stays durable without an fsync-per-frame on the hot path.
            if item is not None and self._queue.empty():
                with self._write_lock:
                    if not self._closed:
                        try:
                            self._handle.flush()
                        except Exception:  # noqa: BLE001
                            pass

    def _write_record(self, frame_id: int, actor_snapshots: list[dict]) -> None:
        if frame_id in self._seen_frames:
            return
        actors_out = []
        for snap in actor_snapshots:
            enriched = enrich_snapshot_with_bbox(self._world, snap)
            location = enriched.get("location")
            if not isinstance(location, dict):
                location = _vec3(location)
            actors_out.append(
                {
                    "id": enriched["id"],
                    "kind": enriched.get("kind", ""),
                    "type_id": enriched.get("type_id", ""),
                    "class_label": enriched.get("class_label", ""),
                    "location": location,
                    "rotation": enriched.get("rotation"),
                    "bbox": enriched.get("bbox"),
                }
            )
        record = {"frame": int(frame_id), "actors": actors_out}
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._write_lock:
            if self._closed:
                return
            self._seen_frames.add(frame_id)
            self._handle.write(line)

    def log_frame(self, frame_id: int, actor_snapshots: list[dict]) -> None:
        # Runs in CARLA's on_tick (streaming) thread: keep it O(1). The snapshot
        # dicts are immutable post-build (also shared with the snapshotter cache),
        # so handing the reference to the writer thread is safe.
        if self._closed:
            return
        self._queue.put((int(frame_id), actor_snapshots))

    def close(self) -> None:
        # Stop accepting new frames, drain everything already queued, then close.
        if self._closed:
            return
        self._queue.put(None)  # sentinel runs after all frames enqueued so far
        self._writer.join(timeout=30.0)
        with self._write_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._handle.flush()
                self._handle.close()
            except Exception:  # noqa: BLE001 - best-effort during shutdown
                pass

    @staticmethod
    def load_by_frame(path: str | Path) -> dict[int, list[dict]]:
        by_frame: dict[int, list[dict]] = {}
        p = Path(path)
        if not p.is_file():
            return by_frame
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                frame = int(record["frame"])
                actors = record.get("actors", [])
                normalized = []
                for actor in actors:
                    item = dict(actor)
                    if "location" in item and not isinstance(item["location"], carla.Location):
                        item["location"] = snapshot_location(item)
                    normalized.append(item)
                by_frame[frame] = normalized
        return by_frame


class TickActorSnapshotter:
    """Eagerly capture + log actor snapshots on every server tick.

    Subscribes to ``world.on_tick`` so the snapshot for each frame is taken
    synchronously with the simulation step that produced it — not lazily when
    a radar message happens to be processed. This guarantees that:

    * Every world tick is logged to ``actor_frames.jsonl`` (no truncation when
      the radar processing loop falls behind).
    * Radar messages waiting in the consumer queue can be matched against actor
      state at their own ``measurement.frame`` via a pure in-memory lookup
      (no CARLA RPC required during the post-Ctrl+C drain).

    The in-memory cache is a sliding window keyed by ``frame_id``; older
    entries are evicted once ``max_frames_in_memory`` is exceeded.

    **Critical**: ``snapshot_fn`` MUST be cheap (zero or near-zero CARLA RPCs).
    CARLA silently stops dispatching on_tick callbacks if the previous callback
    overruns the tick budget. The recommended pattern is to use the
    ``world_snapshot`` already provided by on_tick for actor transforms and
    cache static per-actor metadata (type_id, bbox) the first time each actor
    is seen. See ``make_fast_tick_snapshot_fn`` in ``CaptureRadarCameraData.py``.
    """

    def __init__(
        self,
        world: "carla.World",
        capture_dir: str | Path,
        *,
        snapshot_fn: Callable[["carla.World", "carla.WorldSnapshot"], list[dict]],
        max_frames_in_memory: int = 600,
    ) -> None:
        self._world = world
        self._snapshot_fn = snapshot_fn
        self._max_frames = max(64, int(max_frames_in_memory))
        self._lock = threading.Lock()
        self._by_frame: "OrderedDict[int, list[dict]]" = OrderedDict()
        self._logger = ActorFrameLogger(capture_dir, world)
        self._callback_id: int | None = None
        self._stopped = False
        self._error_count = 0
        self._tick_count = 0
        try:
            self._callback_id = world.on_tick(self._on_tick)
        except RuntimeError as exc:
            print(
                f"[capture] TickActorSnapshotter: world.on_tick failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _on_tick(self, world_snapshot) -> None:
        if self._stopped:
            return
        try:
            frame_id = int(world_snapshot.frame)
            actor_snaps = self._snapshot_fn(self._world, world_snapshot)
        except Exception as exc:  # noqa: BLE001 - runs in CARLA streaming thread
            self._error_count += 1
            if self._error_count <= 3:
                print(
                    f"[capture] TickActorSnapshotter tick error: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return

        with self._lock:
            if frame_id in self._by_frame:
                self._by_frame.move_to_end(frame_id)
                return
            self._by_frame[frame_id] = actor_snaps
            while len(self._by_frame) > self._max_frames:
                self._by_frame.popitem(last=False)
            self._tick_count += 1

        # Re-check after the snapshot work; stop() may have flipped the flag while
        # the snapshot was being built. ActorFrameLogger.log_frame is also lock-
        # guarded internally and no-ops once closed, so this is belt-and-suspenders.
        if self._stopped:
            return
        try:
            self._logger.log_frame(frame_id, actor_snaps)
        except Exception as exc:  # noqa: BLE001 - never abort the tick thread
            self._error_count += 1
            if self._error_count <= 3:
                print(
                    f"[capture] TickActorSnapshotter log_frame error: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    def get(self, frame_id: int) -> list[dict]:
        """Return cached actor snapshots for ``frame_id`` (empty list if evicted)."""
        with self._lock:
            return self._by_frame.get(int(frame_id), [])

    def tick_count(self) -> int:
        return self._tick_count

    def stop(self) -> None:
        """Stop receiving new ticks and close the JSONL handle."""
        if self._stopped:
            return
        self._stopped = True
        if self._callback_id is not None:
            try:
                self._world.remove_on_tick(self._callback_id)
            except Exception:  # noqa: BLE001 - best-effort during shutdown
                pass
            self._callback_id = None
        try:
            self._logger.close()
        except Exception:  # noqa: BLE001
            pass
