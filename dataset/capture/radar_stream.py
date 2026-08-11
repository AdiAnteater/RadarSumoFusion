"""Decouple CARLA radar listen callbacks from heavy per-detection processing."""

from __future__ import annotations

import os
import queue
import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")

DEFAULT_RADAR_QUEUE_MAXSIZE = 4096
DEFAULT_CAPTURE_DEQUE_MAXLEN = 512


def radar_queue_maxsize_from_env() -> int:
    raw = os.environ.get("DATASET_RADAR_QUEUE_MAXSIZE", "").strip()
    if raw:
        try:
            return max(64, min(int(raw), 65536))
        except ValueError:
            pass
    return DEFAULT_RADAR_QUEUE_MAXSIZE


def radar_capture_deque_maxlen_from_env() -> int:
    raw = os.environ.get("DATASET_RADAR_CAPTURE_DEQUE_MAX", "").strip()
    if raw:
        try:
            return max(8, min(int(raw), 8192))
        except ValueError:
            pass
    return DEFAULT_CAPTURE_DEQUE_MAXLEN


def use_per_radar_latest_buffer_from_env() -> bool:
    raw = os.environ.get("DATASET_RADAR_PER_SENSOR_LATEST", "0").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _sensor_sort_key(label: str) -> tuple:
    if len(label) > 1 and label[0] == "R" and label[1:].isdigit():
        return (0, int(label[1:]))
    return (1, label)


class RadarMeasurementQueue(Generic[T]):
    """Thread-safe FIFO queue (single-sensor or low-rate use)."""

    def __init__(self, maxsize: int | None = None) -> None:
        if maxsize is None:
            maxsize = radar_queue_maxsize_from_env()
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.enqueued = 0

    def enqueue(self, item: T) -> bool:
        try:
            self._queue.put_nowait(item)
            self.enqueued += 1
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def pending(self) -> int:
        return self._queue.qsize()

    def drain(self, max_items: int = 64) -> list[T]:
        out: list[T] = []
        while len(out) < max_items:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    def drain_all(self) -> list[T]:
        pending = self.pending()
        if pending == 0:
            return []
        return self.drain(max_items=pending)


class PerRadarLatestBuffer(Generic[T]):
    """Keep one pending measurement per radar (labeling tests under load)."""

    def __init__(self) -> None:
        self._pending: dict[str, T] = {}
        self._lock = threading.Lock()
        self.dropped = 0
        self.enqueued = 0

    def enqueue(self, sensor_label: str, item: T) -> bool:
        with self._lock:
            if sensor_label in self._pending:
                self.dropped += 1
            self._pending[sensor_label] = item
            self.enqueued += 1
            return True

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def _sorted_labels(self) -> list[str]:
        return sorted(self._pending.keys(), key=_sensor_sort_key)

    def drain(self, max_items: int = 64) -> list[T]:
        with self._lock:
            if not self._pending:
                return []
            labels = self._sorted_labels()
            if max_items > 0:
                labels = labels[:max_items]
            return [self._pending.pop(label) for label in labels]

    def drain_all(self) -> list[T]:
        with self._lock:
            if not self._pending:
                return []
            labels = self._sorted_labels()
            return [self._pending.pop(label) for label in labels]


class PerRadarDequeBuffer(Generic[T]):
    """Bounded per-radar FIFO for dataset capture (avoids latest-only loss)."""

    def __init__(self, maxlen: int | None = None) -> None:
        self._maxlen = maxlen if maxlen is not None else radar_capture_deque_maxlen_from_env()
        self._pending: dict[str, deque[T]] = {}
        self._lock = threading.Lock()
        self.dropped = 0
        self.enqueued = 0

    def enqueue(self, sensor_label: str, item: T) -> bool:
        with self._lock:
            dq = self._pending.get(sensor_label)
            if dq is None:
                dq = deque(maxlen=self._maxlen)
                self._pending[sensor_label] = dq
            if len(dq) == self._maxlen:
                self.dropped += 1
            dq.append(item)
            self.enqueued += 1
            return True

    def pending(self) -> int:
        with self._lock:
            return sum(len(dq) for dq in self._pending.values())

    def drain(self, max_items: int = 64) -> list[T]:
        with self._lock:
            if not self._pending:
                return []
            out: list[T] = []
            labels = sorted(self._pending.keys(), key=_sensor_sort_key)
            limit = max_items if max_items > 0 else None
            while True:
                progressed = False
                for label in labels:
                    dq = self._pending.get(label)
                    if dq:
                        out.append(dq.popleft())
                        progressed = True
                        if limit is not None and len(out) >= limit:
                            break
                if not progressed or (limit is not None and len(out) >= limit):
                    break
            self._pending = {k: v for k, v in self._pending.items() if v}
            return out

    def drain_all(self) -> list[T]:
        with self._lock:
            if not self._pending:
                return []
            out: list[T] = []
            for label in sorted(self._pending.keys(), key=_sensor_sort_key):
                out.extend(self._pending[label])
            self._pending.clear()
            return out


def make_radar_measurement_buffer() -> RadarMeasurementQueue[T] | PerRadarLatestBuffer[T]:
    if use_per_radar_latest_buffer_from_env():
        return PerRadarLatestBuffer()
    return RadarMeasurementQueue()


def make_radar_capture_buffer() -> PerRadarDequeBuffer[T]:
    return PerRadarDequeBuffer()


def is_per_radar_buffer(
    buf: RadarMeasurementQueue[T] | PerRadarLatestBuffer[T] | PerRadarDequeBuffer[T],
) -> bool:
    return isinstance(buf, (PerRadarLatestBuffer, PerRadarDequeBuffer))
