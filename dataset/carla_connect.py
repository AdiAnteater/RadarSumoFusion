"""Shared CARLA client connection for DatasetCreation scripts."""

from __future__ import annotations

import os
import time

import carla

# On Windows, "localhost" often resolves to IPv6 (::1) while CARLA listens on IPv4.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2000
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_READY_TIMEOUT_S = 180.0
DEFAULT_READY_POLL_S = 2.0
DEFAULT_PROBE_TIMEOUT_S = 5.0


def carla_host() -> str:
    return os.environ.get("DATASET_CARLA_HOST", DEFAULT_HOST)


def carla_port() -> int:
    return int(os.environ.get("DATASET_CARLA_PORT", str(DEFAULT_PORT)))


def carla_timeout_s() -> float:
    return float(os.environ.get("DATASET_CARLA_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))


def carla_ready_timeout_s() -> float:
    return float(os.environ.get("DATASET_CARLA_READY_TIMEOUT_S", str(DEFAULT_READY_TIMEOUT_S)))


def make_client(timeout_s: float | None = None, host: str | None = None) -> carla.Client:
    client = carla.Client(host or carla_host(), carla_port())
    client.set_timeout(timeout_s if timeout_s is not None else carla_timeout_s())
    return client


def get_world(client: carla.Client | None = None) -> tuple[carla.Client, carla.World]:
    if client is None:
        client = make_client()
    return client, client.get_world()


def wait_for_simulator(
    timeout_s: float | None = None,
    poll_s: float = DEFAULT_READY_POLL_S,
) -> tuple[carla.Client, carla.World]:
    """Block until CARLA answers and the current map is loaded."""
    deadline = time.time() + (timeout_s if timeout_s is not None else carla_ready_timeout_s())
    host = carla_host()
    port = carla_port()
    last_err: Exception | None = None
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        client = make_client(timeout_s=DEFAULT_PROBE_TIMEOUT_S, host=host)
        try:
            world = client.get_world()
            map_name = world.get_map().name
            print(f"CARLA connected at {host}:{port} (map: {map_name})", flush=True)
            return client, world
        except RuntimeError as exc:
            last_err = exc
            remaining = max(0.0, deadline - time.time())
            print(
                f"  [{attempt}] still waiting for CARLA at {host}:{port} "
                f"({remaining:.0f}s left): {exc}",
                flush=True,
            )
            time.sleep(poll_s)

    hint = (
        f"If the simulator window is open, try: $env:DATASET_CARLA_HOST='127.0.0.1' "
        f"(Windows IPv6 localhost can fail against CARLA's IPv4 listener)."
    )
    raise RuntimeError(
        f"Timed out waiting for CARLA at {host}:{port} "
        f"({timeout_s if timeout_s is not None else carla_ready_timeout_s():.0f}s). "
        f"{hint}"
    ) from last_err
