"""Two-process fused run orchestrator.

Design (capture owns the tick; SUMO subscribes):

    preflight heal (traffic/carla_check.py --async)
        -> optional scene cleanup (dataset/world/Clear*.py)
        -> spawn sensor rig  (dataset/setup/RadarCameraSetupN.py, keep-alive)
        -> START CAPTURE      (dataset/capture/CaptureRadarCameraData.py)
                              DATASET_SYNC_MODE=1  => it enables CARLA synchronous
                              mode and owns world.tick() at fixed_delta_s.
        -> START RUNNER       (traffic/runner.py --external-tick)
                              subscribes to the capture tick (one SUMO step per
                              CARLA frame), governs all traffic, mirrors to CARLA.
        -> wait for capture to finish (it auto-stops after duration, then labels
           + post-processes per DatasetCreation defaults)
        -> teardown: stop runner (clears mirrored actors), stop sensor rig
           (destroys sensors), best-effort clear.

Exactly one process (capture) ever calls world.tick(). The runner never ticks;
it paces on world.wait_for_tick(). That is the whole fix carried over from the
tick-rate work, applied to a clean two-process merge.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import FusionConfig

ROOT = Path(__file__).resolve().parent.parent
TRAFFIC_DIR = ROOT / "traffic"
DATASET_DIR = ROOT / "dataset"

IS_WIN = sys.platform == "win32"


@dataclass
class FusionResult:
    ok: bool = False
    capture_dir: str = ""
    error: str = ""
    log: list = field(default_factory=list)


# --------------------------------------------------------------------------
# process helpers
# --------------------------------------------------------------------------

def _base_env(cfg: FusionConfig, extra: dict | None = None) -> dict:
    env = os.environ.copy()
    # Shared CARLA endpoint for BOTH trees (dataset uses DATASET_CARLA_*,
    # traffic reads --carla-host/--carla-port which we pass on the CLI).
    env["DATASET_CARLA_HOST"] = cfg.carla_host
    env["DATASET_CARLA_PORT"] = str(cfg.carla_port)
    # Make both roots importable regardless of cwd (children also self-bootstrap).
    pp = [str(DATASET_DIR), str(TRAFFIC_DIR)]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _popen(cmd: list, cwd: Path, env: dict, *, new_group: bool) -> subprocess.Popen:
    kwargs: dict = {"cwd": str(cwd), "env": env}
    if new_group and IS_WIN:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([sys.executable, *cmd], **kwargs)


def _interrupt(proc: subprocess.Popen | None, label: str, timeout_s: float = 25.0,
               log=None) -> None:
    """Ask a child to stop the polite way so its finally-block runs (runner clears
    mirrored actors; sensor rig destroys its sensors). Falls back to kill."""
    if proc is None or proc.poll() is not None:
        return
    _say(log, f"[fusion] stopping {label} ...")
    try:
        if IS_WIN:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except (OSError, ValueError):
        proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _say(log, f"[fusion] {label} did not exit in {timeout_s:.0f}s; killing.")
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _say(log, msg: str) -> None:
    print(msg, flush=True)
    if log is not None:
        log.append(msg)


# --------------------------------------------------------------------------
# individual stages
# --------------------------------------------------------------------------

def _heal_async(cfg: FusionConfig, log, *, when: str) -> None:
    """Force CARLA back to asynchronous free-run. Runs at the START of a run
    (recover a world a previous run left frozen) and at the END (guarantee the
    server is async so the NEXT process to connect -- another capture, Start.py,
    anything -- does not open onto a stopped, synchronous world and hang/die).

    Root cause of the 'freeze when I start another process after capture' report:
    if capture ever exits without its finally restoring async (hard kill, crash,
    or Ctrl+C on Windows where the finally is skipped), the world stays in
    synchronous mode with nobody ticking. This end-of-run heal closes that hole
    regardless of how the children exited."""
    check = TRAFFIC_DIR / "carla_check.py"
    if not check.is_file():
        return
    _say(log, f"[fusion] {when}: ensuring CARLA is in asynchronous mode ...")
    try:
        subprocess.run(
            [sys.executable, str(check), "--async",
             "--host", cfg.carla_host, "--port", str(cfg.carla_port)],
            cwd=str(TRAFFIC_DIR), env=_base_env(cfg), check=False, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        _say(log, f"[fusion] {when} heal skipped ({exc}).")


def _scene_cleanup(cfg: FusionConfig, log) -> None:
    for name in ("ClearParkedCarsAndMotorcycles.py", "ClearTrashCansAndMailboxes.py"):
        script = DATASET_DIR / "world" / name
        if not script.is_file():
            continue
        _say(log, f"[fusion] scene cleanup: {name} ...")
        try:
            subprocess.run([sys.executable, str(script)], cwd=str(DATASET_DIR),
                           env=_base_env(cfg), check=False, timeout=120)
        except Exception as exc:  # noqa: BLE001
            _say(log, f"[fusion] {name} skipped ({exc}).")


def _spawn_sensor_rig(cfg: FusionConfig, log) -> subprocess.Popen:
    script = DATASET_DIR / "setup" / cfg.setup_script_name()
    if not script.is_file():
        raise FileNotFoundError(f"sensor setup script not found: {script}")
    delta = cfg.fixed_delta_s
    env = _base_env(cfg, {
        "DATASET_KEEP_SENSORS_RUNNING": "1",
        "DATASET_EXPECTED_RADAR_COUNT": cfg.radar_count,
        "DATASET_RADAR_SENSOR_TICK_S": delta,
        "DATASET_TRAFFIC_MANAGER_PORT": cfg.traffic_manager_port,
    })
    _say(log, f"[fusion] spawning {cfg.radar_count}-radar rig "
              f"({cfg.setup_script_name()}), keep-alive ...")
    # New group so we can CTRL_BREAK it later and let its finally destroy sensors.
    proc = _popen([str(script)], cwd=DATASET_DIR, env=env, new_group=True)
    return proc


def _start_capture(cfg: FusionConfig, log) -> subprocess.Popen:
    script = DATASET_DIR / "capture" / "CaptureRadarCameraData.py"
    if not script.is_file():
        raise FileNotFoundError(f"capture script not found: {script}")
    delta = cfg.fixed_delta_s
    extra = {
        # Capture OWNS the tick.
        "DATASET_SYNC_MODE": "1",
        "DATASET_SYNC_FIXED_DELTA_S": delta,
        "DATASET_RADAR_SENSOR_TICK_S": delta,
        "DATASET_CAPTURE_DURATION_S": int(cfg.duration),
        "DATASET_EXPECTED_RADAR_COUNT": cfg.radar_count,
        "DATASET_TRAFFIC_MANAGER_PORT": cfg.traffic_manager_port,
        # Post-capture behavior: honor DatasetCreation defaults unless overridden.
        "DATASET_LABEL_RADAR_AFTER_CAPTURE": "1" if cfg.label else "0",
        "DATASET_POSTPROCESS_AFTER_CAPTURE": "1" if cfg.postprocess else "0",
    }
    if cfg.as_capture_base():
        extra["DATASET_CAPTURE_BASE_DIR"] = cfg.as_capture_base()
    env = _base_env(cfg, extra)
    _say(log, f"[fusion] starting capture (owns tick @ {1.0/delta:.0f} Hz, "
              f"fixed_delta={delta}s); auto-stop after {cfg.duration}s ...")
    return _popen([str(script)], cwd=DATASET_DIR, env=env, new_group=True)


def _start_runner(cfg: FusionConfig, log) -> subprocess.Popen:
    runner = TRAFFIC_DIR / "runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"SUMO runner not found: {runner}")
    delta = cfg.fixed_delta_s
    cmd = [
        str(runner),
        "--scenario", str(cfg.scenario),
        "--density", str(cfg.density),
        "--duration", str(int(cfg.duration) + 120),   # cover capture + margin
        "--direction", cfg.direction,
        "--ambient-vehicles", str(cfg.ambient_vehicles),
        "--pedestrians", str(cfg.pedestrians),
        "--bicycles", str(cfg.bicycles),
        "--ambient-seed", str(cfg.ambient_seed),
        "--step-length", str(delta),
        "--carla-host", cfg.carla_host,
        "--carla-port", str(cfg.carla_port),
        "--render-radius", str(cfg.render_radius),
        "--external-tick",
        "--sync-timeout", str(cfg.sync_timeout),
    ]
    if cfg.sumo_gui:
        cmd.append("--gui")
    if cfg.no_cull:
        cmd.append("--no-cull")
    env = _base_env(cfg, {"DATASET_EXTERNAL_TICK": "1"})
    _say(log, f"[fusion] starting SUMO runner (subscriber) scenario={cfg.scenario} "
              f"density={cfg.density} dir={cfg.direction} ...")
    return _popen(cmd, cwd=TRAFFIC_DIR, env=env, new_group=True)


def _read_capture_dir() -> str:
    pointer = DATASET_DIR / "capture" / ".last_dataset_capture_dir"
    try:
        raw = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not raw:
        return ""
    p = Path(raw)
    return str(p) if p.is_dir() else raw


# --------------------------------------------------------------------------
# top-level run
# --------------------------------------------------------------------------

def run(cfg: FusionConfig, log: list | None = None) -> FusionResult:
    cfg.validate()
    result = FusionResult(log=log if log is not None else [])
    lg = result.log
    if not cfg.ok:
        result.error = "invalid config: " + "; ".join(cfg.errors)
        _say(lg, "[fusion] " + result.error)
        return result

    if not TRAFFIC_DIR.is_dir() or not DATASET_DIR.is_dir():
        result.error = (f"missing component dirs: traffic={TRAFFIC_DIR.is_dir()} "
                        f"dataset={DATASET_DIR.is_dir()}")
        _say(lg, "[fusion] " + result.error)
        return result

    sensor_proc: subprocess.Popen | None = None
    capture_proc: subprocess.Popen | None = None
    runner_proc: subprocess.Popen | None = None

    try:
        _heal_async(cfg, lg, when="preflight")
        if cfg.scene_cleanup:
            _scene_cleanup(cfg, lg)

        sensor_proc = _spawn_sensor_rig(cfg, lg)
        time.sleep(cfg.sensor_settle_s)
        if sensor_proc.poll() is not None:
            raise RuntimeError(f"sensor rig exited early (code {sensor_proc.returncode})")

        capture_proc = _start_capture(cfg, lg)
        time.sleep(cfg.capture_start_gap_s)
        if capture_proc.poll() is not None:
            raise RuntimeError(f"capture exited early (code {capture_proc.returncode})")

        runner_proc = _start_runner(cfg, lg)

        # Wait for capture: it auto-stops after duration, then labels/post-processes.
        _say(lg, "[fusion] running; waiting for capture to finish "
                 "(auto-stops after duration, then labels/post) ...")
        capture_proc.wait()
        code = capture_proc.returncode
        capture_proc = None
        _say(lg, f"[fusion] capture finished (exit {code}).")

        result.capture_dir = _read_capture_dir()
        result.ok = code in (0, None)
        if not result.ok:
            result.error = f"capture exited with code {code}"

    except KeyboardInterrupt:
        result.error = "interrupted by user"
        _say(lg, "\n[fusion] interrupted -- tearing down ...")
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        _say(lg, f"[fusion] ERROR: {exc}")
    finally:
        # Order: runner first (unmirror actors while sensors/world still up),
        # then capture (restores async + writes CSVs), then sensor rig (destroys
        # sensors). All via CTRL_BREAK/SIGINT so finally-blocks run.
        _interrupt(runner_proc, "SUMO runner", log=lg)
        _interrupt(capture_proc, "capture", timeout_s=120.0, log=lg)
        _interrupt(sensor_proc, "sensor rig", log=lg)
        _best_effort_clear(cfg, lg)
        # Guarantee the server is async before we return, so the NEXT process the
        # user starts does not connect onto a frozen synchronous world. This is
        # the fix for the post-run freeze/crash.
        _heal_async(cfg, lg, when="post-run")

    if result.capture_dir:
        _say(lg, f"[fusion] capture dir: {result.capture_dir}")
    _say(lg, f"[fusion] done. ok={result.ok}"
             + (f" error={result.error}" if result.error else ""))
    return result


def _best_effort_clear(cfg: FusionConfig, log) -> None:
    """Sweep any mirrored vehicles left behind. Never touches sensor.* actors."""
    clr = TRAFFIC_DIR / "clear_carla_actors.py"
    if not clr.is_file():
        return
    try:
        subprocess.run([sys.executable, str(clr),
                        "--host", cfg.carla_host, "--port", str(cfg.carla_port)],
                       cwd=str(TRAFFIC_DIR), env=_base_env(cfg),
                       check=False, timeout=60)
    except Exception:  # noqa: BLE001
        pass
