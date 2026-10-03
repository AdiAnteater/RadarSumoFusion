"""Campaign orchestrator: several traffic scenarios, one sensor capture.

    preflight heal -> scene cleanup -> sensor rig (once)
    -> capture (once, CAMPAIGN MODE: owns the tick, records only the frame
       windows it is told to, see dataset/capture/campaign_control.py)
    -> for each run (row):
           sweep vehicles + pedestrians, verify the world is empty
           start traffic/runner.py (--external-tick) for that row
           wait for its first SUMO step -> frame F
           open recording window [F + warm-up, F + warm-up + duration)
           wait until the capture has ticked past the window
           stop the runner (stop-file; it unmirrors its actors), sweep again
    -> tell the capture to stop; it labels + post-processes the combined
       capture (one radar CSV, one actor log, one camera stream, with
       segments.json mapping frames to rows)
    -> stop sensor rig, final sweep, heal async

The capture keeps ticking between rows, so CARLA never stalls and the sensor
rig never respawns; nothing is recorded during cleanup or warm-up.

A failed row (runner crash, no first tick, world not clean) is marked failed
and skipped; frames it already recorded stay in the capture, marked
"truncated" in segments.json. Stop (GUI button / Ctrl+C) truncates the current
row, skips the rest, and still labels + post-processes what was recorded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import FusionConfig, radar_rig_errors, split_radar_count
from . import orchestrator as _orc

SCENARIO_NAMES = {
    1: "Free Flow", 2: "Moderate Demand", 3: "Heavy Demand", 4: "Stop and Go",
    5: "Mixed Vehicles", 6: "Directional Rush Hour", 7: "Aggressive Lane Changing",
    8: "Bottleneck / Work Zone", 9: "Overtaking", 10: "Multi-Lane Overtake",
    11: "Occlusion",
}


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class RunSpec:
    """One row of the campaign table: one scenario with its traffic settings."""
    scenario: int = 1
    density: int = 50
    duration: int = 120          # RECORDED seconds (warm-up is extra, not recorded)
    direction: str = "BOTH"
    ambient_vehicles: int = 20
    pedestrians: int = 20
    bicycles: int = 10
    seed: int = 42
    signals: str = "green"       # green | static (runner --stretch-signals)
    note: str = ""

    @property
    def scenario_name(self) -> str:
        return SCENARIO_NAMES.get(self.scenario, f"Scenario {self.scenario}")

    def validate(self) -> list:
        errs = []
        if self.scenario not in SCENARIO_NAMES:
            errs.append(f"scenario must be 1-11 (got {self.scenario})")
        if not (1 <= self.density <= 100):
            errs.append(f"density must be 1-100 (got {self.density})")
        if self.duration < 5:
            errs.append(f"duration must be >= 5 s (got {self.duration})")
        if self.direction not in ("WB", "EB", "BOTH"):
            errs.append(f"direction must be WB|EB|BOTH (got {self.direction})")
        for name in ("ambient_vehicles", "pedestrians", "bicycles"):
            v = getattr(self, name)
            if not (0 <= v <= 100):
                errs.append(f"{name} must be 0-100 (got {v})")
        if self.signals not in ("green", "static"):
            errs.append(f"signals must be green|static (got {self.signals})")
        return errs

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class CampaignConfig:
    name: str = ""
    runs: list = field(default_factory=list)       # list[RunSpec]

    # Sensors / clock (campaign-wide: the rig and the tick cannot change mid-capture)
    radars_south: int = 4
    radars_north: int = 4
    radar_height_m: float = 3.0
    rate_hz: float = 20.0
    warmup_s: float = 20.0        # per row, not recorded: traffic reaches the stretch
    label: bool = True
    postprocess: bool = True
    scene_cleanup: bool = True
    clear_trees: bool = True
    sumo_gui: bool = False
    render_radius: float = 150.0
    no_cull: bool = False

    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    traffic_manager_port: int = 8000
    capture_base_dir: str = ""

    sync_timeout: float = 180.0
    sensor_settle_s: float = 5.0
    cleanup_verify_s: float = 15.0
    runner_start_timeout_s: float = 240.0

    @property
    def fixed_delta_s(self) -> float:
        return round(1.0 / float(self.rate_hz), 6)

    @property
    def radar_count(self) -> int:
        return int(self.radars_south) + int(self.radars_north)

    def validate(self) -> list:
        errs = []
        if not self.runs:
            errs.append("the campaign has no runs")
        for i, r in enumerate(self.runs, 1):
            errs += [f"run {i}: {e}" for e in r.validate()]
        errs += radar_rig_errors(self.radars_south, self.radars_north, self.radar_height_m)
        if not (1.0 <= self.rate_hz <= 200.0):
            errs.append(f"rate_hz must be 1-200 (got {self.rate_hz})")
        if self.warmup_s < 0:
            errs.append("warm-up must be >= 0 s")
        return errs

    def fusion_config(self) -> FusionConfig:
        """Adapter so the single-run stage helpers can be reused."""
        return FusionConfig(
            radars_south=self.radars_south, radars_north=self.radars_north,
            radar_height_m=self.radar_height_m, rate_hz=self.rate_hz,
            label=self.label, postprocess=self.postprocess,
            capture_base_dir=self.capture_base_dir,
            carla_host=self.carla_host, carla_port=self.carla_port,
            traffic_manager_port=self.traffic_manager_port,
            scene_cleanup=self.scene_cleanup, clear_trees=self.clear_trees,
            sync_timeout=self.sync_timeout, sensor_settle_s=self.sensor_settle_s,
            render_radius=self.render_radius, no_cull=self.no_cull,
            sumo_gui=self.sumo_gui,
        )

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["radar_count"] = self.radar_count
        d["runs"] = [asdict(r) for r in self.runs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CampaignConfig":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d and k != "runs"}
        if ("radars_south" not in d and "radars_north" not in d
                and d.get("radar_count") is not None):
            known["radars_south"], known["radars_north"] = split_radar_count(d["radar_count"])
        cfg = cls(**known)
        cfg.runs = [RunSpec.from_dict(r) for r in d.get("runs", [])]
        return cfg

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CampaignConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class RowOutcome:
    index: int
    status: str = "queued"       # queued|cleanup|starting|warmup|recording|done|failed|skipped|stopped
    detail: str = ""
    start_frame: int | None = None
    end_frame: int | None = None
    recorded_frames: int = 0


@dataclass
class CampaignResult:
    ok: bool = False
    capture_dir: str = ""
    error: str = ""
    rows: list = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json_atomic(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for _ in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.02)
    tmp.replace(path)


class _Control:
    """Writes numbered commands to the capture's control.json."""

    def __init__(self, control_dir: Path):
        self.dir = control_dir
        self.seq = 0

    def send(self, command: str, **payload) -> None:
        self.seq += 1
        _write_json_atomic(self.dir / "control.json",
                           {"seq": self.seq, "command": command, **payload})

    def status(self) -> dict:
        return _read_json(self.dir / "status.json") or {}


def _sweep(fcfg: FusionConfig, log, verify_s: float) -> bool:
    """Remove every vehicle and pedestrian (never sensors), verify empty.
    Runs with DATASET_EXTERNAL_TICK=1: the capture owns the clock."""
    clr = _orc.TRAFFIC_DIR / "clear_carla_actors.py"
    if not clr.is_file():
        _orc._say(log, "[campaign] WARNING: clear_carla_actors.py missing; cannot sweep")
        return False
    env = _orc._base_env(fcfg, {"DATASET_EXTERNAL_TICK": "1"})
    try:
        res = subprocess.run(
            [sys.executable, str(clr), "--walkers", "--verify", str(verify_s),
             "--host", fcfg.carla_host, "--port", str(fcfg.carla_port)],
            cwd=str(_orc.TRAFFIC_DIR), env=env, check=False,
            timeout=verify_s + 60, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        _orc._say(log, f"[campaign] sweep failed to run: {exc}")
        return False
    for line in (res.stdout or "").splitlines():
        if line.strip():
            _orc._say(log, "  " + line.strip())
    return res.returncode == 0


def _start_row_runner(cfg: CampaignConfig, fcfg: FusionConfig, run: RunSpec,
                      status_file: Path, stop_file: Path, log) -> subprocess.Popen:
    runner = _orc.TRAFFIC_DIR / "runner.py"
    # SUMO must outlast warm-up + recording; the orchestrator ends it earlier
    # through the stop file, so the margin is only an upper bound.
    sumo_duration = int(cfg.warmup_s + run.duration + 60)
    cmd = [
        str(runner),
        "--scenario", str(run.scenario),
        "--density", str(run.density),
        "--duration", str(sumo_duration),
        "--direction", run.direction,
        "--ambient-vehicles", str(run.ambient_vehicles),
        "--pedestrians", str(run.pedestrians),
        "--bicycles", str(run.bicycles),
        "--ambient-seed", str(run.seed),
        "--step-length", str(cfg.fixed_delta_s),
        "--carla-host", cfg.carla_host,
        "--carla-port", str(cfg.carla_port),
        "--render-radius", str(cfg.render_radius),
        "--external-tick",
        "--sync-timeout", str(cfg.sync_timeout),
        "--stretch-signals", run.signals,
        "--status-file", str(status_file),
        "--stop-file", str(stop_file),
    ]
    if cfg.sumo_gui:
        cmd.append("--gui")
    if cfg.no_cull:
        cmd.append("--no-cull")
    env = _orc._base_env(fcfg, {"DATASET_EXTERNAL_TICK": "1"})
    return _orc._popen(cmd, cwd=_orc.TRAFFIC_DIR, env=env, new_group=True)


def _stop_runner(proc, stop_file: Path, log, label: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        stop_file.write_text("stop", encoding="utf-8")
    except OSError:
        pass
    try:
        proc.wait(timeout=45)
        return
    except subprocess.TimeoutExpired:
        pass
    _orc._interrupt(proc, label, timeout_s=25.0, log=log)


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

def run_campaign(cfg: CampaignConfig, log: list | None = None, on_event=None,
                 stop_event: threading.Event | None = None) -> CampaignResult:
    """Run every row of ``cfg`` into one capture. ``on_event(dict)`` receives
    progress (row status / capture state) for a UI; ``stop_event`` ends the
    campaign early (current row truncated, rest skipped, capture still labels).
    """
    lg = log if log is not None else []
    say = lambda m: _orc._say(lg, m)  # noqa: E731
    stop_event = stop_event or threading.Event()
    result = CampaignResult(rows=[RowOutcome(i) for i in range(len(cfg.runs))])

    def emit(**ev):
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:  # noqa: BLE001 - never let the UI kill the run
                pass

    def row_status(i, status, detail="", **extra):
        r = result.rows[i]
        r.status, r.detail = status, detail
        for k, v in extra.items():
            setattr(r, k, v)
        emit(type="row", index=i, status=status, detail=detail, **extra)

    errs = cfg.validate()
    if errs:
        result.error = "invalid campaign: " + "; ".join(errs)
        say("[campaign] " + result.error)
        return result

    fcfg = cfg.fusion_config()
    dt = cfg.fixed_delta_s
    control_dir = Path(tempfile.mkdtemp(prefix="rsf_campaign_"))
    ctl = _Control(control_dir)
    sensor_proc = capture_proc = runner_proc = None
    capture_failed = False

    total_rec = sum(r.duration for r in cfg.runs)
    say(f"[campaign] '{cfg.name or 'unnamed'}': {len(cfg.runs)} run(s), "
        f"{total_rec} s recorded + {cfg.warmup_s:.0f} s warm-up each, "
        f"{cfg.radar_count} radars ({cfg.radars_south}S+{cfg.radars_north}N) "
        f"@ {cfg.rate_hz:.0f} Hz")

    try:
        _orc._heal_async(fcfg, lg, when="preflight")
        if cfg.scene_cleanup:
            _orc._scene_cleanup(fcfg, lg)
        if cfg.clear_trees:
            _orc._clear_stretch_trees(fcfg, lg)
        say("[campaign] initial sweep (vehicles + pedestrians) ...")
        _sweep(fcfg, lg, cfg.cleanup_verify_s)

        sensor_proc = _orc._spawn_sensor_rig(fcfg, lg)
        time.sleep(cfg.sensor_settle_s)
        if sensor_proc.poll() is not None:
            raise RuntimeError(f"sensor rig exited early (code {sensor_proc.returncode})")

        # Capture in campaign mode: no fixed duration, sync mode, control dir.
        script = _orc.DATASET_DIR / "capture" / "CaptureRadarCameraData.py"
        extra = {
            **_orc.rig_env(fcfg),
            "DATASET_SYNC_MODE": "1",
            "DATASET_SYNC_FIXED_DELTA_S": dt,
            "DATASET_RADAR_SENSOR_TICK_S": dt,
            "DATASET_TRAFFIC_MANAGER_PORT": cfg.traffic_manager_port,
            "DATASET_LABEL_RADAR_AFTER_CAPTURE": "1" if cfg.label else "0",
            "DATASET_POSTPROCESS_AFTER_CAPTURE": "1" if cfg.postprocess else "0",
            "DATASET_CAMPAIGN_CONTROL_DIR": str(control_dir),
            "DATASET_CAPTURE_NAME": cfg.name,
        }
        if fcfg.as_capture_base():
            extra["DATASET_CAPTURE_BASE_DIR"] = fcfg.as_capture_base()
        env = _orc._base_env(fcfg, extra)
        env.pop("DATASET_CAPTURE_DURATION_S", None)
        say(f"[campaign] starting capture (campaign mode, owns tick @ {cfg.rate_hz:.0f} Hz) ...")
        capture_proc = _orc._popen([str(script)], cwd=_orc.DATASET_DIR, env=env, new_group=True)

        # Wait until the capture is up and ticking.
        t0 = time.monotonic()
        first_frame = None
        while True:
            st = ctl.status()
            if st.get("run_dir"):
                result.capture_dir = st["run_dir"]
            if st.get("latest_frame"):
                if first_frame is None:
                    first_frame = st["latest_frame"]
                elif st["latest_frame"] > first_frame:
                    break
            if capture_proc.poll() is not None:
                raise RuntimeError(f"capture exited during startup (code {capture_proc.returncode})")
            if time.monotonic() - t0 > 120:
                raise RuntimeError("capture did not start ticking within 120 s")
            if stop_event.is_set():
                raise KeyboardInterrupt
            time.sleep(0.25)
        say(f"[campaign] capture ticking; output: {result.capture_dir}")
        emit(type="capture", state="running", run_dir=result.capture_dir)

        warm_ticks = int(round(cfg.warmup_s / dt))
        for i, run in enumerate(cfg.runs):
            if stop_event.is_set():
                break
            if capture_proc.poll() is not None:
                capture_failed = True
                raise RuntimeError(f"capture exited unexpectedly (code {capture_proc.returncode})")
            tag = f"[campaign] run {i + 1}/{len(cfg.runs)} S{run.scenario:02d} {run.scenario_name}"

            # 1. Clean world.
            row_status(i, "cleanup")
            say(f"{tag}: sweeping vehicles + pedestrians ...")
            if not _sweep(fcfg, lg, cfg.cleanup_verify_s):
                say(f"{tag}: world not empty after sweep; retrying once ...")
                if not _sweep(fcfg, lg, cfg.cleanup_verify_s * 2):
                    row_status(i, "failed", "world not empty after cleanup")
                    say(f"{tag}: SKIPPED (leftover actors would contaminate it).")
                    continue

            # 2. Start the runner, wait for its first SUMO step.
            row_status(i, "starting")
            status_file = control_dir / f"runner_{i}.json"
            stop_file = control_dir / f"runner_{i}.stop"
            say(f"{tag}: starting runner (density {run.density}, {run.direction}, "
                f"seed {run.seed}, signals {run.signals}) ...")
            runner_proc = _start_row_runner(cfg, fcfg, run, status_file, stop_file, lg)
            t0 = time.monotonic()
            r_first = None
            while True:
                rs = _read_json(status_file) or {}
                if rs.get("state") == "running" and rs.get("first_frame"):
                    r_first = int(rs["first_frame"])
                    break
                if runner_proc.poll() is not None:
                    break
                if capture_proc.poll() is not None or stop_event.is_set():
                    break
                if time.monotonic() - t0 > cfg.runner_start_timeout_s:
                    break
                time.sleep(0.2)
            if r_first is None:
                code = runner_proc.poll()
                detail = (f"runner exited (code {code}) before its first step" if code is not None
                          else "runner did not start in time")
                _stop_runner(runner_proc, stop_file, lg, "SUMO runner")
                runner_proc = None
                if stop_event.is_set():
                    row_status(i, "stopped", "campaign stopped")
                    break
                row_status(i, "failed", detail)
                say(f"{tag}: FAILED ({detail}); continuing with the next run.")
                continue

            # 3. Recording window in capture frames.
            start = r_first + warm_ticks
            end = start + int(round(run.duration / dt)) - 1
            seg = {"segment_id": i, "scenario": run.scenario,
                   "scenario_name": run.scenario_name, "density": run.density,
                   "duration_s": run.duration, "direction": run.direction,
                   "ambient_vehicles": run.ambient_vehicles,
                   "pedestrians": run.pedestrians, "bicycles": run.bicycles,
                   "seed": run.seed, "signals": run.signals, "note": run.note,
                   "warmup_s": cfg.warmup_s, "rate_hz": cfg.rate_hz,
                   "runner_first_frame": r_first}
            ctl.send("window", segment=seg, start_frame=start, end_frame=end)
            say(f"{tag}: warm-up {cfg.warmup_s:.0f} s, then recording frames {start}..{end} "
                f"({run.duration} s)")

            # 4. Wait for the window to pass.
            outcome = None
            last_emit = 0.0
            while outcome is None:
                st = ctl.status()
                lf = int(st.get("latest_frame") or 0)
                # The capture may have shifted a window that was asked to start
                # in the past; use what it reports.
                for s in st.get("segments", []):
                    if s.get("segment_id") == i:
                        start, end = s["start_frame"], s["end_frame"]
                if capture_proc.poll() is not None:
                    capture_failed = True
                    outcome = ("failed", "capture exited")
                elif stop_event.is_set():
                    ctl.send("close", segment_id=i, reason="campaign stopped")
                    outcome = ("stopped", "campaign stopped by user")
                elif lf >= end:
                    outcome = ("done", "")
                elif runner_proc.poll() is not None:
                    ctl.send("close", segment_id=i,
                             reason=f"runner exited (code {runner_proc.returncode})")
                    outcome = ("failed", f"runner exited early (code {runner_proc.returncode})")
                else:
                    now = time.monotonic()
                    if now - last_emit >= 0.5:
                        last_emit = now
                        if lf < start:
                            frac = 1.0 - (start - lf) / max(warm_ticks, 1)
                            row_status(i, "warmup", f"{max(0.0, frac) * 100:.0f}%")
                        else:
                            frac = (lf - start + 1) / max(end - start + 1, 1)
                            row_status(i, "recording", f"{frac * 100:.0f}%")
                    time.sleep(0.25)

            # 5. Stop the runner (unmirrors its actors), sweep.
            _stop_runner(runner_proc, stop_file, lg, "SUMO runner")
            runner_proc = None
            time.sleep(0.5)
            st = ctl.status()
            rec = 0
            for s in st.get("segments", []):
                if s.get("segment_id") == i:
                    rec = int(s.get("recorded_frames", 0))
                    start, end = s["start_frame"], s["end_frame"]
            status, detail = outcome
            if status == "failed" and rec:
                detail += f"; {rec * dt:.1f} s recorded (kept, marked truncated)"
            row_status(i, status, detail or f"{rec * dt:.1f} s recorded",
                       start_frame=start, end_frame=end, recorded_frames=rec)
            say(f"{tag}: {status.upper()} {detail}".rstrip())
            if capture_failed:
                raise RuntimeError("capture exited unexpectedly")
            if status == "stopped":
                break

        for r in result.rows:
            if r.status == "queued":
                r.status, r.detail = "skipped", "campaign stopped" if stop_event.is_set() else ""
                emit(type="row", index=r.index, status=r.status, detail=r.detail)

        # 6. Finish: last sweep, stop the capture, let it label + post-process.
        say("[campaign] all runs finished; final sweep ...")
        _sweep(fcfg, lg, cfg.cleanup_verify_s)
        ctl.send("stop", reason="campaign complete")
        say("[campaign] capture stopping; labeling + post-processing the combined "
            "capture (this can take a while) ...")
        last_state = None
        while capture_proc.poll() is None:
            st = ctl.status()
            state = st.get("state")
            if state and state != last_state:
                last_state = state
                emit(type="capture", state=state, run_dir=result.capture_dir)
                say(f"[campaign] capture: {state}")
            time.sleep(1.0)
        code = capture_proc.returncode
        capture_proc = None
        result.ok = code in (0, None) and any(r.status == "done" for r in result.rows)
        if code not in (0, None):
            result.error = f"capture exited with code {code}"

    except KeyboardInterrupt:
        stop_event.set()
        result.error = "interrupted by user"
        say("\n[campaign] interrupted -- stopping the capture (it will still label) ...")
        try:
            ctl.send("stop", reason="interrupted")
        except OSError:
            pass
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        say(f"[campaign] ERROR: {exc}")
        try:
            ctl.send("stop", reason=f"error: {exc}")
        except OSError:
            pass
    finally:
        if runner_proc is not None:
            _stop_runner(runner_proc, control_dir / "runner_final.stop", lg, "SUMO runner")
        if capture_proc is not None and capture_proc.poll() is None:
            # We asked it to stop; give labeling all the time it needs unless
            # the user interrupts again.
            say("[campaign] waiting for the capture to finish (Ctrl+C again to force) ...")
            try:
                capture_proc.wait()
            except KeyboardInterrupt:
                _orc._interrupt(capture_proc, "capture", timeout_s=60.0, log=lg)
        _orc._interrupt(sensor_proc, "sensor rig", log=lg)
        _sweep(fcfg, lg, 5.0)
        _orc._heal_async(fcfg, lg, when="post-run")
        _write_campaign_record(cfg, result, control_dir, lg)
        shutil.rmtree(control_dir, ignore_errors=True)

    say(f"[campaign] done. ok={result.ok}"
        + (f" error={result.error}" if result.error else "")
        + (f"\n[campaign] capture dir: {result.capture_dir}" if result.capture_dir else ""))
    emit(type="finished", ok=result.ok, error=result.error, run_dir=result.capture_dir)
    return result


def _write_campaign_record(cfg: CampaignConfig, result: CampaignResult,
                           control_dir: Path, log) -> None:
    """<run_dir>/campaign.json: the plan, every row's outcome, runner statuses."""
    if not result.capture_dir or not Path(result.capture_dir).is_dir():
        return
    runners = {}
    for p in sorted(control_dir.glob("runner_*.json")):
        runners[p.stem] = _read_json(p)
    record = {
        "campaign": cfg.to_dict(),
        "ok": result.ok,
        "error": result.error,
        "rows": [asdict(r) for r in result.rows],
        "runner_status": runners,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        _write_json_atomic(Path(result.capture_dir) / "campaign.json", record)
        cfg.save(Path(result.capture_dir) / "campaign_plan.json")
    except OSError as exc:
        _orc._say(log, f"[campaign] could not write campaign.json: {exc}")
