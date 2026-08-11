import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import sleep, time

from bootstrap import prime_and_prepare, prime_imports

prime_and_prepare(__file__)

from dataset_paths import (
    capture_dir,
    data_output_dir,
    dataset_root,
    pythonpath_env,
    testing_dir,
    venv_site_packages,
    world_dir,
)

# Only one of these runs; others are excluded from the auto-launch list.
RADAR_SETUP_SCRIPTS = frozenset(
    {
        "RadarCameraSetup12.py",
        "RadarCameraSetup4.py",
        "RadarCameraSetup8.py",
        "RadarCameraSetup14.py",
    }
)

# Never auto-launched with the full dataset stack (run explicitly or via test mode).
MANUAL_ONLY_SCRIPTS = frozenset(
    {
        "TestRadarLabeling.py",
        "RadarLabelingTestReport.py",
        "FetchActorSizing.py",
        "PrintRadarLayoutExtrinsics.py",
    }
)

# Preferred order for full pipeline (after radar setup, before capture).
FULL_PIPELINE_MIDDLE_SCRIPTS = (
    "ClearParkedCarsAndMotorcycles.py",
    "ClearTrashCansAndMailboxes.py",
    "SpawnCarsAtPosition14.py",
    "SpawnPedestriansAcrossMap.py",
    "TrafficLightSetup.py",
    "TrafficLightControl.py",
)

DEFAULT_PEDESTRIAN_COUNT = 30

# Radar count -> setup script (each layout is mutually exclusive).
RADAR_COUNT_TO_SETUP = {
    4: "RadarCameraSetup4.py",
    8: "RadarCameraSetup8.py",
    12: "RadarCameraSetup12.py",
    14: "RadarCameraSetup14.py",
}

# Test runs labeling after setup has had time to spawn sensors; traffic spawns in parallel.
TEST_MODE_SCRIPTS_AFTER_SETUP = (
    "SpawnCarsAtPosition14.py",
    "SpawnPedestriansAcrossMap.py",
    "TrafficLightSetup.py",
    "TrafficLightControl.py",
    "TestRadarLabeling.py",
)


def script_path(dc_root: Path, name: str) -> Path:
    if name in RADAR_SETUP_SCRIPTS:
        return dc_root / "setup" / name
    if name in ("CaptureRadarCameraData.py", "ExportRadarExtrinsics.py", "ExportCameraExtrinsics.py"):
        return dc_root / "capture" / name
    if name in ("TestRadarLabeling.py", "RadarLabelingTestReport.py"):
        return dc_root / "testing" / name
    if name in ("FetchActorSizing.py", "PrintRadarLayoutExtrinsics.py"):
        return dc_root / "tools" / name
    return dc_root / "world" / name


def prompt_radar_count() -> int:
    print("How many radars should the dataset use?")
    print("  1) 4   -> setup/RadarCameraSetup4.py")
    print("  2) 8   -> setup/RadarCameraSetup8.py")
    print("  3) 12  -> setup/RadarCameraSetup12.py")
    print("  For 14 -> setup/RadarCameraSetup14.py: type 14 at the prompt.")
    print("Enter menu 1-3, or type the radar count: 4, 8, 12, or 14.")
    allowed = frozenset({4, 8, 12, 14})
    menu = {"1": 4, "2": 8, "3": 12}
    while True:
        choice = input("Choice (default 2 -> 8 radars): ").strip() or "2"
        try:
            n = int(choice)
        except ValueError:
            n = None
        if n is not None and n in allowed:
            return n
        if choice in menu:
            return menu[choice]
        print("Please enter 1-3, or the radar count 4, 8, 12, or 14.")


def prompt_run_mode() -> str:
    print("\nRun mode:")
    print("  1) Full dataset pipeline (all scripts + capture/CaptureRadarCameraData.py)")
    print(
        "  2) Test radar labeling only "
        "(setup + spawn cars/pedestrians + testing/TestRadarLabeling.py)"
    )
    while True:
        choice = input("Choice (default 1): ").strip() or "1"
        if choice in ("1", "full", "dataset"):
            return "full"
        if choice in ("2", "test", "test-labeling"):
            return "test"
        print("Please enter 1 or 2.")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch dataset creation scripts or radar labeling test mode.",
    )
    parser.add_argument(
        "--test-labeling",
        action="store_true",
        help="Test mode: RadarCameraSetup + SpawnCars + TestRadarLabeling (no full capture).",
    )
    parser.add_argument(
        "--radar-count",
        type=int,
        choices=[4, 8, 12, 14],
        help="Radar layout (skips interactive prompt when set).",
    )
    return parser.parse_args()


def get_scripts_to_start(dc_root: Path, radar_setup_filename: str) -> list[Path]:
    setup_path = script_path(dc_root, radar_setup_filename)
    if not setup_path.is_file():
        raise FileNotFoundError(f"Radar setup script not found: {setup_path}")

    ordered_middle = [script_path(dc_root, name) for name in FULL_PIPELINE_MIDDLE_SCRIPTS]
    missing = [p for p in ordered_middle if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Pipeline script not found: {missing[0]}")

    capture_path = script_path(dc_root, "CaptureRadarCameraData.py")
    if not capture_path.is_file():
        raise FileNotFoundError(f"Capture script not found: {capture_path}")

    return [setup_path, *ordered_middle, capture_path]


def get_scripts_for_test_mode(dc_root: Path, radar_setup_filename: str) -> list[Path]:
    """Minimal stack to validate radar labeling (vehicles + pedestrians)."""
    setup_path = script_path(dc_root, radar_setup_filename)
    if not setup_path.is_file():
        raise FileNotFoundError(f"Radar setup script not found: {setup_path}")

    scripts = [setup_path]
    for name in TEST_MODE_SCRIPTS_AFTER_SETUP:
        path = script_path(dc_root, name)
        if not path.is_file():
            raise FileNotFoundError(f"Test mode script not found: {path}")
        scripts.append(path)
    return scripts


def resolve_dataset_export_dir(dc_root: Path) -> Path | None:
    """
    Active capture run from capture/.last_dataset_capture_dir (written by CaptureRadarCameraData.py),
    else the newest sensor_capture_* under Data/ with both radar and camera metadata CSVs.
    """
    cap = capture_dir(dc_root)
    pointer = cap / ".last_dataset_capture_dir"
    if pointer.is_file():
        try:
            raw = pointer.read_text(encoding="utf-8").strip()
            p = Path(raw)
            if not p.is_absolute():
                p = (cap / p).resolve()
            else:
                p = p.resolve()
            if (
                p.is_dir()
                and (p / "radar_data.csv").exists()
                and (p / "camera_data.csv").exists()
            ):
                return p
        except OSError:
            pass
    data_root = data_output_dir(dc_root)
    if not data_root.is_dir():
        return None
    candidates = [
        p
        for p in data_root.glob("sensor_capture_*")
        if p.is_dir()
        and (p / "radar_data.csv").exists()
        and (p / "camera_data.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


REQUEST_STOP_FILENAME = ".request_stop"
REPORT_COMPLETE_FILENAME = ".report_complete"
TEST_LABELING_WAIT_S = 120.0
LIVE_STATS_POLL_INTERVAL_S = 2.0


def resolve_last_test_output_dir(dc_root: Path) -> Path | None:
    pointer = testing_dir(dc_root) / ".last_radar_labeling_test_dir"
    if not pointer.is_file():
        return None
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (testing_dir(dc_root) / path).resolve()
    return path if path.is_dir() else None


def format_live_stats_console_line(data: dict) -> str:
    wc = data.get("with_candidates", 0)
    rate_c = data.get("match_rate_given_candidates", 0.0)
    return (
        "[LiveStats] "
        f"msgs={data.get('radar_messages', 0):,} "
        f"raw={data.get('raw_radar_returns', 0):,} | "
        f"scored={data.get('total_detections', 0):,} | "
        f"matched={data.get('matched_detections', 0):,} | "
        f"w/ candidates={wc:,} → {100 * rate_c:.1f}% | "
        f"q={data.get('queue_pending', 0)} drop={data.get('queue_dropped', 0)} | "
        f"updated={data.get('updated_at', '?')} (#{data.get('seq', 0)})"
    )


def tick_live_stats_display(
    dc_root: Path, last_key: tuple[int, str] | None
) -> tuple[int, str] | None:
    """Print when live_stats.json changes (polled from Start.py)."""
    out_dir = resolve_last_test_output_dir(dc_root)
    if out_dir is None:
        return last_key
    path = out_dir / "live_stats.json"
    if not path.is_file():
        return last_key
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return last_key
    key = (int(data.get("seq", 0)), str(data.get("updated_at", "")))
    if key == last_key:
        return last_key
    print(format_live_stats_console_line(data), flush=True)
    return key


def request_test_labeling_stop(dc_root: Path) -> Path | None:
    out_dir = resolve_last_test_output_dir(dc_root)
    if out_dir is None:
        return None
    (out_dir / REQUEST_STOP_FILENAME).write_text("", encoding="utf-8")
    return out_dir


def wait_for_test_labeling_export(out_dir: Path, timeout_s: float = TEST_LABELING_WAIT_S) -> bool:
    """Wait until TestRadarLabeling finishes plots, summary.txt, and .report_complete."""
    deadline = time() + timeout_s
    meta_path = out_dir / "run_meta.json"
    complete_path = out_dir / REPORT_COMPLETE_FILENAME
    while time() < deadline:
        if complete_path.is_file() and (out_dir / "summary.txt").is_file():
            return True
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            if meta.get("status") in ("completed", "failed") and (out_dir / "summary.json").is_file():
                return True
        sleep(1.0)
    return False


def print_test_labeling_summaries(out_dir: Path) -> None:
    """Echo final test summaries to this console after TestRadarLabeling exits."""
    print("\n" + "=" * 60)
    print("RADAR LABELING TEST — END OF SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Folder: {out_dir.resolve()}\n")

    summary_txt = out_dir / "summary.txt"
    if summary_txt.is_file():
        print(summary_txt.read_text(encoding="utf-8"), end="")
    else:
        print("(summary.txt not found — run may have been interrupted early)")

    summary_json = out_dir / "summary.json"
    if summary_json.is_file():
        try:
            data = json.loads(summary_json.read_text(encoding="utf-8"))
            snap = data.get("summary", {})
            if snap:
                wc = snap.get("with_candidates", 0)
                rate_c = snap.get("match_rate_given_candidates", 0.0)
                print(
                    f"\nKey metrics: scored={snap.get('total_detections', 0):,}, "
                    f"matched={snap.get('matched_detections', 0):,}, "
                    f"w/ candidates={wc:,}, "
                    f"match among candidates={100 * rate_c:.1f}%, "
                    f"PASS={data.get('pass', '?')}"
                )
        except json.JSONDecodeError:
            pass

    print("\nArtifacts:")
    for name in (
        "radar_labeling_summary.png",
        "busiest_frame_summary.png",
        "per_sensor_summary.csv",
        "per_vehicle_summary.csv",
        "per_pedestrian_summary.csv",
        "per_frame_summary.csv",
        "labeling_failure_samples.csv",
        "vehicle_radar_matrix.csv",
        "live_stats.json",
    ):
        path = out_dir / name
        print(f"  [{'ok' if path.is_file() else '—'}] {name}")
    print("=" * 60 + "\n", flush=True)


def _on_term_signal(signum, frame):
    """Make SIGTERM behave like Ctrl+C so a single per-PID signal to this process
    triggers the graceful shutdown handler (reliable even for background launches,
    where Ctrl+C can't be delivered to the whole process group)."""
    raise KeyboardInterrupt


def stop_all(processes: list[subprocess.Popen], timeout_s: float = 5.0) -> None:
    """Stop child scripts, letting each run its own cleanup first.

    SIGINT is sent before terminate()/kill() so every script hits its
    KeyboardInterrupt/finally path and destroys the actors it spawned (radars,
    walkers, ...). Plain terminate() (SIGTERM) skips those finally blocks, which is
    what kept orphaning dataset radars in the world across runs.
    """
    if not processes:
        return

    print("\nStopping all scripts...")

    # 1) Graceful: SIGINT → each script's finally destroys its actors.
    for process in processes:
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass

    # 2) Escalate: terminate() anything that ignored SIGINT.
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass

    # 3) Last resort: SIGKILL.
    for process in processes:
        if process.poll() is None:
            print(f"  - Force killing PID {process.pid}")
            try:
                process.kill()
                process.wait()
            except OSError:
                pass

    print("All scripts stopped.")


def destroy_leftover_dataset_sensors(dc_root: Path) -> None:
    """Belt-and-suspenders world cleanup: destroy any dataset_radar_* / dataset_camera_*
    sensors still present after the child scripts stopped, so the world is left clean
    even if a setup script was force-killed before its own finally ran."""
    try:
        from carla_connect import get_world

        _, world = get_world()
        killed = 0
        for actor in world.get_actors():
            if not actor.type_id.startswith("sensor."):
                continue
            role = actor.attributes.get("role_name", "")
            if role.startswith("dataset_radar_") or role.startswith("dataset_camera_"):
                try:
                    actor.destroy()
                    killed += 1
                except RuntimeError:
                    pass
        print(
            f"Leftover dataset sensors destroyed: {killed}."
            if killed
            else "World clean: no leftover dataset sensors.",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"(leftover-sensor cleanup skipped: {exc})", flush=True)


def export_dataset_extrinsics_in_process(dc_root: Path, dataset_dir: Path | None) -> None:
    """
    Same in-process path as CaptureRadarCameraData: avoids spawning python.exe, which
    often failed to import `carla` and produced no extrinsic files.
    """
    target = dataset_dir
    if target is None:
        target = resolve_dataset_export_dir(dc_root)
    if target is None:
        print(
            "No capture folder found for extrinsics (need sensor_capture_* with both CSVs under Data/).",
            file=sys.stderr,
        )
        return
    cap = capture_dir(dc_root)
    prime_imports(dc_root)
    try:
        from carla_connect import get_world
        from capture.ExportCameraExtrinsics import write_camera_extrinsics_to_dataset_dir
        from capture.ExportRadarExtrinsics import write_radar_extrinsics_live_to_dataset_dir
    except ImportError as e:
        print(f"Extrinsics: could not import CARLA/export modules: {e}", file=sys.stderr)
        return

    print(
        "Exporting camera + radar extrinsics (keep CARLA and sensor scripts running)...",
        flush=True,
    )
    try:
        client, world = get_world()
        write_camera_extrinsics_to_dataset_dir(world, target)
        write_radar_extrinsics_live_to_dataset_dir(world, target)
    except Exception as e:  # noqa: BLE001
        print(f"Extrinsics export error: {e}", file=sys.stderr)


def despawn_all_cars(dc_root: Path) -> None:
    """Run the dedicated car-despawn script."""
    despawn_script = world_dir(dc_root) / "DespawnAllCars.py"
    if not despawn_script.is_file():
        print(f"Despawn script not found: {despawn_script}")
        return

    print("Despawning cars...")
    completed = subprocess.run(
        [sys.executable, str(despawn_script)],
        check=False,
        cwd=str(dc_root),
        env={**os.environ, "PYTHONPATH": pythonpath_env(dc_root)},
    )
    if completed.returncode == 0:
        print("Car despawn completed.")
    else:
        print(f"Car despawn exited with code {completed.returncode}.")


def launch_scripts(
    scripts: list[Path],
    dc_root: Path,
    child_env: dict[str, str],
    *,
    test_mode: bool,
) -> list[subprocess.Popen]:
    processes: list[subprocess.Popen] = []
    for script in scripts:
        try:
            rel = script.relative_to(dc_root)
        except ValueError:
            rel = script
        print(f"  - {rel}")
        popen_kwargs: dict = {
            "env": child_env,
            "cwd": str(dc_root),
        }
        if script.name == "TrafficLightControl.py" and sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        process = subprocess.Popen(
            [sys.executable, str(script)],
            **popen_kwargs,
        )
        processes.append(process)
        if script.name in RADAR_SETUP_SCRIPTS:
            sleep(12.0 if test_mode else 2.0)
        elif script.name == "TrafficLightSetup.py":
            sleep(4.0)
        elif script.name == "TrafficLightControl.py":
            sleep(2.0)
        elif script.name == "SpawnPedestriansAcrossMap.py":
            sleep(3.0)
        elif script.name == "SpawnCarsAtPosition14.py":
            sleep(5.0)
        elif test_mode and script.name == "TestRadarLabeling.py":
            sleep(2.0)
    return processes


def ensure_carla_importable(dc_root: Path) -> None:
    """Fail fast with a clear message if child scripts cannot import carla."""
    probe_env = os.environ.copy()
    probe_env["PYTHONPATH"] = pythonpath_env(dc_root)
    result = subprocess.run(
        [sys.executable, "-c", "import carla"],
        env=probe_env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    venv_sp = venv_site_packages(dc_root)
    lines = [
        "CARLA Python API is not available to child scripts.",
        f"  Interpreter: {sys.executable}",
    ]
    if venv_sp is not None:
        lines.append(f"  Found project venv packages at: {venv_sp}")
        lines.append(
            "  Your .venv may be broken (pyvenv.cfg points at a missing Python). "
            "Recreate it, reinstall carla, then activate before running Start.py."
        )
    else:
        lines.append(
            "  Install the CARLA egg/wheel for your Python version, or create a project .venv "
            "with carla installed."
        )
    if result.stderr.strip():
        lines.append(f"  Import error: {result.stderr.strip()}")
    raise SystemExit("\n".join(lines))


def wait_for_carla_ready(dc_root: Path) -> None:
    """Do not launch child scripts until the simulator responds."""
    prime_imports(dc_root)
    from carla_connect import carla_host, carla_port, wait_for_simulator

    print(
        f"Waiting for CARLA at {carla_host()}:{carla_port()} "
        "(start the simulator if it is not running)...",
        flush=True,
    )
    _, world = wait_for_simulator()
    print(f"CARLA ready — map: {world.get_map().name}", flush=True)


def _resolve_last_capture_dir(dc_root: Path) -> Path | None:
    """Return the active capture run directory from the .last_dataset_capture_dir pointer."""
    pointer = capture_dir(dc_root) / ".last_dataset_capture_dir"
    try:
        raw = pointer.read_text(encoding="utf-8").strip()
        p = Path(raw)
        if not p.is_absolute():
            p = (capture_dir(dc_root) / p).resolve()
        if p.is_dir():
            return p
    except OSError:
        pass
    return None


def _estimate_capture_wait_s(dc_root: Path) -> float:
    """Compute a shutdown wait budget that covers offline radar labeling.

    Strategy: inspect radar_data.csv file size, divide by a conservative
    bytes-per-second throughput to get an estimated labeling duration, then
    add fixed overhead (drain + extrinsics) and a 50 % safety margin.
    Falls back to a generous fixed value if the file can't be found.
    """
    # Fixed overhead: drain queue (≤30s) + extrinsics export + actor-frames load.
    # actor_frames.jsonl can be 600 MB+ for a 2 min capture and parsing it into
    # the in-memory dict takes 30-60 s on its own — bump the overhead so the
    # in-process labeler isn't killed before it even starts iterating.
    OVERHEAD_S = 240.0
    # Conservative labeling throughput observed in practice: ~3 000 rows/s.
    # Each radar CSV row is roughly 200 bytes → ~600 kB/s.
    BYTES_PER_SECOND = 600_000.0
    SAFETY_FACTOR = 1.5
    # Bumped from 300 → 900 because the original budget routinely guillotined
    # the in-process offline labeler on captures with 1 M+ radar rows (which
    # is the common case once you have 8 radars and 2 min of streaming).
    # Forcing users to re-run capture/LabelRadarCapture by hand defeated the
    # whole point of DATASET_LABEL_RADAR_AFTER_CAPTURE=1.
    MIN_WAIT_S = 900.0   # 15 min — covers 1.5 M row captures with safety.
    MAX_WAIT_S = 7200.0  # Cap at 2 h to avoid hanging indefinitely.

    run_dir = _resolve_last_capture_dir(dc_root)
    if run_dir is not None:
        radar_csv = run_dir / "radar_data.csv"
        try:
            size_bytes = radar_csv.stat().st_size
            estimated_label_s = size_bytes / BYTES_PER_SECOND * SAFETY_FACTOR
            total = OVERHEAD_S + estimated_label_s
            wait_s = max(MIN_WAIT_S, min(total, MAX_WAIT_S))
            print(
                f"[shutdown] radar_data.csv is {size_bytes / 1e6:.1f} MB → "
                f"estimated labeling wait: {wait_s:.0f}s",
                flush=True,
            )
            return wait_s
        except OSError:
            pass

    return max(MIN_WAIT_S, OVERHEAD_S)


def main() -> None:
    # SIGTERM triggers the same graceful shutdown as Ctrl+C (KeyboardInterrupt) so the
    # pipeline can be torn down reliably with a single per-PID signal to THIS process.
    signal.signal(signal.SIGTERM, _on_term_signal)
    cli = parse_cli_args()
    dc_root = dataset_root()
    prime_imports(dc_root)
    ensure_carla_importable(dc_root)

    radar_count = cli.radar_count if cli.radar_count is not None else prompt_radar_count()
    run_mode = "test" if cli.test_labeling else prompt_run_mode()
    radar_setup_name = RADAR_COUNT_TO_SETUP[radar_count]

    if run_mode == "test":
        scripts = get_scripts_for_test_mode(dc_root, radar_setup_name)
    else:
        scripts = get_scripts_to_start(dc_root, radar_setup_name)

    if not scripts:
        print("No scripts found to start.")
        return

    test_mode = run_mode == "test"
    # Honor an externally-set capture base dir (e.g. run_sim.sh points this at /scratch,
    # local + fast + roomy) instead of forcing captures onto the repo's Data/ (NFS home).
    _env_capture_base = os.environ.get("DATASET_CAPTURE_BASE_DIR", "").strip()
    data_dir = Path(_env_capture_base) if _env_capture_base else data_output_dir(dc_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = pythonpath_env(dc_root)
    child_env["DATASET_CARLA_HOST"] = os.environ.get("DATASET_CARLA_HOST", "127.0.0.1")
    child_env["DATASET_CARLA_TIMEOUT_S"] = os.environ.get("DATASET_CARLA_TIMEOUT_S", "60")
    child_env["DATASET_EXPECTED_RADAR_COUNT"] = str(radar_count)
    child_env["DATASET_PEDESTRIAN_COUNT"] = str(DEFAULT_PEDESTRIAN_COUNT)
    child_env["DATASET_CAPTURE_BASE_DIR"] = str(data_dir)
    child_env["DATASET_KEEP_PEDESTRIANS_RUNNING"] = "1"
    child_env["DATASET_KEEP_TRAFFIC_RUNNING"] = "1"
    child_env["DATASET_FREE_VEHICLE_DRIVING"] = "1"
    # Default "1": let CARLA cycle the lights so corridor traffic flows. "0" freezes
    # a perimeter ring of lights RED to trap cars in the zone — that starves the
    # corridor (vehicles queue at red and never reach the radars). Override per-run.
    child_env["DATASET_AUTOMATIC_TRAFFIC_LIGHTS"] = os.environ.get(
        "DATASET_AUTOMATIC_TRAFFIC_LIGHTS", "1"
    )
    child_env["DATASET_TRAFFIC_LIGHT_GUI_AUTOCONNECT"] = "1"
    if test_mode:
        child_env["DATASET_KEEP_SENSORS_RUNNING"] = "1"
    mode_label = "TEST (radar labeling)" if test_mode else "FULL dataset"

    print(f"Mode: {mode_label}")
    print(f"Radar layout: {radar_count} sensors via setup/{radar_setup_name}")
    print(f"Dataset output: {data_dir}")
    print(f"Starting {len(scripts)} scripts from {dc_root}:")
    if test_mode:
        print(
            "Radar test: [RadarTest] + [LiveStats] every ~2s (live_stats.json); "
            "autosave every 90s. Press Enter or Ctrl+C to stop — full summary at end "
            f"(waits up to {TEST_LABELING_WAIT_S:.0f}s for plots + summary.txt)."
        )
    else:
        print(
            "Traffic: TrafficLightSetup (perimeter lights cycle) + TrafficLightControl GUI "
            "(auto-connects). Free-roaming TM vehicles and navmesh pedestrians. "
            "On stop: extrinsics + radar_labeling_qa/ under Data/sensor_capture_*."
        )
        print(
            "On Ctrl+C: camera + radar extrinsics are exported into the active sensor_capture_* folder, "
            "then scripts stop. Keep CARLA running until you see the export messages."
        )

    wait_for_carla_ready(dc_root)

    processes = launch_scripts(scripts, dc_root, child_env, test_mode=test_mode)

    print("All scripts started. Press Ctrl+C to stop all.")

    is_test_mode = run_mode == "test"
    live_stats_key: tuple[int, str] | None = None
    last_live_poll = 0.0
    try:
        while True:
            if is_test_mode:
                now = time()
                if now - last_live_poll >= LIVE_STATS_POLL_INTERVAL_S:
                    live_stats_key = tick_live_stats_display(dc_root, live_stats_key)
                    last_live_poll = now
            sleep(0.25)
    except KeyboardInterrupt:
        if is_test_mode:
            out_dir = request_test_labeling_stop(dc_root)
            if out_dir is not None:
                print(
                    f"\nRequested TestRadarLabeling to stop and save to:\n  {out_dir}",
                    flush=True,
                )
                print(
                    f"Waiting up to {TEST_LABELING_WAIT_S:.0f}s for final report...",
                    flush=True,
                )
                labeling_proc = next(
                    (
                        p
                        for p in processes
                        if p.poll() is None and p.args and "TestRadarLabeling" in p.args[-1]
                    ),
                    None,
                )
                if wait_for_test_labeling_export(out_dir, timeout_s=TEST_LABELING_WAIT_S):
                    if labeling_proc is not None:
                        try:
                            labeling_proc.wait(timeout=15.0)
                        except subprocess.TimeoutExpired:
                            pass
                    print_test_labeling_summaries(out_dir)
                else:
                    print(
                        "Timed out waiting for final report. Partial outputs may exist; "
                        "check live_stats.json and the folder above.",
                        flush=True,
                    )
                    print_test_labeling_summaries(out_dir)
            stop_all(processes, timeout_s=12.0)
        else:
            # CaptureRadarCameraData.py also received CTRL_C from the console and is
            # already running its finally block: drain radar queue (up to 30s) →
            # export extrinsics → stop sensors → run offline labeling
            # (LabelRadarCapture.label_radar_capture_dir, ~10-30s on a long run).
            # We MUST wait for it to exit on its own — calling stop_all() right away
            # would TerminateProcess it on Windows and skip the offline labeling step,
            # which is exactly the bug that left captures with no radar_data_labeled.csv.
            capture_proc = next(
                (
                    p
                    for p in processes
                    if p.args and "CaptureRadarCameraData" in str(p.args[-1])
                ),
                None,
            )
            if capture_proc is not None and capture_proc.poll() is None:
                # Explicitly tell the capture to finalize (drain → extrinsics → offline
                # labeling). Don't assume it already got a console Ctrl+C — under a
                # background launch only THIS process may have been signalled.
                try:
                    capture_proc.send_signal(signal.SIGINT)
                except (ProcessLookupError, OSError):
                    pass
                capture_wait_s = _estimate_capture_wait_s(dc_root)
                print(
                    "\nWaiting for CaptureRadarCameraData to finish "
                    "(drain queue + extrinsics + offline labeling). "
                    f"Up to {capture_wait_s:.0f}s — DO NOT close this window.",
                    flush=True,
                )
                try:
                    capture_proc.wait(timeout=capture_wait_s)
                    print(
                        "Capture process exited cleanly. "
                        "Check the run folder for radar_data_labeled.csv.",
                        flush=True,
                    )
                except subprocess.TimeoutExpired:
                    run_dir = _resolve_last_capture_dir(dc_root)
                    hint = (
                        f"--capture-dir \"{run_dir}\""
                        if run_dir
                        else "--capture-dir <run folder>"
                    )
                    print(
                        f"Capture still running after {capture_wait_s:.0f}s; "
                        "terminating. If radar_data_labeled.csv is missing, run:\n"
                        f"  python -m capture.LabelRadarCapture {hint}",
                        flush=True,
                    )
            # Capture handles its own extrinsics export in its finally block while
            # sensors are still alive in the world, so we no longer call
            # export_dataset_extrinsics_in_process here — by the time we got to it,
            # RadarCameraSetup* had already destroyed the sensors and the call failed.
            stop_all(processes)
        despawn_all_cars(dc_root)
        destroy_leftover_dataset_sensors(dc_root)


if __name__ == "__main__":
    main()
