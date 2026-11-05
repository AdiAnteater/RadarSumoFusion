#!/usr/bin/env bash
# One-shot launcher for the CARLA radar/camera dataset pipeline.
#
# What it does:
#   1. Activates the Python venv.
#   2. Starts the CARLA server in the background if it isn't already running.
#   3. Waits for the RPC port to be live, then loads Town10HD_Opt.
#   4. Exports DATASET_CAPTURE_BASE_DIR (so captures go to /scratch, not home).
#   5. Runs Start.py with whatever args you pass through (default: --radar-count 8 --test-labeling).
#   6. On exit (clean or Ctrl+C), tears down anything this script started.
#
# Usage:
#   ./run_sim.sh                                   # test labeling, 8 radars
#   ./run_sim.sh --radar-count 4                   # full dataset capture, 4 radars
#   ./run_sim.sh --radar-count 12 --test-labeling  # 12-radar test
#   KEEP_SERVER=1 ./run_sim.sh ...                 # leave CARLA up after Start.py exits
#   AUTOPILOT=0 ./run_sim.sh ...                   # park vehicles (default: drive, for Doppler)
#   PROFILE=flow ./run_sim.sh --radar-count 8      # fast bidirectional flow preset
#   PROFILE=dense ./run_sim.sh --radar-count 8     # dense stop-and-go (occlusion) preset
#       (presets live in config/profile_<name>.env; unset PROFILE = code defaults)

set -euo pipefail

# --- Configurable paths (override via env if you move things) ---------------
VENV="${VENV:-$HOME/carla_pim_venv}"
CARLA_DIR="${CARLA_DIR:-/scratch/tamoghnd/CARLA_0.9.16}"
CAPTURE_BASE="${DATASET_CAPTURE_BASE_DIR:-/scratch/tamoghnd/dataset_captures}"
RPC_PORT="${RPC_PORT:-2000}"
MAP="${MAP:-Town10HD_Opt}"
SERVER_LOG="${SERVER_LOG:-/tmp/carla_server.log}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# ---------------------------------------------------------------------------

CARLA_STARTED_BY_THIS_SCRIPT=0
CARLA_PID=""
CLEANUP_DONE=0

log() { printf '\033[1;36m[run_sim]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[run_sim]\033[0m %s\n' "$*" >&2; }

cleanup() {
    [[ "$CLEANUP_DONE" == "1" ]] && return
    CLEANUP_DONE=1
    if [[ "${KEEP_SERVER:-0}" == "1" ]]; then
        log "KEEP_SERVER=1, leaving CARLA running."
        return
    fi
    if [[ "$CARLA_STARTED_BY_THIS_SCRIPT" == "1" && -n "$CARLA_PID" ]]; then
        log "Stopping CARLA server (pid $CARLA_PID)…"
        kill "$CARLA_PID" 2>/dev/null || true
        pkill -f CarlaUE4-Linux-Shipping 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- 1. venv ----------------------------------------------------------------
if [[ ! -f "$VENV/bin/activate" ]]; then
    err "venv not found at $VENV (set VENV=… or re-create it)."
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
log "venv:      $VENV"
log "python:    $(python -V 2>&1)"

# --- 2. CARLA server --------------------------------------------------------
if ss -lntp 2>/dev/null | grep -q ":$RPC_PORT "; then
    log "CARLA already listening on :$RPC_PORT — reusing it."
else
    if [[ ! -x "$CARLA_DIR/CarlaUE4.sh" ]]; then
        err "CarlaUE4.sh not found / not executable at $CARLA_DIR"
        exit 1
    fi
    log "Starting CARLA from $CARLA_DIR (log: $SERVER_LOG)…"
    ( cd "$CARLA_DIR" && \
      nohup ./CarlaUE4.sh -RenderOffScreen -nosound \
            -carla-rpc-port="$RPC_PORT" > "$SERVER_LOG" 2>&1 & \
      echo $! > /tmp/run_sim.carla.pid )
    CARLA_PID="$(cat /tmp/run_sim.carla.pid)"
    CARLA_STARTED_BY_THIS_SCRIPT=1

    log "Waiting for CARLA to accept connections on :$RPC_PORT…"
    for _ in {1..60}; do
        if ss -lntp 2>/dev/null | grep -q ":$RPC_PORT "; then
            log "CARLA ready (pid $CARLA_PID)."
            break
        fi
        if ! kill -0 "$CARLA_PID" 2>/dev/null; then
            err "CARLA died during startup. Tail of $SERVER_LOG:"
            tail -40 "$SERVER_LOG" >&2 || true
            exit 1
        fi
        sleep 1
    done
    if ! ss -lntp 2>/dev/null | grep -q ":$RPC_PORT "; then
        err "Timeout waiting for CARLA. See $SERVER_LOG."
        exit 1
    fi
fi

# --- 3. Load map ------------------------------------------------------------
log "Loading map: $MAP"
python - <<PY
import carla, sys, time
c = carla.Client('localhost', $RPC_PORT)
c.set_timeout(30.0)
w = c.get_world()
if w.get_map().name.endswith('$MAP'):
    print('[run_sim]   map already loaded')
else:
    w = c.load_world('$MAP')
    time.sleep(2)
    print(f'[run_sim]   loaded {w.get_map().name}')
PY

# --- 4. Env --------------------------------------------------------------
mkdir -p "$CAPTURE_BASE"
export DATASET_CAPTURE_BASE_DIR="$CAPTURE_BASE"
log "captures:  $DATASET_CAPTURE_BASE_DIR"

# Optional traffic/sensor profile preset: PROFILE=flow|dense ./run_sim.sh ...
# Sources config/profile_<name>.env (a bundle of DATASET_* exports). Unset = code
# defaults (unchanged). Any DATASET_* you export yourself BEFORE the profile is
# overridden by it; export AFTER sourcing to override a single profile value.
if [[ -n "${PROFILE:-}" ]]; then
    PROFILE_FILE="$SCRIPT_DIR/config/profile_${PROFILE}.env"
    if [[ -f "$PROFILE_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$PROFILE_FILE"
        log "profile:   $PROFILE  ($PROFILE_FILE)"
    else
        err "PROFILE='$PROFILE' not found at $PROFILE_FILE. Available profiles:"
        ls "$SCRIPT_DIR"/config/profile_*.env 2>/dev/null | sed 's#.*/profile_##; s#\.env$##' >&2 || true
        exit 1
    fi
fi

# AUTOPILOT maps to DATASET_FREE_VEHICLE_DRIVING (the knob the spawn script reads:
# 1/true/yes/on = drive, else parked). Unset -> leave it to the code default (drive).
if [[ -n "${AUTOPILOT:-}" ]]; then
    export DATASET_FREE_VEHICLE_DRIVING="$AUTOPILOT"
    log "vehicles:  DATASET_FREE_VEHICLE_DRIVING=$AUTOPILOT (from AUTOPILOT)"
else
    log "vehicles:  free driving (default; set AUTOPILOT=0 to park)"
fi

# --- 5. Run Start.py --------------------------------------------------------
cd "$SCRIPT_DIR"
if [[ $# -eq 0 ]]; then
    set -- --radar-count 8 --test-labeling
fi
log "launching: python Start.py $*"
python Start.py "$@"
