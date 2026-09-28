"""RadarSumoFusion -- campaign manager.

A campaign is a list of runs (one traffic scenario each) recorded back to back
into ONE sensor capture: one radar CSV, one actor log, one camera stream, one
labeling + post-processing pass, with segments.json / segment_id columns
mapping every frame to its run. See fusion/campaign.py.

    Runs table      one row per run; empty until you add one
    Add run         opens the traffic settings window; Confirm adds the row
    Edit / double-click, Duplicate, Remove, Up / Down, Save / Load (JSON)
    Sensors + clock campaign-wide: rig, tick rate, warm-up, label / post
    Start campaign  runs every row in order; Stop truncates the current row,
                    skips the rest and still labels what was recorded

Headless equivalent of a saved campaign:
    python run_fusion.py --campaign my_campaign.json

The original component GUIs are unchanged:
    traffic/gui_launcher.py   (SUMO traffic only)
    dataset/Start.py          (DatasetCreation menu)
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from fusion.campaign import CampaignConfig, RunSpec, SCENARIO_NAMES, run_campaign
from fusion.config import VALID_RADAR_COUNTS

ROOT = Path(__file__).resolve().parent
AUTOSAVE = ROOT / ".last_campaign.json"

# Mirrors traffic/runner.py density_to_vph + SCENARIO_DEMAND_SCALE, for the
# veh/h hint in the run window only (the runner does the real mapping).
_MIN_VPH, _MAX_VPH = 50, 900
_DEMAND_SCALE = {2: 1.5, 3: 2.0}


def _vph_hint(scenario: int, density: int) -> int:
    base = _MIN_VPH + (_MAX_VPH - _MIN_VPH) * (density - 1) / 99.0
    return int(round(base * _DEMAND_SCALE.get(scenario, 1.0)))


def _fmt_s(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


COLUMNS = [
    # (id, heading, width, anchor)
    ("idx", "#", 34, "center"),
    ("scenario", "Scenario", 190, "w"),
    ("density", "Density", 64, "center"),
    ("duration", "Rec. (s)", 64, "center"),
    ("direction", "Dir", 50, "center"),
    ("ambient", "Ambient", 72, "center"),
    ("peds", "Peds", 50, "center"),
    ("bikes", "Bikes", 50, "center"),
    ("seed", "Seed", 50, "center"),
    ("signals", "Signals", 60, "center"),
    ("status", "Status", 220, "w"),
]
EMPTY_IID = "__empty__"


class _QueueLog(list):
    """A list that also pushes appended lines onto a queue for the UI."""

    def __init__(self, q: "queue.Queue"):
        super().__init__()
        self._q = q

    def append(self, item):
        super().append(item)
        self._q.put(("log", str(item)))


# --------------------------------------------------------------------------
# Add / Edit run window
# --------------------------------------------------------------------------

class RunDialog(tk.Toplevel):
    """Traffic settings for one run. ``result`` is a RunSpec after Confirm."""

    def __init__(self, parent, title: str, initial: RunSpec):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.result: RunSpec | None = None
        pad = {"padx": 8, "pady": 4}

        frm = ttk.Frame(self, padding=12)
        frm.grid(sticky="nsew")
        r = 0
        ttk.Label(frm, text="Traffic (SUMO)", font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w", **pad); r += 1

        ttk.Label(frm, text="Scenario").grid(row=r, column=0, sticky="w", **pad)
        self._scen_values = [f"{i} - {n}" for i, n in SCENARIO_NAMES.items()]
        self.scenario = tk.StringVar(value=f"{initial.scenario} - {SCENARIO_NAMES[initial.scenario]}")
        cb = ttk.Combobox(frm, textvariable=self.scenario, width=30, state="readonly",
                          values=self._scen_values)
        cb.grid(row=r, column=1, columnspan=2, sticky="w", **pad); r += 1
        cb.bind("<<ComboboxSelected>>", lambda _e: self._update_hint())

        self.density = self._scale(frm, r, "Density (1-100)", 1, 100, initial.density,
                                   command=lambda _v: self._update_hint())
        self.vph_hint = ttk.Label(frm, text="", foreground="#666")
        self.vph_hint.grid(row=r, column=2, sticky="w", **pad); r += 1

        ttk.Label(frm, text="Recorded duration (s)").grid(row=r, column=0, sticky="w", **pad)
        self.duration = tk.IntVar(value=initial.duration)
        ttk.Spinbox(frm, from_=5, to=7200, increment=10, textvariable=self.duration,
                    width=10).grid(row=r, column=1, sticky="w", **pad)
        ttk.Label(frm, text="warm-up is added on top", foreground="#666").grid(
            row=r, column=2, sticky="w", **pad); r += 1

        ttk.Label(frm, text="Direction").grid(row=r, column=0, sticky="w", **pad)
        self.direction = tk.StringVar(value=initial.direction)
        ttk.Combobox(frm, textvariable=self.direction, width=10, state="readonly",
                     values=["WB", "EB", "BOTH"]).grid(row=r, column=1, sticky="w", **pad); r += 1

        self.ambient = self._scale(frm, r, "Ambient vehicles (0-100)", 0, 100,
                                   initial.ambient_vehicles); r += 1
        self.peds = self._scale(frm, r, "Pedestrians (0-100)", 0, 100, initial.pedestrians); r += 1
        self.bikes = self._scale(frm, r, "Bicycles (0-100)", 0, 100, initial.bicycles); r += 1

        ttk.Label(frm, text="Seed").grid(row=r, column=0, sticky="w", **pad)
        self.seed = tk.IntVar(value=initial.seed)
        ttk.Spinbox(frm, from_=0, to=999999, textvariable=self.seed, width=10).grid(
            row=r, column=1, sticky="w", **pad); r += 1

        ttk.Label(frm, text="Signals").grid(row=r, column=0, sticky="w", **pad)
        self.signals = tk.StringVar(value=initial.signals)
        ttk.Combobox(frm, textvariable=self.signals, width=10, state="readonly",
                     values=["green", "static"]).grid(row=r, column=1, sticky="w", **pad)
        ttk.Label(frm, text="green = corridor priority (default)", foreground="#666").grid(
            row=r, column=2, sticky="w", **pad); r += 1

        ttk.Label(frm, text="Note").grid(row=r, column=0, sticky="w", **pad)
        self.note = tk.StringVar(value=initial.note)
        ttk.Entry(frm, textvariable=self.note, width=34).grid(
            row=r, column=1, columnspan=2, sticky="w", **pad); r += 1

        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="Confirm", command=self._confirm).pack(side="right", padx=4)
        self.bind("<Return>", lambda _e: self._confirm())
        self.bind("<Escape>", lambda _e: self.destroy())

        self._update_hint()
        self.update_idletasks()
        # Centre over the parent, then make modal.
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + max(0, (pw - w) // 2)}+{py + max(0, (ph - h) // 3)}")
        self.grab_set()
        self.focus_set()

    def _scale(self, frm, r, label, lo, hi, val, command=None):
        pad = {"padx": 8, "pady": 2}
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", **pad)
        var = tk.IntVar(value=val)
        tk.Scale(frm, from_=lo, to=hi, variable=var, orient="horizontal", length=220,
                 resolution=1, showvalue=True, command=command).grid(
            row=r, column=1, sticky="w", **pad)
        return var

    def _scenario_id(self) -> int:
        return int(self.scenario.get().split(" - ")[0])

    def _update_hint(self):
        try:
            self.vph_hint.configure(
                text=f"~{_vph_hint(self._scenario_id(), int(self.density.get()))} veh/h per direction")
        except (tk.TclError, ValueError):
            pass

    def _confirm(self):
        try:
            run = RunSpec(
                scenario=self._scenario_id(),
                density=int(self.density.get()),
                duration=int(self.duration.get()),
                direction=self.direction.get(),
                ambient_vehicles=int(self.ambient.get()),
                pedestrians=int(self.peds.get()),
                bicycles=int(self.bikes.get()),
                seed=int(self.seed.get()),
                signals=self.signals.get(),
                note=self.note.get().strip(),
            )
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid run", f"Could not read the settings: {exc}", parent=self)
            return
        errs = run.validate()
        if errs:
            messagebox.showerror("Invalid run", "\n".join(errs), parent=self)
            return
        self.result = run
        self.destroy()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class CampaignGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RadarSumoFusion -- Campaign Manager")
        root.minsize(900, 640)
        self._q: "queue.Queue" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self.runs: list[RunSpec] = []
        self.status: list[str] = []
        self._running = False

        pad = {"padx": 6, "pady": 3}
        frm = ttk.Frame(root, padding=10)
        frm.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        # --- campaign name ---------------------------------------------------
        top = ttk.Frame(frm)
        top.grid(row=0, column=0, sticky="ew", **pad)
        ttk.Label(top, text="Campaign", font=("", 11, "bold")).pack(side="left")
        ttk.Label(top, text="  name").pack(side="left")
        self.name = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.name, width=32).pack(side="left", padx=6)
        ttk.Label(top, text="(appended to the sensor_capture_* folder name)",
                  foreground="#666").pack(side="left")

        # --- runs table ------------------------------------------------------
        tbl = ttk.Frame(frm)
        tbl.grid(row=1, column=0, sticky="nsew", **pad)
        frm.rowconfigure(1, weight=1)
        tbl.columnconfigure(0, weight=1)
        tbl.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tbl, columns=[c[0] for c in COLUMNS], show="headings",
                                 height=9, selectmode="browse")
        for cid, head, width, anchor in COLUMNS:
            self.tree.heading(cid, text=head)
            self.tree.column(cid, width=width, anchor=anchor,
                             stretch=(cid in ("scenario", "status")))
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(tbl, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.tag_configure("empty", foreground="#999")
        self.tree.tag_configure("done", foreground="#1a7f37")
        self.tree.tag_configure("failed", foreground="#c62828")
        self.tree.tag_configure("active", foreground="#1565c0")
        self.tree.tag_configure("skipped", foreground="#888")
        self.tree.bind("<Double-1>", lambda _e: self._edit())
        self.tree.bind("<Delete>", lambda _e: self._remove())

        # --- row buttons -----------------------------------------------------
        rowb = ttk.Frame(frm)
        rowb.grid(row=2, column=0, sticky="ew", **pad)
        self._edit_buttons = []
        for text, cmd in (("Add run", self._add), ("Edit", self._edit),
                          ("Duplicate", self._duplicate), ("Remove", self._remove),
                          ("Move up", lambda: self._move(-1)),
                          ("Move down", lambda: self._move(1))):
            b = ttk.Button(rowb, text=text, command=cmd)
            b.pack(side="left", padx=(0, 4))
            self._edit_buttons.append(b)
        for text, cmd in (("Load...", self._load), ("Save...", self._save)):
            b = ttk.Button(rowb, text=text, command=cmd)
            b.pack(side="right", padx=(4, 0))
            self._edit_buttons.append(b)
        self.totals = ttk.Label(frm, text="", foreground="#444")
        self.totals.grid(row=3, column=0, sticky="w", **pad)

        # --- campaign-wide sensors + clock ------------------------------------
        ttk.Separator(frm, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=6)
        sens = ttk.Frame(frm)
        sens.grid(row=5, column=0, sticky="ew", **pad)
        ttk.Label(sens, text="Sensors + clock (whole campaign)", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=6, sticky="w", pady=(0, 4))
        ttk.Label(sens, text="Radar rig").grid(row=1, column=0, sticky="w", padx=(0, 4))
        self.radars = tk.StringVar(value="8")
        ttk.Combobox(sens, textvariable=self.radars, width=6, state="readonly",
                     values=[str(n) for n in VALID_RADAR_COUNTS]).grid(row=1, column=1, sticky="w")
        ttk.Label(sens, text="Rate (Hz)").grid(row=1, column=2, sticky="w", padx=(16, 4))
        self.rate = tk.IntVar(value=20)
        ttk.Spinbox(sens, from_=1, to=100, textvariable=self.rate, width=6).grid(
            row=1, column=3, sticky="w")
        ttk.Label(sens, text="Warm-up per run (s)").grid(row=1, column=4, sticky="w", padx=(16, 4))
        self.warmup = tk.IntVar(value=20)
        ttk.Spinbox(sens, from_=0, to=300, increment=5, textvariable=self.warmup, width=6,
                    command=self._refresh_totals).grid(row=1, column=5, sticky="w")
        self.warmup.trace_add("write", lambda *_: self._refresh_totals())

        checks = ttk.Frame(sens)
        checks.grid(row=2, column=0, columnspan=6, sticky="w", pady=(6, 0))
        self.label = tk.BooleanVar(value=True)
        self.post = tk.BooleanVar(value=True)
        self.cleanup = tk.BooleanVar(value=True)
        self.trees = tk.BooleanVar(value=True)
        self.sumo_gui = tk.BooleanVar(value=False)
        for text, var in (("Label radar after capture", self.label),
                          ("Post-process after capture", self.post),
                          ("Scene cleanup first (parked cars / trash)", self.cleanup),
                          ("Hide stretch trees", self.trees),
                          ("Show sumo-gui", self.sumo_gui)):
            ttk.Checkbutton(checks, text=text, variable=var).pack(side="left", padx=(0, 12))

        # --- start / stop ----------------------------------------------------
        run_row = ttk.Frame(frm)
        run_row.grid(row=6, column=0, sticky="ew", **pad)
        self.start_btn = ttk.Button(run_row, text="Start campaign", command=self._start)
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.stop_btn = ttk.Button(run_row, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.capture_state = ttk.Label(frm, text="", foreground="#444")
        self.capture_state.grid(row=7, column=0, sticky="w", **pad)

        # --- console ---------------------------------------------------------
        self.console = tk.Text(frm, height=12, width=100, state="disabled",
                               bg="#101317", fg="#d0d0d0", wrap="none")
        self.console.grid(row=8, column=0, sticky="nsew", **pad)
        frm.rowconfigure(8, weight=1)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._autoload()
        self._refresh_table()
        self.root.after(120, self._drain)

    # -- table ---------------------------------------------------------------
    def _row_values(self, i: int, run: RunSpec):
        return (i + 1, f"{run.scenario} - {run.scenario_name}", run.density, run.duration,
                run.direction, run.ambient_vehicles, run.pedestrians, run.bicycles,
                run.seed, run.signals, self.status[i] if i < len(self.status) else "")

    def _refresh_table(self, select: int | None = None):
        self.tree.delete(*self.tree.get_children())
        if not self.runs:
            self.tree.insert("", "end", iid=EMPTY_IID, tags=("empty",),
                             values=("", "(no runs yet -- click Add run)", "", "", "", "",
                                     "", "", "", "", ""))
        for i, run in enumerate(self.runs):
            self.tree.insert("", "end", iid=str(i), values=self._row_values(i, run),
                             tags=(self._status_tag(self.status[i]),))
        if select is not None and 0 <= select < len(self.runs):
            self.tree.selection_set(str(select))
            self.tree.see(str(select))
        self._refresh_totals()

    @staticmethod
    def _status_tag(status: str) -> str:
        s = status.lower()
        if s.startswith("done"):
            return "done"
        if s.startswith("failed"):
            return "failed"
        if s.startswith(("skipped", "stopped")):
            return "skipped"
        if s and not s.startswith("queued"):
            return "active"
        return ""

    def _refresh_totals(self):
        try:
            warm = max(0, int(self.warmup.get()))
        except (tk.TclError, ValueError):
            warm = 0
        rec = sum(r.duration for r in self.runs)
        sim = rec + warm * len(self.runs)
        self.totals.configure(
            text=f"{len(self.runs)} run(s)   recorded {_fmt_s(rec)}   "
                 f"simulated incl. warm-up {_fmt_s(sim)}   (wall time depends on how "
                 f"fast CARLA ticks, plus labeling afterwards)")

    def _selected(self) -> int | None:
        sel = self.tree.selection()
        if not sel or sel[0] == EMPTY_IID:
            return None
        return int(sel[0])

    # -- row actions -------------------------------------------------------------
    def _guard(self) -> bool:
        if self._running:
            messagebox.showinfo("Campaign running", "Stop the campaign before editing runs.")
            return False
        return True

    def _add(self):
        if not self._guard():
            return
        i = self._selected()
        base = self.runs[i] if i is not None else (self.runs[-1] if self.runs else RunSpec())
        dlg = RunDialog(self.root, "Add run", RunSpec.from_dict(vars(base)))
        self.root.wait_window(dlg)
        if dlg.result is not None:
            self.runs.append(dlg.result)
            self.status.append("queued")
            self._refresh_table(select=len(self.runs) - 1)
            self._autosave()

    def _edit(self):
        if not self._guard():
            return
        i = self._selected()
        if i is None:
            return
        dlg = RunDialog(self.root, f"Edit run {i + 1}", RunSpec.from_dict(vars(self.runs[i])))
        self.root.wait_window(dlg)
        if dlg.result is not None:
            self.runs[i] = dlg.result
            self.status[i] = "queued"
            self._refresh_table(select=i)
            self._autosave()

    def _duplicate(self):
        if not self._guard():
            return
        i = self._selected()
        if i is None:
            return
        self.runs.insert(i + 1, RunSpec.from_dict(vars(self.runs[i])))
        self.status.insert(i + 1, "queued")
        self._refresh_table(select=i + 1)
        self._autosave()

    def _remove(self):
        if not self._guard():
            return
        i = self._selected()
        if i is None:
            return
        del self.runs[i]
        del self.status[i]
        self._refresh_table(select=min(i, len(self.runs) - 1))
        self._autosave()

    def _move(self, delta: int):
        if not self._guard():
            return
        i = self._selected()
        if i is None:
            return
        j = i + delta
        if not (0 <= j < len(self.runs)):
            return
        self.runs[i], self.runs[j] = self.runs[j], self.runs[i]
        self.status[i], self.status[j] = self.status[j], self.status[i]
        self._refresh_table(select=j)
        self._autosave()

    # -- persistence -------------------------------------------------------------
    def _config(self) -> CampaignConfig:
        return CampaignConfig(
            name=self.name.get().strip(),
            runs=[RunSpec.from_dict(vars(r)) for r in self.runs],
            radar_count=int(self.radars.get()),
            rate_hz=float(self.rate.get()),
            warmup_s=float(self.warmup.get()),
            label=bool(self.label.get()),
            postprocess=bool(self.post.get()),
            scene_cleanup=bool(self.cleanup.get()),
            clear_trees=bool(self.trees.get()),
            sumo_gui=bool(self.sumo_gui.get()),
        )

    def _apply_config(self, cfg: CampaignConfig):
        self.name.set(cfg.name)
        self.runs = list(cfg.runs)
        self.status = ["queued"] * len(self.runs)
        self.radars.set(str(cfg.radar_count))
        self.rate.set(int(cfg.rate_hz))
        self.warmup.set(int(cfg.warmup_s))
        self.label.set(cfg.label)
        self.post.set(cfg.postprocess)
        self.cleanup.set(cfg.scene_cleanup)
        self.trees.set(cfg.clear_trees)
        self.sumo_gui.set(cfg.sumo_gui)
        self._refresh_table()

    def _save(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save campaign", defaultextension=".json",
            initialdir=str(ROOT), filetypes=[("Campaign JSON", "*.json")],
            initialfile=(self.name.get().strip() or "campaign") + ".json")
        if path:
            try:
                self._config().save(path)
                self._log(f"[gui] campaign saved to {path}")
            except (OSError, tk.TclError, ValueError) as exc:
                messagebox.showerror("Save failed", str(exc))

    def _load(self):
        if not self._guard():
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Load campaign", initialdir=str(ROOT),
            filetypes=[("Campaign JSON", "*.json"), ("All files", "*.*")])
        if path:
            try:
                self._apply_config(CampaignConfig.load(path))
                self._log(f"[gui] campaign loaded from {path}")
                self._autosave()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                messagebox.showerror("Load failed", f"Not a campaign file: {exc}")

    def _autosave(self):
        try:
            self._config().save(AUTOSAVE)
        except (OSError, tk.TclError, ValueError):
            pass

    def _autoload(self):
        if AUTOSAVE.is_file():
            try:
                self._apply_config(CampaignConfig.load(AUTOSAVE))
                if self.runs:
                    self._log(f"[gui] restored the last campaign ({len(self.runs)} run(s)) "
                              f"from {AUTOSAVE.name}")
            except (OSError, ValueError, TypeError, KeyError):
                pass

    # -- run -----------------------------------------------------------------------
    def _set_running(self, running: bool):
        self._running = running
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        for b in self._edit_buttons:
            b.configure(state="disabled" if running else "normal")

    def _start(self):
        if self._worker and self._worker.is_alive():
            return
        try:
            cfg = self._config()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        errs = cfg.validate()
        if errs:
            messagebox.showerror("Cannot start", "\n".join(errs))
            return
        self._autosave()
        self.status = ["queued"] * len(self.runs)
        self._refresh_table()
        self._set_running(True)
        self.capture_state.configure(text="Capture: starting ...")
        self._stop_event = threading.Event()
        log = _QueueLog(self._q)
        self._log(f"[gui] starting campaign: {len(cfg.runs)} run(s), "
                  f"{cfg.radar_count} radars @ {cfg.rate_hz:.0f} Hz, warm-up {cfg.warmup_s:.0f} s")

        def _job():
            try:
                run_campaign(cfg, log=log, on_event=lambda ev: self._q.put(("event", ev)),
                             stop_event=self._stop_event)
            except Exception as exc:  # noqa: BLE001
                self._q.put(("log", f"[gui] campaign crashed: {exc}"))
            finally:
                self._q.put(("finished", None))

        self._worker = threading.Thread(target=_job, daemon=True)
        self._worker.start()

    def _stop(self):
        if self._stop_event is not None and not self._stop_event.is_set():
            if messagebox.askyesno(
                    "Stop campaign",
                    "Stop now? The current run is cut short, the remaining runs are "
                    "skipped, and the capture still labels + post-processes what was "
                    "recorded."):
                self._stop_event.set()
                self.stop_btn.configure(state="disabled")
                self._log("[gui] stop requested ...")

    def _on_close(self):
        if self._running:
            messagebox.showinfo(
                "Campaign running",
                "A campaign is running. Press Stop first and wait until labeling "
                "finishes; closing now would orphan the capture and runner processes.")
            return
        self._autosave()
        self.root.destroy()

    # -- UI plumbing ---------------------------------------------------------------
    def _log(self, msg: str):
        self.console.configure(state="normal")
        self.console.insert("end", msg + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _handle_event(self, ev: dict):
        kind = ev.get("type")
        if kind == "row":
            i = ev["index"]
            if 0 <= i < len(self.status):
                status = ev.get("status", "")
                detail = ev.get("detail", "")
                label = {"warmup": "warm-up", "cleanup": "cleaning up",
                         "starting": "starting runner"}.get(status, status)
                if status == "failed":
                    label = "FAILED"
                self.status[i] = f"{label} {detail}".strip() if detail else label
                if self.tree.exists(str(i)):
                    self.tree.item(str(i), values=self._row_values(i, self.runs[i]),
                                   tags=(self._status_tag(self.status[i]),))
                    if status in ("warmup", "recording", "starting"):
                        self.tree.see(str(i))
        elif kind == "capture":
            state = ev.get("state", "")
            run_dir = ev.get("run_dir", "")
            self.capture_state.configure(text=f"Capture: {state}   {run_dir}")
        elif kind == "finished":
            ok = ev.get("ok")
            self.capture_state.configure(
                text=("Capture: done   " if ok else "Capture: finished with errors   ")
                + (ev.get("run_dir") or ""))

    def _drain(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "event":
                    self._handle_event(payload)
                elif kind == "finished":
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(120, self._drain)


def main():
    root = tk.Tk()
    CampaignGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
