"""RadarSumoFusion -- unified Tkinter launcher for a fused capture run.

One panel to set traffic (SUMO) + sensor rig (DatasetCreation) + shared clock,
then Start. The run itself is driven by fusion.orchestrator on a worker thread
so the UI stays responsive; log lines stream into the console box.

The original component GUIs are still available and unchanged:
    traffic/gui_launcher.py   (SUMO traffic only)
    dataset/Start.py          (DatasetCreation menu)
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from fusion.config import FusionConfig, VALID_RADAR_COUNTS
from fusion import orchestrator

SCENARIOS = [
    (1, "Free Flow"), (2, "Moderate Demand"), (3, "Heavy Demand"),
    (4, "Stop and Go"), (5, "Mixed Vehicles"), (6, "Directional Rush Hour"),
    (7, "Aggressive Lane Changing"), (8, "Bottleneck / Work Zone"),
    (9, "Overtaking"), (10, "Multi-Lane Overtake"), (11, "Occlusion"),
]


class _QueueLog(list):
    """A list that also pushes appended lines onto a queue for the UI."""
    def __init__(self, q: "queue.Queue[str]"):
        super().__init__()
        self._q = q

    def append(self, item):
        super().append(item)
        self._q.put(str(item))


class FusionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RadarSumoFusion -- Fused Capture Launcher")
        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None

        pad = {"padx": 6, "pady": 3}
        frm = ttk.Frame(root, padding=10)
        frm.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        r = 0
        ttk.Label(frm, text="Traffic (SUMO)", font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w", **pad); r += 1

        ttk.Label(frm, text="Scenario").grid(row=r, column=0, sticky="w", **pad)
        self.scenario = tk.StringVar(value="1")
        ttk.Combobox(frm, textvariable=self.scenario, width=28, state="readonly",
                     values=[f"{i} - {n}" for i, n in SCENARIOS]).grid(
            row=r, column=1, sticky="w", **pad); r += 1

        self.density = self._scale(frm, r, "Density (1-100)", 1, 100, 50); r += 1
        self.duration = self._spin(frm, r, "Duration (s)", 30, 7200, 300); r += 1

        ttk.Label(frm, text="Direction").grid(row=r, column=0, sticky="w", **pad)
        self.direction = tk.StringVar(value="BOTH")
        ttk.Combobox(frm, textvariable=self.direction, width=10, state="readonly",
                     values=["WB", "EB", "BOTH"]).grid(row=r, column=1, sticky="w", **pad); r += 1

        self.ambient = self._scale(frm, r, "Ambient vehicles (0-100)", 0, 100, 20); r += 1
        self.peds = self._scale(frm, r, "Pedestrians (0-100)", 0, 100, 20); r += 1
        self.bikes = self._scale(frm, r, "Bicycles (0-100)", 0, 100, 10); r += 1
        self.sumo_gui = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Show sumo-gui", variable=self.sumo_gui).grid(
            row=r, column=1, sticky="w", **pad); r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8); r += 1
        ttk.Label(frm, text="Sensors + clock", font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w", **pad); r += 1

        ttk.Label(frm, text="Radar rig").grid(row=r, column=0, sticky="w", **pad)
        self.radars = tk.StringVar(value="8")
        ttk.Combobox(frm, textvariable=self.radars, width=10, state="readonly",
                     values=[str(n) for n in VALID_RADAR_COUNTS]).grid(
            row=r, column=1, sticky="w", **pad); r += 1

        self.rate = self._spin(frm, r, "Rate (Hz)", 1, 100, 20); r += 1
        self.label = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Label radar after capture",
                        variable=self.label).grid(row=r, column=1, sticky="w", **pad); r += 1
        self.post = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Post-process after capture",
                        variable=self.post).grid(row=r, column=1, sticky="w", **pad); r += 1
        self.cleanup = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Scene cleanup first (parked cars / trash)",
                        variable=self.cleanup).grid(row=r, column=1, sticky="w", **pad); r += 1

        self.start_btn = ttk.Button(frm, text="Start fused run", command=self._start)
        self.start_btn.grid(row=r, column=0, columnspan=2, sticky="ew", **pad); r += 1

        self.console = tk.Text(frm, height=14, width=64, state="disabled",
                               bg="#101317", fg="#d0d0d0")
        self.console.grid(row=r, column=0, columnspan=2, sticky="nsew", **pad)
        frm.rowconfigure(r, weight=1)
        frm.columnconfigure(1, weight=1)

        self.root.after(120, self._drain_log)

    def _scale(self, frm, r, label, lo, hi, val):
        pad = {"padx": 6, "pady": 3}
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", **pad)
        var = tk.IntVar(value=val)
        ttk.Scale(frm, from_=lo, to=hi, variable=var, orient="horizontal",
                  length=200).grid(row=r, column=1, sticky="w", **pad)
        return var

    def _spin(self, frm, r, label, lo, hi, val):
        pad = {"padx": 6, "pady": 3}
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", **pad)
        var = tk.IntVar(value=val)
        ttk.Spinbox(frm, from_=lo, to=hi, textvariable=var, width=10).grid(
            row=r, column=1, sticky="w", **pad)
        return var

    def _log(self, msg: str):
        self.console.configure(state="normal")
        self.console.insert("end", msg + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _drain_log(self):
        try:
            while True:
                self._log(self._log_q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(120, self._drain_log)

    def _start(self):
        if self._worker and self._worker.is_alive():
            self._log("[gui] a run is already in progress.")
            return
        scenario = int(self.scenario.get().split(" - ")[0])
        cfg = FusionConfig(
            scenario=scenario,
            density=int(self.density.get()),
            duration=int(self.duration.get()),
            direction=self.direction.get(),
            ambient_vehicles=int(self.ambient.get()),
            pedestrians=int(self.peds.get()),
            bicycles=int(self.bikes.get()),
            sumo_gui=bool(self.sumo_gui.get()),
            radar_count=int(self.radars.get()),
            rate_hz=float(self.rate.get()),
            label=bool(self.label.get()),
            postprocess=bool(self.post.get()),
            scene_cleanup=bool(self.cleanup.get()),
        ).validate()
        if not cfg.ok:
            for e in cfg.errors:
                self._log("[gui] invalid: " + e)
            return
        self.start_btn.configure(state="disabled")
        self._log(f"[gui] starting fused run: scenario {scenario}, "
                  f"{cfg.radar_count} radars, {cfg.rate_hz:.0f} Hz ...")
        log = _QueueLog(self._log_q)

        def _job():
            try:
                orchestrator.run(cfg, log=log)
            finally:
                self.root.after(0, lambda: self.start_btn.configure(state="normal"))

        self._worker = threading.Thread(target=_job, daemon=True)
        self._worker.start()


def main():
    root = tk.Tk()
    FusionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
