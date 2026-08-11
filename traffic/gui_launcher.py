"""
gui_launcher.py
===============
Tkinter GUI for the SUMO traffic scenario system.

Allows the user to:
  - Select one of 11 traffic scenarios
  - Set traffic density (slider 1-100)
  - Set simulation duration
  - Choose traffic direction (NB / SB / Both)
  - Toggle sumo-gui on/off
  - Launch / stop the simulation

Run with:
    python gui_launcher.py
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import queue

#  Paths 
HERE       = os.path.dirname(os.path.abspath(__file__))
RUNNER_PY  = os.path.join(HERE, "runner.py")
NET_FILE   = os.path.join(HERE, "network", "Town10HD_Opt.net.xml")

#  Scenario definitions for display 
SCENARIOS = [
    (1,  "Free Flow Traffic",
         "Baseline smooth flow. Low density, no disruptions.\nVehicles travel at or near speed limit."),
    (2,  "Moderate Demand",
         "Medium density with occasional lane changes.\n20% assertive drivers trigger sporadic overtakes."),
    (3,  "Heavy Demand",
         "High density with frequent lane changes and speed reductions.\nMix of slow and aggressive vehicles creates friction."),
    (4,  "Stop and Go Traffic",
         "Full capacity. Shockwave formation and queue propagation.\nSeed vehicles periodically stop to trigger compression waves."),
    (5,  "Mixed Vehicles",
         "Cars, trucks, motorcycles and buses sharing the road.\nBus stops at the designated stop on the monitored stretch."),
    (6,  "Directional Rush Hour",
         "80% flow on the dominant direction (morning/evening rush).\nOpposing direction carries only 20%."),
    (7,  "Aggressive Lane Changing",
         "70% aggressive drivers + 30% slow vehicles.\nHigh lcAssertive values produce frequent, abrupt lane changes."),
    (8,  "Bottleneck / Work Zone",
         "A stationary vehicle blocks one lane mid-stretch.\nUpstream queue and forced merge into single lane."),
    (9,  "Overtaking",
         "Structured overtaking: slow vehicles on right lane,\nfast vehicles use left lane to pass."),
    (10, "Simultaneous Multi-Lane Overtake",
         "Staggered slow wave followed by fast wave.\nProduces simultaneous overtaking events across both lanes."),
    (11, "Occlusion",
         "Large vehicles on outer lanes travel alongside small cars.\nInner-lane vehicles are occluded from sensor view."),
]

#  Colour palette 
BG          = "#1e2330"
PANEL       = "#252b3b"
ACCENT      = "#4a9eff"
ACCENT_DARK = "#2d6bbf"
TEXT        = "#e0e6f0"
TEXT_DIM    = "#8899aa"
SELECTED_BG = "#2a3a5c"
SELECTED_BD = "#4a9eff"
RED         = "#e05555"
GREEN       = "#4caf7d"
WARN        = "#f0a040"


class TrafficGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("SUMO Traffic Scenario Launcher")
        self.geometry("1050x720")
        self.minsize(900, 600)
        self.configure(bg=BG)
        self.resizable(True, True)

        self._process   = None
        self._log_queue = queue.Queue()
        self._selected  = tk.IntVar(value=1)
        self._density   = tk.IntVar(value=50)
        self._duration  = tk.IntVar(value=600)
        self._direction = tk.StringVar(value="BOTH")
        self._use_gui   = tk.BooleanVar(value=True)

        # City / ambient layer
        self._city_enabled = tk.BooleanVar(value=False)
        self._amb_vehicles = tk.IntVar(value=40)
        self._pedestrians  = tk.IntVar(value=30)
        self._bicycles     = tk.IntVar(value=20)
        self._ambient_seed = tk.IntVar(value=42)
        self._render_radius= tk.IntVar(value=120)
        self._cull         = tk.BooleanVar(value=True)

        self._build_ui()
        self._check_net_file()
        self._poll_log()

    #  UI construction 

    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        # Left panel: scenario list
        left = tk.Frame(self, bg=BG, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        tk.Label(left, text="Traffic Scenarios",
                 font=("Segoe UI", 14, "bold"), bg=BG, fg=ACCENT
                 ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        scroll_frame = tk.Frame(left, bg=BG)
        scroll_frame.grid(row=1, column=0, sticky="nsew")
        scroll_frame.columnconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical",
                                  command=canvas.yview)
        self._sc_inner = tk.Frame(canvas, bg=BG)
        self._sc_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._sc_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        scroll_frame.rowconfigure(0, weight=1)
        scroll_frame.columnconfigure(0, weight=1)
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        self._cards = {}
        for sid, name, desc in SCENARIOS:
            self._build_scenario_card(self._sc_inner, sid, name, desc)

        # Right panel: controls + log
        right = tk.Frame(self, bg=PANEL, padx=16, pady=16)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(7, weight=1)

        tk.Label(right, text="Configuration",
                 font=("Segoe UI", 13, "bold"), bg=PANEL, fg=ACCENT
                 ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        self._build_controls(right)
        self._build_city_panel(right, 4)

        # Action buttons
        btn_frame = tk.Frame(right, bg=PANEL)
        btn_frame.grid(row=5, column=0, sticky="ew", pady=(12, 8))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self._run_btn = tk.Button(
            btn_frame, text="▶  Launch Simulation",
            font=("Segoe UI", 11, "bold"),
            bg=GREEN, fg="white", activebackground="#3a9060",
            relief="flat", cursor="hand2", padx=10, pady=8,
            command=self._launch
        )
        self._run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._stop_btn = tk.Button(
            btn_frame, text="■  Stop",
            font=("Segoe UI", 11, "bold"),
            bg=RED, fg="white", activebackground="#b03030",
            relief="flat", cursor="hand2", padx=10, pady=8,
            state="disabled",
            command=self._stop
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # Log output
        tk.Label(right, text="Output Log",
                 font=("Segoe UI", 10, "bold"), bg=PANEL, fg=TEXT_DIM
                 ).grid(row=6, column=0, sticky="w", pady=(4, 2))

        self._log = scrolledtext.ScrolledText(
            right, height=10, bg="#111620", fg="#aaccee",
            font=("Consolas", 9), relief="flat",
            insertbackground=ACCENT, state="disabled"
        )
        self._log.grid(row=7, column=0, sticky="nsew")
        right.rowconfigure(7, weight=1)

    def _build_scenario_card(self, parent, sid, name, desc):
        frame = tk.Frame(parent, bg=PANEL, bd=1, relief="flat",
                         cursor="hand2", padx=10, pady=8)
        frame.pack(fill="x", padx=4, pady=3)
        frame.columnconfigure(1, weight=1)

        # Number badge
        badge = tk.Label(frame,
                         text=f"{sid:02d}",
                         font=("Segoe UI", 11, "bold"),
                         bg=ACCENT_DARK, fg="white",
                         width=3, padx=4, pady=2)
        badge.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))

        tk.Label(frame, text=name,
                 font=("Segoe UI", 10, "bold"), bg=PANEL, fg=TEXT,
                 anchor="w").grid(row=0, column=1, sticky="ew")

        tk.Label(frame, text=desc,
                 font=("Segoe UI", 8), bg=PANEL, fg=TEXT_DIM,
                 anchor="w", justify="left", wraplength=340
                 ).grid(row=1, column=1, sticky="ew")

        self._cards[sid] = (frame, badge)
        for widget in (frame, badge) + tuple(frame.winfo_children()):
            widget.bind("<Button-1>", lambda e, s=sid: self._select(s))

        if sid == 1:
            self._highlight_card(sid)

    def _build_controls(self, parent):
        # Density slider
        tk.Label(parent, text="Traffic Density",
                 font=("Segoe UI", 10, "bold"), bg=PANEL, fg=TEXT
                 ).grid(row=1, column=0, sticky="w")

        density_row = tk.Frame(parent, bg=PANEL)
        density_row.grid(row=2, column=0, sticky="ew", pady=(2, 10))
        density_row.columnconfigure(0, weight=1)

        self._density_slider = ttk.Scale(
            density_row, from_=1, to=100,
            variable=self._density, orient="horizontal",
            command=self._update_density_label
        )
        self._density_slider.grid(row=0, column=0, sticky="ew")

        self._density_label = tk.Label(
            density_row, text="50  (~477 veh/h/dir)",
            font=("Consolas", 9), bg=PANEL, fg=ACCENT, width=24
        )
        self._density_label.grid(row=0, column=1, padx=(8, 0))

        # Duration
        dur_frame = tk.Frame(parent, bg=PANEL)
        dur_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        tk.Label(dur_frame, text="Duration (seconds):",
                 font=("Segoe UI", 10, "bold"), bg=PANEL, fg=TEXT
                 ).pack(side="left")

        vcmd = (self.register(lambda s: s.isdigit() or s == ""), "%P")
        dur_entry = tk.Entry(dur_frame, textvariable=self._duration,
                             width=7, bg="#111620", fg=TEXT,
                             insertbackground=ACCENT, font=("Consolas", 10),
                             relief="flat", validate="key",
                             validatecommand=vcmd)
        dur_entry.pack(side="left", padx=(8, 0))

        # Duration presets
        for label, val in [("10m", 600), ("20m", 1200), ("30m", 1800)]:
            tk.Button(dur_frame, text=label,
                      bg=ACCENT_DARK, fg="white",
                      font=("Segoe UI", 8), relief="flat",
                      cursor="hand2", padx=5,
                      command=lambda v=val: self._duration.set(v)
                      ).pack(side="left", padx=2)

        # Direction + GUI toggle row
        opt_frame = tk.Frame(parent, bg=PANEL)
        opt_frame.grid(row=3, column=0, sticky="ew", pady=(36, 0))

        tk.Label(opt_frame, text="Direction:",
                 font=("Segoe UI", 10, "bold"), bg=PANEL, fg=TEXT
                 ).pack(side="left")

        for label, val in [("Both", "BOTH"), ("-> WB", "WB"), ("<- EB", "EB")]:
            tk.Radiobutton(opt_frame, text=label,
                           variable=self._direction, value=val,
                           bg=PANEL, fg=TEXT, selectcolor=ACCENT_DARK,
                           activebackground=PANEL, font=("Segoe UI", 9)
                           ).pack(side="left", padx=4)

        tk.Checkbutton(opt_frame, text="Open SUMO-GUI",
                       variable=self._use_gui,
                       bg=PANEL, fg=TEXT, selectcolor=ACCENT_DARK,
                       activebackground=PANEL, font=("Segoe UI", 9)
                       ).pack(side="right")

    #  City ambient panel 

    def _build_city_panel(self, parent, row):
        frame = tk.Frame(parent, bg=PANEL)
        frame.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        frame.columnconfigure(0, weight=1)

        tk.Checkbutton(
            frame, text="Enable City Ambient  (whole-map traffic)",
            variable=self._city_enabled, command=self._toggle_city,
            bg=PANEL, fg=ACCENT, selectcolor=ACCENT_DARK,
            activebackground=PANEL, font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=0, sticky="w")

        body = tk.Frame(frame, bg=PANEL)
        body.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        body.columnconfigure(0, weight=1)
        self._city_body = body

        self._city_slider(body, 0, "Ambient vehicles", self._amb_vehicles)
        self._city_slider(body, 1, "Pedestrians",      self._pedestrians)
        self._city_slider(body, 2, "Bicycles",         self._bicycles)

        opt = tk.Frame(body, bg=PANEL)
        opt.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        vcmd = (self.register(lambda s: s.isdigit() or s == ""), "%P")

        tk.Label(opt, text="Seed:", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(opt, textvariable=self._ambient_seed, width=6,
                 bg="#111620", fg=TEXT, insertbackground=ACCENT,
                 font=("Consolas", 9), relief="flat",
                 validate="key", validatecommand=vcmd
                 ).pack(side="left", padx=(4, 10))

        tk.Label(opt, text="Radius m:", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(opt, textvariable=self._render_radius, width=5,
                 bg="#111620", fg=TEXT, insertbackground=ACCENT,
                 font=("Consolas", 9), relief="flat",
                 validate="key", validatecommand=vcmd
                 ).pack(side="left", padx=(4, 10))

        tk.Checkbutton(opt, text="Cull far actors", variable=self._cull,
                       bg=PANEL, fg=TEXT, selectcolor=ACCENT_DARK,
                       activebackground=PANEL, font=("Segoe UI", 9)
                       ).pack(side="left")

        body.grid_remove()  # start collapsed

    def _city_slider(self, parent, row, label, var):
        rowf = tk.Frame(parent, bg=PANEL)
        rowf.grid(row=row, column=0, sticky="ew", pady=1)
        rowf.columnconfigure(1, weight=1)
        tk.Label(rowf, text=label, bg=PANEL, fg=TEXT, width=16, anchor="w",
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        val = tk.Label(rowf, text=str(var.get()), bg=PANEL, fg=ACCENT, width=4,
                       font=("Consolas", 9))
        val.grid(row=0, column=2, padx=(6, 0))
        ttk.Scale(rowf, from_=0, to=100, variable=var, orient="horizontal",
                  command=lambda v, lb=val: lb.configure(
                      text=str(int(float(v))))
                  ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _toggle_city(self):
        if self._city_enabled.get():
            self._city_body.grid()
        else:
            self._city_body.grid_remove()

    #  Interaction 

    def _select(self, sid):
        self._selected.set(sid)
        for s, (frame, badge) in self._cards.items():
            if s == sid:
                self._highlight_card(s)
            else:
                self._unhighlight_card(s)

    def _highlight_card(self, sid):
        frame, badge = self._cards[sid]
        frame.configure(bg=SELECTED_BG, highlightbackground=SELECTED_BD,
                        highlightthickness=2)
        badge.configure(bg=ACCENT)
        for w in frame.winfo_children():
            if isinstance(w, tk.Label):
                w.configure(bg=SELECTED_BG)

    def _unhighlight_card(self, sid):
        frame, badge = self._cards[sid]
        frame.configure(bg=PANEL, highlightthickness=0)
        badge.configure(bg=ACCENT_DARK)
        for w in frame.winfo_children():
            if isinstance(w, tk.Label):
                w.configure(bg=PANEL)

    def _update_density_label(self, _=None):
        d   = int(self._density.get())
        vph = int(50 + (900 - 50) * (d - 1) / 99)
        self._density_label.configure(text=f"{d:3d}  (~{vph} veh/h/dir)")

    #  Launch / stop 

    def _check_net_file(self):
        if not os.path.isfile(NET_FILE):
            self._log_write(
                f"[WARN] Network file not found: {NET_FILE}\n"
                "       Run build_network.py first to compile the .net.xml\n"
            )

    def _launch(self):
        if self._process and self._process.poll() is None:
            messagebox.showwarning("Already Running",
                                   "A simulation is already running.\n"
                                   "Stop it before launching a new one.")
            return

        sid      = self._selected.get()
        density  = int(self._density.get())
        duration = self._duration.get()
        direction= self._direction.get()
        use_gui  = self._use_gui.get()

        cmd = [
            sys.executable, RUNNER_PY,
            "--scenario",  str(sid),
            "--density",   str(density),
            "--duration",  str(duration),
            "--direction", direction,
        ]
        if use_gui:
            cmd.append("--gui")

        # City / ambient layer (only when enabled). --extend-routes moves
        # scenario spawn/despawn out to the far city gateways.
        if self._city_enabled.get():
            cmd += [
                "--extend-routes",
                "--ambient-vehicles", str(int(self._amb_vehicles.get())),
                "--pedestrians",      str(int(self._pedestrians.get())),
                "--bicycles",         str(int(self._bicycles.get())),
                "--ambient-seed",     str(int(self._ambient_seed.get())),
                "--render-radius",    str(int(self._render_radius.get())),
            ]
            if not self._cull.get():
                cmd.append("--no-cull")

        self._log_write(f"Launching: {' '.join(cmd)}\n")
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=HERE,
        )

        threading.Thread(target=self._stream_output, daemon=True).start()

    def _stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log_write("\n[Simulation terminated by user]\n")
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")

    def _stream_output(self):
        for line in self._process.stdout:
            self._log_queue.put(line)
        self._process.wait()
        self._log_queue.put(
            f"\n[Process exited with code {self._process.returncode}]\n"
        )
        self.after(0, lambda: self._run_btn.configure(state="normal"))
        self.after(0, lambda: self._stop_btn.configure(state="disabled"))

    def _poll_log(self):
        try:
            while True:
                line = self._log_queue.get_nowait()
                self._log_write(line)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _log_write(self, text: str):
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")


#  Entry point 

if __name__ == "__main__":
    app = TrafficGUI()
    app.mainloop()
