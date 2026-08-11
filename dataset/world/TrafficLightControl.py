import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dc_entry", _root / "_entry.py")
_dc_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_dc_entry)
_dc_entry.bootstrap(__file__)

import os
import tkinter as tk
from tkinter import messagebox, ttk

import carla

from carla_connect import carla_host, carla_port, carla_timeout_s

HOST = carla_host()
PORT = carla_port()
TIMEOUT_SECONDS = carla_timeout_s()
REFRESH_INTERVAL_MS = 1000


def pipeline_autoconnect_from_env() -> bool:
    return os.environ.get("DATASET_TRAFFIC_LIGHT_GUI_AUTOCONNECT", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def state_to_text(state):
    if state == carla.TrafficLightState.Red:
        return "Red"
    if state == carla.TrafficLightState.Yellow:
        return "Yellow"
    if state == carla.TrafficLightState.Green:
        return "Green"
    return str(state)


class TrafficLightControlUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CARLA Traffic Light Control")
        self.root.geometry("980x560")

        self.client = None
        self.world = None
        self.light_by_id = {}
        self._auto_refresh_job = None

        self.host_var = tk.StringVar(value=HOST)
        self.port_var = tk.StringVar(value=str(PORT))
        self.freeze_var = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Host:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.host_var, width=20).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(top, text="Port:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.port_var, width=8).pack(side=tk.LEFT, padx=(6, 12))

        ttk.Button(top, text="Connect", command=self.connect).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Refresh Lights", command=self.refresh_lights).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(top, text="Freeze when applying state", variable=self.freeze_var).pack(
            side=tk.LEFT, padx=(12, 0)
        )

        mid = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        mid.pack(fill=tk.BOTH, expand=True)

        columns = ("id", "state", "location")
        self.tree = ttk.Treeview(mid, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("id", text="Traffic Light ID")
        self.tree.heading("state", text="Current State")
        self.tree.heading("location", text="Location (x, y, z)")
        self.tree.column("id", width=140, anchor=tk.CENTER)
        self.tree.column("state", width=140, anchor=tk.CENTER)
        self.tree.column("location", width=620, anchor=tk.W)

        scrollbar = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)

        ttk.Button(bottom, text="Set Red", command=lambda: self.apply_state(carla.TrafficLightState.Red)).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(
            bottom,
            text="Set Yellow",
            command=lambda: self.apply_state(carla.TrafficLightState.Yellow),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            bottom,
            text="Set Green",
            command=lambda: self.apply_state(carla.TrafficLightState.Green),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="Unfreeze Selected", command=self.unfreeze_selected).pack(side=tk.LEFT, padx=(18, 4))
        ttk.Button(
            bottom,
            text="Unfreeze All + Resume Cycle",
            command=self.unfreeze_all_and_resume_cycle,
        ).pack(side=tk.LEFT, padx=(8, 4))

        self.status_var = tk.StringVar(value="Not connected.")
        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 0, 10, 10)).pack(fill=tk.X)

    def _cancel_auto_refresh(self) -> None:
        if self._auto_refresh_job is not None:
            try:
                self.root.after_cancel(self._auto_refresh_job)
            except tk.TclError:
                pass
            self._auto_refresh_job = None

    def connect(self):
        host = self.host_var.get().strip() or HOST
        port_text = self.port_var.get().strip()
        try:
            port = int(port_text)
        except ValueError:
            messagebox.showerror("Invalid Port", "Port must be an integer.")
            return

        self._cancel_auto_refresh()
        self.world = None
        self.client = None

        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(TIMEOUT_SECONDS)
            self.world = self.client.get_world()
        except Exception as exc:
            messagebox.showerror("Connection Failed", f"Could not connect to CARLA:\n{exc}")
            self.status_var.set("Connection failed.")
            return

        self.status_var.set(f"Connected to {host}:{port}.")
        self.refresh_lights()
        self._auto_refresh_job = self.root.after(REFRESH_INTERVAL_MS, self._auto_refresh)

    def _auto_refresh(self):
        if self.world is None:
            self._auto_refresh_job = None
            return
        try:
            self.refresh_lights(silent=True)
        except Exception:
            pass
        if self.world is not None:
            self._auto_refresh_job = self.root.after(REFRESH_INTERVAL_MS, self._auto_refresh)

    def refresh_lights(self, silent=False):
        if self.world is None:
            if not silent:
                messagebox.showwarning("Not Connected", "Connect to CARLA first.")
            return

        selected_ids = set(self._selected_ids())

        try:
            lights = sorted(self.world.get_actors().filter("traffic.traffic_light*"), key=lambda light: light.id)
        except Exception as exc:
            if not silent:
                messagebox.showerror("Refresh Failed", f"Could not retrieve traffic lights:\n{exc}")
            return

        self.light_by_id = {light.id: light for light in lights}
        self.tree.delete(*self.tree.get_children())

        for light in lights:
            loc = light.get_location()
            item_id = str(light.id)
            self.tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(
                    light.id,
                    state_to_text(light.get_state()),
                    f"{loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}",
                ),
            )
            if light.id in selected_ids:
                self.tree.selection_add(item_id)

        self.status_var.set(f"Loaded {len(lights)} traffic lights.")

    def _selected_ids(self):
        ids = []
        for item in self.tree.selection():
            try:
                ids.append(int(item))
            except ValueError:
                continue
        return ids

    def apply_state(self, state):
        if self.world is None:
            messagebox.showwarning("Not Connected", "Connect to CARLA first.")
            return

        selected_ids = self._selected_ids()
        if not selected_ids:
            messagebox.showinfo("No Selection", "Select one or more traffic lights first.")
            return

        updated = 0
        failed = []
        freeze = self.freeze_var.get()

        for light_id in selected_ids:
            light = self.light_by_id.get(light_id)
            if light is None:
                failed.append(light_id)
                continue
            try:
                light.set_state(state)
                if freeze:
                    light.freeze(True)
                updated += 1
            except Exception:
                failed.append(light_id)

        self.refresh_lights(silent=True)

        state_name = state_to_text(state)
        if failed:
            self.status_var.set(
                f"Set {updated} light(s) to {state_name}. Failed: {', '.join(str(light_id) for light_id in failed)}"
            )
        else:
            self.status_var.set(f"Set {updated} light(s) to {state_name}.")

    def unfreeze_selected(self):
        if self.world is None:
            messagebox.showwarning("Not Connected", "Connect to CARLA first.")
            return

        selected_ids = self._selected_ids()
        if not selected_ids:
            messagebox.showinfo("No Selection", "Select one or more traffic lights first.")
            return

        updated = 0
        for light_id in selected_ids:
            light = self.light_by_id.get(light_id)
            if light is None:
                continue
            try:
                light.freeze(False)
                updated += 1
            except Exception:
                pass

        self.refresh_lights(silent=True)
        self.status_var.set(f"Unfroze {updated} selected light(s).")

    def unfreeze_all_and_resume_cycle(self):
        if self.world is None:
            messagebox.showwarning("Not Connected", "Connect to CARLA first.")
            return

        try:
            from world.TrafficLightSetup import reset_light_phase_times

            lights = list(self.world.get_actors().filter("traffic.traffic_light*"))
        except Exception as exc:
            messagebox.showerror("Failed", f"Could not list traffic lights:\n{exc}")
            return

        updated = 0
        for light in lights:
            try:
                reset_light_phase_times(light)
                light.freeze(False)
                updated += 1
            except RuntimeError:
                continue

        self.refresh_lights(silent=True)
        self.status_var.set(f"Unfroze {updated} light(s) with default phase times.")


def main():
    root = tk.Tk()
    ui = TrafficLightControlUI(root)
    if pipeline_autoconnect_from_env():
        root.after(600, ui.connect)
        ui.status_var.set("Auto-connecting to CARLA (pipeline mode)...")
    root.mainloop()


if __name__ == "__main__":
    main()
