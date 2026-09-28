"""Corridor-priority signal programs for the traffic lights along the monitored
boulevard (SUMO side, applied through TraCI after the simulation starts).

Why this exists
---------------
The net is imported from CARLA's OpenDRIVE, whose signals arrive as plain
two-phase programs (about 34 s green / 33 s red, no amber) at every junction.
The monitored stretch (edges 20 / -20, 55 m) ends at junction 189 westbound and
the next light (532) is 25 m past its eastbound end, so with those programs a
red phase queues traffic back across the whole stretch roughly half the time.
Headless validation of all 11 scenarios (traffic/validate_scenarios.py) showed
the consequence: every scenario, "free flow" and "overtaking" included, had a
mean stretch speed of ~1.3 m/s and vehicles stopped ~70% of the time, and the
captured dataset had 82% of vehicle-frames on the stretch stationary. The
scenario definitions (demand, vehicle mix, lane-change parameters, controllers)
were being masked by signal queuing that no scenario asked for.

What it does
------------
For every traffic light with a corridor through-movement, keep the imported
state strings (they were generated with the junction's right-of-way logic; a
hand-written string is rejected as "incompatible with logic at junction") and
re-time them: the main corridor phase gets 90 s (default), the main side-street
phase 20 s, every other phase (amber, all-red, pedestrian) is untouched, and
every corridor light starts in its corridor phase together (common offset). Side
streets (ambient traffic) and the pedestrian crossings over the boulevard are
still served every cycle, they just no longer own half of it. The corridor is
whatever ambient_traffic.WB_ROUTE_EDGES / EB_ROUTE_EDGES say, so this follows
the routes if they move. The three lights involved here are 532, 189 and 719:
the stretch (55 m) sits between 532 and 189 with only 25 m and 19 m of road to
the neighbouring lights, so any red at either neighbour backs up over it.

Modes (runner --stretch-signals):
    green   apply the corridor-priority program (default)
    static  leave the imported programs alone (old behaviour, for comparison)
"""

import os
import re

import traci

# Signal programs are defined in seconds of simulation time.
DEFAULT_GREEN_S = 90.0        # phase A: corridor green
DEFAULT_CROSS_GREEN_S = 20.0  # phase B: side streets + boulevard crossings
DEFAULT_SPLIT_S = 30.0        # both main phases at a "split" light (see below)
PROGRAM_ID = "corridor_priority"


def _routes():
    """(wb_edges, eb_edges) as lists, from ambient_traffic when importable."""
    try:
        import ambient_traffic

        def _as_list(v):
            return v.split() if isinstance(v, str) else list(v)
        wb = _as_list(ambient_traffic.WB_ROUTE_EDGES)
        eb = _as_list(ambient_traffic.EB_ROUTE_EDGES)
        if wb and eb:
            return wb, eb
    except Exception:  # noqa: BLE001
        pass
    return (["-1", "-2", "21", "20", "19", "18", "5"],
            ["-5", "-18", "-19", "-20", "-21", "2", "1"])


def _corridor_edges():
    wb, eb = _routes()
    return set(wb) | set(eb)


def _exit_edges():
    """Edges a vehicle drives on AFTER leaving the stretch, per direction. A
    red for a movement starting on one of these is what queues traffic back
    onto the stretch, so these movements decide which phase gets the long
    green at each light."""
    out = set()
    for route, stretch in ((_routes()[0], "20"), (_routes()[1], "-20")):
        if stretch in route:
            out.update(route[route.index(stretch):])
    return out


def _edge_of_lane(lane_id: str) -> str:
    return lane_id.rsplit("_", 1)[0]


def _crossing_edges_from_net(net_file: str) -> dict:
    """{crossing_edge_id: set(crossed edge ids)} from the net XML."""
    out = {}
    try:
        with open(net_file, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return out
    for m in re.finditer(r'<edge id="([^"]+)" function="crossing" crossingEdges="([^"]+)"', text):
        out[m.group(1)] = set(m.group(2).split())
    return out


def _right_turn_links_from_net(net_file: str) -> dict:
    """{tls_id: set(linkIndex)} of connections netconvert marked dir="r"."""
    out = {}
    try:
        with open(net_file, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return out
    for m in re.finditer(r'<connection\b[^>]*\btl="([^"]+)"[^>]*\blinkIndex="(\d+)"[^>]*\bdir="r"', text):
        out.setdefault(m.group(1), set()).add(int(m.group(2)))
    return out


def _link_class(in_lane, out_lane, via_lane, corridor, crossings):
    """Return one of: 'cor_straight', 'cor_turn', 'side', 'x_cor', 'x_side'."""
    via_edge = _edge_of_lane(via_lane) if via_lane else ""
    out_edge = _edge_of_lane(out_lane) if out_lane else ""
    in_edge = _edge_of_lane(in_lane) if in_lane else ""
    for cand in (via_edge, out_edge, in_edge):
        if cand in crossings:
            return "x_cor" if crossings[cand] & corridor else "x_side"
    if in_edge.startswith(":") or out_edge.startswith(":"):
        return "x_unknown"    # walkingarea / internal link, not a vehicle link
    in_cor = _edge_of_lane(in_lane) in corridor
    out_cor = _edge_of_lane(out_lane) in corridor
    if in_cor and out_cor:
        return "cor_straight"
    if in_cor:
        return "cor_turn"
    return "side"


def apply_corridor_priority(net_file: str, *, green_s: float = DEFAULT_GREEN_S,
                            cross_green_s: float = DEFAULT_CROSS_GREEN_S,
                            split_s: float = DEFAULT_SPLIT_S,
                            verbose: bool = True, **_ignored):
    """Re-time the imported program of every traffic light on the corridor.

    The imported programs (10 phases here: main green, green extension, amber,
    all-red, for each of two directions, plus a pedestrian phase) were generated
    together with each junction's right-of-way logic, so the phase sequence and
    every state string are kept verbatim (a re-ordered or hand-written program
    is flagged "incompatible with logic at junction" by SUMO). Only the two
    main-phase durations change.

    Two kinds of corridor light:

    * "through" lights (189, 719 here): the phase that greens the movements
      LEAVING the stretch also greens the movements entering it. That phase gets
      a long green, the side-street phase ``cross_green_s``.
    * "split" lights (532 here): the corridor turns at the junction and the
      westbound ENTRY (-2 -> 21) and eastbound EXIT (-21 -> 2) sit in different
      phases. Favouring either starves the other (a long eastbound phase left
      the westbound bus queued for the whole run; a long westbound phase queued
      eastbound traffic back over the stretch). Both main phases get a short,
      equal ``split_s`` so the eastbound queue during red (~7 vehicles at 900
      veh/h) fits in the 25 m of edge -21 and never reaches the stretch.

    Through lights are then given exactly TWICE the split lights' cycle so all
    corridor lights stay phase-locked for the whole run (common offset: every
    light starts in its corridor phase). Call after traci.start(). Returns the
    list of TLS ids changed.
    """
    corridor = _corridor_edges()
    exits = _exit_edges()
    crossings = _crossing_edges_from_net(net_file)
    right_turns = _right_turn_links_from_net(net_file)
    changed = []
    plans = []
    installed = {}
    link_info = {}
    for tls in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls)
        if not links:
            continue
        classes = []
        info = {}
        for idx, entry in enumerate(links):
            if not entry:
                classes.append("x_unknown")   # no lane info: never treat as a vehicle link
                continue
            in_lane, out_lane, via = entry[0]
            cls = _link_class(in_lane, out_lane, via, corridor, crossings)
            classes.append(cls)
            info[idx] = (cls, _edge_of_lane(in_lane), _edge_of_lane(out_lane))
        link_info[tls] = info
        straight_idx = [i for i, c in enumerate(classes) if c == "cor_straight"]
        exit_idx = [i for i in straight_idx
                    if _edge_of_lane(links[i][0][0]) in exits]
        entry_idx = [i for i in straight_idx if i not in exit_idx]
        side_idx = [i for i, c in enumerate(classes) if c == "side"]
        if not straight_idx:
            continue
        logics = traci.trafficlight.getAllProgramLogics(tls)
        if not logics:
            continue
        base = logics[0]

        def score(state, idx):
            # Full green counts 1, permissive (yielding) green 0.7.
            return sum(1.0 if state[i] == "G" else 0.7 if state[i] == "g" else 0.0
                       for i in idx)

        main = [i for i, p in enumerate(base.phases) if p.duration >= 10.0]
        if not main:
            continue
        cor_phase = max(main, key=lambda i: (score(base.phases[i].state, exit_idx),
                                             score(base.phases[i].state, straight_idx),
                                             base.phases[i].duration))
        if score(base.phases[cor_phase].state, straight_idx) <= 0.0:
            continue
        side_phase = max((i for i in main if i != cor_phase),
                         key=lambda i: (score(base.phases[i].state, side_idx),
                                        base.phases[i].duration), default=None)
        entry_ok = all(base.phases[cor_phase].state[i] in "Gg" for i in entry_idx)
        kind = "through" if entry_ok else "split"
        plans.append((tls, base, cor_phase, side_phase, len(straight_idx), kind))

    def cycle_of(base, cor_phase, side_phase, cor_s, side_s):
        return sum(cor_s if i == cor_phase else side_s if i == side_phase else p.duration
                   for i, p in enumerate(base.phases))

    split = [pl for pl in plans if pl[5] == "split"]
    t_split = max((cycle_of(b, c, sp, split_s, split_s) for _, b, c, sp, _, _ in split),
                  default=0.0)
    # Through lights: corridor phase fills 2 x the split cycle (phase lock), or
    # green_s when there is no split light; side phase absorbs rounding.
    for tls, base, cor_phase, side_phase, n_links, kind in plans:
        if kind == "split":
            cor_s, side_s = split_s, split_s
        else:
            cor_s, side_s = green_s, cross_green_s
            if t_split > 0:
                fixed = cycle_of(base, cor_phase, side_phase, 0.0, 0.0)
                cor_s = max(green_s, 2.0 * t_split - fixed - side_s)
        # Split light: the corridor exit links that are RIGHT turns (dir="r" in
        # the net) are permitted, yielding ('g'), during the entry phase too.
        # A right turn only merges with the traffic it yields to, so this is
        # SUMO-safe (no "mutual conflict" between two minor movements), and it
        # is how a right-turn-on-red rule behaves. Without it the eastbound
        # curb lane queued back over the stretch for the whole entry phase.
        # Left-turn exits stay red (they would cross the entering left turn).
        permit = set()
        if kind == "split":
            permit = {i for i in right_turns.get(tls, set())
                      if link_info[tls].get(i, ("",))[0] == "cor_straight"
                      and link_info[tls][i][1] in exits}
        vehicle_links = [i for i, (cls, _, _) in link_info[tls].items() if not cls.startswith("x_")]
        phases = []
        for i, p in enumerate(base.phases):
            dur = p.duration
            if i == cor_phase:
                dur = cor_s
            elif i == side_phase:
                dur = side_s
            state = p.state
            if permit and i != cor_phase:
                st = list(state)
                if any(state[k] == "y" for k in vehicle_links):
                    # Amber for the permitted links too, so 'g' never drops
                    # straight to 'r' (SUMO logs that as a missing yellow phase
                    # and the affected vehicles brake at emergency rates).
                    for k in permit:
                        if st[k] == "r":
                            st[k] = "y"
                elif any(state[k] in "Gg" for k in vehicle_links):
                    for k in permit:
                        if st[k] == "r":
                            st[k] = "g"
                state = "".join(st)
            phases.append(traci.trafficlight.Phase(dur, state, p.minDur, p.maxDur))
        logic = traci.trafficlight.Logic(PROGRAM_ID, base.type, cor_phase, phases)
        traci.trafficlight.setProgramLogic(tls, logic)
        traci.trafficlight.setProgram(tls, PROGRAM_ID)
        traci.trafficlight.setPhase(tls, cor_phase)
        changed.append(tls)
        installed[tls] = (phases, cor_phase, side_phase, kind)
        if verbose:
            print(f"[signals] {tls} ({kind}): corridor phase {cor_phase} -> {cor_s:.0f}s, "
                  f"side phase {side_phase} -> {side_s:.0f}s, "
                  f"cycle {sum(p.duration for p in phases):.0f}s "
                  f"({n_links} corridor through-links)")
    if installed and os.environ.get("DATASET_SIGNAL_GREEN_WAVE", "1").strip() not in ("0", "false", "no"):
        _green_wave_offsets(net_file, installed, link_info, verbose)
    if verbose and not changed:
        print("[signals] no corridor traffic lights found; programs unchanged")
    return changed


def _green_wave_offsets(net_file, installed, link_info, verbose):
    """Offset each through light so its red window falls between the platoons
    a split light releases into the corridor (a one-direction green wave).

    At a split light the corridor entry is green only in the short side phase,
    so vehicles enter the stretch in platoons once per split cycle. A through
    light downstream whose red happens to coincide with a platoon's arrival
    queues that platoon on the stretch. We compute when the platoon leaves the
    split light, add the travel time (route distance / 10 m/s), and start the
    through light's corridor phase with just enough remaining green that its
    red begins right after the platoon has cleared it.
    """
    try:
        import sumolib
        net = sumolib.net.readNet(net_file)
    except Exception:  # noqa: BLE001
        return
    wb, eb = _routes()

    def phase_starts(phases, first):
        """Start time of every phase, running from ``first`` at t=0."""
        order = list(range(first, len(phases))) + list(range(0, first))
        t, out = 0.0, {}
        for i in order:
            out[i] = t
            t += phases[i].duration
        return out, t

    def route_distance(route, from_edge, to_edge):
        if from_edge not in route or to_edge not in route:
            return None
        a, b = route.index(from_edge), route.index(to_edge)
        if b < a:
            return None
        return sum(net.getEdge(e).getLength() for e in route[a:b + 1])

    for s_tls, (s_phases, s_cor, s_side, s_kind) in installed.items():
        if s_kind != "split" or s_side is None:
            continue
        starts, s_cycle = phase_starts(s_phases, s_cor)
        # Entry green = side phase plus the following extension phase(s) that
        # keep the entry links green.
        entry_links = [i for i, (cls, in_e, out_e) in link_info[s_tls].items()
                       if cls == "cor_straight" and in_e not in _exit_edges()]
        if not entry_links:
            continue
        g0 = starts[s_side]
        g1 = g0 + s_phases[s_side].duration
        j = (s_side + 1) % len(s_phases)
        while j != s_side and all(s_phases[j].state[i] in "Gg" for i in entry_links):
            g1 += s_phases[j].duration
            j = (j + 1) % len(s_phases)
        out_edges = {link_info[s_tls][i][2] for i in entry_links}
        for t_tls, (t_phases, t_cor, t_side, t_kind) in installed.items():
            if t_kind != "through":
                continue
            in_edges = {in_e for cls, in_e, _ in link_info[t_tls].values() if cls == "cor_straight"}
            dists = [route_distance(r, oe, ie) for r in (wb, eb)
                     for oe in out_edges for ie in in_edges]
            dists = [d for d in dists if d is not None]
            if not dists:
                continue
            tau = min(dists) / 10.0 + 3.0
            red_start = g1 + tau                     # platoon tail has passed
            # Remaining green in the corridor phase so that red starts then:
            # corridor phase + any green extension phases before the yellow.
            ext = 0.0
            j = (t_cor + 1) % len(t_phases)
            cor_links = [i for i, (cls, _, _) in link_info[t_tls].items() if cls == "cor_straight"]
            while j != t_cor and all(t_phases[j].state[i] in "Gg" for i in cor_links):
                ext += t_phases[j].duration
                j = (j + 1) % len(t_phases)
            remaining = red_start - ext
            t_cycle = sum(p.duration for p in t_phases)
            while remaining <= 1.0:
                remaining += s_cycle
            remaining = min(remaining, t_phases[t_cor].duration)
            try:
                traci.trafficlight.setPhase(t_tls, t_cor)
                traci.trafficlight.setPhaseDuration(t_tls, remaining)
            except traci.TraCIException:
                continue
            if verbose:
                print(f"[signals] {t_tls}: green wave from {s_tls} (platoon leaves at "
                      f"t={g1:.0f}s, +{tau:.0f}s travel) -> first red at t={red_start:.0f}s, "
                      f"cycle {t_cycle:.0f}s")
