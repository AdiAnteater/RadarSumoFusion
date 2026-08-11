"""
sync_route_files.py
===================
Optional one-off tidy-up. Rewrites the WB_route / EB_route edge lists in all 11
scenario/routes/*.rou.xml files so what is on disk matches what actually runs.

runner.py applies ambient_traffic.WB_ROUTE_EDGES / EB_ROUTE_EDGES to the temp
route file on every run regardless, so this changes no behaviour. It exists so
the repository does not document routing that is no longer used -- which matters
if the scenario files are cited or reproduced from the methods section.

Run from an activated venv, from sumo_traffic\\:

    python sync_route_files.py            # show what would change
    python sync_route_files.py --write    # apply it
"""

import argparse
import glob
import os
import re
import sys

import ambient_traffic

HERE       = os.path.dirname(os.path.abspath(__file__))
ROUTES_DIR = os.path.join(HERE, "scenarios", "routes")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Actually write the files (default is a dry run)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(ROUTES_DIR, "s*.rou.xml")))
    if not paths:
        sys.exit(f"ERROR: no route files found in {ROUTES_DIR}")

    print(f"WB_route -> {ambient_traffic.WB_ROUTE_EDGES}")
    print(f"EB_route -> {ambient_traffic.EB_ROUTE_EDGES}\n")

    changed = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            before = f.read()
        after = ambient_traffic.apply_map_edge_routes(before)
        name = os.path.basename(path)
        if before == after:
            print(f"  ok       {name}")
            continue
        changed += 1
        old = re.findall(r'<route\s+id="(?:WB|EB)_route"\s+edges="([^"]*)"', before)
        print(f"  rewrite  {name}   (was: {' | '.join(old)})")
        if args.write:
            with open(path, "w", encoding="utf-8") as f:
                f.write(after)

    print()
    if not args.write:
        print(f"{changed} file(s) would change. Re-run with --write to apply.")
    else:
        print(f"{changed} file(s) rewritten.")


if __name__ == "__main__":
    main()
