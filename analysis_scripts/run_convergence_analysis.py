#!/usr/bin/env python
"""Launch forward WHAM convergence analyses over the configured time grids."""

import argparse
import logging
import shutil
from pathlib import Path

from .convergence_analysis import generate_dg_convergence


RMSD_TIMES_NS = [10, 12, 14, 16, 18, 20]
SEPARATION_TIMES_NS = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


def rmsd_state(dof):
    """Return whether an RMSD DOF is bound or bulk."""
    dof_lower = dof.lower()
    if "_bulk" in dof_lower:
        return "bulk"
    if dof_lower.endswith("_only") or "with" in dof_lower:
        return "bound"
    return "bulk"


def discover_calculations(root, systems=None, stages=("separation", "RMSD")):
    """Discover separation and RMSD RED repeat directories."""
    systems = set(systems) if systems else None
    calculations = []

    if "separation" in stages:
        for red_dir in sorted(root.glob("*/US/separation/results/run*/RED")):
            parts = red_dir.relative_to(root).parts
            system = parts[0]
            if systems is not None and system not in systems:
                continue
            calculations.append({
                "system": system,
                "stage": "separation",
                "dof": None,
                "rmsd_unbound": False,
                "run_number": int(parts[4].replace("run", "")),
                "red_dir": red_dir,
            })

    if "RMSD" in stages:
        for red_dir in sorted(root.glob("*/US/RMSD/results/*/run*/RED")):
            parts = red_dir.relative_to(root).parts
            system = parts[0]
            dof = parts[4]
            if systems is not None and system not in systems:
                continue
            calculations.append({
                "system": system,
                "stage": "RMSD",
                "dof": dof,
                "rmsd_unbound": rmsd_state(dof) == "bulk",
                "run_number": int(parts[5].replace("run", "")),
                "red_dir": red_dir,
            })

    return calculations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="MolecularGluesBFE repository root",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="Analyse only this system; may be supplied multiple times",
    )
    parser.add_argument(
        "--stage",
        choices=["separation", "RMSD", "all"],
        default="all",
    )
    parser.add_argument("--run", type=int, help="Analyse only this repeat")
    parser.add_argument(
        "--wham",
        default=shutil.which("wham") or "wham",
        help="WHAM executable",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional log file; defaults to convergence_analysis.log in root",
    )
    args = parser.parse_args()

    log_path = args.log or args.root / "convergence_analysis.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stages = ("separation", "RMSD") if args.stage == "all" else (args.stage,)
    calculations = discover_calculations(args.root, args.systems, stages)
    if args.run is not None:
        calculations = [
            calculation for calculation in calculations
            if calculation["run_number"] == args.run
        ]

    logging.info("Found %d calculations", len(calculations))
    print(f"Found {len(calculations)} calculations")

    successes = 0
    failures = 0
    for calculation in calculations:
        times = (
            SEPARATION_TIMES_NS
            if calculation["stage"] == "separation"
            else RMSD_TIMES_NS
        )

        label = (
            f"{calculation['system']} {calculation['stage']} "
            f"{calculation['dof'] or ''} run{calculation['run_number']}"
        ).strip()
        print(f"Starting {label}")
        logging.info("Starting %s", label)

        try:
            generate_dg_convergence(
                calculation["system"],
                calculation["stage"],
                times,
                dof=calculation["dof"],
                run_number=calculation["run_number"],
                wham_executable=args.wham,
                rmsd_unbound=calculation["rmsd_unbound"],
            )
            successes += 1
            logging.info("Completed %s", label)
        except Exception as exc:
            failures += 1
            logging.exception("Failed %s: %s", label, exc)
            print(f"  FAILED: {exc}")

    print(f"Completed: {successes}; failed/skipped: {failures}")
    logging.info("Completed: %d; failed/skipped: %d", successes, failures)


if __name__ == "__main__":
    main()
