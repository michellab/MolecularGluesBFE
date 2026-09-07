#!/usr/bin/env python
"""Aggregate per-repeat forward DG convergence files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"sampling_time_ns", "dg_kcal_mol"}

# DOFs excluded only from the overall RMSD total for these systems. The
# directory names are used here because the requested ``*_only`` names are
# represented without that suffix for the second DOF in this repository.
RMSD_TOTAL_EXCLUSIONS = {
    "CRBN_CK1a": {"CK1a_only", "CRBNwithCK1a"},
    "CRBN_len_CK1a": {"CK1a_only", "CRBN_lenwithCK1a"},
}


def _summarize_repeats(data):
    """Return repeat values and inter-repeat statistics in the standard format."""
    data = data.copy()
    data["repeat_number"] = data["repeat"].str.extract(r"run(\d+)")[0].astype(int)
    repeat_values = data.pivot_table(
        index="sampling_time_ns",
        columns="repeat_number",
        values="dg_kcal_mol",
        aggfunc="first",
    ).reindex(columns=[1, 2, 3])
    repeat_values.columns = ["dg_repeat1", "dg_repeat2", "dg_repeat3"]

    summary = repeat_values.copy()
    n_repeats = repeat_values.notna().sum(axis=1)
    summary["av_dg_kcal_mol"] = repeat_values.mean(axis=1)
    summary["std_dev_kcal_mol"] = repeat_values.std(axis=1, ddof=1)
    summary["sem_kcal_mol"] = (
        summary["std_dev_kcal_mol"] / np.sqrt(n_repeats)
    )
    return summary.reset_index()[
        [
            "sampling_time_ns",
            "dg_repeat1",
            "dg_repeat2",
            "dg_repeat3",
            "av_dg_kcal_mol",
            "std_dev_kcal_mol",
            "sem_kcal_mol",
        ]
    ]


def aggregate_repeat_convergence(
    results_dir, filename="dg_convergence.csv", boresch=False
):
    """Aggregate repeat-level convergence files in one simulation directory."""
    results_dir = Path(results_dir)
    pattern = (
        f"run*/{filename}"
        if boresch
        else f"run*/RED/{filename}"
    )
    repeat_files = sorted(results_dir.glob(pattern))
    if not repeat_files:
        raise FileNotFoundError(
            f"No repeat convergence files found in {results_dir}"
        )

    frames = []
    for path in repeat_files:
        data = pd.read_csv(path)
        if not REQUIRED_COLUMNS.issubset(data.columns):
            raise ValueError(
                f"{path} must contain columns {sorted(REQUIRED_COLUMNS)}"
            )
        data = data[["sampling_time_ns", "dg_kcal_mol"]].copy()
        data["repeat"] = path.relative_to(results_dir).parts[0]
        frames.append(data)

    output = _summarize_repeats(pd.concat(frames, ignore_index=True))
    output.to_csv(results_dir / filename, index=False)
    return output


def aggregate_overall_rmsd(system_dir, filename="dg_convergence.csv"):
    """Aggregate complete-repeat RMSD totals for one molecular-glue system.

    RMSD contributions are summed across all RMSD degrees of freedom within
    each repeat before statistics are calculated across repeats. A repeat is
    included at a given sampling time only when every RMSD degree of freedom
    has a value at that time.
    """
    system_dir = Path(system_dir)
    rmsd_dir = system_dir / "US" / "RMSD"
    excluded_dofs = RMSD_TOTAL_EXCLUSIONS.get(system_dir.name, set())
    repeat_files = sorted(
        path
        for path in rmsd_dir.glob(f"results/*/run*/RED/{filename}")
        if (
            path.name == filename
            and path.relative_to(rmsd_dir).parts[1] not in excluded_dofs
        )
    )
    if not repeat_files:
        raise FileNotFoundError(f"No RMSD convergence files found in {rmsd_dir}")

    frames = []
    for path in repeat_files:
        relative = path.relative_to(rmsd_dir).parts
        dof = relative[1]
        repeat = relative[2]
        data = pd.read_csv(path)
        if not REQUIRED_COLUMNS.issubset(data.columns):
            raise ValueError(
                f"{path} must contain columns {sorted(REQUIRED_COLUMNS)}"
            )
        data = data[["sampling_time_ns", "dg_kcal_mol"]].copy()
        data["dof"] = dof
        data["repeat"] = repeat
        frames.append(data)

    all_data = pd.concat(frames, ignore_index=True)
    required_dofs = all_data["dof"].nunique()

    complete_keys = (
        all_data.groupby(["repeat", "sampling_time_ns"])
        .agg(
            n_dofs=("dof", "nunique"),
            n_valid=("dg_kcal_mol", "count"),
        )
        .loc[lambda values: (values["n_dofs"] == required_dofs)
              & (values["n_valid"] == required_dofs)]
        .reset_index()[["repeat", "sampling_time_ns"]]
    )
    complete_data = all_data.merge(
        complete_keys,
        on=["repeat", "sampling_time_ns"],
        how="inner",
    )
    repeat_totals = (
        complete_data.groupby(["repeat", "sampling_time_ns"], as_index=False)
        ["dg_kcal_mol"]
        .sum()
    )

    output = _summarize_repeats(repeat_totals.rename(columns={"repeat": "repeat"}))
    output.to_csv(rmsd_dir / filename, index=False)
    return output



def aggregate_overall_boresch(system_dir, filename="dg_convergence.csv"):
    """Aggregate the five Boresch DOF contributions for one system."""
    system_dir = Path(system_dir)
    boresch_dir = system_dir / "US" / "Boresch"
    repeat_files = sorted(
        boresch_dir.glob(f"results/*/run*/{filename}")
    )
    if not repeat_files:
        raise FileNotFoundError(
            f"No Boresch convergence files found in {boresch_dir}"
        )

    frames = []
    for path in repeat_files:
        relative = path.relative_to(boresch_dir).parts
        dof = relative[1]
        repeat = relative[2]
        data = pd.read_csv(path)
        if not REQUIRED_COLUMNS.issubset(data.columns):
            raise ValueError(
                f"{path} must contain columns {sorted(REQUIRED_COLUMNS)}"
            )
        data = data[["sampling_time_ns", "dg_kcal_mol"]].copy()
        data["dof"] = dof
        data["repeat"] = repeat
        frames.append(data)

    all_data = pd.concat(frames, ignore_index=True)
    required_dofs = all_data["dof"].nunique()
    complete_keys = (
        all_data.groupby(["repeat", "sampling_time_ns"])
        .agg(
            n_dofs=("dof", "nunique"),
            n_valid=("dg_kcal_mol", "count"),
        )
        .loc[lambda values: (values["n_dofs"] == required_dofs)
              & (values["n_valid"] == required_dofs)]
        .reset_index()[["repeat", "sampling_time_ns"]]
    )
    complete_data = all_data.merge(
        complete_keys,
        on=["repeat", "sampling_time_ns"],
        how="inner",
    )
    repeat_totals = (
        complete_data.groupby(["repeat", "sampling_time_ns"], as_index=False)
        ["dg_kcal_mol"]
        .sum()
    )
    output = _summarize_repeats(repeat_totals)
    output.to_csv(boresch_dir / filename, index=False)
    return output


def discover_results_directories(root):
    """Find results directories containing repeat convergence files."""
    candidates = list(root.glob("*/US/separation/results"))
    candidates.extend(root.glob("*/US/RMSD/results/*"))
    return sorted(
        path for path in candidates
        if path.is_dir() and (
            any(path.glob("run*/RED/dg_convergence*.csv"))
            or any(path.glob("run*/dg_convergence*.csv"))
        )
    )


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
        help="Aggregate only this system; may be supplied multiple times",
    )
    args = parser.parse_args()

    results_directories = discover_results_directories(args.root)
    if args.systems:
        systems = set(args.systems)
        results_directories = [
            path for path in results_directories
            if path.relative_to(args.root).parts[0] in systems
        ]

    print(f"Found {len(results_directories)} simulation results directories")
    for filename in ("dg_convergence.csv", "dg_convergence_reverse.csv"):
        for results_dir in results_directories:
            try:
                summary = aggregate_repeat_convergence(results_dir, filename=filename)
            except FileNotFoundError:
                continue
            print(f"Wrote {results_dir / filename} ({len(summary)} times)")

    selected_systems = set(args.systems) if args.systems else None
    boresch_directories = sorted(
        path for path in args.root.glob("*/US/Boresch/results/*")
        if path.is_dir()
        and (selected_systems is None or path.relative_to(args.root).parts[0] in selected_systems)
        and any(path.glob("run*/dg_convergence*.csv"))
    )
    for filename in ("dg_convergence.csv", "dg_convergence_reverse.csv"):
        for results_dir in boresch_directories:
            try:
                summary = aggregate_repeat_convergence(
                    results_dir, filename=filename, boresch=True
                )
            except FileNotFoundError:
                continue
            print(f"Wrote {results_dir / filename} ({len(summary)} times)")

    systems = sorted({path.relative_to(args.root).parts[0] for path in results_directories})
    for system in systems:
        for filename in ("dg_convergence.csv", "dg_convergence_reverse.csv"):
            try:
                summary = aggregate_overall_rmsd(args.root / system, filename=filename)
            except FileNotFoundError:
                pass
            else:
                print(
                    f"Wrote {args.root / system / 'US' / 'RMSD' / filename} "
                    f"({len(summary)} times)"
                )
            try:
                summary = aggregate_overall_boresch(args.root / system, filename=filename)
            except FileNotFoundError:
                pass
            else:
                print(
                    f"Wrote {args.root / system / 'US' / 'Boresch' / filename} "
                    f"({len(summary)} times)"
                )


if __name__ == "__main__":
    main()
