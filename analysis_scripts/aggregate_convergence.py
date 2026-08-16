#!/usr/bin/env python
"""Aggregate per-repeat forward DG convergence files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"sampling_time_ns", "dg_kcal_mol"}


def aggregate_repeat_convergence(results_dir):
    """Aggregate repeat-level convergence files in one simulation directory."""
    results_dir = Path(results_dir)
    repeat_files = sorted(results_dir.glob("run*/RED/dg_convergence.csv"))
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
        data["repeat_file"] = str(path)
        frames.append(data)

    all_data = pd.concat(frames, ignore_index=True)

    def sample_std(values):
        return np.std(values, ddof=1) if len(values) > 1 else np.nan

    grouped = all_data.groupby("sampling_time_ns", sort=True)
    summary = grouped["dg_kcal_mol"].agg(
        av_dg_kcal_mol="mean",
        std_dev_kcal_mol=sample_std,
    ).reset_index()

    n_repeats = grouped["dg_kcal_mol"].count().reset_index(name="n_repeats")
    summary = summary.merge(n_repeats, on="sampling_time_ns")
    summary["sem_kcal_mol"] = (
        summary["std_dev_kcal_mol"] / np.sqrt(summary["n_repeats"])
    )

    output = summary[
        [
            "sampling_time_ns",
            "av_dg_kcal_mol",
            "std_dev_kcal_mol",
            "sem_kcal_mol",
        ]
    ]
    output.to_csv(results_dir / "dg_convergence.csv", index=False)
    return output


def discover_results_directories(root):
    """Find results directories containing repeat convergence files."""
    candidates = list(root.glob("*/US/separation/results"))
    candidates.extend(root.glob("*/US/RMSD/results/*"))
    return sorted(
        path for path in candidates
        if path.is_dir() and any(path.glob("run*/RED/dg_convergence.csv"))
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
    for results_dir in results_directories:
        summary = aggregate_repeat_convergence(results_dir)
        print(f"Wrote {results_dir / 'dg_convergence.csv'} ({len(summary)} times)")


if __name__ == "__main__":
    main()
