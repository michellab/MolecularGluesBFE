"""Helpers for retrospective umbrella-sampling convergence analysis."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from .us_analysis import (
    RMSDContribution,
    SepContribution,
    obtain_dirpath,
)


SAMPLES_PER_NS = 2000


import subprocess

from .us_analysis import (
    RMSDContribution,
    SepContribution,
    obtain_dirpath,
)


def _numeric_cv_files(directory):
    """Return numeric CV filenames sorted by their umbrella centre."""
    cv_files = []
    for path in Path(directory).glob("*.txt"):
        try:
            cv_files.append((float(path.stem), path))
        except ValueError:
            continue
    return sorted(cv_files)


def write_convergence_metafile(target_dir, free_energy_step, k_cv):
    """Write a WHAM metadata file for a staged sampling directory."""
    target_dir = Path(target_dir)
    cv_files = _numeric_cv_files(target_dir)
    if not cv_files:
        raise RuntimeError(f"No numeric CV files found in {target_dir}")

    lines = []
    for cv_value, path in cv_files:
        wham_center = cv_value * 0.1 if free_energy_step == "RMSD" else cv_value
        lines.append(f"{path.resolve()} {wham_center:.3f} {k_cv}\n")

    metafile = target_dir / "metafile.txt"
    metafile.write_text("".join(lines))
    return metafile


def run_wham_and_calculate_dg(
    target_dir,
    free_energy_step,
    wham_executable="wham",
    rmsd_unbound=False,
):
    """Run WHAM in a staged directory and calculate its stage ΔG."""
    target_dir = Path(target_dir)
    cv_files = _numeric_cv_files(target_dir)
    cv_values = np.array([cv_value for cv_value, _ in cv_files])

    if free_energy_step == "separation":
        k_cv = 1000
        bins = 200
        lower_limit = np.min(cv_values)
        upper_limit = np.max(cv_values)
    elif free_energy_step == "RMSD":
        k_cv = 500
        bins = 100
        lower_limit = np.round(np.min(cv_values) * 0.1 + 0.02, 4)
        upper_limit = np.round(np.max(cv_values) * 0.1 + 0.02, 4)
    else:
        raise ValueError("free_energy_step must be 'separation' or 'RMSD'")

    metafile = write_convergence_metafile(target_dir, free_energy_step, k_cv)
    pmf_file = target_dir / "pmf.txt"
    log_file = target_dir / "wham.log"
    command = [
        wham_executable,
        str(lower_limit), str(upper_limit), str(bins), "1e-6", "300", "0",
        str(metafile), str(pmf_file),
    ]

    with log_file.open("w") as log_handle:
        subprocess.run(
            command,
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    pmf = np.loadtxt(pmf_file)
    finite = np.isfinite(pmf[:, 1])
    x = pmf[finite, 0]
    free_energy = pmf[finite, 1]

    if free_energy_step == "separation":
        delta_g = SepContribution(x, free_energy, upper_limit - 0.1)
    else:
        delta_g = RMSDContribution(
            x,
            free_energy,
            k_rmsd=500,
            unbound=rmsd_unbound,
        )

    return delta_g, pmf_file

def generate_dg_convergence(
    system,
    free_energy_step,
    sampling_times_ns,
    dof=None,
    run_number=1,
    wham_executable="wham",
    rmsd_unbound=False,
    max_equilibration_time_ns=10.0,
    keep_temporary=False,
):
    """Generate forward DG estimates for one US repeat.

    The target sampling time includes equilibration. A repeat containing an
    equilibration time above the configured maximum is rejected so that an
    anomalous window cannot produce a partially sampled WHAM calculation.
    """
    red_dir = Path(
        obtain_dirpath(
            system,
            free_energy_step,
            dof,
            equilibration="RED",
            run_number=run_number,
        )
    )
    equilibration_file = red_dir / "equil_times.csv"
    equilibration_df = pd.read_csv(equilibration_file)

    if equilibration_df["equilibration_time_ns"].max() > max_equilibration_time_ns:
        raise ValueError(
            f"Skipping {red_dir}: equilibration time exceeds "
            f"{max_equilibration_time_ns} ns"
        )

    results = []
    for sampling_time_ns in sampling_times_ns:
        target_dir = None
        try:
            target_dir, _ = prepare_sampling_directory(
                system,
                free_energy_step,
                sampling_time_ns,
                dof=dof,
                run_number=run_number,
            )
            delta_g, _ = run_wham_and_calculate_dg(
                target_dir,
                free_energy_step,
                wham_executable=wham_executable,
                rmsd_unbound=rmsd_unbound,
            )
            results.append({
                "sampling_time_ns": sampling_time_ns,
                "dg_kcal_mol": delta_g,
            })
        finally:
            if target_dir is not None and not keep_temporary:
                shutil.rmtree(target_dir)

    convergence_df = pd.DataFrame(
        results,
        columns=["sampling_time_ns", "dg_kcal_mol"],
    )
    convergence_df.to_csv(red_dir / "dg_convergence.csv", index=False)
    return convergence_df


def prepare_sampling_directory(
    system,
    free_energy_step,
    sampling_time_ns,
    dof=None,
    run_number=1,
    overwrite=False,
):
    """Prepare prefix-truncated CV files for one total sampling time.

    ``sampling_time_ns`` is the total simulation time, including the
    equilibration removed by RED. Each RED window therefore contributes
    ``(sampling_time_ns - equilibration_time_ns)`` of data.

    Parameters
    ----------
    system : str
        System name.
    free_energy_step : str
        ``'separation'`` or ``'RMSD'``.
    sampling_time_ns : float
        Total target simulation time in ns.
    dof : str, optional
        RMSD degree of freedom.
    run_number : int, default=1
        Repeat number.
    overwrite : bool, default=False
        Whether to replace an existing target directory.

    Returns
    -------
    pathlib.Path
        Directory containing the truncated CV files.
    pandas.DataFrame
        Per-window truncation information.
    """

    if sampling_time_ns <= 0:
        raise ValueError("sampling_time_ns must be positive")

    red_dir = Path(
        obtain_dirpath(
            system,
            free_energy_step,
            dof,
            equilibration="RED",
            run_number=run_number,
        )
    )
    if not red_dir.is_dir():
        raise FileNotFoundError(f"RED directory not found: {red_dir}")

    equilibration_file = red_dir / "equil_times.csv"
    if not equilibration_file.is_file():
        raise FileNotFoundError(f"Equilibration file not found: {equilibration_file}")

    equilibration_df = pd.read_csv(equilibration_file)
    required_columns = {"CV", "equilibration_time_ns"}
    if not required_columns.issubset(equilibration_df.columns):
        raise ValueError(
            f"{equilibration_file} must contain columns {sorted(required_columns)}"
        )

    target_dir = red_dir / str(int(sampling_time_ns))
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {target_dir}. "
                "Use overwrite=True only for a deliberately regenerated target."
            )
        raise NotImplementedError(
            "Refusing to remove an existing target directory; choose a new target "
            "or remove it manually after inspection."
        )

    # Resolve CV values to their actual filenames, preserving names such as
    # 1.0.txt rather than reconstructing them from floating-point values.
    cv_files = {}
    for path in red_dir.glob("*.txt"):
        try:
            cv_files[float(path.stem)] = path
        except ValueError:
            continue

    if not cv_files:
        raise RuntimeError(f"No numeric CV files found in {red_dir}")

    rows = []
    for record in equilibration_df.itertuples(index=False):
        cv_value = float(record.CV)
        equilibration_time_ns = float(record.equilibration_time_ns)

        if cv_value not in cv_files:
            raise FileNotFoundError(
                f"No RED CV file found for CV value {cv_value} in {red_dir}"
            )

        available_time_ns = sampling_time_ns - equilibration_time_ns
        if available_time_ns <= 0:
            raise ValueError(
                f"Sampling time {sampling_time_ns} ns is shorter than or equal to "
                f"the equilibration time {equilibration_time_ns} ns for CV {cv_value}"
            )

        input_path = cv_files[cv_value]
        data = np.atleast_2d(np.loadtxt(input_path))
        n_target = int(round(available_time_ns * SAMPLES_PER_NS))

        if n_target > len(data):
            available_total_ns = equilibration_time_ns + len(data) / SAMPLES_PER_NS
            raise ValueError(
                f"Insufficient data for CV {cv_value}: target {sampling_time_ns} ns, "
                f"available {available_total_ns:.4f} ns"
            )

        rows.append({
            "CV": cv_value,
            "equilibration_time_ns": equilibration_time_ns,
            "n_target": n_target,
            "available_time_ns": available_time_ns,
            "source_file": str(input_path),
        })

    target_dir.mkdir()
    for row in rows:
        data = np.atleast_2d(np.loadtxt(row["source_file"]))
        output_path = target_dir / Path(row["source_file"]).name
        np.savetxt(output_path, data[:row["n_target"]])

    return target_dir, pd.DataFrame(rows)
