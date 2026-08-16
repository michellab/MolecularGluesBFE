"""Bootstrap statistics for calculated-vs-experimental binding free energies (ΔG).

Computes the headline benchmark set — R² (squared Pearson), Kendall tau, MUE and
RMSE — from four plain arrays, with parametric-bootstrap 95% confidence intervals
and, for the correlation/ranking metrics (R², Kendall), a significance test
against a random-ranking null model.

Inputs (one entry per ligand, same order in every array), all in kcal/mol:
    exp        experimental ΔG
    exp_err    1-sigma error on the experimental ΔG
    calc       calculated ΔG
    calc_err   1-sigma error on the calculated ΔG (inter-repeat SEM)

Method (parametric bootstrap):

* ``n_bootstrap`` times, every point is perturbed by Gaussian noise with std taken
  from the error arrays and the metric recomputed; ``[ci_low, ci_high]`` are the
  2.5 / 97.5 percentiles and ``point`` is the metric on the raw, un-jittered data.
* For R² / Kendall a dummy (null) model draws both coordinates independently from
  the experimental values; the one-sided ``p_value`` is the probability the real
  metric does not beat the null. MUE/RMSE are error metrics (lower is better), so
  this null test is not meaningful for them (``show_significance=False``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats

DEFAULT_SEED = 42
DEFAULT_N_BOOTSTRAP = 1000

# Assumed experimental ΔG standard error (kcal/mol) used when no per-ligand
# experimental error is supplied. Affects only the bootstrap CIs and the null
# model, not the point estimates.
DEFAULT_EXPERIMENTAL_DG_ERROR = 0.23


@dataclasses.dataclass
class StatResult:
    """One metric: point estimate, bootstrap 95% CI and dummy-model p-value.

    ``point`` (the statistic on the raw, un-jittered data) is the headline value,
    paired with the bootstrap percentile interval [ci_low, ci_high]. ``value`` (the
    bootstrap mean) is kept for reference. ``show_significance`` is False for error
    metrics (MUE/RMSE), where beating the random null is not a meaningful claim.
    """

    name: str
    value: float  # bootstrap mean (kept for reference)
    ci_low: float
    ci_high: float
    p_value: float
    significance: str  # "p-value < 0.05" or "p-value > 0.05"
    point: float = float("nan")  # statistic on raw data (headline)
    show_significance: bool = True

    def format(self) -> str:
        return f"{self.name} = {self.point:.2f} [{self.ci_low:.2f}, {self.ci_high:.2f}], ({self.significance})"


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _ci_from_values(values: list[float], percentage: float = 95) -> tuple[float, float]:
    low = (100 - percentage) / 2
    high = 100 - low
    return tuple(float(x) for x in np.percentile(values, q=[low, high]))  # type: ignore[return-value]


def _compare_two_distributions(distrib_a: list[float], distrib_b: list[float]) -> float:
    """One-sided p-value for the hypothesis that distrib_a > distrib_b: the
    fraction of all (a, b) pairs with a < b."""
    reshaped_a = np.reshape(distrib_a, (len(distrib_a), 1))
    reshaped_b = np.reshape(distrib_b, (1, len(distrib_b)))
    differences = reshaped_a - reshaped_b
    return float(np.sum(differences < 0) / len(differences.flatten()))


# --------------------------------------------------------------------------- #
# per-sample metric functions (operate on (predicted, experimental) arrays)    #
# --------------------------------------------------------------------------- #
def _r2(pred: np.ndarray, exp: np.ndarray) -> float:
    return float(stats.pearsonr(pred, exp).statistic) ** 2


def _kendall(pred: np.ndarray, exp: np.ndarray) -> float:
    return float(stats.kendalltau(pred, exp).statistic)


def _mue(pred: np.ndarray, exp: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - exp)))


def _rmse(pred: np.ndarray, exp: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - exp) ** 2)))


# --------------------------------------------------------------------------- #
# noisy-sample generation                                                      #
# --------------------------------------------------------------------------- #
def _draw_parametric(rng, calc, calc_err, exp, exp_err):
    """Perturb each real point by Gaussian noise (predicted then experimental)."""
    n = len(calc)
    pred_out = np.empty(n)
    exp_out = np.empty(n)
    for k in range(n):
        pred_out[k] = calc[k] + rng.normal(loc=0, scale=calc_err[k])
        exp_out[k] = exp[k] + rng.normal(loc=0, scale=exp_err[k])
    return pred_out, exp_out


def _draw_dummy(rng, exp, exp_err):
    """Null model: both coordinates are independent random draws from the
    experimental (value, error) pairs, then perturbed by their experimental noise."""
    experimental_values = list(zip(exp.tolist(), exp_err.tolist(), strict=True))
    n = len(experimental_values)
    new_x = rng.choice(experimental_values, size=n, replace=True)
    new_y = rng.choice(experimental_values, size=n, replace=True)
    pred_out = np.empty(n)
    exp_out = np.empty(n)
    for k in range(n):
        x, y = new_x[k], new_y[k]
        pred_out[k] = x[0] + rng.normal(loc=0, scale=x[1])
        exp_out[k] = y[0] + rng.normal(loc=0, scale=y[1])
    return pred_out, exp_out


# --------------------------------------------------------------------------- #
# core bootstrap runner                                                        #
# --------------------------------------------------------------------------- #
def _run_metric(metric_fn, name, calc, calc_err, exp, exp_err, *, seed, n_bootstrap,
                show_significance=True) -> StatResult:
    rng = np.random.default_rng(seed=seed)

    real_dist: list[float] = []
    for _ in range(n_bootstrap):
        pred, expv = _draw_parametric(rng, calc, calc_err, exp, exp_err)
        real_dist.append(metric_fn(pred, expv))

    dummy_dist: list[float] = []
    for _ in range(n_bootstrap):
        pred, expv = _draw_dummy(rng, exp, exp_err)
        dummy_dist.append(metric_fn(pred, expv))

    point = float(metric_fn(np.asarray(calc, dtype=float), np.asarray(exp, dtype=float)))
    ci_low, ci_high = _ci_from_values(real_dist, percentage=95)
    value = float(np.mean(real_dist))
    p_value = _compare_two_distributions(real_dist, dummy_dist)
    significance = "p-value < 0.05" if p_value < 0.05 else "p-value > 0.05"
    return StatResult(name, value, ci_low, ci_high, p_value, significance,
                      point=point, show_significance=show_significance)


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def _as_arrays(exp, exp_err, calc, calc_err):
    arrs = [np.asarray(x, dtype=float) for x in (exp, exp_err, calc, calc_err)]
    if len({len(x) for x in arrs}) != 1:
        raise ValueError("All input arrays must have the same length.")
    return arrs  # exp, exp_err, calc, calc_err


def compute_report_stats(
    exp, exp_err, calc, calc_err,
    *, seed: int = DEFAULT_SEED, n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> dict[str, StatResult]:
    """Headline reporting set — R², Kendall tau, MUE, RMSE — in the input ΔG units.

    Returns a dict keyed by metric name ("R2", "Kendall", "MUE", "RMSE").
    """
    e, ee, c, ce = _as_arrays(exp, exp_err, calc, calc_err)
    kw = dict(seed=seed, n_bootstrap=n_bootstrap)
    results = [
        _run_metric(_r2, "R2", c, ce, e, ee, **kw),
        _run_metric(_kendall, "Kendall", c, ce, e, ee, **kw),
        _run_metric(_mue, "MUE", c, ce, e, ee, show_significance=False, **kw),
        _run_metric(_rmse, "RMSE", c, ce, e, ee, show_significance=False, **kw),
    ]
    return {r.name: r for r in results}


if __name__ == "__main__":
    # tiny smoke test
    rng = np.random.default_rng(0)
    exp = rng.normal(-9, 2, 12)
    calc = exp + rng.normal(0, 0.8, 12)
    for res in compute_report_stats(exp, np.full(12, DEFAULT_EXPERIMENTAL_DG_ERROR),
                                    calc, np.full(12, 0.3)).values():
        print(res.format())