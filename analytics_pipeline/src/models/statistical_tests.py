"""Statistical significance tests for strategy and model comparisons.

The functions in this module answer the question every examiner of a
financial-ML project is trained to ask: *is the reported difference between
two strategies statistically meaningful, or is it noise?*

Four tests are implemented, each with a clear, well-cited motivation:

1. ``paired_bootstrap_p_value`` - paired bootstrap test on the mean
   difference of two return streams. Non-parametric, robust to non-normal
   returns, and is the standard recommendation when sample sizes are small
   and the distribution is heavy-tailed (a defining feature of crypto
   returns).
2. ``diebold_mariano_test`` - the Diebold-Mariano (1995) test on the
   difference of forecast losses. Standard practice in econometrics for
   comparing two forecasting models on the same series.
3. ``probabilistic_sharpe_ratio`` - the probability that an observed Sharpe
   exceeds a benchmark Sharpe given finite-sample noise (Bailey and Lopez
   de Prado, 2012). The benchmark is typically zero; values above 95
   percent indicate statistical confidence the strategy is non-trivial.
4. ``deflated_sharpe_ratio`` - the same idea adjusted for multiple-testing
   inflation: when N strategies are tried and the best is reported, the
   raw Sharpe of the winner overstates skill. DSR penalises by the count
   of trials, kurtosis, and skewness of returns (Bailey and Lopez de
   Prado, 2014).

References
----------
Bailey, D. H. and Lopez de Prado, M. (2012) 'The Sharpe Ratio Efficient
    Frontier', *Journal of Risk*, 15(2), pp. 3-44.
Bailey, D. H. and Lopez de Prado, M. (2014) 'The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality',
    *Journal of Portfolio Management*, 40(5), pp. 94-107.
Diebold, F. X. and Mariano, R. S. (1995) 'Comparing Predictive Accuracy',
    *Journal of Business and Economic Statistics*, 13(3), pp. 253-263.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy import stats

from src.reproducibility import LIVESTRAT_SEED


def paired_bootstrap_p_value(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    *,
    n_iterations: int = 5000,
    seed: int = LIVESTRAT_SEED,
) -> Tuple[float, float, Tuple[float, float]]:
    """Return p-value, observed mean diff, and 95% CI for mean(a) - mean(b).

    The null hypothesis is that the two return streams have the same mean.
    A two-sided p-value is returned. The bootstrap resampling pairs the
    observations (same index from each series), so the two strategies must
    be evaluated on the same time stamps - which is exactly the LiveStrat
    walk-forward setting.

    Parameters
    ----------
    returns_a, returns_b : array-like of equal length
        Per-period returns of strategies A and B. NaNs are dropped pairwise.
    n_iterations : int
        Number of bootstrap resamples. 5000 is the literature default for a
        stable p-value at 1e-3 resolution.
    seed : int
        Random seed for reproducibility. Defaults to ``LIVESTRAT_SEED``.

    Returns
    -------
    p_value : float
        Two-sided bootstrap p-value.
    observed_diff : float
        Observed mean(a) - mean(b).
    ci_95 : (float, float)
        Bootstrap 95 percent confidence interval for the mean difference.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap_p_value expects equal-length series")

    finite_mask = np.isfinite(a) & np.isfinite(b)
    a = a[finite_mask]
    b = b[finite_mask]
    n = len(a)
    if n < 5:
        return float("nan"), float("nan"), (float("nan"), float("nan"))

    diff = a - b
    observed = diff.mean()

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()

    # Two-sided p-value under the null that the true mean diff is zero.
    centred = boot_means - observed
    p_value = float((np.abs(centred) >= abs(observed)).mean())

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return p_value, float(observed), (float(ci_lo), float(ci_hi))


def diebold_mariano_test(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    *,
    h: int = 1,
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    The losses can be any per-period error metric (squared error, absolute
    error, negative log likelihood, etc.). A positive test statistic with
    a small p-value means model A has *higher* loss than model B - i.e. B
    is significantly better.

    Parameters
    ----------
    losses_a, losses_b : array-like of equal length
        Per-period loss values for two models on the same series.
    h : int
        Forecast horizon, used to set the Newey-West lag length. ``h=1``
        is correct for one-step-ahead forecasts; longer horizons require a
        larger lag to handle the induced autocorrelation in the loss
        differential. Default 1.

    Returns
    -------
    dm_stat : float
        The Diebold-Mariano test statistic. Asymptotically N(0, 1).
    p_value : float
        Two-sided p-value.
    """
    a = np.asarray(losses_a, dtype=float)
    b = np.asarray(losses_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("diebold_mariano_test expects equal-length loss series")

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = len(a)
    if n < 5:
        return float("nan"), float("nan")

    d = a - b
    mean_d = d.mean()

    # Newey-West variance with lag h - 1.
    gamma_0 = np.var(d, ddof=0)
    variance = gamma_0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        weight = 1.0 - lag / float(h)
        variance += 2.0 * weight * cov
    if variance <= 0:
        return float("nan"), float("nan")

    dm_stat = mean_d / math.sqrt(variance / n)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 365 * 6,
) -> Tuple[float, float]:
    """Probabilistic Sharpe Ratio (Bailey and Lopez de Prado, 2012).

    Returns the probability that the *true* (population) Sharpe of the
    underlying strategy exceeds a benchmark Sharpe, given finite-sample
    noise and the higher-moment structure of the realised returns.

    Parameters
    ----------
    returns : array-like
        Per-period strategy returns (not cumulative).
    benchmark_sharpe : float
        Annualised benchmark Sharpe to test against. Default 0 (the strategy
        outperforms a flat return).
    periods_per_year : int
        Used to annualise the realised Sharpe. The LiveStrat default of
        ``365 * 6`` corresponds to 4-hour candles, 365 days a year.

    Returns
    -------
    realised_sharpe : float
        Annualised Sharpe computed from ``returns``.
    psr : float
        Probability the true Sharpe is above ``benchmark_sharpe``. Values
        above 0.95 indicate statistical confidence.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 30:
        return float("nan"), float("nan")

    mean_r = r.mean()
    std_r = r.std(ddof=1)
    if std_r <= 0:
        return float("nan"), float("nan")

    sr_periodic = mean_r / std_r
    realised_sr = sr_periodic * math.sqrt(periods_per_year)

    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=True, bias=False)

    # Convert benchmark from annualised to periodic to match.
    benchmark_periodic = benchmark_sharpe / math.sqrt(periods_per_year)

    numerator = (sr_periodic - benchmark_periodic) * math.sqrt(n - 1)
    denominator = math.sqrt(1.0 - skew * sr_periodic + ((kurt - 1.0) / 4.0) * sr_periodic ** 2)
    if denominator <= 0:
        return realised_sr, float("nan")

    psr = float(stats.norm.cdf(numerator / denominator))
    return float(realised_sr), psr


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    n_trials: int,
    periods_per_year: int = 365 * 6,
) -> Tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014).

    Adjusts the realised Sharpe of the *winning* strategy for the fact that
    it was selected from ``n_trials`` candidates. Reports DSR as a
    probability that the true Sharpe is positive, after correcting for
    multiple testing.

    Parameters
    ----------
    returns : array-like
        Per-period returns of the winning strategy.
    n_trials : int
        How many strategy variants were tried. For LiveStrat this is the
        product of (asset count) x (timeframe count) x (strategy family
        count) x (target horizon count).
    periods_per_year : int
        Used to annualise. Default 365*6 (4-hour cadence).

    Returns
    -------
    sharpe : float
        Annualised Sharpe of the winning strategy.
    dsr : float
        Deflated probability that the true Sharpe is positive.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be a positive integer")

    # Use PSR machinery but inflate the benchmark by the expected maximum
    # Sharpe under the null when n_trials are tried (Bailey and Lopez de
    # Prado, 2014, eq. 4).
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 30:
        return float("nan"), float("nan")

    sr_periodic = r.mean() / r.std(ddof=1)
    realised_sr = sr_periodic * math.sqrt(periods_per_year)

    # Expected maximum Sharpe under the null with n_trials independent
    # candidates and a sample variance of Sharpe of ((1 - skew * SR + ...) / (n-1)).
    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=True, bias=False)

    euler_mascheroni = 0.5772156649
    expected_max_z = (
        (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    variance_of_sharpe = (1.0 - skew * sr_periodic + ((kurt - 1.0) / 4.0) * sr_periodic ** 2) / (n - 1)
    if variance_of_sharpe <= 0:
        return realised_sr, float("nan")
    expected_max_sharpe_periodic = expected_max_z * math.sqrt(variance_of_sharpe)

    numerator = (sr_periodic - expected_max_sharpe_periodic) * math.sqrt(n - 1)
    denominator = math.sqrt(1.0 - skew * sr_periodic + ((kurt - 1.0) / 4.0) * sr_periodic ** 2)
    if denominator <= 0:
        return realised_sr, float("nan")

    dsr = float(stats.norm.cdf(numerator / denominator))
    return float(realised_sr), dsr


def summarise_strategy_vs_benchmark(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    n_trials: int = 1,
    periods_per_year: int = 365 * 6,
) -> dict:
    """Produce the standard statistical-evidence dictionary used in reports.

    Combines all four tests into one summary dict that can be written
    straight into a CSV or JSON output. Keys are stable and self-describing
    so report figures can be auto-generated from these rows.
    """
    p_boot, observed_diff, ci_95 = paired_bootstrap_p_value(
        strategy_returns, benchmark_returns
    )
    dm_stat, dm_p = diebold_mariano_test(
        -strategy_returns, -benchmark_returns  # losses = negative returns
    )
    sharpe, psr = probabilistic_sharpe_ratio(
        strategy_returns, benchmark_sharpe=0.0, periods_per_year=periods_per_year
    )
    _, dsr = deflated_sharpe_ratio(
        strategy_returns, n_trials=n_trials, periods_per_year=periods_per_year
    )

    return {
        "n_observations": int(np.isfinite(strategy_returns).sum()),
        "mean_strategy_return": float(np.nanmean(strategy_returns)),
        "mean_benchmark_return": float(np.nanmean(benchmark_returns)),
        "mean_difference": observed_diff,
        "bootstrap_p_value": p_boot,
        "bootstrap_ci_low": ci_95[0],
        "bootstrap_ci_high": ci_95[1],
        "diebold_mariano_stat": dm_stat,
        "diebold_mariano_p_value": dm_p,
        "annualised_sharpe": sharpe,
        "probabilistic_sharpe_ratio": psr,
        "deflated_sharpe_ratio": dsr,
        "n_trials_for_dsr": n_trials,
        "evidence_label": _evidence_label(p_boot, psr, dsr),
    }


def _evidence_label(
    bootstrap_p: float, psr: Optional[float], dsr: Optional[float]
) -> str:
    """Translate the three test results into a single, examiner-friendly label.

    Labels are deliberately ordered from strongest to weakest evidence so
    the UI and report can colour-code consistently.
    """
    if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), [bootstrap_p, psr, dsr])):
        return "insufficient_data"
    if bootstrap_p < 0.05 and (psr or 0) > 0.95 and (dsr or 0) > 0.95:
        return "strong_evidence"
    if bootstrap_p < 0.05 and (psr or 0) > 0.90:
        return "moderate_evidence"
    if bootstrap_p < 0.10:
        return "weak_evidence"
    return "no_evidence"
