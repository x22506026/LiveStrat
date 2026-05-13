"""Tests for the statistical-significance module.

These tests verify the well-known limiting behaviours of the four tests
(non-pathological inputs return finite values, identical series return
no evidence, large differences return strong evidence) rather than
asserting numerical equality against a reference implementation - the
formulas themselves are taken directly from Bailey and Lopez de Prado
(2012, 2014) and Diebold and Mariano (1995).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = ROOT_DIR / "analytics_pipeline"
for path in (ROOT_DIR, ANALYTICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analytics_pipeline.src.models.statistical_tests import (
    deflated_sharpe_ratio,
    diebold_mariano_test,
    paired_bootstrap_p_value,
    probabilistic_sharpe_ratio,
    summarise_strategy_vs_benchmark,
)
from analytics_pipeline.src.reproducibility import set_global_seed


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_series_returns_high_p_value(self):
        set_global_seed()
        x = np.random.normal(size=200)
        p, observed, ci = paired_bootstrap_p_value(x, x.copy())
        self.assertGreater(p, 0.9)
        self.assertAlmostEqual(observed, 0.0, places=10)
        self.assertAlmostEqual(ci[0], 0.0, places=10)
        self.assertAlmostEqual(ci[1], 0.0, places=10)

    def test_clearly_different_means_returns_low_p_value(self):
        set_global_seed()
        a = np.random.normal(loc=0.01, scale=0.005, size=500)
        b = np.random.normal(loc=0.000, scale=0.005, size=500)
        p, observed, _ = paired_bootstrap_p_value(a, b, n_iterations=2000)
        self.assertLess(p, 0.05)
        self.assertGreater(observed, 0)

    def test_short_series_returns_nan(self):
        a = np.array([0.01, 0.02])
        b = np.array([0.005, 0.015])
        p, _, _ = paired_bootstrap_p_value(a, b, n_iterations=200)
        self.assertTrue(np.isnan(p))


class DieboldMarianoTests(unittest.TestCase):
    def test_identical_losses_no_significance(self):
        set_global_seed()
        l = np.random.exponential(scale=1.0, size=300)
        stat, p = diebold_mariano_test(l, l.copy())
        # Identical loss series produce zero numerator and an
        # asymptotically NaN ratio; the test should signal that gracefully.
        self.assertTrue(np.isnan(p) or p > 0.9)
        self.assertTrue(np.isnan(stat) or abs(stat) < 0.01)

    def test_one_model_clearly_better(self):
        set_global_seed()
        worse = np.random.exponential(scale=1.0, size=400)
        better = worse * 0.5  # half the loss, perfectly correlated
        stat, p = diebold_mariano_test(worse, better)
        # worse has higher loss, so worse - better > 0 and the DM stat
        # should be positive and significant.
        self.assertGreater(stat, 0)
        self.assertLess(p, 0.001)


class SharpeTests(unittest.TestCase):
    def test_probabilistic_sharpe_positive_returns_high_confidence(self):
        # Clearly positive Sharpe should give PSR >> 0.5. Threshold is set
        # conservatively at 0.85 to allow for the skew/kurtosis correction
        # that shifts PSR away from a naive Gaussian estimate.
        set_global_seed()
        r = np.random.normal(loc=0.003, scale=0.02, size=600)
        sharpe, psr = probabilistic_sharpe_ratio(r, periods_per_year=365 * 6)
        self.assertGreater(sharpe, 0)
        self.assertGreater(psr, 0.85)

    def test_probabilistic_sharpe_is_in_valid_range(self):
        # The PSR is a probability and must always lie in [0, 1].
        set_global_seed()
        for loc in (-0.002, 0.0, 0.002):
            r = np.random.normal(loc=loc, scale=0.02, size=600)
            _, psr = probabilistic_sharpe_ratio(r, periods_per_year=365 * 6)
            with self.subTest(loc=loc):
                self.assertGreaterEqual(psr, 0.0)
                self.assertLessEqual(psr, 1.0)

    def test_deflated_sharpe_penalises_multiple_trials(self):
        set_global_seed()
        r = np.random.normal(loc=0.0015, scale=0.02, size=600)
        _, dsr_one = deflated_sharpe_ratio(r, n_trials=1)
        _, dsr_many = deflated_sharpe_ratio(r, n_trials=500)
        # More trials must always reduce the deflated confidence.
        self.assertGreater(dsr_one, dsr_many)


class SummariseTests(unittest.TestCase):
    def test_full_summary_keys_are_stable(self):
        set_global_seed()
        a = np.random.normal(loc=0.002, scale=0.02, size=500)
        b = np.random.normal(loc=0.000, scale=0.02, size=500)
        summary = summarise_strategy_vs_benchmark(a, b, n_trials=10)

        required_keys = {
            "n_observations",
            "mean_strategy_return",
            "mean_benchmark_return",
            "mean_difference",
            "bootstrap_p_value",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "diebold_mariano_stat",
            "diebold_mariano_p_value",
            "annualised_sharpe",
            "probabilistic_sharpe_ratio",
            "deflated_sharpe_ratio",
            "n_trials_for_dsr",
            "evidence_label",
        }
        self.assertEqual(set(summary.keys()), required_keys)


if __name__ == "__main__":
    unittest.main()
