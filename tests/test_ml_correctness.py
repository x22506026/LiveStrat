"""ML correctness tests for the LiveStrat analytics pipeline.

These tests verify the most common failure modes in time-series machine
learning - look-ahead bias, training/test contamination, feature leakage,
and non-chronological splits. They are the first thing a critical examiner
will look for when reviewing a financial-ML project, so they are kept
narrowly focused and fast (no model training, no network calls).

References
----------
Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2014)
    'Pseudo-Mathematics and Financial Charlatanism: The Effects of
    Backtest Overfitting on Out-of-Sample Performance', Notices of the
    American Mathematical Society, 61(5), pp. 458-471.

López de Prado, M. (2018) Advances in Financial Machine Learning.
    Hoboken, NJ: Wiley.

Pineau, J. et al. (2021) 'Improving Reproducibility in Machine Learning
    Research', JMLR 22, 1-20.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = ROOT_DIR / "analytics_pipeline"
for path in (ROOT_DIR, ANALYTICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analytics_pipeline.src.config import MARKET_FEATURE_COLUMNS, FUTURES_FEATURE_COLUMNS
from analytics_pipeline.src.models.evaluate import make_time_based_split
from analytics_pipeline.src.reproducibility import (
    LIVESTRAT_SEED,
    set_global_seed,
    sklearn_random_state,
)


class ChronologicalSplitTests(unittest.TestCase):
    """make_time_based_split must produce a strictly chronological partition."""

    def test_split_preserves_row_order(self):
        x = pd.DataFrame({"feature": range(100)})
        y = pd.Series(range(100))
        x_train, x_test, y_train, y_test = make_time_based_split(x, y, train_ratio=0.7)

        # Train rows must all come before test rows.
        self.assertTrue(x_train["feature"].max() < x_test["feature"].min())
        self.assertTrue(y_train.max() < y_test.min())

    def test_split_ratio_is_respected(self):
        x = pd.DataFrame({"feature": range(100)})
        y = pd.Series(range(100))
        for ratio in (0.5, 0.6, 0.7, 0.8, 0.9):
            with self.subTest(ratio=ratio):
                x_train, x_test, _, _ = make_time_based_split(x, y, train_ratio=ratio)
                self.assertEqual(len(x_train), int(100 * ratio))
                self.assertEqual(len(x_train) + len(x_test), 100)

    def test_split_does_not_shuffle(self):
        # If shuffling were happening, the train set would contain at least one
        # of the last 10 rows. We assert that is never the case across many
        # ratios.
        x = pd.DataFrame({"feature": range(200)})
        y = pd.Series(range(200))
        for ratio in (0.5, 0.6, 0.7, 0.8):
            with self.subTest(ratio=ratio):
                x_train, _, _, _ = make_time_based_split(x, y, train_ratio=ratio)
                self.assertTrue((x_train["feature"] < 200 - 10).any())
                self.assertFalse((x_train["feature"] >= 190).any())


class FeatureCausalityTests(unittest.TestCase):
    """Engineered features must use only information available up to time t."""

    # Substrings that, in column names, indicate forward-looking content.
    # NB. "future_" with underscore catches the label column 'future_close' but
    # does not match the Binance derivatives 'futures_*' columns (different
    # word, different domain).
    FORWARD_LOOKING_SUBSTRINGS = ("future_", "ahead_", "_ahead", "next_period", "_shift_neg", "lead_")

    def test_no_market_feature_is_a_forward_shift(self):
        # build_market_features uses pct_change, rolling means and stds. These
        # are causal. The only forward-shifted column produced by the pipeline
        # is `future_close` / `future_return`, and that lives in the LABEL
        # construction step (build_labels.py), not in MARKET_FEATURE_COLUMNS.
        for column in MARKET_FEATURE_COLUMNS:
            with self.subTest(column=column):
                lowered = column.lower()
                for forbidden in self.FORWARD_LOOKING_SUBSTRINGS:
                    self.assertNotIn(
                        forbidden,
                        lowered,
                        msg=f"Market feature '{column}' contains forward-looking substring '{forbidden}'",
                    )

    def test_no_futures_feature_is_a_forward_shift(self):
        # FUTURES_FEATURE_COLUMNS contains Binance perpetual-futures features
        # (funding rate, open interest, basis, etc). Each column legitimately
        # starts with 'futures_' or contains 'futures'; that is the derivatives
        # market domain, not a forward-looking shift. The forbidden patterns
        # below are tight enough to ignore those names but still catch any
        # accidental forward shift like 'future_close' or 'price_ahead_24h'.
        for column in FUTURES_FEATURE_COLUMNS:
            with self.subTest(column=column):
                lowered = column.lower()
                for forbidden in self.FORWARD_LOOKING_SUBSTRINGS:
                    self.assertNotIn(
                        forbidden,
                        lowered,
                        msg=f"Futures feature '{column}' contains forward-looking substring '{forbidden}'",
                    )


class LabelLeakageTests(unittest.TestCase):
    """Labels are built by forward-shifting close prices. After construction
    the last horizon rows must be dropped so that no row has a label derived
    from data beyond the dataset edge."""

    def test_forward_shift_leaves_no_nan_labels(self):
        # Simulate the label construction pattern from build_labels.py.
        closes = pd.Series(np.linspace(100, 110, 50))
        horizon = 6
        future_close = closes.shift(-horizon)
        future_return = (future_close / closes) - 1.0

        # The last `horizon` rows must be NaN before they are dropped.
        self.assertEqual(future_return.isna().sum(), horizon)

        cleaned = future_return.dropna()
        self.assertEqual(len(cleaned), len(closes) - horizon)
        self.assertFalse(cleaned.isna().any())

    def test_horizon_never_negative(self):
        # The horizon shift in build_labels uses .shift(-horizon_steps). A
        # positive horizon means "look forward N bars". A zero or negative
        # horizon would either return the present (no signal to learn) or the
        # past (leak the label into features). We assert any horizon used is
        # strictly positive.
        from analytics_pipeline.src.config import LABEL_HORIZON_STEPS

        self.assertGreater(LABEL_HORIZON_STEPS, 0)


class CalibrationIsolationTests(unittest.TestCase):
    """If a calibration set is used to tune thresholds, it must not overlap
    with the held-out test set."""

    def test_four_way_split_is_chronological_and_disjoint(self):
        # The market+futures binary walk-forward uses a four-way split:
        # train / calibration / validation / test. The helper returns four
        # frames whose internal indices are reset, so we attach an explicit
        # row marker before calling it and then verify the markers form a
        # disjoint, strictly increasing partition - the only configuration
        # that prevents look-ahead in walk-forward backtests.
        from analytics_pipeline.src.models.evaluate_market_futures_binary_backtests import (
            split_train_calibration_validation,
        )

        n_rows = 200
        train_df = pd.DataFrame({"row_marker": range(n_rows)})
        fit_df, calibration_df, validation_df, test_df = split_train_calibration_validation(
            train_df
        )

        fit_markers = list(fit_df["row_marker"])
        cal_markers = list(calibration_df["row_marker"])
        val_markers = list(validation_df["row_marker"])
        test_markers = list(test_df["row_marker"])

        # Each partition holds a contiguous block of original rows...
        self.assertEqual(fit_markers, list(range(min(fit_markers), max(fit_markers) + 1)))
        self.assertEqual(cal_markers, list(range(min(cal_markers), max(cal_markers) + 1)))
        self.assertEqual(val_markers, list(range(min(val_markers), max(val_markers) + 1)))
        self.assertEqual(test_markers, list(range(min(test_markers), max(test_markers) + 1)))

        # ...and the four blocks are strictly chronological in order
        # fit < calibration < validation < test (no overlap, no jumbling).
        self.assertLess(max(fit_markers), min(cal_markers))
        self.assertLess(max(cal_markers), min(val_markers))
        self.assertLess(max(val_markers), min(test_markers))

        # And their union is exactly the original frame.
        union = set(fit_markers) | set(cal_markers) | set(val_markers) | set(test_markers)
        self.assertEqual(union, set(range(n_rows)))


class ReproducibilityTests(unittest.TestCase):
    """Seeding must produce identical numpy and Python random draws."""

    def test_set_global_seed_is_deterministic(self):
        set_global_seed()
        first_np = np.random.rand(10)

        set_global_seed()
        second_np = np.random.rand(10)

        np.testing.assert_array_equal(first_np, second_np)

    def test_seed_value_is_explicit(self):
        # The chosen seed should be a fixed integer documented in the
        # reproducibility module. This catches accidental drift.
        self.assertEqual(LIVESTRAT_SEED, 1729)

    def test_sklearn_random_state_is_seed_compatible(self):
        # sklearn_random_state must return an int that scikit-learn can
        # pass to a model constructor's random_state argument.
        rs = sklearn_random_state()
        self.assertIsInstance(rs, int)
        # And an override must take effect.
        self.assertEqual(sklearn_random_state(42), 42)


if __name__ == "__main__":
    unittest.main()
