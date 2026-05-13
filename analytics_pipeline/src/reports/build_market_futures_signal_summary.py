"""Build a user-facing summary from the preferred market + futures model results."""

import pandas as pd

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    get_market_futures_dataset_path,
    get_market_futures_preferred_model_summary_path,
    get_market_futures_signal_summary_path,
)
from src.io_paths import ensure_dirs


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE


def rank_model_rows(df):
    """Rank model rows so the best-performing backend recommendation wins per symbol."""
    ranked = df.copy()
    ranked["macro_f1_rank"] = ranked["test_macro_f1"].rank(method="dense", ascending=False)
    ranked["balanced_accuracy_rank"] = ranked["test_balanced_accuracy"].rank(method="dense", ascending=False)
    ranked["accuracy_rank"] = ranked["test_accuracy"].rank(method="dense", ascending=False)
    ranked = ranked.sort_values(
        ["macro_f1_rank", "balanced_accuracy_rank", "accuracy_rank", "latest_signal_confidence"],
        ascending=[True, True, True, False],
    )
    return ranked


def build_market_futures_signal_summary(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Create one best-current backend summary row per tracked symbol."""
    ensure_dirs()
    preferred_summary_path = get_market_futures_preferred_model_summary_path(
        timeframe,
        start_date,
        end_date,
    )
    summary_df = pd.read_csv(preferred_summary_path)

    rows = []
    for symbol, symbol_df in summary_df.groupby("symbol"):
        ranked = rank_model_rows(symbol_df)
        best = ranked.iloc[0]
        dataset_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
        latest_dataset_row = {}
        if dataset_path.exists():
            dataset_df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
            if not dataset_df.empty:
                latest_dataset_row = dataset_df.sort_values("open_time").iloc[-1].to_dict()
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "window_start": start_date,
                "window_end": end_date,
                "selected_backend_model": best["model_name"],
                "selected_target_name": best["target_name"],
                "latest_signal": best["latest_signal"],
                "latest_signal_confidence": best["latest_signal_confidence"],
                "test_accuracy": best["test_accuracy"],
                "test_macro_f1": best["test_macro_f1"],
                "test_balanced_accuracy": best["test_balanced_accuracy"],
                "futures_feature_completeness_score": latest_dataset_row.get("futures_feature_completeness_score"),
                "futures_completeness_label": latest_dataset_row.get("futures_completeness_label"),
                "futures_context_resilience_score": latest_dataset_row.get("futures_context_resilience_score"),
                "futures_context_resilience_label": latest_dataset_row.get("futures_context_resilience_label"),
                "futures_basis_reliance_score": latest_dataset_row.get("futures_basis_reliance_score"),
                "basis_feature_mode": latest_dataset_row.get("basis_feature_mode", "unavailable"),
                "basis_proxy_active": latest_dataset_row.get("basis_proxy_active", False),
                "effective_basis_feature_available": latest_dataset_row.get("effective_basis_feature_available", False),
                "funding_feature_available": latest_dataset_row.get("funding_feature_available"),
                "open_interest_feature_available": latest_dataset_row.get("open_interest_feature_available"),
                "positioning_feature_available": latest_dataset_row.get("positioning_feature_available"),
                "taker_flow_feature_available": latest_dataset_row.get("taker_flow_feature_available"),
                "basis_feature_available": latest_dataset_row.get("basis_feature_available"),
                "backend_summary": (
                    f"{symbol} currently uses {best['model_name']} with target {best['target_name']}. "
                    f"The latest signal is {best['latest_signal']} with confidence "
                    f"{float(best['latest_signal_confidence']) * 100:.1f}%. "
                    f"Basis mode is {str(latest_dataset_row.get('basis_feature_mode', 'unavailable')).replace('_', ' ')}."
                ),
            }
        )

    output_df = pd.DataFrame(rows)
    output_path = get_market_futures_signal_summary_path(timeframe, start_date, end_date)
    output_df.to_csv(output_path, index=False)

    print("market + futures signal summary generated")
    print(f"rows saved: {len(output_df)}")
    print(f"summary saved to: {output_path}")
    return output_df


if __name__ == "__main__":
    build_market_futures_signal_summary()
