"""Compare news-event, cautious multimodal, and on-chain structural overlays."""

from pathlib import Path

import pandas as pd

from src.config import (
    PROCESSED_DIR,
    get_all_symbols,
    get_evaluation_metrics_path,
    get_supported_onchain_assets,
    get_strategy_summary_path,
)


INTRADAY_WINDOWS = {
    "4h": ("2026-04-01", "2026-04-30"),
    "1h": ("2026-04-16", "2026-04-30"),
}
CORE_SYMBOLS = get_all_symbols()
DAILY_ASSETS = get_supported_onchain_assets()


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_metrics(symbol, timeframe, model_name, start_date, end_date):
    path = get_evaluation_metrics_path(symbol, timeframe, model_name, start_date, end_date)
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return {
        str(row["metric"]): _safe_float(row["value"])
        for _, row in df.iterrows()
    }


def _load_best_daily_overlay_map():
    path = get_strategy_summary_path("onchain_structural_overlay", "1d")
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    grouped = {}
    for asset_symbol, asset_df in df.groupby("asset_symbol"):
        best_row = asset_df.sort_values(
            ["test_macro_f1", "test_balanced_accuracy", "test_accuracy"],
            ascending=[False, False, False],
        ).iloc[0]
        grouped[str(asset_symbol)] = best_row.to_dict()
    return grouped


def _classify_context_role(row):
    news_macro = _safe_float(row.get("news_event_macro_f1"))
    multimodal_macro = _safe_float(row.get("multimodal_veto_macro_f1"))
    daily_macro = _safe_float(row.get("daily_onchain_macro_f1"))
    news_risk = _safe_float(row.get("latest_news_event_risk_score"))
    onchain_risk = _safe_float(row.get("daily_onchain_risk_score"))
    onchain_support = _safe_float(row.get("daily_onchain_support_score"))
    onchain_confidence = _safe_float(row.get("daily_onchain_confidence"))
    news_mode = str(row.get("latest_news_event_mode", "quiet") or "quiet")

    if (
        daily_macro >= max(news_macro, multimodal_macro)
        and (onchain_support > 0.20 or onchain_risk > 0.20)
        and onchain_confidence >= 0.35
    ):
        return "onchain_structural_confirmation_lead"
    if news_risk >= 0.35 and news_mode in {"risk_event", "watch_event"} and news_macro >= multimodal_macro - 0.03:
        return "news_event_veto_lead"
    if multimodal_macro >= news_macro and multimodal_macro >= 0.30:
        return "cautious_multimodal_overlay"
    return "market_futures_core_keep_context_secondary"


def compare_context_overlays():
    """Create a compact comparison summary for context-overlay families."""
    daily_overlay_map = _load_best_daily_overlay_map()
    summary_rows = []

    for timeframe, (start_date, end_date) in INTRADAY_WINDOWS.items():
        multimodal_path = PROCESSED_DIR / f"market_multimodal_strategy_summary_{timeframe}_{start_date}_{end_date}.csv"
        if not multimodal_path.exists():
            continue

        multimodal_df = pd.read_csv(multimodal_path)
        multimodal_map = {
            str(row["symbol"]): row.to_dict()
            for _, row in multimodal_df.iterrows()
        }

        for symbol in CORE_SYMBOLS:
            news_metrics = _load_metrics(symbol, timeframe, "market_multimodal_news_event_veto", start_date, end_date)
            cautious_metrics = _load_metrics(symbol, timeframe, "market_multimodal_context_veto", start_date, end_date)
            symbol_summary = multimodal_map.get(symbol, {})
            asset_symbol = symbol.replace("USDT", "")
            daily_overlay = daily_overlay_map.get(asset_symbol, {})

            latest_news_mode = str(symbol_summary.get("latest_news_event_regime_label", "quiet") or "quiet")
            latest_news_risk = _safe_float(symbol_summary.get("latest_news_event_risk_score"))
            daily_overlay_mode = str(daily_overlay.get("latest_overlay_mode", "structural_mixed") or "structural_mixed")
            daily_overlay_risk = _safe_float(daily_overlay.get("latest_overlay_risk_score"))
            daily_overlay_support = _safe_float(daily_overlay.get("latest_overlay_support_score"))
            daily_overlay_confidence = _safe_float(daily_overlay.get("latest_overlay_confidence"))

            row = {
                "symbol": symbol,
                "asset_symbol": asset_symbol,
                "timeframe": timeframe,
                "window_start": start_date,
                "window_end": end_date,
                "news_event_macro_f1": _safe_float(news_metrics.get("macro_f1")),
                "news_event_balanced_accuracy": _safe_float(news_metrics.get("balanced_accuracy")),
                "multimodal_veto_macro_f1": _safe_float(cautious_metrics.get("macro_f1")),
                "multimodal_veto_balanced_accuracy": _safe_float(cautious_metrics.get("balanced_accuracy")),
                "daily_onchain_overlay": daily_overlay.get("strategy_name", "unavailable"),
                "daily_onchain_macro_f1": _safe_float(daily_overlay.get("test_macro_f1")),
                "daily_onchain_balanced_accuracy": _safe_float(daily_overlay.get("test_balanced_accuracy")),
                "latest_news_event_mode": latest_news_mode,
                "latest_news_event_risk_score": latest_news_risk,
                "latest_onchain_overlay_mode": daily_overlay_mode,
                "daily_onchain_risk_score": daily_overlay_risk,
                "daily_onchain_support_score": daily_overlay_support,
                "daily_onchain_confidence": daily_overlay_confidence,
            }
            row["context_overlay_lead"] = _classify_context_role(row)
            row["context_overlay_summary"] = (
                f"{symbol} on {timeframe} currently shows {latest_news_mode.replace('_', ' ')} news "
                f"(risk {latest_news_risk:.2f}) and {daily_overlay_mode.replace('_', ' ')} on-chain overlay "
                f"(risk {daily_overlay_risk:.2f}, support {daily_overlay_support:.2f}, confidence {daily_overlay_confidence:.2f}). "
                f"Recommended context lead is {row['context_overlay_lead'].replace('_', ' ')}."
            )
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    output_path = get_strategy_summary_path("context_overlay_comparison", "1d")
    summary_df.to_csv(output_path, index=False)

    print("context overlay comparison summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {output_path}")
    return summary_df


if __name__ == "__main__":
    compare_context_overlays()
