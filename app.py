"""Flask entry point for LiveStrat.

This file holds the web routes. It reads the CSV outputs that the
analytics pipeline wrote into analytics_pipeline/data/processed/ and
turns them into the five pages of the app plus the JSON endpoints that
the page JavaScript uses.

If a page looks slow or out of date the fix is almost always somewhere
in the pipeline, not in this file.
"""

import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session, url_for

# Make the analytics package importable without installing it.
PROJECT_DIR = Path(__file__).resolve().parent
ANALYTICS_DIR = PROJECT_DIR / "analytics_pipeline"
if str(ANALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYTICS_DIR))

from analytics_pipeline.src.models.alert_engine import build_alert_response
from analytics_pipeline.src.models.strategy_registry import (
    build_asset_capability_state,
    build_capability_notes,
    build_timeframe_strategy_policy,
    get_preset_strategy,
    get_strategy_registry,
    resolve_custom_strategy_config,
    resolve_preset_strategy_config,
)
from analytics_pipeline.src.models.system_blueprint import build_system_blueprint
from analytics_pipeline.src.models.runtime_support import build_runtime_support_snapshot
from analytics_pipeline.src.models.strategy_governance import (
    build_context_layer_assessment,
    build_family_governance,
    build_multimodal_assessment,
    build_strategy_governance,
)
from analytics_pipeline.src.models.family_governance_matrix import build_family_governance_matrix
from analytics_pipeline.src.models.data_strategy_health import build_data_strategy_health_snapshot
from analytics_pipeline.src.models.evaluation_coverage import build_evaluation_coverage_snapshot
from analytics_pipeline.src.models.market_futures_targets import describe_target_for_timeframe
from analytics_pipeline.src.models.pipeline_refresh import build_pipeline_refresh_snapshot
from analytics_pipeline.src.models.pipeline_refresh import build_pipeline_refresh_guidance
from analytics_pipeline.src.models.strategy_family_evidence import build_strategy_family_evidence_snapshot
from analytics_pipeline.src.models.strategy_family_scope import build_strategy_family_scope_snapshot
from analytics_pipeline.src.models.source_governance import build_context_source_governance_snapshot
from analytics_pipeline.src.models.timeframe_readiness import build_timeframe_readiness_snapshot
from analytics_pipeline.src.reports.build_backend_readiness_report import build_backend_readiness_report
from analytics_pipeline.src.config import ASSET_REGISTRY, get_all_symbols
from account_services import (
    alert_preferences_payload,
    dispatch_telegram_ready_messages,
    get_current_user,
    get_demo_user,
    get_or_create_alert_preferences,
    normalize_email,
    normalize_username,
    persist_alert_events_for_user,
    serialize_notification_event,
    serialize_strategy_profile,
    telegram_status_payload,
)
from database import db
from models import AlertPreference, NotificationEvent, SavedStrategyProfile, User

load_dotenv()

# Flask app and database wiring. The connection string falls back to
# the local docker compose Postgres when DATABASE_URL is not set.
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "livestrat-development-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://livestrat:livestrat_password@localhost:5432/livestrat",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
db.init_app(app)

PROCESSED_DIR = PROJECT_DIR / "analytics_pipeline" / "data" / "processed"

# in-process TTL cache for the two slowest endpoints so the page feels instant
import time as _time
import threading as _threading
_PAYLOAD_CACHE = {}
_PAYLOAD_CACHE_LOCK = _threading.Lock()
_PAYLOAD_CACHE_TTL_SECONDS = 300

def cached_payload(cache_key, build_fn, ttl=_PAYLOAD_CACHE_TTL_SECONDS):
    now = _time.time()
    with _PAYLOAD_CACHE_LOCK:
        entry = _PAYLOAD_CACHE.get(cache_key)
        if entry and (now - entry[0]) < ttl:
            return entry[1]
    payload = build_fn()
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_CACHE[cache_key] = (now, payload)
    return payload

RAW_BINANCE_DIR = PROJECT_DIR / "analytics_pipeline" / "data" / "raw" / "binance"
SENTIMENT_SUMMARY_PATH = PROCESSED_DIR / "sentiment_summary_1d.csv"
SUPPORTED_MARKET_TIMEFRAMES = ("1h", "4h", "1d")

NUMERIC_SUMMARY_FIELDS = {
    "latest_close",
    "predicted_return",
    "predicted_price",
    "forecast_directional_accuracy",
    "forecast_rmse",
    "classification_macro_f1",
    "futures_confidence",
    "indicator_score",
    "trend_score",
    "momentum_score",
    "participation_score",
    "volatility_penalty",
    "context_adjustment",
    "final_strategy_score",
    "latest_return_4h_pct",
    "latest_return_24h_pct",
    "latest_return_3d_pct",
    "latest_volatility_20",
    "latest_volume_zscore",
    "latest_high_low_range_pct",
    "latest_taker_buy_ratio",
    "scaled_model_confidence",
    "rule_based_test_accuracy",
    "baseline_unscaled_test_accuracy",
    "baseline_scaled_test_accuracy",
    "top_feature_importance",
    "latest_signal_confidence",
    "test_accuracy",
    "test_macro_f1",
    "test_balanced_accuracy",
    "selected_primary_confidence",
    "strategy_total_return",
    "buy_hold_total_return",
    "excess_return",
    "annualized_strategy_return",
    "annualized_strategy_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "exposure_ratio",
    "trade_count",
    "hit_rate",
    "buy_threshold",
    "exit_threshold",
    "latest_position",
    "calibration_temperature",
    "multimodal_latest_signal_confidence",
    "multimodal_selected_context_feature_count",
    "multimodal_validation_macro_f1",
    "multimodal_validation_balanced_accuracy",
    "multimodal_test_accuracy",
    "multimodal_test_macro_f1",
    "multimodal_test_balanced_accuracy",
    "latest_sentiment_value",
    "latest_gdelt_sentiment_mean",
    "latest_gdelt_article_count",
    "latest_gdelt_coverage_quality_score",
    "gdelt_age_days",
    "latest_onchain_snapshot_score",
    "latest_gdelt_snapshot_age_days",
    "latest_onchain_snapshot_age_days",
    "latest_defi_tvl_usd",
    "defi_tvl_change_pct_1d",
    "defi_tvl_change_pct_7d",
    "defi_tvl_change_pct_30d",
    "defi_tvl_zscore_30d",
    "defi_tvl_drawdown_30d",
    "defi_regime_score",
    "defi_snapshot_age_days",
    "futures_feature_completeness_score",
    "futures_context_resilience_score",
    "futures_basis_reliance_score",
    "gdelt_reliability_score",
    "broad_sentiment_reliability_score",
    "effective_sentiment_reliability_score",
    "onchain_reliability_score",
    "feature_count",
    "walkforward_fold_count",
    "walkforward_avg_accuracy",
    "walkforward_avg_macro_f1",
    "walkforward_avg_balanced_accuracy",
    "walkforward_avg_strategy_total_return",
    "walkforward_avg_buy_hold_return",
    "walkforward_avg_excess_return",
    "walkforward_avg_sharpe",
    "walkforward_avg_max_drawdown",
    "walkforward_deployment_activity_rate",
    "ablation_latest_signal_confidence",
    "ablation_best_accuracy",
    "ablation_best_macro_f1",
    "ablation_best_balanced_accuracy",
    "ablation_market_futures_macro_f1",
    "ablation_market_futures_balanced_accuracy",
    "delta_macro_f1_vs_market_futures",
    "delta_balanced_accuracy_vs_market_futures",
}

USER_STRATEGY_LABELS = {
    "recommended": "Balanced Default",
    "market_futures_backend": "Balanced Default",
    "conservative_trend": "Trend Confirmation",
    "momentum_breakout": "Momentum Breakout",
    "futures_crowd_reversal": "Crowd Reversal",
    "multimodal_balanced": "Context-Aware Balanced",
    "daily_structural_confirmation": "Structural Confirmation",
    "rule_based": "Transparent Rule Benchmark",
    "scaled_model": "Scaled Market Baseline",
    "unscaled_model": "Unscaled Market Baseline",
    "enhanced_market_futures": "Balanced Default",
    "core_market_rule_based": "Transparent Rule Benchmark",
    "core_market_model": "Market Baseline",
    "market_futures_core": "Market + Futures Core",
}


def format_user_label(value):
    if value in (None, ""):
        return "n/a"
    text = str(value).replace("_", " ").replace("-", " ")
    parts = [part for part in text.split() if part]
    return " ".join(part.capitalize() if not part.isupper() else part for part in parts)


def resolve_strategy_display_name(strategy_key, raw_model_name="", raw_family_name=""):
    raw_model_name = str(raw_model_name or "")
    raw_family_name = str(raw_family_name or "")

    if strategy_key in USER_STRATEGY_LABELS:
        return USER_STRATEGY_LABELS[strategy_key]
    if raw_family_name in USER_STRATEGY_LABELS:
        return USER_STRATEGY_LABELS[raw_family_name]

    normalized = raw_model_name.lower()
    if "multimodal" in normalized:
        return "Context-Aware Balanced"
    if "onchain" in normalized:
        return "On-Chain Specialist"
    if "rule" in normalized:
        return "Transparent Rule Benchmark"
    if "scaled_market_baseline" in normalized:
        return "Scaled Market Baseline"
    if "unscaled_market_baseline" in normalized:
        return "Unscaled Market Baseline"
    if "market_futures" in normalized and "preferred" in normalized:
        return "Balanced Default"
    if "market_futures" in normalized:
        return "Market + Futures Core"
    if "logistic" in normalized and "market" in normalized:
        return "Market Core Logistic"
    return format_user_label(raw_model_name or raw_family_name or strategy_key)


def resolve_engine_label(raw_model_name="", strategy_key="", raw_family_name=""):
    raw_model_name = str(raw_model_name or "")
    normalized = raw_model_name.lower()

    if strategy_key == "rule_based" or "rule" in normalized:
        return "Transparent rules"
    if strategy_key == "scaled_model" or normalized == "scaled_market_baseline":
        return "Scaled market baseline"
    if strategy_key == "unscaled_model" or normalized == "unscaled_market_baseline":
        return "Unscaled market baseline"
    if "multimodal" in normalized:
        return "Context-aware engine"
    if "onchain" in normalized:
        return "On-chain specialist"
    if "market_futures" in normalized:
        return "Market + futures engine"
    if "market" in normalized:
        return "Market engine"
    return resolve_strategy_display_name(strategy_key, raw_model_name, raw_family_name)


def resolve_strategy_family_label(raw_family_name="", strategy_key="", raw_model_name=""):
    raw_family_name = str(raw_family_name or "")
    if raw_family_name in USER_STRATEGY_LABELS:
        return USER_STRATEGY_LABELS[raw_family_name]
    return resolve_strategy_display_name(strategy_key, raw_model_name, raw_family_name)


def get_latest_processed_file(pattern):
    # pick the newest processed artifact for a given filename pattern
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def get_latest_raw_binance_file(symbol, timeframe):
    matches = list(RAW_BINANCE_DIR.glob(f"{symbol}_{timeframe}_*.csv"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def safe_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or value is pd.NA or str(value).strip() == "":
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        result = float(value)
        # NaN and inf are not valid JSON so collapse them to the default
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def clamp(value, lower=-1.0, upper=1.0):
    return max(lower, min(upper, value))


def signal_from_score(score, buy_threshold=0.25, sell_threshold=-0.25):
    if score >= buy_threshold:
        return "buy"
    if score <= sell_threshold:
        return "dont_buy"
    return "hold"


def extract_timeframe_from_name(filename):
    for timeframe in SUPPORTED_MARKET_TIMEFRAMES:
        if f"_{timeframe}_" in filename:
            return timeframe
    return None


def build_target_semantics(target_name, timeframe):
    if not target_name or target_name == "n/a":
        return {
            "target_name": target_name or "n/a",
            "requested_timeframe": timeframe,
            "requested_horizon_hours": None,
            "effective_horizon_hours": None,
            "horizon_steps": None,
            "exact_horizon_match": False,
            "horizon_resolution_note": "No timeframe-aware target semantics are available for this strategy.",
        }
    return describe_target_for_timeframe(target_name, timeframe)


def load_market_summaries_from_path(path):
    # load a generated summary table into a symbol keyed dict
    summaries = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if key in NUMERIC_SUMMARY_FIELDS and value not in (None, ""):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = value

            summaries[cleaned["symbol"]] = cleaned

    return summaries


def load_market_summaries():
    bundle = load_market_summaries_bundle()
    return bundle["summaries"]


def _asset_display_name(symbol):
    for asset_config in ASSET_REGISTRY.values():
        if asset_config.get("market_symbol") == symbol:
            return asset_config.get("display_name") or symbol
    return symbol


def ensure_supported_asset_summaries(summaries, timeframe=None):
    """Keep the UI universe aligned with the seven configured market assets."""
    selected_timeframe = timeframe or "4h"
    for symbol in get_all_symbols():
        summaries.setdefault(
            symbol,
            {
                "symbol": symbol,
                "asset_name": _asset_display_name(symbol),
                "timeframe": selected_timeframe,
                "current_pipeline_mode": "pending_refresh",
                "selected_primary_model": "pending_refresh",
                "selected_primary_signal": "n/a",
                "selected_primary_confidence": 0,
                "latest_signal": "n/a",
                "latest_signal_confidence": 0,
                "latest_effective_sentiment_source": "unavailable",
                "latest_effective_sentiment_label": "unavailable",
                "latest_onchain_regime_label": "unavailable",
                "primary_summary": (
                    f"{symbol} is part of the LiveStrat seven-asset universe, but this timeframe needs "
                    "a refreshed app-facing summary before stronger claims should be shown."
                ),
            },
        )
    return summaries


def _load_per_asset_summary(symbol, timeframe):
    # latest per-asset market summary CSV; one row, return as dict
    pattern = f"{symbol}_{timeframe}_market_summary_*.csv"
    path = get_latest_processed_file(pattern)
    if path is None:
        # also try undated file
        legacy = PROCESSED_DIR / f"{symbol}_{timeframe}_market_summary.csv"
        if legacy.exists():
            path = legacy
        else:
            return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return {k: v for k, v in row.items() if v is not None and not (isinstance(v, float) and pd.isna(v))}


def fill_market_behaviour_fields(summaries, timeframe):
    # the consolidated overview leaves volatility_status / rule_signal blank for some
    # assets and never writes activity_status. fill from per-asset summary csvs.
    if not timeframe:
        timeframe = "4h"
    for symbol, summary in summaries.items():
        per_asset = _load_per_asset_summary(symbol, timeframe)
        for key in ("volatility_status", "rule_signal", "latest_volume_zscore", "latest_volatility_20"):
            current = summary.get(key)
            is_missing = current in (None, "", "n/a", "nan") or (isinstance(current, float) and pd.isna(current))
            if is_missing and key in per_asset:
                summary[key] = per_asset[key]
        if summary.get("activity_status") in (None, "", "n/a", "nan") or (
            isinstance(summary.get("activity_status"), float) and pd.isna(summary.get("activity_status"))
        ):
            summary["activity_status"] = classify_activity_status(summary.get("latest_volume_zscore"))


def get_merged_market_summaries(requested_timeframe=None):
    summary_bundle = load_market_summaries_bundle(requested_timeframe)
    summaries = summary_bundle["summaries"]
    market_trend_summaries = load_market_trend_benchmark_summaries(summary_bundle["timeframe"] or requested_timeframe)
    for summary_symbol, trend_summary in market_trend_summaries.items():
        summaries.setdefault(summary_symbol, {})
        for key, value in trend_summary.items():
            if key not in summaries[summary_symbol] or summaries[summary_symbol].get(key) in (None, "", "n/a"):
                summaries[summary_symbol][key] = value
    user_strategy_summaries = load_user_strategy_signal_summaries(summary_bundle["timeframe"] or requested_timeframe)
    for summary_symbol, strategy_summary in user_strategy_summaries.items():
        summaries.setdefault(summary_symbol, {})
        summaries[summary_symbol].update(strategy_summary)
    if summaries and not any("current_pipeline_mode" in summary for summary in summaries.values()):
        futures_summaries = load_market_futures_signal_summaries(summary_bundle["timeframe"])
        for summary_symbol, futures_summary in futures_summaries.items():
            summaries.setdefault(summary_symbol, {})
            summaries[summary_symbol].update(futures_summary)
    comparison_summaries = load_strategy_backbone_comparison_summaries(summary_bundle["timeframe"])
    for summary_symbol, comparison_summary in comparison_summaries.items():
        summaries.setdefault(summary_symbol, {})
        summaries[summary_symbol].update(comparison_summary)
    structural_break_summaries = load_structural_break_summaries(summary_bundle["timeframe"])
    for summary_symbol, break_summary in structural_break_summaries.items():
        summaries.setdefault(summary_symbol, {})
        summaries[summary_symbol].update(break_summary)
    cross_asset_summaries = load_cross_asset_relative_strength_summaries(
        summary_bundle["timeframe"] or requested_timeframe
    )
    for summary_symbol, ranking_summary in cross_asset_summaries.items():
        summaries.setdefault(summary_symbol, {})
        summaries[summary_symbol].update(ranking_summary)
    context_reliability_summaries = load_context_reliability_summaries()
    for summary_symbol, context_summary in context_reliability_summaries.items():
        summaries.setdefault(summary_symbol, {})
        summaries[summary_symbol].update(context_summary)
    daily_structural_summaries = load_daily_structural_confirmation_summaries()
    for summary_symbol, daily_summary in daily_structural_summaries.items():
        summaries.setdefault(summary_symbol, {})
        summaries[summary_symbol].update(daily_summary)
    context_overlay_summaries = load_context_overlay_comparison_summaries(
        summary_bundle["timeframe"] or requested_timeframe
    )
    for summary_symbol, overlay_summary in context_overlay_summaries.items():
        summaries.setdefault(summary_symbol, {})
        summaries[summary_symbol].update(overlay_summary)
    ensure_supported_asset_summaries(summaries, summary_bundle["timeframe"] or requested_timeframe)
    fill_market_behaviour_fields(summaries, summary_bundle["timeframe"] or requested_timeframe)
    return summaries, summary_bundle


def load_context_reliability_summaries():
    path = get_latest_processed_file("context_reliability_strategy_summary_1d.csv")
    if path is None or not path.exists():
        return {}

    summaries = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if key in NUMERIC_SUMMARY_FIELDS and value not in (None, ""):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = value

            symbol = cleaned.get("market_symbol")
            if symbol:
                summaries[symbol] = cleaned
    return summaries


def load_context_overlay_comparison_summaries(requested_timeframe=None):
    path = get_latest_processed_file("context_overlay_comparison_strategy_summary_1d.csv")
    if path is None or not path.exists():
        return {}

    summaries = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = str(row.get("symbol", "") or "").strip().upper()
            timeframe = str(row.get("timeframe", "") or "").strip()
            if not symbol:
                continue
            if requested_timeframe and timeframe and timeframe != requested_timeframe:
                continue

            cleaned = {}
            for key, value in row.items():
                if value in (None, ""):
                    cleaned[key] = value
                    continue
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    cleaned[key] = value
            summaries[symbol] = cleaned
    return summaries


def load_daily_structural_confirmation_summaries():
    path = get_latest_processed_file("daily_structural_confirmation_strategy_summary_1d.csv")
    if path is None or not path.exists():
        return {}

    summaries = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = str(row.get("market_symbol", "") or "").strip().upper()
            if not symbol:
                continue

            cleaned = {}
            for key, value in row.items():
                if value in (None, ""):
                    cleaned[key] = value
                    continue
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    cleaned[key] = value
            summaries[symbol] = cleaned
    return summaries


def load_market_summaries_bundle(requested_timeframe=None):
    # prefer the unified market intelligence overview, then fall back to older market overviews
    patterns = []
    if requested_timeframe in SUPPORTED_MARKET_TIMEFRAMES:
        patterns.extend(
            [
                f"market_intelligence_overview_{requested_timeframe}_*.csv",
                f"market_overview_{requested_timeframe}_*.csv",
            ]
        )
    patterns.extend(
        [
            "market_intelligence_overview_*.csv",
            "market_overview_*.csv",
        ]
    )

    for pattern in patterns:
        candidate = get_latest_processed_file(pattern)
        if candidate is not None and candidate.exists():
            return {
                "summaries": load_market_summaries_from_path(candidate),
                "path": candidate,
                "timeframe": extract_timeframe_from_name(candidate.name),
            }

    legacy_path = PROCESSED_DIR / "market_overview_4h.csv"
    if legacy_path.exists():
        return {
            "summaries": load_market_summaries_from_path(legacy_path),
            "path": legacy_path,
            "timeframe": "4h",
        }

    return {
        "summaries": {},
        "path": None,
        "timeframe": requested_timeframe or "4h",
    }


def load_market_futures_signal_summaries(requested_timeframe=None):
    # load the latest market + futures backend selections if they exist
    patterns = []
    if requested_timeframe in SUPPORTED_MARKET_TIMEFRAMES:
        patterns.append(f"market_futures_signal_summary_{requested_timeframe}_*.csv")
    patterns.append("market_futures_signal_summary_*.csv")

    signal_summary_path = None
    for pattern in patterns:
        signal_summary_path = get_latest_processed_file(pattern)
        if signal_summary_path is not None and signal_summary_path.exists():
            break

    if signal_summary_path is None or not signal_summary_path.exists():
        return {}

    summaries = {}
    with open(signal_summary_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if key in NUMERIC_SUMMARY_FIELDS and value not in (None, ""):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = value
            summaries[cleaned["symbol"]] = cleaned

    return summaries


def load_user_strategy_signal_summaries(requested_timeframe=None):
    patterns = []
    if requested_timeframe in SUPPORTED_MARKET_TIMEFRAMES:
        patterns.append(f"user_strategy_signal_summary_{requested_timeframe}_*.csv")
    patterns.append("user_strategy_signal_summary_*.csv")

    summary_path = None
    for pattern in patterns:
        summary_path = get_latest_processed_file(pattern)
        if summary_path is not None and summary_path.exists():
            break

    if summary_path is None or not summary_path.exists():
        return {}

    summaries = {}
    with open(summary_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if key in NUMERIC_SUMMARY_FIELDS and value not in (None, ""):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = value
            symbol = cleaned.get("symbol")
            if symbol:
                summaries[symbol] = cleaned

    return summaries


def load_strategy_backbone_comparison_summaries(requested_timeframe=None):
    # load the latest benchmark-vs-backbone comparison rows if they exist
    patterns = []
    if requested_timeframe in SUPPORTED_MARKET_TIMEFRAMES:
        patterns.append(f"strategy_backbone_comparison_{requested_timeframe}_*.csv")
    patterns.append("strategy_backbone_comparison_*.csv")

    comparison_path = None
    for pattern in patterns:
        comparison_path = get_latest_processed_file(pattern)
        if comparison_path is not None and comparison_path.exists():
            break

    if comparison_path is None or not comparison_path.exists():
        return {}

    summaries = {}
    with open(comparison_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if value in (None, ""):
                    cleaned[key] = value
                    continue
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    cleaned[key] = value
            summaries[cleaned["symbol"]] = cleaned

    return summaries


def load_market_trend_benchmark_summaries(requested_timeframe=None):
    patterns = []
    if requested_timeframe in SUPPORTED_MARKET_TIMEFRAMES:
        patterns.append(f"market_trend_forecast_summary_{requested_timeframe}_*.csv")
    patterns.append("market_trend_forecast_summary_*.csv")

    summary_path = None
    for pattern in patterns:
        summary_path = get_latest_processed_file(pattern)
        if summary_path is not None and summary_path.exists():
            break

    if summary_path is None or not summary_path.exists():
        return {}

    rows = []
    with open(summary_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if value in (None, ""):
                    cleaned[key] = value
                    continue
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    cleaned[key] = value
            rows.append(cleaned)

    best_by_symbol = {}
    for row in sorted(rows, key=lambda item: (item.get("macro_f1", 0), item.get("accuracy", 0)), reverse=True):
        symbol = row.get("symbol")
        if symbol and symbol not in best_by_symbol:
            best_by_symbol[symbol] = {
                "symbol": symbol,
                "timeframe": row.get("timeframe", requested_timeframe),
                "window_start": row.get("window_start"),
                "window_end": row.get("window_end"),
                "selected_primary_model": row.get("model_name", "market_trend_benchmark"),
                "selected_backend_model": "",
                "selected_primary_signal": row.get("latest_prediction", "n/a"),
                "latest_signal": row.get("latest_prediction", "n/a"),
                "selected_primary_confidence": 0,
                "latest_signal_confidence": 0,
                "test_accuracy": row.get("accuracy", 0),
                "test_macro_f1": row.get("macro_f1", 0),
                "test_balanced_accuracy": row.get("balanced_accuracy", 0),
                "latest_close": row.get("latest_close", 0),
                "current_pipeline_mode": "market_only_expansion",
                "primary_summary": (
                    f"{symbol} is available as a market-first asset using the market trend benchmark on "
                    f"{row.get('timeframe', requested_timeframe)}."
                ),
            }

    return best_by_symbol


def load_cross_asset_relative_strength_summaries(requested_timeframe=None):
    patterns = []
    if requested_timeframe in SUPPORTED_MARKET_TIMEFRAMES:
        patterns.append(f"cross_asset_relative_strength_summary_{requested_timeframe}_*.csv")
    patterns.append("cross_asset_relative_strength_summary_*.csv")

    summary_path = None
    for pattern in patterns:
        summary_path = get_latest_processed_file(pattern)
        if summary_path is not None and summary_path.exists():
            break

    if summary_path is None or not summary_path.exists():
        return {}

    summaries = {}
    with open(summary_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if value in (None, ""):
                    cleaned[key] = value
                    continue
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    cleaned[key] = value
            summaries[cleaned["symbol"]] = cleaned

    return summaries


def load_structural_break_summaries(requested_timeframe=None):
    # load the latest structural-break summaries if they exist
    patterns = []
    if requested_timeframe in SUPPORTED_MARKET_TIMEFRAMES:
        patterns.append(f"structural_break_summary_{requested_timeframe}_*.csv")
    patterns.append("structural_break_summary_*.csv")

    summary_path = None
    for pattern in patterns:
        summary_path = get_latest_processed_file(pattern)
        if summary_path is not None and summary_path.exists():
            break

    if summary_path is None or not summary_path.exists():
        return {}

    summaries = {}
    with open(summary_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if value in (None, ""):
                    cleaned[key] = value
                    continue
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    cleaned[key] = value
            summaries[cleaned["symbol"]] = cleaned

    return summaries


def build_market_context(timeframe=None):
    # select a stable default asset for the market driven pages
    summaries, _ = get_merged_market_summaries(timeframe)

    preferred_order = get_all_symbols()
    market_symbols = [symbol for symbol in preferred_order if symbol in summaries]
    market_symbols.extend([symbol for symbol in summaries.keys() if symbol not in market_symbols])
    selected_symbol = "BTCUSDT" if "BTCUSDT" in summaries else (market_symbols[0] if market_symbols else None)
    selected_summary = summaries.get(selected_symbol, {})
    return summaries, market_symbols, selected_symbol, selected_summary


def build_context_mode(summary):
    sentiment_source = str(summary.get("latest_effective_sentiment_source", "unavailable") or "unavailable")
    onchain_label = str(summary.get("latest_onchain_regime_label", "unavailable") or "unavailable")

    if sentiment_source == "fear_greed_market_fallback":
        return "Fallback"
    if sentiment_source == "unavailable" and onchain_label == "unavailable":
        return "Reduced"
    return "Ready"


def build_display_layer_list(capabilities):
    layers = ["market_spot"]
    if capabilities.get("futures_available"):
        layers.append("futures_structure")
    if capabilities.get("effective_sentiment_available"):
        layers.append("sentiment")
    if capabilities.get("onchain_available"):
        layers.append("onchain")
    if capabilities.get("defi_available"):
        layers.append("defi_ecosystem")
    return layers


def build_defi_context_payload(summary):
    return {
        "chain_name": summary.get("defi_chain_name"),
        "latest_tvl_usd": safe_float(summary.get("latest_defi_tvl_usd")),
        "change_pct_1d": safe_float(summary.get("defi_tvl_change_pct_1d")),
        "change_pct_7d": safe_float(summary.get("defi_tvl_change_pct_7d")),
        "change_pct_30d": safe_float(summary.get("defi_tvl_change_pct_30d")),
        "zscore_30d": safe_float(summary.get("defi_tvl_zscore_30d")),
        "drawdown_30d": safe_float(summary.get("defi_tvl_drawdown_30d")),
        "regime_score": safe_float(summary.get("defi_regime_score")),
        "regime_label": summary.get("defi_regime_label", "unavailable"),
        "summary": summary.get("defi_summary", "DeFi ecosystem context is unavailable."),
        "snapshot_status": summary.get("defi_snapshot_status", "unavailable"),
        "snapshot_age_days": safe_float(summary.get("defi_snapshot_age_days")),
        "available": safe_bool(summary.get("defi_context_available")) or summary.get("defi_snapshot_status") in {"fresh", "aging", "stale"},
    }


def get_runtime_support():
    market_symbols = build_market_context()[1]
    return build_runtime_support_snapshot(
        PROCESSED_DIR,
        RAW_BINANCE_DIR,
        market_symbols=tuple(market_symbols),
    )


def get_evaluation_coverage():
    return build_evaluation_coverage_snapshot(PROCESSED_DIR)


def get_timeframe_readiness():
    market_symbols = build_market_context()[1]
    return build_timeframe_readiness_snapshot(
        PROCESSED_DIR,
        RAW_BINANCE_DIR,
        market_symbols=tuple(market_symbols),
    )


def get_strategy_family_evidence():
    return build_strategy_family_evidence_snapshot(PROCESSED_DIR)


def get_strategy_family_scope():
    return build_strategy_family_scope_snapshot(PROJECT_DIR)


def get_data_strategy_health():
    return build_data_strategy_health_snapshot(PROJECT_DIR)


def get_backend_readiness_report():
    return build_backend_readiness_report()


def get_context_source_governance():
    return build_context_source_governance_snapshot()


def get_pipeline_refresh():
    return build_pipeline_refresh_snapshot(PROCESSED_DIR)


def apply_pipeline_refresh_to_summary(summary, refresh_entry):
    summary = dict(summary or {})
    refresh_entry = refresh_entry or {}
    summary["pipeline_freshness_label"] = refresh_entry.get("freshness_label", "unknown")
    summary["pipeline_refresh_status"] = refresh_entry.get("status", "missing")
    summary["pipeline_window_age_days"] = refresh_entry.get("window_age_days")
    summary["pipeline_window_end"] = refresh_entry.get("window_end")
    return summary


def get_family_governance_matrix():
    return build_family_governance_matrix(PROCESSED_DIR)


def get_family_governance_snapshot(asset=None, timeframe=None):
    matrix = get_family_governance_matrix()
    normalized_asset = (asset or "").strip().upper()
    normalized_timeframe = timeframe or "4h"

    asset_summary = matrix["assets"].get(
        normalized_asset,
        {"asset": normalized_asset, "timeframes": {}},
    ) if normalized_asset else None
    timeframe_summary = matrix["timeframes"].get(
        normalized_timeframe,
        {
            "timeframe": normalized_timeframe,
            "assets": {},
            "lead_market_futures_assets": [],
            "lead_market_only_assets": [],
            "mixed_assets": [],
            "summary": "No family-governance comparison has been generated for this timeframe yet.",
        },
    )

    entry = None
    if asset_summary:
        entry = asset_summary.get("timeframes", {}).get(normalized_timeframe)

    recommended_actions = []
    if normalized_asset:
        recommended_actions = [
            action
            for action in matrix.get("recommended_actions", [])
            if action.get("asset") == normalized_asset
            and action.get("timeframe") == normalized_timeframe
        ]

    live_governance = None
    if normalized_asset:
        live_summaries, _ = get_merged_market_summaries(normalized_timeframe)
        live_governance = build_family_governance(
            live_summaries.get(normalized_asset, {}),
            "market_futures_backend",
        )

    return {
        "matrix": matrix,
        "asset": normalized_asset,
        "timeframe": normalized_timeframe,
        "entry": entry,
        "live_governance": live_governance,
        "asset_summary": asset_summary,
        "timeframe_summary": timeframe_summary,
        "recommended_actions": recommended_actions,
    }


def get_context_overlay_snapshot(asset=None, timeframe=None):
    normalized_asset = (asset or "").strip().upper()
    normalized_timeframe = timeframe or "4h"
    entry = None

    if normalized_asset:
        overlay_summaries = load_context_overlay_comparison_summaries(normalized_timeframe)
        entry = overlay_summaries.get(normalized_asset)

    recommended_role = "secondary_context"
    recommended_action = "keep_context_secondary"
    if entry:
        lead = str(
            entry.get("context_overlay_lead", "market_futures_core_keep_context_secondary")
            or "market_futures_core_keep_context_secondary"
        )
        if lead == "news_event_veto_lead":
            recommended_role = "event_veto"
            recommended_action = "elevate_news_risk_checks"
        elif lead == "onchain_structural_confirmation_lead":
            recommended_role = "structural_confirmation"
            recommended_action = "elevate_onchain_confirmation"
        elif lead == "cautious_multimodal_overlay":
            recommended_role = "cautious_overlay"
            recommended_action = "use_combined_context_cautiously"

    return {
        "asset": normalized_asset,
        "timeframe": normalized_timeframe,
        "entry": entry,
        "recommended_role": recommended_role,
        "recommended_action": recommended_action,
    }


def classify_activity_status(volume_zscore):
    if volume_zscore is None:
        return "n/a"
    volume_zscore = safe_float(volume_zscore, default=None)
    if volume_zscore is None:
        return "n/a"
    if volume_zscore >= 1.0:
        return "Elevated"
    if volume_zscore <= -0.5:
        return "Quiet"
    return "Normal"


def load_recent_market_chart(symbol, requested_timeframe="4h", max_points=48):
    candidate_timeframes = [requested_timeframe] if requested_timeframe in SUPPORTED_MARKET_TIMEFRAMES else []
    candidate_timeframes.extend([timeframe for timeframe in SUPPORTED_MARKET_TIMEFRAMES if timeframe not in candidate_timeframes])

    chart_path = None
    resolved_timeframe = requested_timeframe or "4h"
    for timeframe in candidate_timeframes:
        chart_path = get_latest_raw_binance_file(symbol, timeframe)
        if chart_path is not None:
            resolved_timeframe = timeframe
            break

    if chart_path is None:
        return {
            "asset": symbol,
            "requested_timeframe": requested_timeframe,
            "resolved_timeframe": resolved_timeframe,
            "points": [],
            "refreshed_at": None,
        }

    df = pd.read_csv(chart_path, parse_dates=["open_time", "close_time"])
    df = df.sort_values("open_time").tail(max_points).reset_index(drop=True)
    points = [
        {
            "time": row["open_time"].isoformat(),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for _, row in df.iterrows()
    ]
    return {
        "asset": symbol,
        "requested_timeframe": requested_timeframe,
        "resolved_timeframe": resolved_timeframe,
        "refreshed_at": datetime.fromtimestamp(chart_path.stat().st_mtime).isoformat(),
        "points": points,
    }


def load_backtest_curve(symbol, timeframe="4h", variant="binary"):
    # read the most recent backtest equity curve for this asset and timeframe
    if variant == "binary":
        pattern = f"{symbol}_{timeframe}_market_futures_binary_backtest_curve_*.csv"
    else:
        pattern = f"{symbol}_{timeframe}_market_futures_backtest_curve_*.csv"
    path = get_latest_processed_file(pattern)
    empty = {"available": False, "points": [], "window": None, "source_file": None}
    if not path:
        return empty
    try:
        df = pd.read_csv(path)
    except Exception:
        return empty
    if df.empty:
        return empty

    points = []
    for _, row in df.iterrows():
        points.append({
            "time": str(row.get("open_time", "")),
            "close": safe_float(row.get("close")),
            "strategy_equity": safe_float(row.get("strategy_equity_curve", 1.0), 1.0),
            "buy_hold_equity": safe_float(row.get("buy_hold_equity_curve", 1.0), 1.0),
            "position": safe_float(row.get("position", 0.0)),
            "action": str(row.get("action", "")),
        })

    final_strategy = points[-1]["strategy_equity"] if points else 1.0
    final_hold = points[-1]["buy_hold_equity"] if points else 1.0

    peak = 1.0
    max_dd = 0.0
    for point in points:
        peak = max(peak, point["strategy_equity"])
        if peak > 0:
            dd = (point["strategy_equity"] / peak) - 1.0
            if dd < max_dd:
                max_dd = dd

    return {
        "available": True,
        "asset": symbol,
        "timeframe": timeframe,
        "variant": variant,
        "source_file": path.name,
        "point_count": len(points),
        "points": points,
        "strategy_total_return_pct": (final_strategy - 1.0) * 100.0,
        "buy_hold_total_return_pct": (final_hold - 1.0) * 100.0,
        "excess_return_pct": (final_strategy - final_hold) * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "window": {
            "start": points[0]["time"] if points else None,
            "end": points[-1]["time"] if points else None,
        },
    }


def load_walkforward_folds(symbol, timeframe="4h", variant="binary"):
    # read the per-fold table for this asset and timeframe
    if variant == "binary":
        pattern = f"market_futures_binary_walkforward_detail_{timeframe}_*.csv"
    else:
        pattern = f"market_futures_walkforward_detail_{timeframe}_*.csv"
    path = get_latest_processed_file(pattern)
    empty = {"available": False, "folds": [], "fold_count": 0, "source_file": None}
    if not path:
        return empty
    try:
        df = pd.read_csv(path)
    except Exception:
        return empty
    if df.empty or "symbol" not in df.columns:
        return empty

    asset_rows = df[df["symbol"] == symbol]
    if asset_rows.empty:
        return empty

    folds = []
    for _, row in asset_rows.iterrows():
        folds.append({
            "fold_number": int(safe_float(row.get("fold_number", 0))),
            "test_start": str(row.get("test_start_time", "")),
            "test_end": str(row.get("test_end_time", "")),
            "train_rows": int(safe_float(row.get("train_rows", 0))),
            "test_rows": int(safe_float(row.get("test_rows", 0))),
            "accuracy": safe_float(row.get("fold_accuracy", 0.0)),
            "macro_f1": safe_float(row.get("fold_macro_f1", 0.0)),
            "balanced_accuracy": safe_float(row.get("fold_balanced_accuracy", 0.0)),
            "strategy_return": safe_float(row.get("selected_strategy_total_return", 0.0)),
            "buy_hold_return": safe_float(row.get("selected_buy_hold_total_return", 0.0)),
            "excess_return": safe_float(row.get("selected_excess_return", 0.0)),
            "sharpe": safe_float(row.get("selected_sharpe_ratio", 0.0)),
            "trade_count": int(safe_float(row.get("selected_trade_count", 0))),
            "deployment_active": safe_bool(row.get("selected_deployment_active", False)),
        })
    folds.sort(key=lambda fold: fold["fold_number"])
    return {
        "available": True,
        "asset": symbol,
        "timeframe": timeframe,
        "variant": variant,
        "source_file": path.name,
        "fold_count": len(folds),
        "folds": folds,
    }


def build_backtest_view_payload(symbol, timeframe="4h", variant="binary"):
    return {
        "curve": load_backtest_curve(symbol, timeframe, variant),
        "folds": load_walkforward_folds(symbol, timeframe, variant),
    }


def load_significance_summary(timeframe="4h"):
    # read the strategy significance summary written by the pipeline for one timeframe
    path = PROCESSED_DIR / f"strategy_significance_summary_{timeframe}.csv"
    empty = {"available": False, "rows": [], "timeframe": timeframe, "source_file": None}
    if not path.exists():
        return empty
    try:
        df = pd.read_csv(path)
    except Exception:
        return empty
    if df.empty:
        return empty
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "asset": str(row.get("symbol", "")),
            "timeframe": str(row.get("timeframe", timeframe)),
            "n_observations": int(safe_float(row.get("n_observations", 0))),
            "mean_strategy_return": safe_float(row.get("mean_strategy_return", 0.0)),
            "mean_benchmark_return": safe_float(row.get("mean_benchmark_return", 0.0)),
            "mean_difference": safe_float(row.get("mean_difference", 0.0)),
            "bootstrap_p_value": safe_float(row.get("bootstrap_p_value", 0.0)),
            "bootstrap_ci_low": safe_float(row.get("bootstrap_ci_low", 0.0)),
            "bootstrap_ci_high": safe_float(row.get("bootstrap_ci_high", 0.0)),
            "diebold_mariano_stat": safe_float(row.get("diebold_mariano_stat", 0.0)),
            "diebold_mariano_p_value": safe_float(row.get("diebold_mariano_p_value", 0.0)),
            "annualised_sharpe": safe_float(row.get("annualised_sharpe", 0.0)),
            "probabilistic_sharpe_ratio": safe_float(row.get("probabilistic_sharpe_ratio", 0.0)),
            "deflated_sharpe_ratio": safe_float(row.get("deflated_sharpe_ratio", 0.0)),
            "n_trials_for_dsr": int(safe_float(row.get("n_trials_for_dsr", 0))),
            "evidence_label": str(row.get("evidence_label", "no_evidence")),
        })
    return {
        "available": True,
        "timeframe": timeframe,
        "source_file": path.name,
        "rows": rows,
    }


def build_cross_asset_comparison(timeframe="4h"):
    # merge per-asset walk-forward metrics with the significance summary so the
    # Analytics page can render one heatmap row per asset
    summaries, _bundle = get_merged_market_summaries(timeframe)
    significance = load_significance_summary(timeframe)
    significance_by_asset = {row["asset"]: row for row in significance.get("rows", [])}

    metric_definitions = [
        {"key": "accuracy", "label": "Accuracy", "hint": "Held-out classification accuracy. Higher is better."},
        {"key": "walkforward_sharpe", "label": "WF Sharpe", "hint": "Average Sharpe across walk-forward folds. Higher is better."},
        {"key": "excess_return", "label": "Excess vs hold", "hint": "Average per-fold strategy return minus buy-and-hold return."},
        {"key": "macro_f1", "label": "Macro-F1", "hint": "Held-out macro-F1. Fair across classes when long signals are rare."},
        {"key": "dsr", "label": "DSR", "hint": "Deflated Sharpe Ratio. Adjusts for multiple trials and non-normal returns."},
    ]

    assets = []
    for symbol, summary in summaries.items():
        sig_row = significance_by_asset.get(symbol, {})
        assets.append({
            "asset": symbol,
            "asset_short": symbol.replace("USDT", ""),
            "metrics": {
                "accuracy": safe_float(summary.get("test_accuracy", summary.get("walkforward_avg_accuracy", 0.0))),
                "walkforward_sharpe": safe_float(summary.get("walkforward_avg_sharpe", summary.get("sharpe_ratio", 0.0))),
                "excess_return": safe_float(summary.get("walkforward_avg_excess_return", summary.get("excess_return", 0.0))),
                "macro_f1": safe_float(summary.get("test_macro_f1", summary.get("walkforward_avg_macro_f1", 0.0))),
                "dsr": safe_float(sig_row.get("deflated_sharpe_ratio", 0.0)),
            },
            "fold_count": int(safe_float(summary.get("walkforward_fold_count", 0))),
            "evidence_label": sig_row.get("evidence_label", "no_data"),
        })

    return {
        "timeframe": timeframe,
        "assets": assets,
        "metric_definitions": metric_definitions,
        "significance_window": significance.get("source_file"),
    }


def build_market_display_payload(symbol, requested_timeframe="4h"):
    summaries, summary_bundle = get_merged_market_summaries(requested_timeframe)
    market_summary = summaries.get(symbol, {})
    capabilities = build_asset_capability_state(symbol.replace("USDT", ""), market_summary)
    notes = build_capability_notes(capabilities)
    summary_path = summary_bundle["path"]
    resolved_timeframe = summary_bundle["timeframe"] or requested_timeframe or "4h"
    runtime_support = get_runtime_support()["assets"].get(symbol, {})
    context_assessment = build_context_layer_assessment(market_summary, runtime_support)
    context_overlay_snapshot = get_context_overlay_snapshot(symbol, resolved_timeframe)
    pipeline_refresh = get_pipeline_refresh().get("timeframes", {}).get(resolved_timeframe, {})
    pipeline_guidance = build_pipeline_refresh_guidance(pipeline_refresh)

    return {
        "data_lane": "display",
        "asset": symbol,
        "requested_timeframe": requested_timeframe,
        "resolved_timeframe": resolved_timeframe,
        "exact_timeframe_match": resolved_timeframe == requested_timeframe,
        "available_timeframes": runtime_support.get("market_summary_timeframes", []),
        "chart_timeframes": runtime_support.get("chart_timeframes", []),
        "refreshed_at": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat() if summary_path else None,
        "latest_close": float(market_summary.get("latest_close", 0) or 0),
        "latest_return_24h_pct": float(market_summary.get("latest_return_24h_pct", 0) or 0),
        "volatility_status": market_summary.get("volatility_status", "n/a"),
        "trend_status": market_summary.get("trend_status", "n/a"),
        "activity_status": classify_activity_status(market_summary.get("latest_volume_zscore")),
        "summary_text": (
            market_summary.get("primary_summary")
            or market_summary.get("analysis_summary")
            or "No generated market snapshot is available yet."
        ),
        "context_mode": build_context_mode(market_summary),
        "rule_signal": market_summary.get("rule_signal", "n/a"),
        "policy_return": float(market_summary.get("strategy_total_return", 0) or 0),
        "sharpe_ratio": float(market_summary.get("sharpe_ratio", 0) or 0),
        "max_drawdown": float(market_summary.get("max_drawdown", 0) or 0),
        "capabilities": capabilities,
        "capability_notes": notes,
        "futures_completeness_label": market_summary.get("futures_completeness_label", "unavailable"),
        "futures_feature_completeness_score": float(market_summary.get("futures_feature_completeness_score", 0) or 0),
        "futures_context_resilience_label": market_summary.get("futures_context_resilience_label", "unavailable"),
        "futures_context_resilience_score": float(market_summary.get("futures_context_resilience_score", 0) or 0),
        "futures_basis_reliance_score": float(market_summary.get("futures_basis_reliance_score", 0) or 0),
        "basis_feature_available": safe_bool(market_summary.get("basis_feature_available")),
        "basis_feature_mode": market_summary.get("basis_feature_mode", "unavailable"),
        "basis_proxy_active": safe_bool(market_summary.get("basis_proxy_active")),
        "effective_basis_feature_available": safe_bool(market_summary.get("effective_basis_feature_available")),
        "effective_sentiment_source": market_summary.get("latest_effective_sentiment_source", "unavailable"),
        "effective_sentiment_label": market_summary.get("latest_effective_sentiment_label", "unavailable"),
        "gdelt_status": market_summary.get("latest_gdelt_regime_label", "unavailable"),
        "gdelt_article_count": float(market_summary.get("latest_gdelt_article_count", 0) or 0),
        "gdelt_dominant_event_theme": market_summary.get("latest_gdelt_dominant_event_theme", market_summary.get("gdelt_dominant_event_theme")),
        "gdelt_risk_event_theme": market_summary.get("latest_gdelt_risk_event_theme", market_summary.get("gdelt_risk_event_theme")),
        "gdelt_supportive_event_theme": market_summary.get("latest_gdelt_supportive_event_theme", market_summary.get("gdelt_supportive_event_theme")),
        "onchain_status": market_summary.get("latest_onchain_regime_label", "unavailable"),
        "onchain_primary_support_driver": market_summary.get("latest_onchain_primary_support_driver"),
        "onchain_primary_risk_driver": market_summary.get("latest_onchain_primary_risk_driver"),
        "onchain_participation_breadth_score": float(market_summary.get("latest_onchain_participation_breadth_score", 0) or 0),
        "onchain_structural_fragility_score": float(market_summary.get("latest_onchain_structural_fragility_score", 0) or 0),
        "onchain_snapshot_status": market_summary.get("latest_onchain_snapshot_status", "unavailable"),
        "onchain_snapshot_label": market_summary.get("latest_onchain_snapshot_label", "unavailable"),
        "onchain_snapshot_age_days": float(market_summary.get("latest_onchain_snapshot_age_days", 0) or 0),
        "defi_context": build_defi_context_payload(market_summary),
        "defi_chain_name": market_summary.get("defi_chain_name"),
        "defi_regime_label": market_summary.get("defi_regime_label", "unavailable"),
        "defi_snapshot_status": market_summary.get("defi_snapshot_status", "unavailable"),
        "latest_defi_tvl_usd": safe_float(market_summary.get("latest_defi_tvl_usd")),
        "defi_tvl_change_pct_30d": safe_float(market_summary.get("defi_tvl_change_pct_30d")),
        "defi_summary": market_summary.get("defi_summary", "DeFi ecosystem context is unavailable."),
        "multimodal_selected_context_variant": market_summary.get("multimodal_selected_context_variant"),
        "display_layers": build_display_layer_list(capabilities),
        "pipeline_refresh": pipeline_refresh,
        "pipeline_refresh_guidance": pipeline_guidance,
        "context_assessment": context_assessment,
        "context_overlay_entry": context_overlay_snapshot.get("entry"),
        "context_overlay_role": context_overlay_snapshot.get("recommended_role"),
        "context_overlay_action": context_overlay_snapshot.get("recommended_action"),
    }


def build_named_strategy_decision(summary, strategy_key, capabilities):
    strategy_key = "recommended" if strategy_key == "market_futures_backend" else strategy_key
    forecast_score = clamp(safe_float(summary.get("predicted_return")) * 12)
    indicator_score = clamp(safe_float(summary.get("indicator_score")))
    final_score = clamp(safe_float(summary.get("final_strategy_score")))
    momentum_score = clamp(safe_float(summary.get("momentum_score")))
    trend_score = clamp(safe_float(summary.get("trend_score")))
    participation_score = clamp(safe_float(summary.get("participation_score")))
    volatility_penalty = clamp(safe_float(summary.get("volatility_penalty")), 0.0, 1.0)
    context_adjustment = clamp(safe_float(summary.get("context_adjustment")), -0.35, 0.35)
    futures_confidence = clamp(safe_float(summary.get("futures_confidence")), 0.0, 1.0)
    daily_score = clamp(safe_float(summary.get("daily_confirmation_score", summary.get("latest_onchain_overlay_confidence", 0))))

    futures_signal = str(summary.get("futures_signal", "hold") or "hold")
    classification_signal = str(summary.get("classification_signal", "hold") or "hold")
    daily_signal = str(summary.get("best_daily_signal", "hold") or "hold")

    strategy_rules = {
        "recommended": {
            "strategy_family": "enhanced_market_futures",
            "score": final_score,
            "signal": summary.get("final_strategy_signal"),
            "model_name": summary.get("forecast_model", summary.get("selected_primary_model", "selected_market_futures_backend")),
            "policy_name": summary.get("policy_name", "forecast_indicator_futures_blend"),
            "target_name": summary.get("selected_target_name", summary.get("current_pipeline_mode", "future_return")),
            "basis": "Combines the latest regression forecast, directional classifier, technical indicators, futures read, and available context adjustment.",
        },
        "conservative_trend": {
            "strategy_family": "trend_confirmation",
            "score": clamp((0.38 * trend_score) + (0.28 * indicator_score) + (0.22 * forecast_score) + (0.12 * context_adjustment) - (0.18 * volatility_penalty)),
            "model_name": summary.get("classification_model", "trend_confirmation_policy"),
            "policy_name": "strict_trend_confirmation",
            "target_name": "future_return_with_trend_filter",
            "basis": "Requires trend, indicator, and forecast evidence to line up before it becomes bullish.",
            "buy_threshold": 0.35,
            "sell_threshold": -0.25,
        },
        "momentum_breakout": {
            "strategy_family": "momentum_breakout",
            "score": clamp((0.34 * momentum_score) + (0.24 * participation_score) + (0.24 * forecast_score) + (0.12 * indicator_score) - (0.12 * volatility_penalty)),
            "model_name": summary.get("classification_model", "momentum_breakout_policy"),
            "policy_name": "momentum_participation_breakout",
            "target_name": "short_horizon_continuation",
            "basis": "Looks for short-horizon momentum, stronger participation, and a supportive forecast before entering.",
            "buy_threshold": 0.30,
            "sell_threshold": -0.20,
        },
        "futures_crowd_reversal": {
            "strategy_family": "futures_crowd_reversal",
            "score": clamp((0.42 * final_score) + (0.28 * (1 if futures_signal == "buy" else -1 if futures_signal == "dont_buy" else 0) * max(futures_confidence, 0.25)) + (0.18 * forecast_score) - (0.12 * volatility_penalty)),
            "model_name": "futures_crowd_reversal_policy",
            "policy_name": "positioning_pressure_reversal",
            "target_name": "positioning_adjusted_future_return",
            "basis": "Weights the futures structure heavily, then checks whether forecast and volatility support the setup.",
            "buy_threshold": 0.30,
            "sell_threshold": -0.25,
        },
        "multimodal_balanced": {
            "strategy_family": "context_aware_balanced",
            "score": clamp((0.58 * final_score) + (0.22 * context_adjustment) + (0.10 * indicator_score) + (0.10 * forecast_score)),
            "model_name": summary.get("forecast_model", "context_aware_balanced_policy"),
            "policy_name": "context_adjusted_balanced_policy",
            "target_name": "context_adjusted_future_return",
            "basis": "Starts from the default strategy and lets available context nudge the decision without replacing market evidence.",
            "buy_threshold": 0.28,
            "sell_threshold": -0.25,
        },
        "daily_structural_confirmation": {
            "strategy_family": "daily_structural_confirmation",
            "score": clamp((0.42 * daily_score) + (0.22 * final_score) + (0.16 * trend_score) + (0.12 * context_adjustment) + (0.08 if daily_signal == "buy" else -0.08 if daily_signal == "dont_buy" else 0)),
            "model_name": summary.get("best_daily_specialist", "daily_structural_confirmation"),
            "policy_name": "higher_timeframe_confirmation",
            "target_name": "structural_confirmation",
            "basis": "Uses the daily structural read as a higher-timeframe confirmation layer for the selected asset.",
            "buy_threshold": 0.42,
            "sell_threshold": -0.20,
        },
    }

    rule = dict(strategy_rules.get(strategy_key, strategy_rules["recommended"]))
    score = clamp(safe_float(rule["score"]))
    current_signal = signal_from_score(
        score,
        buy_threshold=safe_float(rule.get("buy_threshold"), 0.25),
        sell_threshold=safe_float(rule.get("sell_threshold"), -0.25),
    )
    if rule.get("signal") in {"buy", "hold", "dont_buy"}:
        current_signal = rule["signal"]
    if strategy_key == "conservative_trend" and classification_signal == "dont_buy" and current_signal == "buy":
        current_signal = "hold"
    if strategy_key == "multimodal_balanced" and not (capabilities["effective_sentiment_available"] or capabilities["onchain_available"]):
        rule["basis"] += " Current context layers are reduced, so market and futures evidence remain dominant."

    predicted_price = safe_float(summary.get("predicted_price"))
    predicted_return = safe_float(summary.get("predicted_return"))
    summary_text = (
        f"{resolve_strategy_display_name(strategy_key)} reads {current_signal} with score {score:.2f}. "
        f"Forecast move is {predicted_return:.2%}"
        + (f" toward an estimated price near {predicted_price:,.2f}. " if predicted_price else ". ")
        + f"Indicator read is {summary.get('indicator_signal', 'n/a')}; futures read is {futures_signal}. "
        + rule["basis"]
    )

    return {
        "strategy_family": rule["strategy_family"],
        "current_signal": current_signal,
        "confidence": abs(score),
        "held_out_accuracy": safe_float(summary.get("test_accuracy", summary.get("forecast_directional_accuracy", 0))),
        "macro_f1": safe_float(summary.get("test_macro_f1", summary.get("classification_macro_f1", 0))),
        "model_name": rule["model_name"],
        "policy_name": rule["policy_name"],
        "target_name": rule["target_name"],
        "data_layers_used": build_display_layer_list(capabilities),
        "evaluation_basis": rule["basis"],
        "summary_text": summary_text,
        "strategy_variant_score": score,
    }


def build_strategy_decision_payload(symbol, strategy_key="market_futures_backend", requested_timeframe="4h"):
    summaries, summary_bundle = get_merged_market_summaries(requested_timeframe)
    summary = summaries.get(symbol, {})
    capabilities = build_asset_capability_state(symbol.replace("USDT", ""), summary)
    resolved_timeframe = summary_bundle["timeframe"] or requested_timeframe or "4h"
    summary_path = summary_bundle["path"]
    runtime_support = get_runtime_support()["assets"].get(symbol, {})
    pipeline_refresh = get_pipeline_refresh().get("timeframes", {}).get(resolved_timeframe, {})
    pipeline_guidance = build_pipeline_refresh_guidance(pipeline_refresh)
    summary = apply_pipeline_refresh_to_summary(summary, pipeline_refresh)
    named_strategy_keys = {
        "recommended",
        "market_futures_backend",
        "conservative_trend",
        "momentum_breakout",
        "futures_crowd_reversal",
        "multimodal_balanced",
        "daily_structural_confirmation",
    }
    named_strategy_decision = build_named_strategy_decision(summary, strategy_key, capabilities)

    decision_map = {
        "recommended": build_named_strategy_decision(summary, "recommended", capabilities),
        "market_futures_backend": build_named_strategy_decision(summary, "recommended", capabilities),
        "conservative_trend": build_named_strategy_decision(summary, "conservative_trend", capabilities),
        "momentum_breakout": build_named_strategy_decision(summary, "momentum_breakout", capabilities),
        "futures_crowd_reversal": build_named_strategy_decision(summary, "futures_crowd_reversal", capabilities),
        "multimodal_balanced": build_named_strategy_decision(summary, "multimodal_balanced", capabilities),
        "daily_structural_confirmation": build_named_strategy_decision(summary, "daily_structural_confirmation", capabilities),
        "rule_based": {
            "strategy_family": "core_market_rule_based",
            "current_signal": summary.get("rule_signal", "n/a"),
            "confidence": 0.0,
            "held_out_accuracy": float(summary.get("rule_based_test_accuracy", 0) or 0),
            "macro_f1": 0.0,
            "model_name": "rule_based_benchmark",
            "policy_name": "transparent_market_rule_set",
            "target_name": "future_return_bucket",
            "data_layers_used": ["market_spot"],
            "evaluation_basis": "Chronological rule benchmark evaluated on the held-out section of the market dataset.",
            "summary_text": "Rule-based benchmark uses transparent market conditions rather than a trained classifier.",
        },
        "scaled_model": {
            "strategy_family": "core_market_model",
            "current_signal": summary.get("scaled_model_signal", "n/a"),
            "confidence": float(summary.get("scaled_model_confidence", 0) or 0),
            "held_out_accuracy": float(summary.get("baseline_scaled_test_accuracy", 0) or 0),
            "macro_f1": 0.0,
            "model_name": "scaled_market_baseline",
            "policy_name": "classification_only",
            "target_name": "future_return_bucket",
            "data_layers_used": ["market_spot"],
            "evaluation_basis": "Scaled market-only benchmark on the leakage-safe market feature set.",
            "summary_text": "Scaled market baseline is the cleanest interpretable ML-style baseline before additional context layers are added.",
        },
        "unscaled_model": {
            "strategy_family": "core_market_model",
            "current_signal": "comparison_view",
            "confidence": 0.0,
            "held_out_accuracy": float(summary.get("baseline_unscaled_test_accuracy", 0) or 0),
            "macro_f1": 0.0,
            "model_name": "unscaled_market_baseline",
            "policy_name": "classification_only",
            "target_name": "future_return_bucket",
            "data_layers_used": ["market_spot"],
            "evaluation_basis": "Unscaled market-only benchmark for comparison against the scaled baseline.",
            "summary_text": "Unscaled market baseline is kept mainly as a benchmark comparison rather than a production-facing choice.",
        },
    }

    payload = decision_map.get(strategy_key, named_strategy_decision if strategy_key in named_strategy_keys else decision_map["recommended"])
    strategy_display_name = resolve_strategy_display_name(
        strategy_key,
        payload.get("model_name"),
        payload.get("strategy_family"),
    )
    engine_display_name = resolve_engine_label(
        payload.get("model_name"),
        strategy_key,
        payload.get("strategy_family"),
    )
    family_display_name = resolve_strategy_family_label(
        payload.get("strategy_family"),
        strategy_key,
        payload.get("model_name"),
    )
    timeframe_policy = build_timeframe_strategy_policy(
        requested_timeframe=requested_timeframe,
        resolved_timeframe=resolved_timeframe,
        selected_timeframes=[requested_timeframe],
    )
    target_semantics = build_target_semantics(payload.get("target_name"), resolved_timeframe)
    governance = build_strategy_governance(
        strategy_key,
        summary,
        requested_timeframe,
        resolved_timeframe,
        runtime_status=runtime_support.get("strategy_statuses", {}).get(strategy_key, {}),
        capabilities=capabilities,
    )
    context_assessment = build_context_layer_assessment(summary, runtime_support)
    family_governance = build_family_governance(summary, strategy_key)
    governance_snapshot = get_family_governance_snapshot(symbol, resolved_timeframe)
    context_overlay_snapshot = get_context_overlay_snapshot(symbol, resolved_timeframe)
    payload.update(
        {
            "data_lane": "decision",
            "asset": symbol,
            "strategy_key": strategy_key,
            "requested_timeframe": requested_timeframe,
            "resolved_timeframe": resolved_timeframe,
            "exact_timeframe_match": resolved_timeframe == requested_timeframe,
              "available_timeframes": runtime_support.get("decision_timeframes", []),
              "strategy_display_name": strategy_display_name,
              "engine_display_name": engine_display_name,
              "strategy_family_display_name": family_display_name,
            "multimodal_timeframes": runtime_support.get("multimodal_timeframes", []),
            "sentiment_timeframes": runtime_support.get("sentiment_timeframes", []),
            "onchain_timeframes": runtime_support.get("onchain_timeframes", []),
            "defi_timeframes": runtime_support.get("defi_timeframes", []),
            "runtime_strategy_status": runtime_support.get("strategy_statuses", {}).get(strategy_key, {}),
            "governance": governance,
            "timeframe_policy": timeframe_policy,
            "refreshed_at": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat() if summary_path else None,
              "context_mode": build_context_mode(summary),
              "target_semantics": target_semantics,
              "walkforward_fold_count": float(summary.get("walkforward_fold_count", 0) or 0),
              "walkforward_avg_accuracy": float(summary.get("walkforward_avg_accuracy", 0) or 0),
            "walkforward_avg_excess_return": float(summary.get("walkforward_avg_excess_return", 0) or 0),
            "sharpe_ratio": float(summary.get("sharpe_ratio", 0) or 0),
            "policy_return": float(summary.get("strategy_total_return", 0) or 0),
            "max_drawdown": float(summary.get("max_drawdown", 0) or 0),
            "predicted_return": float(summary.get("predicted_return", 0) or 0),
            "predicted_price": float(summary.get("predicted_price", 0) or 0),
            "forecast_directional_accuracy": float(summary.get("forecast_directional_accuracy", 0) or 0),
            "indicator_signal": summary.get("indicator_signal", "n/a"),
            "indicator_score": float(summary.get("indicator_score", 0) or 0),
            "final_strategy_score": float(payload.get("strategy_variant_score", summary.get("final_strategy_score", 0)) or 0),
            "base_strategy_score": float(summary.get("final_strategy_score", 0) or 0),
            "indicator_summary": summary.get("indicator_summary", ""),
            "pipeline_refresh": pipeline_refresh,
            "pipeline_refresh_guidance": pipeline_guidance,
            "context_assessment": context_assessment,
            "family_governance": family_governance,
            "family_governance_entry": governance_snapshot.get("entry"),
            "family_governance_actions": governance_snapshot.get("recommended_actions", []),
            "timeframe_family_governance": governance_snapshot.get("timeframe_summary", {}),
            "context_overlay_entry": context_overlay_snapshot.get("entry"),
            "context_overlay_role": context_overlay_snapshot.get("recommended_role"),
            "context_overlay_action": context_overlay_snapshot.get("recommended_action"),
            "futures_completeness_label": summary.get("futures_completeness_label", "unavailable"),
            "futures_feature_completeness_score": float(summary.get("futures_feature_completeness_score", 0) or 0),
            "futures_context_resilience_label": summary.get("futures_context_resilience_label", "unavailable"),
            "futures_context_resilience_score": float(summary.get("futures_context_resilience_score", 0) or 0),
            "futures_basis_reliance_score": float(summary.get("futures_basis_reliance_score", 0) or 0),
            "basis_feature_available": safe_bool(summary.get("basis_feature_available")),
            "basis_feature_mode": summary.get("basis_feature_mode", "unavailable"),
            "basis_proxy_active": safe_bool(summary.get("basis_proxy_active")),
            "effective_basis_feature_available": safe_bool(summary.get("effective_basis_feature_available")),
            "gdelt_dominant_event_theme": summary.get("latest_gdelt_dominant_event_theme", summary.get("gdelt_dominant_event_theme")),
            "gdelt_risk_event_theme": summary.get("latest_gdelt_risk_event_theme", summary.get("gdelt_risk_event_theme")),
            "gdelt_supportive_event_theme": summary.get("latest_gdelt_supportive_event_theme", summary.get("gdelt_supportive_event_theme")),
            "onchain_primary_support_driver": summary.get("latest_onchain_primary_support_driver"),
            "onchain_primary_risk_driver": summary.get("latest_onchain_primary_risk_driver"),
            "onchain_participation_breadth_score": float(summary.get("latest_onchain_participation_breadth_score", 0) or 0),
            "onchain_structural_fragility_score": float(summary.get("latest_onchain_structural_fragility_score", 0) or 0),
            "defi_context": build_defi_context_payload(summary),
            "defi_chain_name": summary.get("defi_chain_name"),
            "defi_regime_label": summary.get("defi_regime_label", "unavailable"),
            "defi_snapshot_status": summary.get("defi_snapshot_status", "unavailable"),
            "latest_defi_tvl_usd": safe_float(summary.get("latest_defi_tvl_usd")),
            "defi_tvl_change_pct_30d": safe_float(summary.get("defi_tvl_change_pct_30d")),
            "defi_summary": summary.get("defi_summary", "DeFi ecosystem context is unavailable."),
        }
    )
    return payload


def build_alert_strategy_summary(asset, strategy_key="recommended", timeframe="4h"):
    market_summaries, _, _, _ = build_market_context(timeframe)
    market_summary = dict(market_summaries.get(asset, {}))
    decision = build_strategy_decision_payload(asset, strategy_key, timeframe)
    market_summary.update(
        {
            "final_strategy_signal": decision.get("current_signal"),
            "final_strategy_score": decision.get("final_strategy_score"),
            "predicted_return": decision.get("predicted_return"),
            "predicted_price": decision.get("predicted_price"),
            "indicator_signal": decision.get("indicator_signal"),
            "strategy_display_name": decision.get("strategy_display_name"),
            "strategy_key": decision.get("strategy_key"),
            "strategy_timeframe": decision.get("resolved_timeframe"),
            "strategy_summary_text": decision.get("summary_text"),
        }
    )
    return market_summary


def build_dashboard_payload(symbol, requested_timeframe="4h"):
    display_payload = build_market_display_payload(symbol, requested_timeframe)
    decision_payload = build_strategy_decision_payload(symbol, "market_futures_backend", requested_timeframe)
    governance_snapshot = get_family_governance_snapshot(symbol, decision_payload["resolved_timeframe"])
    context_overlay_snapshot = get_context_overlay_snapshot(symbol, decision_payload["resolved_timeframe"])
    pipeline_refresh = get_pipeline_refresh().get("timeframes", {}).get(decision_payload["resolved_timeframe"], {})
    pipeline_guidance = build_pipeline_refresh_guidance(pipeline_refresh)
    return {
        "data_lane": "mixed_overview",
        "asset": symbol,
        "requested_timeframe": requested_timeframe,
        "resolved_timeframe": decision_payload["resolved_timeframe"],
        "refreshed_at": decision_payload["refreshed_at"] or display_payload["refreshed_at"],
        "pipeline_refresh": pipeline_refresh,
        "pipeline_refresh_guidance": pipeline_guidance,
        "source_policy_decision": get_context_source_governance()["source_decision"],
        "display": display_payload,
        "decision": decision_payload,
        "defi_context": display_payload.get("defi_context"),
        "family_governance_entry": governance_snapshot.get("entry"),
        "family_governance_actions": governance_snapshot.get("recommended_actions", []),
        "timeframe_family_governance": governance_snapshot.get("timeframe_summary", {}),
        "context_overlay_entry": context_overlay_snapshot.get("entry"),
        "context_overlay_role": context_overlay_snapshot.get("recommended_role"),
        "context_overlay_action": context_overlay_snapshot.get("recommended_action"),
    }


def build_analytics_payload(symbol, requested_timeframe="4h"):
    summaries, summary_bundle = get_merged_market_summaries(requested_timeframe)
    summary = summaries.get(symbol, {})
    resolved_timeframe = summary_bundle["timeframe"] or requested_timeframe
    pipeline_refresh = get_pipeline_refresh().get("timeframes", {}).get(resolved_timeframe, {})
    pipeline_guidance = build_pipeline_refresh_guidance(pipeline_refresh)
    summary = apply_pipeline_refresh_to_summary(summary, pipeline_refresh)
    decision_payload = build_strategy_decision_payload(symbol, "market_futures_backend", requested_timeframe)
    gdelt_unavailable = summary.get("latest_gdelt_regime_label") in (None, "", "unavailable")
    runtime_support = get_runtime_support()["assets"].get(symbol, {})
    evaluation_coverage = get_evaluation_coverage()["assets"].get(symbol, {})
    family_scope = get_strategy_family_scope()
    timeframe_readiness = get_timeframe_readiness()
    context_assessment = build_context_layer_assessment(summary, runtime_support)
    multimodal_assessment = build_multimodal_assessment(summary, runtime_support)
    family_governance = build_family_governance(summary, "market_futures_backend")
    governance_snapshot = get_family_governance_snapshot(symbol, resolved_timeframe)
    context_overlay_snapshot = get_context_overlay_snapshot(symbol, resolved_timeframe)

    for coverage_key in (
        "market_trend_benchmark",
        "market_futures",
        "multimodal",
        "onchain_specialist",
        "cross_asset_relative_strength",
        "market_baseline_scaled",
        "market_baseline_unscaled",
    ):
        coverage_section = evaluation_coverage.get(coverage_key, {})
        if coverage_section.get("best_model_name"):
            coverage_section["best_model_display_name"] = resolve_strategy_display_name(
                coverage_key,
                coverage_section.get("best_model_name"),
                coverage_key,
            )

    return {
        "data_lane": "decision_analytics",
        "asset": symbol,
        "requested_timeframe": requested_timeframe,
        "resolved_timeframe": summary_bundle["timeframe"] or requested_timeframe,
        "exact_timeframe_match": (summary_bundle["timeframe"] or requested_timeframe) == requested_timeframe,
        "available_timeframes": runtime_support.get("decision_timeframes", []),
        "refreshed_at": (
            datetime.fromtimestamp(summary_bundle["path"].stat().st_mtime).isoformat()
            if summary_bundle["path"]
            else None
        ),
        "decision": decision_payload,
        "family_governance": family_governance,
        "family_governance_entry": governance_snapshot.get("entry"),
        "family_governance_actions": governance_snapshot.get("recommended_actions", []),
        "timeframe_family_governance": governance_snapshot.get("timeframe_summary", {}),
        "context_overlay_entry": context_overlay_snapshot.get("entry"),
        "context_overlay_role": context_overlay_snapshot.get("recommended_role"),
        "context_overlay_action": context_overlay_snapshot.get("recommended_action"),
        "target_semantics": decision_payload.get("target_semantics", {}),
        "context_assessment": context_assessment,
        "source_policy_decision": get_context_source_governance()["source_decision"],
        "pipeline_refresh": pipeline_refresh,
        "pipeline_refresh_guidance": pipeline_guidance,
        "defi_context": build_defi_context_payload(summary),
        "futures_completeness_label": summary.get("futures_completeness_label", "unavailable"),
        "futures_feature_completeness_score": float(summary.get("futures_feature_completeness_score", 0) or 0),
        "futures_context_resilience_label": summary.get("futures_context_resilience_label", "unavailable"),
        "futures_context_resilience_score": float(summary.get("futures_context_resilience_score", 0) or 0),
        "futures_basis_reliance_score": float(summary.get("futures_basis_reliance_score", 0) or 0),
        "basis_feature_available": safe_bool(summary.get("basis_feature_available")),
        "basis_feature_mode": summary.get("basis_feature_mode", "unavailable"),
        "basis_proxy_active": safe_bool(summary.get("basis_proxy_active")),
        "effective_basis_feature_available": safe_bool(summary.get("effective_basis_feature_available")),
        "evaluation_coverage": evaluation_coverage,
        "strategy_family_scope": {
            "selected_asset": family_scope.get("assets", {}).get(symbol, {}),
            "defensibility_summary": family_scope.get("defensibility_summary", ""),
            "recommended_demo_framing": family_scope.get("recommended_demo_framing", []),
            "families": family_scope.get("families", {}),
        },
        "timeframe_readiness": timeframe_readiness,
        "performance_overview": {
            "summary_text": (
                summary.get("walkforward_summary")
                or summary.get("primary_summary")
                or summary.get("backend_summary")
                or summary.get("analysis_summary")
                or "No generated analytics summary is available yet."
            ),
            "accuracy": float(summary.get("test_accuracy", summary.get("baseline_scaled_test_accuracy", 0)) or 0),
            "balanced_accuracy": float(summary.get("test_balanced_accuracy", summary.get("rule_based_test_accuracy", 0)) or 0),
            "macro_f1": float(summary.get("test_macro_f1", 0) or 0),
            "target_name": summary.get("selected_target_name", summary.get("top_feature_name", "n/a")),
            "policy_return": float(summary.get("strategy_total_return", 0) or 0),
            "buy_hold_return": float(summary.get("buy_hold_total_return", 0) or 0),
            "sharpe_ratio": float(summary.get("sharpe_ratio", 0) or 0),
            "walkforward_fold_count": float(summary.get("walkforward_fold_count", 0) or 0),
            "walkforward_avg_accuracy": float(summary.get("walkforward_avg_accuracy", 0) or 0),
            "walkforward_avg_excess_return": float(summary.get("walkforward_avg_excess_return", 0) or 0),
            "walkforward_deployment_activity_rate": float(summary.get("walkforward_deployment_activity_rate", 0) or 0),
            "predicted_return": float(summary.get("predicted_return", 0) or 0),
            "predicted_price": float(summary.get("predicted_price", 0) or 0),
            "forecast_directional_accuracy": float(summary.get("forecast_directional_accuracy", 0) or 0),
            "forecast_rmse": float(summary.get("forecast_rmse", 0) or 0),
            "indicator_score": float(summary.get("indicator_score", 0) or 0),
            "final_strategy_score": float(summary.get("final_strategy_score", 0) or 0),
            "final_strategy_signal": summary.get("final_strategy_signal", "n/a"),
        },
        "engine_summary": {
            "primary_model": summary.get("forecast_model", summary.get("selected_primary_model", summary.get("selected_backend_model", "n/a"))),
            "primary_model_display_name": resolve_strategy_display_name(
                "market_futures_backend",
                summary.get("forecast_model", summary.get("selected_primary_model", summary.get("selected_backend_model", "n/a"))),
                "enhanced_market_futures",
            ),
            "engine_label": resolve_engine_label(
                summary.get("forecast_model", summary.get("selected_primary_model", summary.get("selected_backend_model", "n/a"))),
                "market_futures_backend",
                "enhanced_market_futures",
            ),
            "policy_name": summary.get("policy_name", "n/a"),
            "probability_mode": summary.get("probability_mode", "n/a"),
            "calibration_temperature": float(summary.get("calibration_temperature", 0) or 0),
            "backtest_summary": summary.get("strategy_signal_summary", summary.get("backtest_summary", summary.get("backend_summary", "No backtest summary is available yet."))),
            "lead_family_label": family_governance.get("lead_family_label", "Mixed evidence"),
            "family_recommendation": family_governance.get("recommended_family", "mixed_evidence_keep_both_visible"),
            "family_evidence_summary": family_governance.get("evidence_summary", ""),
        },
        "multimodal_context": {
            "best_strategy": summary.get("multimodal_best_strategy", "n/a"),
            "context_label": summary.get("latest_multimodal_context_label", "n/a"),
            "latest_signal": summary.get("multimodal_latest_signal", "n/a"),
            "selected_context_variant": summary.get("multimodal_selected_context_variant", "n/a"),
            "test_macro_f1": float(summary.get("multimodal_test_macro_f1", 0) or 0),
            "validation_macro_f1": float(summary.get("multimodal_validation_macro_f1", 0) or 0),
            "effective_sentiment_source": summary.get("latest_effective_sentiment_source", "n/a"),
            "effective_sentiment_label": summary.get("latest_effective_sentiment_label", "n/a"),
            "gdelt_regime": "Not Available" if gdelt_unavailable else summary.get("latest_gdelt_regime_label", "n/a"),
            "gdelt_article_count": None if gdelt_unavailable else float(summary.get("latest_gdelt_article_count", 0) or 0),
            "gdelt_dominant_event_theme": None if gdelt_unavailable else summary.get("latest_gdelt_dominant_event_theme", summary.get("gdelt_dominant_event_theme")),
            "gdelt_risk_event_theme": None if gdelt_unavailable else summary.get("latest_gdelt_risk_event_theme", summary.get("gdelt_risk_event_theme")),
            "gdelt_supportive_event_theme": None if gdelt_unavailable else summary.get("latest_gdelt_supportive_event_theme", summary.get("gdelt_supportive_event_theme")),
            "onchain_primary_support_driver": summary.get("latest_onchain_primary_support_driver"),
            "onchain_primary_risk_driver": summary.get("latest_onchain_primary_risk_driver"),
            "onchain_participation_breadth_score": float(summary.get("latest_onchain_participation_breadth_score", 0) or 0),
            "onchain_structural_fragility_score": float(summary.get("latest_onchain_structural_fragility_score", 0) or 0),
            "defi_context": build_defi_context_payload(summary),
            "detail": summary.get("multimodal_summary", "Multimodal evaluation summary not generated yet."),
            "assessment": multimodal_assessment,
        },
        "ablation_study": {
            "best_variant": summary.get("ablation_best_variant", "n/a"),
            "best_macro_f1": float(summary.get("ablation_best_macro_f1", 0) or 0),
            "market_futures_macro_f1": float(summary.get("ablation_market_futures_macro_f1", 0) or 0),
            "delta_macro_f1": float(summary.get("delta_macro_f1_vs_market_futures", 0) or 0),
            "detail": summary.get("ablation_summary", "Ablation summary not generated yet."),
        },
    }


def build_strategy_catalog_detail_payload(preset_id, symbol="BTCUSDT", requested_timeframe="4h"):
    summaries, summary_bundle = get_merged_market_summaries(requested_timeframe)
    summary = summaries.get(symbol, {})
    preset = get_preset_strategy(preset_id) or get_preset_strategy("recommended")
    resolved_config = resolve_preset_strategy_config(
        preset.get("id", "recommended"),
        symbol,
        market_summary=summary,
        requested_timeframe=requested_timeframe,
        resolved_timeframe=summary_bundle["timeframe"] or requested_timeframe,
    )
    runtime_support = get_runtime_support()["assets"].get(symbol, {})
    context_assessment = build_context_layer_assessment(summary, runtime_support)
    family_governance = build_family_governance(summary, "market_futures_backend")
    context_overlay_snapshot = get_context_overlay_snapshot(
        symbol,
        resolved_config.get("resolved_timeframe", requested_timeframe),
    )

    return {
        "preset": preset,
        "resolved_config": resolved_config,
        "asset": symbol,
        "requested_timeframe": requested_timeframe,
        "resolved_timeframe": resolved_config.get("resolved_timeframe", requested_timeframe),
        "family_governance": family_governance,
        "context_assessment": context_assessment,
        "context_overlay_entry": context_overlay_snapshot.get("entry"),
        "context_overlay_role": context_overlay_snapshot.get("recommended_role"),
        "context_overlay_action": context_overlay_snapshot.get("recommended_action"),
        "daily_structural_fit": {
            "fit": resolved_config.get("daily_confirmation_fit"),
            "posture": resolved_config.get("daily_posture_label"),
            "label": resolved_config.get("daily_structural_label"),
            "confidence": resolved_config.get("daily_structural_confidence"),
            "primary_support_driver": summary.get("latest_onchain_primary_support_driver"),
            "primary_risk_driver": summary.get("latest_onchain_primary_risk_driver"),
        },
        "strategy_page_brief": {
            "name": resolved_config.get("strategy_name"),
            "tagline": preset.get("tagline", ""),
            "best_for": resolved_config.get("best_for", preset.get("best_for", "")),
            "explanation": preset.get("explanation", ""),
            "evaluation_basis_note": resolved_config.get("evaluation_basis_note", ""),
            "timeframe_note": resolved_config.get("timeframe_note", ""),
        },
    }


def load_sentiment_summary():
    # load the latest broad market mood summary if it exists
    if not SENTIMENT_SUMMARY_PATH.exists():
        return {}

    with open(SENTIMENT_SUMMARY_PATH, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            return {}
        return rows[0]


def load_strategy_registry():
    # load the user-facing strategy preset and custom builder schema
    return get_strategy_registry()


def classify_model_family(model_name):
    normalized = str(model_name or "").lower()
    if "lstm" in normalized:
        return "LSTM sequence model"
    if "forest" in normalized:
        return "Tree ensemble"
    if "logistic" in normalized:
        return "Logistic classifier"
    if "gradient" in normalized:
        return "Gradient boosting"
    if "rule" in normalized:
        return "Rule benchmark"
    if "linear" in normalized or "ridge" in normalized:
        return "Regression forecast"
    return "Model"


def load_model_metric_rows(symbol=None, timeframe=None):
    pattern = "*evaluation_metrics*.csv"
    rows_by_key = {}
    for metrics_path in PROCESSED_DIR.glob(pattern):
        name = metrics_path.name
        if symbol and not name.startswith(f"{symbol}_"):
            continue
        if timeframe and f"_{timeframe}_" not in name:
            continue
        with open(metrics_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["model_name", "symbol", "timeframe", "metric", "value"]:
                continue
            for row in reader:
                row_symbol = str(row.get("symbol", "")).strip().upper()
                row_timeframe = str(row.get("timeframe", "")).strip()
                model_name = row.get("model_name", "n/a")
                key = (row_symbol, row_timeframe, model_name)
                entry = rows_by_key.setdefault(
                    key,
                    {
                        "symbol": row_symbol,
                        "timeframe": row_timeframe,
                        "model_name": model_name,
                        "model_family": classify_model_family(model_name),
                        "metric_source": metrics_path.name,
                        "updated_at": datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat(),
                    },
                )
                metric = row.get("metric")
                if metric:
                    entry[metric] = safe_float(row.get("value"))

    rows = list(rows_by_key.values())
    rows.sort(
        key=lambda item: (
            item.get("symbol", ""),
            item.get("timeframe", ""),
            -safe_float(item.get("macro_f1")),
            -safe_float(item.get("accuracy")),
            item.get("model_name", ""),
        )
    )
    return rows


def build_evaluation_evidence_payload(requested_timeframe="4h"):
    requested_timeframe = requested_timeframe if requested_timeframe in {"1h", "4h"} else "4h"
    _, market_symbols, _, _ = build_market_context(requested_timeframe)
    strategy_keys = [
        "recommended",
        "conservative_trend",
        "momentum_breakout",
        "futures_crowd_reversal",
        "multimodal_balanced",
        "daily_structural_confirmation",
    ]

    strategy_rows = []
    for symbol in market_symbols:
        for strategy_key in strategy_keys:
            decision = build_strategy_decision_payload(symbol, strategy_key, requested_timeframe)
            strategy_rows.append(
                {
                    "symbol": symbol,
                    "timeframe": decision.get("resolved_timeframe"),
                    "strategy_key": strategy_key,
                    "strategy_name": decision.get("strategy_display_name"),
                    "signal": decision.get("current_signal"),
                    "score": safe_float(decision.get("final_strategy_score")),
                    "predicted_return": safe_float(decision.get("predicted_return")),
                    "predicted_price": safe_float(decision.get("predicted_price")),
                    "accuracy": safe_float(decision.get("held_out_accuracy")),
                    "macro_f1": safe_float(decision.get("macro_f1")),
                    "policy_return": safe_float(decision.get("policy_return")),
                    "sharpe_ratio": safe_float(decision.get("sharpe_ratio")),
                    "walkforward_folds": safe_float(decision.get("walkforward_fold_count")),
                    "walkforward_accuracy": safe_float(decision.get("walkforward_avg_accuracy")),
                }
            )

    model_rows = load_model_metric_rows(timeframe=requested_timeframe)
    deep_learning_rows = [
        row for row in model_rows
        if "lstm" in str(row.get("model_name", "")).lower()
    ]
    best_deep_learning_by_asset = {}
    for row in deep_learning_rows:
        symbol = row.get("symbol")
        current = best_deep_learning_by_asset.get(symbol)
        candidate_score = (safe_float(row.get("macro_f1")), safe_float(row.get("accuracy")))
        current_score = (
            safe_float(current.get("macro_f1")),
            safe_float(current.get("accuracy")),
        ) if current else (-1, -1)
        if candidate_score > current_score:
            best_deep_learning_by_asset[symbol] = row

    export_rows = []
    for row in strategy_rows:
        export_rows.append(
            {
                "row_type": "strategy",
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "name": row["strategy_name"],
                "family": row["strategy_key"],
                "signal": row["signal"],
                "score": row["score"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "policy_return": row["policy_return"],
                "sharpe_ratio": row["sharpe_ratio"],
            }
        )
    for row in model_rows:
        export_rows.append(
            {
                "row_type": "model",
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "name": row.get("model_name"),
                "family": row.get("model_family"),
                "signal": "",
                "score": "",
                "accuracy": safe_float(row.get("accuracy")),
                "macro_f1": safe_float(row.get("macro_f1")),
                "policy_return": "",
                "sharpe_ratio": "",
            }
        )

    return {
        "timeframe": requested_timeframe,
        "asset_count": len(market_symbols),
        "strategy_rows": strategy_rows,
        "model_rows": model_rows,
        "deep_learning_rows": deep_learning_rows,
        "best_deep_learning_by_asset": best_deep_learning_by_asset,
        "export_rows": export_rows,
        "summary": (
            f"Evidence bundle covers {len(market_symbols)} assets on {requested_timeframe}, "
            f"{len(strategy_rows)} strategy decisions, {len(model_rows)} evaluated model rows, "
            f"and {len(deep_learning_rows)} LSTM/deep-learning model rows."
        ),
    }


@app.context_processor
def inject_demo_context():
    current_user = get_current_user()
    return {
        "current_user": current_user,
        "demo_user": get_demo_user(),
        "account_email": current_user.email if current_user else None,
        "demo_mode": False,
        "today_label": datetime.now().strftime("%A, %B %d, %Y"),
    }


def require_demo_access():
    if not get_current_user():
        return redirect(url_for("login"))
    return None


# Page routes start here. Each one loads the relevant CSVs, builds a payload, and renders a template. None of the heavy lifting happens in this file.
@app.route("/")
@app.route("/login", methods=["GET"])
def login():
    if get_current_user():
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/signup")
def signup():
    if get_current_user():
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/login", methods=["POST"])
def login_user():
    email = normalize_email(request.form.get("email"))
    password = request.form.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password):
        flash("Invalid email or password.", "error")
        return redirect(url_for("login"))

    session.clear()
    session["user_id"] = user.id
    user.last_login_at = datetime.utcnow()
    db.session.commit()
    return redirect(url_for("dashboard"))


@app.route("/signup_user", methods=["POST"])
def signup_user():
    full_name = (request.form.get("full_name") or "").strip()
    email = normalize_email(request.form.get("email"))
    username = normalize_username(request.form.get("username"))
    password = request.form.get("password") or ""

    if not full_name or not email or not username or not password:
        flash("Please complete all signup fields.", "error")
        return redirect(url_for("signup"))

    if len(password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("signup"))

    if User.query.filter_by(email=email).first():
        flash("An account already exists for that email.", "error")
        return redirect(url_for("signup"))

    if User.query.filter_by(username=username).first():
        flash("That username is already taken.", "error")
        return redirect(url_for("signup"))

    user = User(email=email, username=username, full_name=full_name[:120])
    user.set_password(password)
    user.alert_preferences = AlertPreference()
    db.session.add(user)
    db.session.commit()

    session.clear()
    session["user_id"] = user.id
    return redirect(url_for("dashboard"))


@app.route("/start-here")
def start_here():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    market_summaries, market_symbols, selected_symbol, selected_summary = build_market_context()
    return render_template(
        "start_here.html",
        active_page="dashboard",
        market_summaries=market_summaries,
        market_symbols=market_symbols,
        selected_symbol=selected_symbol,
        selected_summary=selected_summary,
    )


@app.route("/dashboard")
def dashboard():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    market_summaries, market_symbols, selected_symbol, selected_summary = build_market_context()
    sentiment_summary = load_sentiment_summary()
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        market_summaries=market_summaries,
        market_symbols=market_symbols,
        selected_symbol=selected_symbol,
        selected_summary=selected_summary,
        sentiment_summary=sentiment_summary,
    )


@app.route("/markets")
def markets():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    market_summaries, market_symbols, selected_symbol, selected_summary = build_market_context()
    return render_template(
        "markets.html",
        active_page="markets",
        market_summaries=market_summaries,
        market_symbols=market_symbols,
        selected_symbol=selected_symbol,
        selected_summary=selected_summary,
    )


@app.route("/strategies")
def strategies():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    market_summaries, market_symbols, selected_symbol, selected_summary = build_market_context()
    strategy_registry = load_strategy_registry()
    return render_template(
        "strategies.html",
        active_page="strategies",
        market_summaries=market_summaries,
        market_symbols=market_symbols,
        selected_symbol=selected_symbol,
        selected_summary=selected_summary,
        strategy_registry=strategy_registry,
    )


@app.route("/strategies/<preset_id>")
def strategy_detail(preset_id):
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    market_summaries, market_symbols, selected_symbol, selected_summary = build_market_context()
    requested_timeframe = (request.args.get("timeframe") or "4h").strip()
    asset = (request.args.get("asset") or selected_symbol or "BTCUSDT").strip().upper()
    detail_payload = build_strategy_catalog_detail_payload(preset_id, asset, requested_timeframe)
    return render_template(
        "strategy_detail.html",
        active_page="strategies",
        market_summaries=market_summaries,
        market_symbols=market_symbols,
        selected_symbol=asset,
        selected_summary=selected_summary,
        preset_id=preset_id,
        strategy_detail_payload=detail_payload,
    )


# JSON endpoints below. The page JavaScript calls these so charts and tables can refresh without a full page reload.
@app.route("/api/strategy-config", methods=["POST"])
def strategy_config_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    payload = request.get_json(silent=True) or {}
    requested_timeframe = (payload.get("timeframe") or "4h").strip()
    market_summaries, summary_bundle = get_merged_market_summaries(requested_timeframe)

    asset = payload.get("asset") or "BTCUSDT"
    market_summary = market_summaries.get(asset, {})
    mode = payload.get("mode", "custom")

    if mode == "preset":
        config = resolve_preset_strategy_config(
            payload.get("preset_id", "recommended"),
            asset,
            market_summary=market_summary,
            requested_timeframe=requested_timeframe,
            resolved_timeframe=summary_bundle["timeframe"] or requested_timeframe,
        )
    else:
        config = resolve_custom_strategy_config(
            payload.get("selection", {}),
            asset,
            market_summary=market_summary,
            requested_timeframe=requested_timeframe,
            resolved_timeframe=summary_bundle["timeframe"] or requested_timeframe,
        )

    config["data_lane"] = "decision"
    config["requested_timeframe"] = requested_timeframe
    config["resolved_timeframe"] = summary_bundle["timeframe"] or requested_timeframe
    config["refreshed_at"] = (
        datetime.fromtimestamp(summary_bundle["path"].stat().st_mtime).isoformat()
        if summary_bundle["path"]
        else None
    )
    return jsonify(config)


@app.route("/api/strategy-catalog/<preset_id>")
def strategy_catalog_detail_api(preset_id):
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    asset = (request.args.get("asset") or "BTCUSDT").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    return jsonify(build_strategy_catalog_detail_payload(preset_id, asset, timeframe))


@app.route("/api/market-snapshot")
def market_snapshot_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    asset = (request.args.get("asset") or "BTCUSDT").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    return jsonify(build_market_display_payload(asset, timeframe))


@app.route("/api/market-chart")
def market_chart_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    asset = (request.args.get("asset") or "BTCUSDT").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    points = max(12, min(int(request.args.get("points", 48) or 48), 120))
    return jsonify(load_recent_market_chart(asset, timeframe, max_points=points))


@app.route("/api/backtest-curve")
def backtest_curve_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    asset = (request.args.get("asset") or "BTCUSDT").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    variant = (request.args.get("variant") or "binary").strip().lower()
    if variant not in {"binary", "tri"}:
        variant = "binary"
    return jsonify(build_backtest_view_payload(asset, timeframe, variant))


@app.route("/api/cross-asset-metrics")
def cross_asset_metrics_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    timeframe = (request.args.get("timeframe") or "4h").strip()
    if timeframe not in ("1h", "4h", "1d"):
        timeframe = "4h"
    return jsonify(build_cross_asset_comparison(timeframe))


@app.route("/api/significance-summary")
def significance_summary_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    timeframe = (request.args.get("timeframe") or "4h").strip()
    if timeframe not in ("1h", "4h", "1d"):
        timeframe = "4h"
    return jsonify(load_significance_summary(timeframe))


@app.route("/api/dashboard-overview")
def dashboard_overview_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    asset = (request.args.get("asset") or "BTCUSDT").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    return jsonify(build_dashboard_payload(asset, timeframe))


@app.route("/api/strategy-decision")
def strategy_decision_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    asset = (request.args.get("asset") or "BTCUSDT").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    strategy_key = (request.args.get("strategy") or "market_futures_backend").strip()
    return jsonify(build_strategy_decision_payload(asset, strategy_key, timeframe))


@app.route("/api/analytics-summary")
def analytics_summary_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    asset = (request.args.get("asset") or "BTCUSDT").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    payload = cached_payload(
        f"analytics:{asset}:{timeframe}",
        lambda: build_analytics_payload(asset, timeframe),
    )
    return jsonify(payload)


@app.route("/api/evaluation-evidence")
def evaluation_evidence_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    timeframe = (request.args.get("timeframe") or "4h").strip()
    payload = cached_payload(
        f"evidence:{timeframe}",
        lambda: build_evaluation_evidence_payload(timeframe),
    )
    if request.args.get("format") == "csv":
        fieldnames = [
            "row_type",
            "symbol",
            "timeframe",
            "name",
            "family",
            "signal",
            "score",
            "accuracy",
            "macro_f1",
            "policy_return",
            "sharpe_ratio",
        ]
        lines = [",".join(fieldnames)]
        for row in payload["export_rows"]:
            values = [str(row.get(field, "")).replace(",", " ") for field in fieldnames]
            lines.append(",".join(values))
        return Response(
            "\n".join(lines),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=livestrat_evaluation_evidence_{payload['timeframe']}.csv"},
        )
    return jsonify(payload)


@app.route("/api/system-blueprint")
def system_blueprint_api():
    return jsonify(build_system_blueprint())


@app.route("/api/runtime-support")
def runtime_support_api():
    return jsonify(get_runtime_support())


@app.route("/api/pipeline-refresh")
def pipeline_refresh_api():
    timeframe = (request.args.get("timeframe") or "").strip()
    snapshot = get_pipeline_refresh()
    if timeframe:
        entry = snapshot.get("timeframes", {}).get(timeframe, {})
        return jsonify(
            {
                "timeframe": timeframe,
                "entry": entry,
                "guidance": build_pipeline_refresh_guidance(entry),
                "overall_label": snapshot.get("overall_label", "unknown"),
                "updated_at": snapshot.get("updated_at"),
            }
        )
    guided = {
        key: {
            **value,
            "guidance": build_pipeline_refresh_guidance(value),
        }
        for key, value in snapshot.get("timeframes", {}).items()
    }
    return jsonify(
        {
            **snapshot,
            "timeframes": guided,
        }
    )


@app.route("/api/live-market-check")
def live_market_check_api():
    asset = (request.args.get("asset") or "BTCUSDT").strip().upper()
    if asset not in get_all_symbols():
        return jsonify({"error": f"{asset} is not in the supported LiveStrat asset universe."}), 400

    query = urlencode({"symbol": asset})
    request_url = f"https://api.binance.com/api/v3/ticker/24hr?{query}"
    request_headers = {"User-Agent": "LiveStrat submission app; public Binance spot ticker check"}

    try:
        with urlopen(Request(request_url, headers=request_headers), timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return jsonify({"error": f"Binance returned HTTP {error.code} for {asset}."}), 502
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        return jsonify({"error": f"Latest market check failed for {asset}: {error}"}), 502

    return jsonify(
        {
            "asset": asset,
            "source": "Binance spot 24h ticker",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "latest_price": safe_float(payload.get("lastPrice")),
            "price_change_pct_24h": safe_float(payload.get("priceChangePercent")),
            "high_24h": safe_float(payload.get("highPrice")),
            "low_24h": safe_float(payload.get("lowPrice")),
            "base_volume_24h": safe_float(payload.get("volume")),
            "quote_volume_24h": safe_float(payload.get("quoteVolume")),
            "note": (
                "This is a lightweight live market check only. "
                "It updates the latest displayed market fact, not the saved ML evaluation outputs."
            ),
        }
    )


@app.route("/api/transaction-cost-sensitivity")
def transaction_cost_sensitivity_api():
    """Return the most recent transaction-cost sensitivity sweep."""
    timeframe = (request.args.get("timeframe") or "4h").strip()
    asset = (request.args.get("asset") or "").strip().upper()

    pattern = f"market_futures_cost_sensitivity_summary_{timeframe}*.csv"
    matches = list(PROCESSED_DIR.glob(pattern))
    if not matches:
        return jsonify({"available": False, "timeframe": timeframe, "rows": []})
    latest = max(matches, key=lambda p: p.stat().st_mtime)

    df = pd.read_csv(latest)
    if asset:
        df = df[df["symbol"].astype(str).str.upper() == asset]
    return jsonify({
        "available": True,
        "timeframe": timeframe,
        "source_file": latest.name,
        "rows": df.to_dict(orient="records"),
    })


@app.route("/api/evaluation-coverage")
def evaluation_coverage_api():
    asset = (request.args.get("asset") or "").strip().upper()
    coverage = get_evaluation_coverage()
    if asset:
        return jsonify({
            "asset": asset,
            "coverage": coverage["assets"].get(asset, {}),
            "families": coverage["families"],
        })
    return jsonify(coverage)


@app.route("/api/family-governance")
def family_governance_api():
    asset = (request.args.get("asset") or "").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    snapshot = get_family_governance_snapshot(asset, timeframe)
    matrix = snapshot.get("matrix", {})

    if asset:
        return jsonify(
            {
                "asset": asset,
                "timeframe": timeframe,
                "entry": snapshot.get("entry"),
                "live_governance": snapshot.get("live_governance"),
                "asset_summary": snapshot.get("asset_summary", {}),
                "timeframe_summary": snapshot.get("timeframe_summary", {}),
                "recommended_actions": snapshot.get("recommended_actions", []),
            }
        )

    return jsonify(
        {
            "timeframe": timeframe,
            "summary": snapshot.get("timeframe_summary", {}),
            "assets": matrix.get("assets", {}),
            "recommended_actions": [
                action for action in matrix.get("recommended_actions", [])
                if action.get("timeframe") == timeframe
            ],
        }
    )


@app.route("/api/context-overlays")
def context_overlays_api():
    asset = (request.args.get("asset") or "").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    snapshot = get_context_overlay_snapshot(asset, timeframe)

    if asset:
        return jsonify(snapshot)

    return jsonify(
        {
            "timeframe": timeframe,
            "assets": load_context_overlay_comparison_summaries(timeframe),
        }
    )


@app.route("/api/cross-asset-relative-strength")
def cross_asset_relative_strength_api():
    asset = (request.args.get("asset") or "").strip().upper()
    timeframe = (request.args.get("timeframe") or "4h").strip()
    rows = load_cross_asset_relative_strength_summaries(timeframe)

    if asset:
        return jsonify(
            {
                "asset": asset,
                "timeframe": timeframe,
                "entry": rows.get(asset, {}),
                "universe": rows,
            }
        )

    return jsonify(
        {
            "timeframe": timeframe,
            "assets": rows,
            "strategy_role": "Ranks supported assets so the system can compare opportunities instead of forecasting each coin in isolation.",
        }
    )


@app.route("/api/timeframe-readiness")
def timeframe_readiness_api():
    asset = (request.args.get("asset") or "").strip().upper()
    readiness = get_timeframe_readiness()
    if asset:
        return jsonify(
            {
                "asset": asset,
                "timeframes": readiness["assets"].get(asset, {}).get("timeframes", {}),
                "global": readiness["timeframes"],
                "recommended_next_runs": readiness["recommended_next_runs"],
            }
        )
    return jsonify(readiness)


@app.route("/api/strategy-family-evidence")
def strategy_family_evidence_api():
    asset = (request.args.get("asset") or "").strip().upper()
    timeframe = (request.args.get("timeframe") or "").strip().lower()
    evidence = get_strategy_family_evidence()

    rows = evidence["rows"]
    if asset:
        rows = [row for row in rows if row["asset"] == asset]
    if timeframe:
        rows = [row for row in rows if row["timeframe"] == timeframe]

    return jsonify(
        {
            "rows": rows,
            "families": evidence["families"],
            "rubric_alignment": evidence["rubric_alignment"],
            "filters": {
                "asset": asset or "all",
                "timeframe": timeframe or "all",
            },
        }
    )


@app.route("/api/strategy-family-scope")
def strategy_family_scope_api():
    asset = (request.args.get("asset") or "").strip().upper()
    family = (request.args.get("family") or "").strip()
    scope = get_strategy_family_scope()

    if asset:
        scope = {
            **scope,
            "selected_asset": scope["assets"].get(asset, {}),
        }
    if family:
        scope = {
            **scope,
            "selected_family": scope["families"].get(family, {}),
        }

    return jsonify(scope)


@app.route("/api/data-strategy-health")
def data_strategy_health_api():
    return jsonify(get_data_strategy_health())


@app.route("/api/backend-readiness")
def backend_readiness_api():
    return jsonify(get_backend_readiness_report())


@app.route("/api/source-governance")
def source_governance_api():
    return jsonify(get_context_source_governance())


@app.route("/api/strategy-profiles", methods=["GET", "POST"])
def strategy_profiles_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    current_user = get_current_user()

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        config = payload.get("config") or {}
        name = (payload.get("name") or config.get("strategy_name") or "Untitled strategy").strip()
        asset = (payload.get("asset") or config.get("asset") or "BTCUSDT").strip().upper()
        profile_tag = (payload.get("tag") or "research").strip()

        profile = SavedStrategyProfile(
            user_id=current_user.id,
            name=name[:120],
            asset=asset[:20],
            profile_tag=profile_tag[:80],
            config_json=config,
        )
        db.session.add(profile)
        db.session.commit()

        return jsonify({"profile": serialize_strategy_profile(profile)}), 201

    profiles = (
        SavedStrategyProfile.query.filter_by(user_id=current_user.id)
        .order_by(SavedStrategyProfile.updated_at.desc())
        .all()
    )
    return jsonify({"profiles": [serialize_strategy_profile(profile) for profile in profiles]})


@app.route("/api/strategy-profiles/<int:profile_id>", methods=["PUT", "DELETE"])
def strategy_profile_detail_api(profile_id):
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    current_user = get_current_user()
    profile = SavedStrategyProfile.query.filter_by(
        id=profile_id,
        user_id=current_user.id,
    ).first_or_404()

    if request.method == "DELETE":
        db.session.delete(profile)
        db.session.commit()
        return jsonify({"deleted": True, "profile_id": str(profile_id)})

    payload = request.get_json(silent=True) or {}
    if "name" in payload:
        name = (payload.get("name") or "").strip()
        profile.name = (name or profile.name)[:120]
    if "tag" in payload:
        profile.profile_tag = (payload.get("tag") or "research").strip()[:80]

    db.session.commit()
    return jsonify({"profile": serialize_strategy_profile(profile)})


@app.route("/api/alerts", methods=["POST"])
def alerts_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    payload = request.get_json(silent=True) or {}
    market_summaries, _, selected_symbol, _ = build_market_context()
    asset = payload.get("asset") or selected_symbol or "BTCUSDT"
    strategy_key = (payload.get("strategy") or "recommended").strip()
    timeframe = (payload.get("timeframe") or "4h").strip()
    preferences = payload.get("preferences", {})
    current_user = get_current_user()
    if not preferences and current_user and current_user.alert_preferences:
        preferences = current_user.alert_preferences.to_alert_engine_preferences()
    market_summary = build_alert_strategy_summary(asset, strategy_key, timeframe)

    return jsonify(build_alert_response(asset, market_summary, preferences))


@app.route("/api/alert-preferences", methods=["GET", "POST"])
def alert_preferences_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    current_user = get_current_user()
    preferences = get_or_create_alert_preferences(current_user)

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        raw_preferences = payload.get("preferences", {})
        preferences.price_alerts = bool(raw_preferences.get("price"))
        preferences.strategy_alerts = bool(raw_preferences.get("strategy"))
        preferences.sentiment_alerts = bool(raw_preferences.get("sentiment"))
        preferences.onchain_alerts = bool(raw_preferences.get("onchain"))
        preferences.volatility_alerts = bool(raw_preferences.get("volatility"))
        preferences.telegram_enabled = bool(raw_preferences.get("telegram_enabled"))
        telegram_chat_id = (raw_preferences.get("telegram_chat_id") or "").strip()
        preferences.telegram_chat_id = telegram_chat_id[:120] if telegram_chat_id else None
        db.session.commit()

    return jsonify(
        {
            "user_id": current_user.id,
            "preferences": alert_preferences_payload(preferences),
            "telegram": telegram_status_payload(preferences),
        }
    )


@app.route("/api/telegram-status")
def telegram_status_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    current_user = get_current_user()
    preferences = get_or_create_alert_preferences(current_user)
    return jsonify(
        {
            "user_id": current_user.id,
            "telegram": telegram_status_payload(preferences),
        }
    )


@app.route("/api/notification-events", methods=["GET", "POST"])
def notification_events_api():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    current_user = get_current_user()

    if request.method == "POST":
        try:
            payload = request.get_json(silent=True) or {}
            asset = (payload.get("asset") or "BTCUSDT").strip().upper()
            strategy_key = (payload.get("strategy") or "recommended").strip()
            timeframe = (payload.get("timeframe") or "4h").strip()
            market_summary = build_alert_strategy_summary(asset, strategy_key, timeframe)
            preferences = payload.get("preferences", {})
            saved_preferences = get_or_create_alert_preferences(current_user)
            if not preferences and current_user.alert_preferences:
                preferences = current_user.alert_preferences.to_alert_engine_preferences()
            alert_payload = build_alert_response(asset, market_summary, preferences)
            persisted = persist_alert_events_for_user(current_user, alert_payload.get("alerts", []))
            telegram_attempts = dispatch_telegram_ready_messages(persisted)
            return jsonify(
                {
                    "symbol": asset,
                    "saved_count": len(persisted["events"]),
                    "events": persisted["events"],
                    "telegram_ready_messages": persisted["telegram_ready_messages"],
                    "telegram_delivery_attempts": telegram_attempts,
                }
            ), 201
        except Exception as error:  # pragma: no cover - runtime diagnostics path
            db.session.rollback()
            return jsonify(
                {
                    "error": "notification_event_save_failed",
                    "detail": str(error),
                }
            ), 500

    events = (
        NotificationEvent.query.filter_by(user_id=current_user.id)
        .order_by(NotificationEvent.created_at.desc())
        .limit(40)
        .all()
    )
    return jsonify({"events": [serialize_notification_event(event) for event in events]})


@app.route("/api/notification-events/<int:event_id>/read", methods=["POST"])
def notification_event_read_api(event_id):
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response

    current_user = get_current_user()
    event = NotificationEvent.query.filter_by(id=event_id, user_id=current_user.id).first_or_404()
    if event.read_at is None:
        event.read_at = datetime.utcnow()
        db.session.commit()
    return jsonify({"event": serialize_notification_event(event)})


@app.route("/analytics")
def analytics():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    market_summaries, market_symbols, selected_symbol, selected_summary = build_market_context()
    return render_template(
        "analytics.html",
        active_page="analytics",
        market_summaries=market_summaries,
        market_symbols=market_symbols,
        selected_symbol=selected_symbol,
        selected_summary=selected_summary,
    )


@app.route("/notifications")
def notifications():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    market_summaries, market_symbols, selected_symbol, selected_summary = build_market_context()
    sentiment_summary = load_sentiment_summary()
    alert_preferences = alert_preferences_payload(get_or_create_alert_preferences(get_current_user()))
    return render_template(
        "notifications.html",
        active_page="notifications",
        market_summaries=market_summaries,
        market_symbols=market_symbols,
        selected_symbol=selected_symbol,
        selected_summary=selected_summary,
        sentiment_summary=sentiment_summary,
        alert_preferences=alert_preferences,
    )


@app.route("/account")
def account():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    market_summaries, market_symbols, selected_symbol, selected_summary = build_market_context()
    current_user = get_current_user()
    saved_strategy_profiles = (
        SavedStrategyProfile.query.filter_by(user_id=current_user.id)
        .order_by(SavedStrategyProfile.updated_at.desc())
        .limit(5)
        .all()
    )
    return render_template(
        "account.html",
        active_page="account",
        market_summaries=market_summaries,
        market_symbols=market_symbols,
        selected_symbol=selected_symbol,
        selected_summary=selected_summary,
        saved_strategy_profiles=saved_strategy_profiles,
    )


@app.route("/logout")
def logout():
    redirect_response = require_demo_access()
    if redirect_response:
        return redirect_response
    return render_template("logout_confirm.html", active_page="account")


@app.route("/logout_confirmed")
def logout_confirmed():
    session.clear()
    return redirect(url_for("login"))


@app.cli.command("init-db")
def init_db_command():
    db.create_all()
    print("LiveStrat database tables created.")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
