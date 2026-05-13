"""Data-to-strategy health checks for LiveStrat refresh decisions."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import GDELT_ASSET_QUERY_MAP, get_all_symbols

DEFAULT_SYMBOLS = tuple(get_all_symbols())
EXPANSION_SYMBOLS = ("BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
GDELT_ASSETS = set(GDELT_ASSET_QUERY_MAP)
ASSET_CHUNK_DAYS = {"BTC": 7, "ETH": 5, "SOL": 3, "BNB": 5, "XRP": 5, "ADA": 5, "DOGE": 5}
GDELT_MIN_REQUEST_INTERVAL_SECONDS = 8.0


SOURCE_ROLES = {
    "spot_market": {
        "label": "Binance spot market",
        "role": "price, volatility, volume, trend, labels",
        "strategy_families": [
            "market_trend_benchmark",
            "market_futures_core",
            "market_futures_binary",
            "cross_asset_relative_strength",
        ],
        "current_days": 2,
        "recent_days": 5,
    },
    "futures_structure": {
        "label": "Binance futures structure",
        "role": "funding, open interest, positioning, taker flow, basis",
        "strategy_families": ["market_futures_core", "market_futures_binary"],
        "current_days": 2,
        "recent_days": 5,
    },
    "broad_sentiment": {
        "label": "Fear & Greed sentiment",
        "role": "market-wide mood fallback",
        "strategy_families": ["context_confirmation", "multimodal_context"],
        "current_days": 3,
        "recent_days": 10,
    },
    "asset_news": {
        "label": "Asset news sentiment",
        "role": "event risk and confirmation context",
        "strategy_families": ["context_confirmation", "multimodal_context", "structural_break_governance"],
        "current_days": 2,
        "recent_days": 5,
    },
    "onchain_daily": {
        "label": "Coin Metrics on-chain",
        "role": "daily structural confirmation",
        "strategy_families": ["daily_structural_confirmation", "context_confirmation", "structural_break_governance"],
        "current_days": 3,
        "recent_days": 10,
    },
    "defi_ecosystem": {
        "label": "DeFiLlama ecosystem TVL",
        "role": "chain-level ecosystem activity and structural context",
        "strategy_families": ["context_confirmation", "multimodal_context", "structural_break_governance"],
        "current_days": 3,
        "recent_days": 10,
    },
}


def _utc_now():
    return datetime.now(timezone.utc)


def _safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def _parse_latest_datetime(df, candidates):
    if df.empty:
        return None
    for column in candidates:
        if column not in df.columns:
            continue
        values = pd.to_datetime(df[column], errors="coerce", utc=True)
        if values.notna().any():
            return values.max().to_pydatetime()
    return None


def _file_latest(paths, datetime_columns):
    latest_data_time = None
    latest_mtime = None
    row_count = 0
    latest_file = None
    files = list(paths)

    for path in files:
        df = _safe_read_csv(path)
        row_count += len(df)
        data_time = _parse_latest_datetime(df, datetime_columns)
        if data_time and (latest_data_time is None or data_time > latest_data_time):
            latest_data_time = data_time
            latest_file = path
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            if latest_file is None:
                latest_file = path

    return {
        "files": len(files),
        "row_count": row_count,
        "latest_data_time": latest_data_time,
        "latest_mtime": latest_mtime,
        "latest_file": latest_file,
    }


def _age_days(dt, now=None):
    if dt is None:
        return None
    now = now or _utc_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max((now - dt.astimezone(timezone.utc)).total_seconds() / 86400.0, 0.0)


def _freshness_status(source_key, latest_dt, now=None):
    if latest_dt is None:
        return "missing"
    limits = SOURCE_ROLES[source_key]
    age = _age_days(latest_dt, now=now)
    if age <= limits["current_days"]:
        return "current"
    if age <= limits["recent_days"]:
        return "recent"
    return "stale"


def _layer_record(source_key, summary, note="", now=None):
    latest_dt = summary.get("latest_data_time") or summary.get("latest_mtime")
    status = _freshness_status(source_key, latest_dt, now=now)
    age = _age_days(latest_dt, now=now)
    source = SOURCE_ROLES[source_key]
    return {
        "source_key": source_key,
        "source_label": source["label"],
        "status": status,
        "age_days": round(age, 2) if age is not None else None,
        "files": summary.get("files", 0),
        "row_count": int(summary.get("row_count", 0) or 0),
        "latest_data_time": latest_dt.isoformat() if latest_dt else None,
        "latest_file": summary.get("latest_file").name if summary.get("latest_file") else None,
        "role": source["role"],
        "strategy_families": source["strategy_families"],
        "note": note,
    }


def _estimate_gdelt_requests(start_date, end_date, symbols):
    if not start_date or not end_date:
        return {"estimated_requests": 0, "estimated_min_seconds": 0, "assets": []}

    start = pd.to_datetime(start_date, utc=True).date()
    end = pd.to_datetime(end_date, utc=True).date()
    days = max((end - start).days + 1, 1)
    assets = [symbol.replace("USDT", "") for symbol in symbols if symbol.replace("USDT", "") in GDELT_ASSETS]
    request_count = 0
    per_asset = []
    for asset in assets:
        chunk_days = ASSET_CHUNK_DAYS.get(asset, 7)
        chunks = (days + chunk_days - 1) // chunk_days
        request_count += chunks
        per_asset.append({"asset": asset, "chunk_days": chunk_days, "estimated_chunks": chunks})
    return {
        "estimated_requests": request_count,
        "estimated_min_seconds": round(request_count * GDELT_MIN_REQUEST_INTERVAL_SECONDS, 1),
        "assets": per_asset,
    }


def _manifest(processed_dir):
    path = Path(processed_dir) / "market_intelligence_refresh_manifest.json"
    if not path.exists():
        return {}
    try:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def build_data_strategy_health_snapshot(project_dir):
    """Return a compact view of whether acquisition supports current strategies."""
    project_dir = Path(project_dir)
    raw_dir = project_dir / "analytics_pipeline" / "data" / "raw"
    processed_dir = project_dir / "analytics_pipeline" / "data" / "processed"
    symbols = DEFAULT_SYMBOLS
    now = _utc_now()

    layers = [
        _layer_record(
            "spot_market",
            _file_latest((raw_dir / "binance").glob("*.csv"), ["open_time", "close_time", "timestamp"]),
            "Core market acquisition is the cleanest and most complete layer.",
            now=now,
        ),
        _layer_record(
            "futures_structure",
            _file_latest((raw_dir / "binance_futures").glob("*.csv"), ["open_time", "timestamp", "time"]),
            "Futures endpoints are recent-only in places, so partial coverage is expected.",
            now=now,
        ),
        _layer_record(
            "broad_sentiment",
            _file_latest((raw_dir / "sentiment").glob("*.csv"), ["timestamp", "window_end_utc"]),
            "Fast to refresh; should not be allowed to remain stale because it is cheap context.",
            now=now,
        ),
        _layer_record(
            "asset_news",
            _file_latest((raw_dir / "gdelt_news").glob("*.csv"), ["seendate", "published_at", "window_end_utc"]),
            "GDELT/RSS can be slow; use targeted context-only refreshes and reuse existing windows when possible.",
            now=now,
        ),
        _layer_record(
            "onchain_daily",
            _file_latest((raw_dir / "coinmetrics").glob("*.csv"), ["window_end_utc", "time"]),
            "Daily structural layer; stale data weakens confirmation and report defensibility.",
            now=now,
        ),
        _layer_record(
            "defi_ecosystem",
            _file_latest((raw_dir / "defillama").glob("*.csv"), ["window_end_utc", "date"]),
            "Free chain-level TVL context; useful for SOL/BNB ecosystem display and confirmation, not a direct wallet-level substitute.",
            now=now,
        ),
    ]

    stale_or_missing = [layer for layer in layers if layer["status"] in {"stale", "missing"}]
    manifest = _manifest(processed_dir)
    timeframes = manifest.get("timeframes", {})
    gdelt_estimates = {
        timeframe: _estimate_gdelt_requests(
            entry.get("start_date"),
            entry.get("end_date"),
            entry.get("symbols") or symbols,
        )
        for timeframe, entry in timeframes.items()
    }

    expansion_assets = []
    for symbol in EXPANSION_SYMBOLS:
        market_feature_files = list(processed_dir.glob(f"{symbol}_*_market_features_*.csv"))
        raw_market_files = list((raw_dir / "binance").glob(f"{symbol}_*.csv"))
        expansion_assets.append(
            {
                "symbol": symbol,
                "tier": "market_first_expansion",
                "market_raw_files": len(raw_market_files),
                "processed_market_feature_files": len(market_feature_files),
                "recommended_status": "ready_to_evaluate" if market_feature_files else "needs_market_pull",
                "strategy_scope": "market/futures and cross-asset ranking first; on-chain only after source validation.",
            }
        )

    recommendations = []
    if any(layer["source_key"] == "broad_sentiment" and layer["status"] == "stale" for layer in layers):
        recommendations.append("Refresh Fear & Greed first; it is cheap and removes an avoidable stale context weakness.")
    if any(layer["source_key"] == "onchain_daily" and layer["status"] == "stale" for layer in layers):
        recommendations.append("Refresh Coin Metrics next, then rebuild on-chain features and daily structural summaries.")
    if any(layer["source_key"] == "asset_news" and layer["status"] != "current" for layer in layers):
        recommendations.append("Treat GDELT as targeted and expensive: run it only for the final selected window/assets unless raw files are missing.")
    recommendations.append("Keep market/futures refreshes separate from context refreshes so acquisition failures do not block core strategy outputs.")

    return {
        "generated_at": now.isoformat(),
        "layers": layers,
        "stale_or_missing_count": len(stale_or_missing),
        "gdelt_refresh_estimates": gdelt_estimates,
        "expansion_assets": expansion_assets,
        "recommendations": recommendations,
        "linkage_summary": (
            "Spot and futures feed the core decision families. Sentiment/news and on-chain should act as confirmation, "
            "veto, and structural context unless their evaluated uplift beats the core benchmark."
        ),
    }
