"""Retrieve Coin Metrics community asset metrics for supported on-chain assets."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.config import (
    COINMETRICS_CANDIDATE_METRICS,
    COINMETRICS_COMMUNITY_API_BASE,
    ONCHAIN_FREQUENCY,
    ONCHAIN_LOOKBACK_DAYS,
    get_asset_config,
    get_raw_onchain_path,
    get_supported_onchain_assets,
)
from src.io_paths import ONCHAIN_LOGS_DIR, ensure_dirs


CATALOG_URL = f"{COINMETRICS_COMMUNITY_API_BASE}/catalog-v2/asset-metrics"
TIMESERIES_URL = f"{COINMETRICS_COMMUNITY_API_BASE}/timeseries/asset-metrics"


def fetch_asset_metric_catalog(asset_code):
    """Fetch available asset metrics and frequencies for one Coin Metrics asset."""
    response = requests.get(
        CATALOG_URL,
        params={"assets": asset_code},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("data", [])


def pick_supported_metrics(catalog_rows, asset_code, frequency=ONCHAIN_FREQUENCY):
    """Choose only candidate metrics that are available for the asset and frequency."""
    supported = []
    minimum_recent_time = datetime.now(timezone.utc) - timedelta(days=ONCHAIN_LOOKBACK_DAYS)
    for row in catalog_rows:
        if row.get("asset") != asset_code:
            continue

        for metric_row in row.get("metrics", []):
            metric = metric_row.get("metric")
            if metric not in COINMETRICS_CANDIDATE_METRICS:
                continue

            frequencies = metric_row.get("frequencies", [])
            if any(
                item.get("frequency") == frequency
                and _frequency_is_recent_enough(item, minimum_recent_time)
                for item in frequencies
            ):
                supported.append(metric)

    return supported


def _frequency_is_recent_enough(frequency_row, minimum_recent_time):
    max_time = frequency_row.get("max_time")
    if not max_time:
        return False
    try:
        parsed = pd.Timestamp(max_time).to_pydatetime()
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc) >= minimum_recent_time


def fetch_asset_metric_timeseries(asset_code, metrics, frequency=ONCHAIN_FREQUENCY, lookback_days=ONCHAIN_LOOKBACK_DAYS):
    """Fetch daily Coin Metrics community time series for selected metrics."""
    if not metrics:
        return pd.DataFrame()

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)
    response = requests.get(
        TIMESERIES_URL,
        params={
            "assets": asset_code,
            "metrics": ",".join(metrics),
            "frequency": frequency,
            "start_time": start_time.date().isoformat(),
            "end_time": end_time.date().isoformat(),
            "page_size": 10000,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    return pd.DataFrame(data)


def write_onchain_audit_log(asset_symbol, output_path, metrics):
    """Save a small Coin Metrics audit log for the retrieved asset."""
    log_path = ONCHAIN_LOGS_DIR / f"{asset_symbol}_coinmetrics_audit_log.json"
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "asset_symbol": asset_symbol,
                "selected_metrics": metrics,
                "output_file": str(output_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
        )
    return log_path


def run_coinmetrics_audit_for_asset(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    """Retrieve raw community on-chain metrics for one configured asset."""
    ensure_dirs()
    asset_code = get_asset_config(asset_symbol)["coinmetrics_asset"]
    catalog_rows = fetch_asset_metric_catalog(asset_code)
    selected_metrics = pick_supported_metrics(catalog_rows, asset_code, frequency=frequency)
    raw_df = fetch_asset_metric_timeseries(asset_code, selected_metrics, frequency=frequency)

    if not raw_df.empty:
        raw_df = raw_df.rename(columns={"asset": "coinmetrics_asset", "time": "window_end_utc"})
        raw_df.insert(0, "asset_symbol", asset_symbol)

    output_path = get_raw_onchain_path(asset_symbol, frequency=frequency)
    raw_df.to_csv(output_path, index=False)
    log_path = write_onchain_audit_log(asset_symbol, output_path, selected_metrics)

    print(f"Coin Metrics audit completed for {asset_symbol}")
    print(f"selected metrics: {selected_metrics}")
    print(f"rows saved: {len(raw_df)}")
    print(f"raw data saved to: {output_path}")
    print(f"audit log saved to: {log_path}")

    return raw_df


def run_coinmetrics_audit_for_all_supported_assets(frequency=ONCHAIN_FREQUENCY):
    """Retrieve raw community on-chain metrics for all enabled assets."""
    outputs = {}
    for asset_symbol in get_supported_onchain_assets():
        outputs[asset_symbol] = run_coinmetrics_audit_for_asset(asset_symbol, frequency=frequency)
    return outputs


if __name__ == "__main__":
    run_coinmetrics_audit_for_all_supported_assets()
