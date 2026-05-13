"""Retrieve Binance futures public market data for richer market-structure analysis."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    FUTURES_BASE_URL,
    FUTURES_CONTRACT_TYPE,
    FUTURES_RATIO_LIMIT,
    FUTURES_RECENT_LOOKBACK_DAYS,
    get_all_symbols,
    get_raw_futures_basis_path,
    get_raw_futures_funding_rate_path,
    get_raw_futures_long_short_ratio_path,
    get_raw_futures_mark_price_path,
    get_raw_futures_open_interest_path,
    get_raw_futures_taker_volume_path,
)
from src.io_paths import FUTURES_LOGS_DIR, ensure_dirs


MARK_PRICE_KLINES_URL = f"{FUTURES_BASE_URL}/fapi/v1/markPriceKlines"
FUNDING_RATE_URL = f"{FUTURES_BASE_URL}/fapi/v1/fundingRate"
OPEN_INTEREST_HIST_URL = f"{FUTURES_BASE_URL}/futures/data/openInterestHist"
LONG_SHORT_RATIO_URL = f"{FUTURES_BASE_URL}/futures/data/globalLongShortAccountRatio"
TAKER_VOLUME_URL = f"{FUTURES_BASE_URL}/futures/data/takerlongshortRatio"
BASIS_URL = f"{FUTURES_BASE_URL}/futures/data/basis"

MARK_PRICE_LIMIT = 1500
FUNDING_RATE_LIMIT = 1000

SYMBOL = DEFAULT_SYMBOL
TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
RUN_ALL_SYMBOLS = False

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def to_milliseconds(date_str, end_of_day=False):
    """Convert YYYY-MM-DD into a UTC timestamp in milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def interval_to_milliseconds(interval):
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"unsupported interval: {interval}")
    return INTERVAL_TO_MS[interval]


def request_json(url, params, timeout=30):
    """Issue one GET request to Binance and return JSON."""
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_mark_price_klines(symbol, interval, start_ms, end_ms):
    """Fetch full mark-price kline history for the requested range."""
    interval_ms = interval_to_milliseconds(interval)
    cursor = start_ms
    rows = []

    while cursor <= end_ms:
        batch = request_json(
            MARK_PRICE_KLINES_URL,
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": MARK_PRICE_LIMIT,
            },
            timeout=60,
        )
        if not batch:
            break

        rows.extend(batch)
        next_cursor = batch[-1][0] + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    return rows


def build_mark_price_dataframe(raw_rows):
    """Convert raw mark-price kline rows into a typed dataframe."""
    df = pd.DataFrame(
        raw_rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "ignore_0",
            "close_time",
            "ignore_1",
            "number_of_trades",
            "ignore_2",
            "ignore_3",
            "ignore_4",
        ],
    )
    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    numeric_columns = ["open", "high", "low", "close", "number_of_trades"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    keep_columns = ["open_time", "open", "high", "low", "close", "close_time", "number_of_trades"]
    return df[keep_columns].sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)


def fetch_funding_rate_history(symbol, start_ms, end_ms):
    """Fetch full funding-rate history for the requested range."""
    cursor = start_ms
    rows = []

    while cursor <= end_ms:
        batch = request_json(
            FUNDING_RATE_URL,
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": FUNDING_RATE_LIMIT,
            },
            timeout=60,
        )
        if not batch:
            break

        rows.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    return rows


def build_funding_rate_dataframe(raw_rows):
    """Convert funding-rate rows into a typed dataframe."""
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return df

    df["funding_time"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df.get("markPrice"), errors="coerce")
    keep_columns = ["funding_time", "funding_rate", "mark_price"]
    return df[keep_columns].sort_values("funding_time").drop_duplicates(subset=["funding_time"]).reset_index(drop=True)


def clamp_recent_window(start_ms, end_ms):
    """Respect Binance's recent-history limit for some public futures endpoints."""
    now = datetime.now(timezone.utc)
    min_allowed = int((now - timedelta(days=FUTURES_RECENT_LOOKBACK_DAYS)).timestamp() * 1000)
    max_allowed = int(now.timestamp() * 1000)

    effective_end = min(end_ms, max_allowed)
    effective_start = max(start_ms, min_allowed)
    return effective_start, effective_end


def fetch_recent_futures_series(url, symbol, period, start_ms, end_ms,
                                extra_params=None, symbol_param_name="symbol"):
    """Fetch one of Binance's recent-only futures market-structure datasets."""
    effective_start, effective_end = clamp_recent_window(start_ms, end_ms)
    if effective_start > effective_end:
        return [], effective_start, effective_end

    params = {
        "period": period,
        "startTime": effective_start,
        "endTime": effective_end,
        "limit": FUTURES_RATIO_LIMIT,
    }
    if symbol_param_name:
        params[symbol_param_name] = symbol
    if extra_params:
        params.update(extra_params)

    cursor = effective_start
    rows = []
    while cursor <= effective_end:
        params["startTime"] = cursor
        batch = request_json(url, params, timeout=60)
        if not batch:
            break

        rows.extend(batch)
        next_cursor = int(batch[-1]["timestamp"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    return rows, effective_start, effective_end


def fetch_recent_futures_series_safe(url, symbol, period, start_ms, end_ms,
                                     extra_params=None, symbol_param_name="symbol"):
    """Fetch a recent-only futures dataset, returning an empty result on endpoint failure."""
    try:
        return fetch_recent_futures_series(
            url,
            symbol,
            period,
            start_ms,
            end_ms,
            extra_params=extra_params,
            symbol_param_name=symbol_param_name,
        )
    except requests.RequestException as exc:
        print(f"warning: failed to fetch recent futures dataset from {url} for {symbol}: {exc}")
        return [], None, None


def build_recent_series_dataframe(raw_rows, rename_map):
    """Convert recent-only market-structure rows into a typed dataframe."""
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return df

    df = df.rename(columns=rename_map)
    df["event_time"] = pd.to_datetime(pd.to_numeric(df["event_time"]), unit="ms", utc=True)
    for column in df.columns:
        if column in {"event_time", "symbol"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values("event_time").drop_duplicates(subset=["event_time"]).reset_index(drop=True)


def audit_timeframe_dataframe(df, timestamp_column, expected_ms=None):
    """Create a small audit summary for one time-indexed dataframe."""
    audit = {
        "rows_retrieved": int(len(df)),
        "missing_numeric_values": 0,
        "duplicate_timestamps": 0,
        "is_time_sorted": True,
    }
    if df.empty:
        audit["first_timestamp"] = "n/a"
        audit["last_timestamp"] = "n/a"
        audit["missing_time_gaps"] = 0
        return audit

    audit["first_timestamp"] = str(df[timestamp_column].min())
    audit["last_timestamp"] = str(df[timestamp_column].max())
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    audit["missing_numeric_values"] = int(df[numeric_columns].isnull().sum().sum()) if numeric_columns else 0
    audit["duplicate_timestamps"] = int(df[timestamp_column].duplicated().sum())
    audit["is_time_sorted"] = bool(df[timestamp_column].is_monotonic_increasing)

    if expected_ms is None:
        audit["missing_time_gaps"] = "not_checked"
    else:
        expected_diff = pd.Timedelta(milliseconds=expected_ms)
        diffs = df[timestamp_column].diff().dropna()
        audit["missing_time_gaps"] = int((diffs != expected_diff).sum())

    return audit


def run_binance_futures_audit_for_symbol(symbol, timeframe, start_date, end_date,
                                         contract_type=FUTURES_CONTRACT_TYPE):
    """Retrieve futures market-structure datasets for one symbol."""
    ensure_dirs()
    start_ms = to_milliseconds(start_date)
    end_ms = to_milliseconds(end_date, end_of_day=True)

    mark_price_rows = fetch_mark_price_klines(symbol, timeframe, start_ms, end_ms)
    mark_price_df = build_mark_price_dataframe(mark_price_rows)
    mark_price_path = get_raw_futures_mark_price_path(symbol, timeframe, start_date, end_date)
    mark_price_df.to_csv(mark_price_path, index=False)

    funding_rows = fetch_funding_rate_history(symbol, start_ms, end_ms)
    funding_df = build_funding_rate_dataframe(funding_rows)
    funding_path = get_raw_futures_funding_rate_path(symbol, start_date, end_date)
    funding_df.to_csv(funding_path, index=False)

    open_interest_rows, oi_start, oi_end = fetch_recent_futures_series_safe(
        OPEN_INTEREST_HIST_URL,
        symbol,
        timeframe,
        start_ms,
        end_ms,
    )
    open_interest_df = build_recent_series_dataframe(
        open_interest_rows,
        {
            "timestamp": "event_time",
            "sumOpenInterest": "open_interest",
            "sumOpenInterestValue": "open_interest_value",
            "CMCCirculatingSupply": "circulating_supply",
        },
    )
    open_interest_path = get_raw_futures_open_interest_path(symbol, timeframe, start_date, end_date)
    open_interest_df.to_csv(open_interest_path, index=False)

    long_short_rows, ls_start, ls_end = fetch_recent_futures_series_safe(
        LONG_SHORT_RATIO_URL,
        symbol,
        timeframe,
        start_ms,
        end_ms,
    )
    long_short_df = build_recent_series_dataframe(
        long_short_rows,
        {
            "timestamp": "event_time",
            "longShortRatio": "long_short_ratio",
            "longAccount": "long_account_share",
            "shortAccount": "short_account_share",
        },
    )
    long_short_path = get_raw_futures_long_short_ratio_path(symbol, timeframe, start_date, end_date)
    long_short_df.to_csv(long_short_path, index=False)

    taker_rows, taker_start, taker_end = fetch_recent_futures_series_safe(
        TAKER_VOLUME_URL,
        symbol,
        timeframe,
        start_ms,
        end_ms,
    )
    taker_df = build_recent_series_dataframe(
        taker_rows,
        {
            "timestamp": "event_time",
            "buySellRatio": "taker_buy_sell_ratio",
            "buyVol": "taker_buy_volume",
            "sellVol": "taker_sell_volume",
        },
    )
    taker_path = get_raw_futures_taker_volume_path(symbol, timeframe, start_date, end_date)
    taker_df.to_csv(taker_path, index=False)

    basis_rows, basis_start, basis_end = fetch_recent_futures_series_safe(
        BASIS_URL,
        symbol,
        timeframe,
        start_ms,
        end_ms,
        extra_params={"contractType": contract_type},
        symbol_param_name="pair",
    )
    basis_df = build_recent_series_dataframe(
        basis_rows,
        {
            "timestamp": "event_time",
            "basisRate": "basis_rate",
            "futuresPrice": "futures_price",
            "indexPrice": "index_price",
            "annualizedBasisRate": "annualized_basis_rate",
            "basis": "basis_value",
            "pair": "symbol",
        },
    )
    basis_path = get_raw_futures_basis_path(
        symbol,
        timeframe,
        start_date,
        end_date,
        contract_type=contract_type,
    )
    basis_df.to_csv(basis_path, index=False)

    audit_payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "contract_type": contract_type,
        "requested_start": str(pd.to_datetime(start_ms, unit="ms", utc=True)),
        "requested_end": str(pd.to_datetime(end_ms, unit="ms", utc=True)),
        "saved_files": {
            "mark_price": str(mark_price_path),
            "funding_rate": str(funding_path),
            "open_interest": str(open_interest_path),
            "long_short_ratio": str(long_short_path),
            "taker_volume": str(taker_path),
            "basis": str(basis_path),
        },
        "datasets": {
            "mark_price": audit_timeframe_dataframe(
                mark_price_df,
                "open_time",
                expected_ms=interval_to_milliseconds(timeframe),
            ),
            "funding_rate": audit_timeframe_dataframe(funding_df, "funding_time"),
            "open_interest": audit_timeframe_dataframe(open_interest_df, "event_time"),
            "long_short_ratio": audit_timeframe_dataframe(long_short_df, "event_time"),
            "taker_volume": audit_timeframe_dataframe(taker_df, "event_time"),
            "basis": audit_timeframe_dataframe(basis_df, "event_time"),
        },
        "recent_window_effective_ranges": {
            "open_interest": {
                "effective_start": str(pd.to_datetime(oi_start, unit="ms", utc=True)) if oi_start is not None else "n/a",
                "effective_end": str(pd.to_datetime(oi_end, unit="ms", utc=True)) if oi_end is not None else "n/a",
            },
            "long_short_ratio": {
                "effective_start": str(pd.to_datetime(ls_start, unit="ms", utc=True)) if ls_start is not None else "n/a",
                "effective_end": str(pd.to_datetime(ls_end, unit="ms", utc=True)) if ls_end is not None else "n/a",
            },
            "taker_volume": {
                "effective_start": str(pd.to_datetime(taker_start, unit="ms", utc=True)) if taker_start is not None else "n/a",
                "effective_end": str(pd.to_datetime(taker_end, unit="ms", utc=True)) if taker_end is not None else "n/a",
            },
            "basis": {
                "effective_start": str(pd.to_datetime(basis_start, unit="ms", utc=True)) if basis_start is not None else "n/a",
                "effective_end": str(pd.to_datetime(basis_end, unit="ms", utc=True)) if basis_end is not None else "n/a",
            },
        },
    }

    log_path = FUTURES_LOGS_DIR / (
        f"audit_binance_futures_{symbol}_{timeframe}_{start_date}_{end_date}.json"
    )
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2)

    print(f"binance futures audit completed for {symbol} on {timeframe}")
    print(f"mark-price rows: {len(mark_price_df)}")
    print(f"funding-rate rows: {len(funding_df)}")
    print(f"open-interest rows: {len(open_interest_df)}")
    print(f"long/short rows: {len(long_short_df)}")
    print(f"taker-volume rows: {len(taker_df)}")
    print(f"basis rows: {len(basis_df)}")
    print(f"log saved to: {log_path}")

    return {
        "mark_price": mark_price_df,
        "funding_rate": funding_df,
        "open_interest": open_interest_df,
        "long_short_ratio": long_short_df,
        "taker_volume": taker_df,
        "basis": basis_df,
    }


def run_binance_futures_audit():
    """Retrieve futures market data for the configured symbol set."""
    symbols = get_all_symbols() if RUN_ALL_SYMBOLS else [SYMBOL]
    outputs = {}
    for symbol in symbols:
        outputs[symbol] = run_binance_futures_audit_for_symbol(
            symbol=symbol,
            timeframe=TIMEFRAME,
            start_date=START_DATE,
            end_date=END_DATE,
        )
    return outputs


if __name__ == "__main__":
    run_binance_futures_audit()
