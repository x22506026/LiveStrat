import requests
import pandas as pd
from datetime import datetime, timezone

from src.io_paths import ensure_dirs, LOGS_DIR
from src.config import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    get_all_symbols,
    get_raw_binance_path,
)


# this file retrieves historical ohlcv data from the binance public api
# it paginates through the full requested period and validates the result

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_LIMIT = 1000

SYMBOL = DEFAULT_SYMBOL
INTERVAL = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
RUN_ALL_SYMBOLS = False
UPDATE_MODE = False

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
    # convert yyyy-mm-dd to unix milliseconds
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def interval_to_milliseconds(interval):
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"unsupported interval: {interval}")
    return INTERVAL_TO_MS[interval]


def fetch_ohlcv_batch(symbol, interval, start_ms, end_ms):
    # fetch one batch of ohlcv rows from binance
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": BINANCE_LIMIT
    }

    response = requests.get(BINANCE_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_full_ohlcv_history(symbol, interval, start_ms, end_ms):
    # paginate until the full requested date range is covered
    interval_ms = interval_to_milliseconds(interval)
    all_rows = []
    cursor = start_ms

    while cursor <= end_ms:
        batch = fetch_ohlcv_batch(symbol, interval, cursor, end_ms)
        if not batch:
            break

        all_rows.extend(batch)

        # move just beyond the last open time to avoid duplicates
        last_open_time = batch[-1][0]
        next_cursor = last_open_time + interval_ms

        if next_cursor <= cursor:
            break

        cursor = next_cursor

    return all_rows


def build_market_dataframe(raw_rows):
    # convert raw api response into a clean dataframe
    df = pd.DataFrame(
        raw_rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore"
        ]
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    return df


def load_existing_market_dataframe(path):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df


def audit_market_dataframe(df, interval, requested_start_ms, requested_end_ms):
    # calculate quality checks for the downloaded dataset
    interval_ms = interval_to_milliseconds(interval)
    expected_diff = pd.Timedelta(milliseconds=interval_ms)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    duplicate_count = int(df["open_time"].duplicated().sum())
    missing_numeric_values = int(df[numeric_cols].isnull().sum().sum())
    diffs = df["open_time"].diff().dropna()
    missing_candle_gaps = int((diffs != expected_diff).sum())

    audit = {
        "rows_retrieved": int(len(df)),
        "first_open_time": str(df["open_time"].min()) if not df.empty else "n/a",
        "last_open_time": str(df["open_time"].max()) if not df.empty else "n/a",
        "requested_start": str(pd.to_datetime(requested_start_ms, unit="ms", utc=True)),
        "requested_end": str(pd.to_datetime(requested_end_ms, unit="ms", utc=True)),
        "missing_numeric_values": missing_numeric_values,
        "duplicate_open_time_rows": duplicate_count,
        "missing_candle_gaps": missing_candle_gaps,
        "is_time_sorted": bool(df["open_time"].is_monotonic_increasing),
    }

    if not df.empty:
        coverage_end = df["open_time"].max() + expected_diff - pd.Timedelta(milliseconds=1)
        audit["covers_requested_start"] = bool(df["open_time"].min() <= pd.to_datetime(requested_start_ms, unit="ms", utc=True))
        audit["covers_requested_end"] = bool(coverage_end >= pd.to_datetime(requested_end_ms, unit="ms", utc=True))
    else:
        audit["covers_requested_start"] = False
        audit["covers_requested_end"] = False

    return audit


def run_binance_audit_for_symbol(symbol, interval, start_date, end_date, update_mode=False):
    start_ms = to_milliseconds(start_date)
    end_ms = to_milliseconds(end_date, end_of_day=True)
    raw_path = get_raw_binance_path(symbol, interval, start_date, end_date)
    existing_df = load_existing_market_dataframe(raw_path) if update_mode else pd.DataFrame()

    request_start_ms = start_ms
    if not existing_df.empty:
        request_start_ms = int(existing_df["open_time"].max().timestamp() * 1000) + interval_to_milliseconds(interval)

    print("starting binance ohlcv audit...")
    print(f"symbol={symbol}, interval={interval}, update_mode={update_mode}")

    if request_start_ms <= end_ms:
        raw_rows = fetch_full_ohlcv_history(symbol, interval, request_start_ms, end_ms)
        fetched_df = build_market_dataframe(raw_rows)
    else:
        fetched_df = pd.DataFrame(columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
            "taker_buy_quote_volume", "ignore"
        ])

    if not existing_df.empty and not fetched_df.empty:
        df = pd.concat([existing_df, fetched_df], ignore_index=True)
        df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    elif not existing_df.empty:
        df = existing_df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    else:
        df = fetched_df

    audit = audit_market_dataframe(df, interval, start_ms, end_ms)

    # save raw data
    output_file = raw_path
    df.to_csv(output_file, index=False)

    # write audit log
    log_file = LOGS_DIR / f"audit_binance_{symbol}_{interval}_{start_date}_{end_date}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("binance market data audit\n")
        f.write(f"symbol: {symbol}\n")
        f.write(f"timeframe: {interval}\n")
        f.write(f"date range: {start_date} to {end_date}\n")
        f.write(f"update mode: {update_mode}\n")
        for key, value in audit.items():
            f.write(f"{key.replace('_', ' ')}: {value}\n")
        f.write(f"raw file saved to: {output_file}\n")

    print("binance audit completed successfully")
    print(f"rows retrieved: {audit['rows_retrieved']}")
    print(f"first open time: {audit['first_open_time']}")
    print(f"last open time: {audit['last_open_time']}")
    print(f"missing candle gaps: {audit['missing_candle_gaps']}")
    print(f"raw data saved to: {output_file}")
    print(f"audit log saved to: {log_file}")


def run_binance_audit():
    # ensure required folders exist
    ensure_dirs()

    symbols = get_all_symbols() if RUN_ALL_SYMBOLS else [SYMBOL]
    for symbol in symbols:
        run_binance_audit_for_symbol(
            symbol=symbol,
            interval=INTERVAL,
            start_date=START_DATE,
            end_date=END_DATE,
            update_mode=UPDATE_MODE,
        )


if __name__ == "__main__":
    run_binance_audit()
