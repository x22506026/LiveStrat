"""Retrieve asset-specific crypto news from the GDELT DOC 2.0 API."""

import json
import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from requests import HTTPError

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GDELT_ASSET_QUERY_MAP,
    GDELT_DEFAULT_MAX_RECORDS,
    GDELT_DOC_API_BASE,
    get_raw_gdelt_articles_path,
)
from src.io_paths import GDELT_LOGS_DIR, ensure_dirs


MAX_GDELT_LOOKBACK_DAYS = 90
DEFAULT_CHUNK_DAYS = 7
MIN_CHUNK_DAYS = 1
GDELT_MIN_REQUEST_INTERVAL_SECONDS = 8.0
GDELT_RATE_LIMIT_RETRY_SECONDS = 15.0
GDELT_MAX_RETRIES = 5
ASSET_CHUNK_DAYS = {
    "BTC": 7,
    "ETH": 5,
    "SOL": 3,
    "BNB": 5,
    "XRP": 5,
    "ADA": 5,
    "DOGE": 5,
}
ASSET_MAX_RECORDS = {
    "BTC": 250,
    "ETH": 200,
    "SOL": 125,
    "BNB": 125,
    "XRP": 125,
    "ADA": 125,
    "DOGE": 125,
}

_LAST_GDELT_REQUEST_TS = None


def to_gdelt_timestamp(date_string, end_of_day=False):
    """Convert a YYYY-MM-DD date string into the DOC API timestamp format."""
    if end_of_day:
        dt = datetime.strptime(date_string, "%Y-%m-%d").replace(
            hour=23,
            minute=59,
            second=59,
            tzinfo=timezone.utc,
        )
    else:
        dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y%m%d%H%M%S")


def parse_window(date_string, end_of_day=False):
    """Convert a YYYY-MM-DD date string into a timezone-aware datetime."""
    if end_of_day:
        return datetime.strptime(date_string, "%Y-%m-%d").replace(
            hour=23,
            minute=59,
            second=59,
            tzinfo=timezone.utc,
        )
    return datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def validate_recent_window(start_date, end_date):
    """Ensure the requested date window stays within the recent DOC API coverage."""
    start_dt = parse_window(start_date)
    end_dt = parse_window(end_date, end_of_day=True)
    max_age = datetime.now(timezone.utc) - timedelta(days=MAX_GDELT_LOOKBACK_DAYS)
    if start_dt < max_age:
        raise ValueError(
            "GDELT DOC API only supports recent windows. "
            f"Requested start {start_date} is older than {MAX_GDELT_LOOKBACK_DAYS} days."
        )
    if end_dt < start_dt:
        raise ValueError(f"End date {end_date} must not be earlier than start date {start_date}.")
    return start_dt, end_dt


def build_date_chunks(start_date, end_date, chunk_days=DEFAULT_CHUNK_DAYS):
    """Split a requested window into smaller API-friendly chunks."""
    start_dt, end_dt = validate_recent_window(start_date, end_date)
    chunks = []
    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days) - timedelta(seconds=1), end_dt)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(seconds=1)
    return chunks


def throttle_gdelt_requests():
    """Respect the public DOC API pacing guidance."""
    global _LAST_GDELT_REQUEST_TS
    now = time.monotonic()
    if _LAST_GDELT_REQUEST_TS is not None:
        elapsed = now - _LAST_GDELT_REQUEST_TS
        if elapsed < GDELT_MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(GDELT_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _LAST_GDELT_REQUEST_TS = time.monotonic()


def cool_off_after_rate_limit(attempt):
    """Back off more aggressively when GDELT explicitly rate-limits a request."""
    sleep_seconds = GDELT_RATE_LIMIT_RETRY_SECONDS * max(attempt, 1)
    time.sleep(sleep_seconds)


def fetch_gdelt_articles_for_chunk(asset_symbol, chunk_start, chunk_end, max_records=GDELT_DEFAULT_MAX_RECORDS):
    """Fetch one chunk of article-list results for a configured asset query."""
    query = GDELT_ASSET_QUERY_MAP[asset_symbol]
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sort": "datedesc",
        "maxrecords": max_records,
        "STARTDATETIME": chunk_start.strftime("%Y%m%d%H%M%S"),
        "ENDDATETIME": chunk_end.strftime("%Y%m%d%H%M%S"),
    }
    response = None
    for attempt in range(1, GDELT_MAX_RETRIES + 1):
        throttle_gdelt_requests()
        response = requests.get(
            GDELT_DOC_API_BASE,
            params=params,
            timeout=60,
        )
        if response.status_code != 429:
            break
        if attempt == GDELT_MAX_RETRIES:
            break
        cool_off_after_rate_limit(attempt)

    try:
        response.raise_for_status()
    except HTTPError as exc:
        body_preview = response.text[:500].strip()
        raise RuntimeError(
            f"GDELT returned HTTP {response.status_code} for {asset_symbol}. "
            f"Response preview: {body_preview}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        body_preview = response.text[:500].strip()
        raise RuntimeError(
            f"GDELT returned a non-JSON response for {asset_symbol}. "
            f"Response preview: {body_preview}"
        ) from exc

    rows = payload.get("articles", [])
    return rows, query


def split_chunk(chunk_start, chunk_end):
    """Split one chunk into two smaller chunks."""
    midpoint = chunk_start + (chunk_end - chunk_start) / 2
    left_end = midpoint
    right_start = midpoint + timedelta(seconds=1)
    return [
        (chunk_start, left_end),
        (right_start, chunk_end),
    ]


def fetch_chunk_with_fallback(asset_symbol, chunk_start, chunk_end, max_records, min_chunk_days=MIN_CHUNK_DAYS):
    """Fetch one chunk and recursively subdivide it if the request times out."""
    try:
        rows, query_used = fetch_gdelt_articles_for_chunk(
            asset_symbol,
            chunk_start,
            chunk_end,
            max_records=max_records,
        )
        return [(rows, query_used, chunk_start, chunk_end)]
    except requests.exceptions.RequestException as exc:
        chunk_duration_days = max((chunk_end - chunk_start).total_seconds() / 86400.0, 0.0)
        if chunk_duration_days <= min_chunk_days:
            raise RuntimeError(
                f"GDELT request failed for {asset_symbol} even at minimum chunk size. "
                f"Window {chunk_start.isoformat()} to {chunk_end.isoformat()}: {exc}"
            ) from exc

        child_max_records = max(50, math.ceil(max_records / 2))
        child_results = []
        for child_start, child_end in split_chunk(chunk_start, chunk_end):
            child_results.extend(
                fetch_chunk_with_fallback(
                    asset_symbol,
                    child_start,
                    child_end,
                    max_records=child_max_records,
                    min_chunk_days=min_chunk_days,
                )
            )
        return child_results


def normalise_gdelt_articles(rows, asset_symbol, query, chunk_start, chunk_end):
    """Project raw GDELT article rows into a stable schema."""
    if not rows:
        return pd.DataFrame(
            columns=[
                "asset_symbol",
                "query_used",
                "chunk_start_utc",
                "chunk_end_utc",
                "title",
                "url",
                "url_mobile",
                "domain",
                "language",
                "sourcecountry",
                "socialimage",
                "seendate",
            ]
        )

    df = pd.DataFrame(rows)
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], utc=True, errors="coerce")
    else:
        df["seendate"] = pd.NaT

    for column in ["title", "url", "url_mobile", "domain", "language", "sourcecountry", "socialimage"]:
        if column not in df.columns:
            df[column] = pd.NA

    df.insert(0, "chunk_end_utc", chunk_end.isoformat())
    df.insert(0, "chunk_start_utc", chunk_start.isoformat())
    df.insert(0, "query_used", query)
    df.insert(0, "asset_symbol", asset_symbol)
    return df[
        [
            "asset_symbol",
            "query_used",
            "chunk_start_utc",
            "chunk_end_utc",
            "title",
            "url",
            "url_mobile",
            "domain",
            "language",
            "sourcecountry",
            "socialimage",
            "seendate",
        ]
    ]


def deduplicate_articles(df):
    """Remove exact repeated URLs and obvious repeated title/date pairs."""
    if df.empty:
        return df
    deduped = df.copy()
    deduped = deduped.drop_duplicates(subset=["url"], keep="first")
    deduped = deduped.drop_duplicates(subset=["title", "seendate"], keep="first")
    return deduped.reset_index(drop=True)


def write_gdelt_audit_log(asset_symbol, start_date, end_date, output_path, rows_saved, query, unique_domains):
    """Persist a compact audit record for one GDELT pull."""
    log_path = GDELT_LOGS_DIR / f"{asset_symbol}_gdelt_audit_{start_date}_{end_date}.json"
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "asset_symbol": asset_symbol,
                "query_used": query,
                "window_start": start_date,
                "window_end": end_date,
                "rows_saved": rows_saved,
                "unique_domains": unique_domains,
                "output_file": str(output_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
        )
    return log_path


def run_gdelt_audit_for_asset(asset_symbol, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                              chunk_days=DEFAULT_CHUNK_DAYS, max_records=GDELT_DEFAULT_MAX_RECORDS):
    """Retrieve and save recent asset-specific GDELT article data."""
    ensure_dirs()
    if asset_symbol not in GDELT_ASSET_QUERY_MAP:
        raise KeyError(f"No GDELT query configured for asset {asset_symbol}.")

    chunk_frames = []
    query_used = GDELT_ASSET_QUERY_MAP[asset_symbol]
    resolved_chunk_days = ASSET_CHUNK_DAYS.get(asset_symbol, chunk_days)
    resolved_max_records = ASSET_MAX_RECORDS.get(asset_symbol, max_records)
    for chunk_start, chunk_end in build_date_chunks(start_date, end_date, chunk_days=resolved_chunk_days):
        chunk_results = fetch_chunk_with_fallback(
            asset_symbol,
            chunk_start,
            chunk_end,
            max_records=resolved_max_records,
        )
        for rows, query_used, resolved_start, resolved_end in chunk_results:
            chunk_frames.append(
                normalise_gdelt_articles(rows, asset_symbol, query_used, resolved_start, resolved_end)
            )

    raw_df = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
    raw_df = deduplicate_articles(raw_df)

    output_path = get_raw_gdelt_articles_path(asset_symbol, start_date, end_date)
    raw_df.to_csv(output_path, index=False)
    unique_domains = sorted(raw_df["domain"].dropna().astype(str).unique().tolist()) if not raw_df.empty else []
    log_path = write_gdelt_audit_log(
        asset_symbol,
        start_date,
        end_date,
        output_path,
        len(raw_df),
        query_used,
        unique_domains,
    )

    print(f"GDELT audit completed for {asset_symbol}")
    print(f"window: {start_date} to {end_date}")
    print(f"rows saved: {len(raw_df)}")
    print(f"raw data saved to: {output_path}")
    print(f"audit log saved to: {log_path}")
    return raw_df


def run_gdelt_audit_for_assets(asset_symbols, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                               chunk_days=DEFAULT_CHUNK_DAYS, max_records=GDELT_DEFAULT_MAX_RECORDS):
    """Retrieve and save GDELT article data for multiple configured assets."""
    outputs = {}
    failures = {}
    for asset_symbol in asset_symbols:
        try:
            outputs[asset_symbol] = run_gdelt_audit_for_asset(
                asset_symbol,
                start_date=start_date,
                end_date=end_date,
                chunk_days=chunk_days,
                max_records=max_records,
            )
        except Exception as exc:  # pragma: no cover - network reliability is environment-dependent
            failures[asset_symbol] = str(exc)
            print(f"GDELT audit failed for {asset_symbol}")
            print(str(exc))

    if failures:
        print("GDELT audit completed with partial failures")
        print(json.dumps(failures, indent=2))
    return outputs


if __name__ == "__main__":
    run_gdelt_audit_for_assets(["BTC", "ETH", "SOL"])
