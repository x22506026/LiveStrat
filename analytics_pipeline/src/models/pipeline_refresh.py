"""Shared refresh manifest and currentness snapshot for LiveStrat pipelines."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    SUPPORTED_TIMEFRAMES,
    get_market_intelligence_refresh_manifest_path,
)
from src.io_paths import PROCESSED_DIR


FRESHNESS_LIMITS = {
    "1h": {"current": 1, "recent": 3},
    "4h": {"current": 1, "recent": 3},
    "1d": {"current": 2, "recent": 7},
}

OVERVIEW_FILE_RE = re.compile(
    r"^market_intelligence_overview_(?P<timeframe>1h|4h|1d)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.csv$"
)


def _utc_now():
    return datetime.now(timezone.utc)


def _parse_iso_datetime(value):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_date(value):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _default_manifest():
    return {
        "schema_version": 1,
        "updated_at": None,
        "timeframes": {},
    }


def _merge_bool(existing_entry, new_entry, key, default=False):
    return bool(existing_entry.get(key, default)) or bool(new_entry.get(key, default))


def _normalise_manifest_entry(entry):
    if not isinstance(entry, dict):
        return {}

    normalised = dict(entry)
    status = str(normalised.get("status", "missing") or "missing")
    refreshed_at = normalised.get("refreshed_at")

    if "include_core_suite" not in normalised:
        normalised["include_core_suite"] = status == "completed" and (
            bool(normalised.get("include_strategy_suite")) or int(normalised.get("row_count", 0) or 0) > 0
        )
    if "include_context_suite" not in normalised:
        normalised["include_context_suite"] = status == "completed" and bool(normalised.get("include_gdelt_context"))

    if normalised.get("include_core_suite") and not normalised.get("core_refreshed_at"):
        normalised["core_refreshed_at"] = refreshed_at
    if normalised.get("include_strategy_suite") and not normalised.get("strategy_refreshed_at"):
        normalised["strategy_refreshed_at"] = refreshed_at
    if normalised.get("include_context_suite") and not normalised.get("context_refreshed_at"):
        normalised["context_refreshed_at"] = refreshed_at
    if normalised.get("include_gdelt_context") and normalised.get("include_context_suite") and not normalised.get("gdelt_context_refreshed_at"):
        normalised["gdelt_context_refreshed_at"] = refreshed_at

    return normalised


def load_pipeline_refresh_manifest(processed_dir=None):
    processed_dir = Path(processed_dir or PROCESSED_DIR)
    manifest_path = processed_dir / get_market_intelligence_refresh_manifest_path().name
    if not manifest_path.exists():
        return _default_manifest()

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return _default_manifest()

    if not isinstance(manifest, dict):
        return _default_manifest()
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("updated_at", None)
    manifest.setdefault("timeframes", {})
    manifest["timeframes"] = {
        timeframe: _normalise_manifest_entry(entry)
        for timeframe, entry in manifest["timeframes"].items()
    }
    return manifest


def write_pipeline_refresh_manifest(
    timeframe,
    start_date,
    end_date,
    status,
    row_count=0,
    include_strategy_suite=True,
    include_gdelt_context=True,
    error=None,
    processed_dir=None,
    extra=None,
):
    processed_dir = Path(processed_dir or PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / get_market_intelligence_refresh_manifest_path().name
    manifest = load_pipeline_refresh_manifest(processed_dir)
    existing_entry = dict(manifest.get("timeframes", {}).get(timeframe, {}))

    refreshed_at = _utc_now().isoformat()
    entry = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "status": status,
        "row_count": int(row_count or 0),
        "include_strategy_suite": bool(include_strategy_suite),
        "include_gdelt_context": bool(include_gdelt_context),
        "refreshed_at": refreshed_at,
        "error": str(error) if error else None,
    }
    if extra:
        entry.update(extra)

    if status == "completed":
        core_ran = bool(entry.get("include_core_suite", False))
        context_ran = bool(entry.get("include_context_suite", False))
        strategy_ran = bool(include_strategy_suite and core_ran)
        gdelt_ran = bool(include_gdelt_context and context_ran)

        entry["include_core_suite"] = _merge_bool(existing_entry, entry, "include_core_suite", False)
        entry["include_strategy_suite"] = _merge_bool(existing_entry, entry, "include_strategy_suite", False)
        entry["include_gdelt_context"] = _merge_bool(existing_entry, entry, "include_gdelt_context", False)
        entry["include_context_suite"] = _merge_bool(existing_entry, entry, "include_context_suite", False)

        entry["core_refreshed_at"] = refreshed_at if core_ran else existing_entry.get("core_refreshed_at")
        entry["strategy_refreshed_at"] = refreshed_at if strategy_ran else existing_entry.get("strategy_refreshed_at")
        entry["context_refreshed_at"] = refreshed_at if context_ran else existing_entry.get("context_refreshed_at")
        entry["gdelt_context_refreshed_at"] = (
            refreshed_at if gdelt_ran else existing_entry.get("gdelt_context_refreshed_at")
        )
    else:
        for key in (
            "include_core_suite",
            "include_strategy_suite",
            "include_gdelt_context",
            "include_context_suite",
            "core_refreshed_at",
            "strategy_refreshed_at",
            "context_refreshed_at",
            "gdelt_context_refreshed_at",
        ):
            if key in existing_entry and key not in entry:
                entry[key] = existing_entry.get(key)

    manifest["updated_at"] = refreshed_at
    manifest["timeframes"][timeframe] = entry

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest_path


def _classify_freshness(timeframe, end_date, refreshed_at, now_utc):
    limits = FRESHNESS_LIMITS.get(timeframe, {"current": 2, "recent": 7})
    if end_date is None:
        return "unknown", None, "No pipeline window end date is recorded yet."

    age_days = max((now_utc.date() - end_date).days, 0)
    if age_days <= limits["current"]:
        label = "current"
    elif age_days <= limits["recent"]:
        label = "recent"
    else:
        label = "stale"

    refreshed_note = ""
    if refreshed_at is not None:
        refreshed_hours = max((now_utc - refreshed_at).total_seconds() / 3600.0, 0.0)
        refreshed_note = f" The latest refresh completed about {refreshed_hours:.1f} hours ago."

    summary = (
        f"{timeframe} covers data through {end_date.isoformat()} and is currently marked as {label}."
        f"{refreshed_note}"
    )
    return label, age_days, summary


def build_pipeline_refresh_snapshot(processed_dir=None, reference_now=None):
    processed_dir = Path(processed_dir or PROCESSED_DIR)
    manifest = load_pipeline_refresh_manifest(processed_dir)
    now_utc = reference_now or _utc_now()

    timeframe_entries = {}
    freshness_labels = []

    for timeframe in SUPPORTED_TIMEFRAMES:
        entry = dict(manifest.get("timeframes", {}).get(timeframe, {}))
        refreshed_at = _parse_iso_datetime(entry.get("refreshed_at"))
        end_date = _parse_iso_date(entry.get("end_date"))
        start_date = _parse_iso_date(entry.get("start_date"))
        freshness_label, age_days, freshness_summary = _classify_freshness(
            timeframe,
            end_date,
            refreshed_at,
            now_utc,
        )

        latest_overview_matches = sorted(
            processed_dir.glob(f"market_intelligence_overview_{timeframe}_*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        latest_overview = latest_overview_matches[0] if latest_overview_matches else None

        if latest_overview is not None:
            filename_match = OVERVIEW_FILE_RE.match(latest_overview.name)
            if end_date is None and filename_match:
                end_date = _parse_iso_date(filename_match.group("end"))
            if start_date is None and filename_match:
                start_date = _parse_iso_date(filename_match.group("start"))
            if refreshed_at is None:
                refreshed_at = datetime.fromtimestamp(latest_overview.stat().st_mtime, tz=timezone.utc)
            freshness_label, age_days, freshness_summary = _classify_freshness(
                timeframe,
                end_date,
                refreshed_at,
                now_utc,
            )

        status = entry.get("status", "missing")
        has_outputs = latest_overview is not None
        if status == "completed" and not has_outputs:
            status = "completed_no_overview"
        elif status == "missing" and has_outputs:
            status = "outputs_present_manifest_missing"
        elif status == "failed":
            freshness_label = "failed"
            freshness_summary = (
                f"{timeframe} attempted a refresh ending on "
                f"{end_date.isoformat() if end_date else 'an unknown date'}, but the pipeline failed and should not be treated as current."
            )

        timeframe_entry = {
            "timeframe": timeframe,
            "status": status,
            "freshness_label": freshness_label,
            "window_start": start_date.isoformat() if start_date else entry.get("start_date"),
            "window_end": end_date.isoformat() if end_date else entry.get("end_date"),
            "window_age_days": age_days,
            "refreshed_at": refreshed_at.isoformat() if refreshed_at else entry.get("refreshed_at"),
            "row_count": int(entry.get("row_count", 0) or 0),
            "include_core_suite": bool(entry.get("include_core_suite", False)),
            "include_strategy_suite": bool(entry.get("include_strategy_suite", False)),
            "include_gdelt_context": bool(entry.get("include_gdelt_context", False)),
            "include_context_suite": bool(entry.get("include_context_suite", False)),
            "resume_existing": bool(entry.get("resume_existing", False)),
            "core_refreshed_at": entry.get("core_refreshed_at"),
            "strategy_refreshed_at": entry.get("strategy_refreshed_at"),
            "context_refreshed_at": entry.get("context_refreshed_at"),
            "gdelt_context_refreshed_at": entry.get("gdelt_context_refreshed_at"),
            "has_overview_output": has_outputs,
            "latest_overview_path": str(latest_overview) if latest_overview else None,
            "error": entry.get("error"),
            "summary": freshness_summary,
        }
        timeframe_entries[timeframe] = timeframe_entry
        freshness_labels.append(freshness_label)

    if any(label == "failed" for label in freshness_labels):
        overall_label = "failed"
    elif all(label == "current" for label in freshness_labels):
        overall_label = "current"
    elif any(label == "current" for label in freshness_labels) or any(label == "recent" for label in freshness_labels):
        overall_label = "mixed"
    elif any(label == "stale" for label in freshness_labels):
        overall_label = "stale"
    else:
        overall_label = "unknown"

    return {
        "updated_at": manifest.get("updated_at"),
        "overall_label": overall_label,
        "timeframes": timeframe_entries,
    }


def build_pipeline_refresh_guidance(refresh_entry):
    refresh_entry = refresh_entry or {}
    freshness_label = str(refresh_entry.get("freshness_label", "unknown") or "unknown")
    status = str(refresh_entry.get("status", "missing") or "missing")
    timeframe = str(refresh_entry.get("timeframe", "n/a") or "n/a")
    window_end = refresh_entry.get("window_end")
    age_days = refresh_entry.get("window_age_days")

    if status == "failed":
        return {
            "label": "refresh_failed",
            "trust_mode": "reduced_trust",
            "headline": f"{timeframe} refresh most recently failed, so saved outputs should be treated cautiously.",
            "recommended_action": "rerun_pipeline",
        }
    if status in {"missing", "completed_no_overview"}:
        return {
            "label": "refresh_missing",
            "trust_mode": "reduced_trust",
            "headline": f"{timeframe} does not currently have a complete refresh snapshot, so user-facing confidence should stay conservative.",
            "recommended_action": "generate_missing_outputs",
        }
    if freshness_label == "stale":
        age_note = f" through {window_end}" if window_end else ""
        days_note = f" ({age_days:.0f} days old)" if isinstance(age_days, (int, float)) and age_days is not None else ""
        return {
            "label": "stale_window",
            "trust_mode": "reduced_trust",
            "headline": f"{timeframe} still relies on a stale evaluation window{age_note}{days_note}, so live confidence should be softened.",
            "recommended_action": "refresh_recent_window",
        }
    if freshness_label == "recent":
        return {
            "label": "recent_window",
            "trust_mode": "normal",
            "headline": f"{timeframe} is reasonably recent, but it is not fully current yet.",
            "recommended_action": "refresh_when_convenient",
        }
    if freshness_label == "current":
        return {
            "label": "current_window",
            "trust_mode": "normal",
            "headline": f"{timeframe} is backed by a current evaluation window.",
            "recommended_action": "monitor_only",
        }

    return {
        "label": "unknown_window",
        "trust_mode": "reduced_trust",
        "headline": f"{timeframe} does not have enough refresh metadata to justify strong user-facing confidence.",
        "recommended_action": "inspect_pipeline_state",
    }
