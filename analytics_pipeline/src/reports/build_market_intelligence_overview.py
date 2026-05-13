"""Build one app-facing market intelligence overview per dated pipeline run."""

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import (
    ASSET_REGISTRY,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    get_defillama_summary_path,
    get_market_context_ablation_summary_path,
    get_market_multimodal_strategy_summary_path,
    get_market_futures_backtest_summary_path,
    get_market_futures_signal_summary_path,
    get_market_futures_walkforward_summary_path,
    get_market_intelligence_overview_path,
    get_market_overview_path,
    get_onchain_features_path,
    get_sentiment_summary_path,
)
from src.io_paths import ensure_dirs


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE


def load_csv_or_empty(path):
    """Load a CSV if it exists, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _is_blank_context(value):
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "unavailable"}


def build_primary_summary(row):
    """Create one concise explanation for the app to display."""
    backend_model = row.get("selected_backend_model")
    latest_signal = row.get("latest_signal")
    backend_summary = row.get("backend_summary")
    backtest_summary = row.get("backtest_summary")
    walkforward_summary = row.get("walkforward_summary")
    analysis_summary = row.get("analysis_summary")

    backend_model = _clean_text(backend_model)
    latest_signal = _clean_text(latest_signal)
    backend_summary = _clean_text(backend_summary)
    backtest_summary = _clean_text(backtest_summary)
    walkforward_summary = _clean_text(walkforward_summary)
    analysis_summary = _clean_text(analysis_summary)
    symbol = _clean_text(row.get("symbol")) or "This asset"

    if backend_model:
        return walkforward_summary or backtest_summary or backend_summary or analysis_summary or (
            f"{symbol} currently uses {backend_model} with a {latest_signal or 'hold'} signal."
        )

    return analysis_summary or f"{symbol} currently relies on the market-only baseline view."


def coalesce_suffix_columns(df, base_name):
    """Combine merge-suffixed columns back into one canonical field."""
    left_name = f"{base_name}_x"
    right_name = f"{base_name}_y"
    if left_name in df.columns or right_name in df.columns:
        left_series = df[left_name] if left_name in df.columns else pd.Series(pd.NA, index=df.index)
        right_series = df[right_name] if right_name in df.columns else pd.Series(pd.NA, index=df.index)
        df[base_name] = left_series.combine_first(right_series)
        drop_columns = [column for column in (left_name, right_name) if column in df.columns]
        df.drop(columns=drop_columns, inplace=True)
    return df


def build_latest_onchain_snapshot_df(reference_end_date):
    """Load the latest available on-chain snapshot for each supported asset."""
    rows = []
    reference_timestamp = pd.Timestamp(reference_end_date)
    if reference_timestamp.tzinfo is None:
        reference_timestamp = reference_timestamp.tz_localize("UTC")
    else:
        reference_timestamp = reference_timestamp.tz_convert("UTC")

    for asset_symbol, asset_config in ASSET_REGISTRY.items():
        market_symbol = asset_config["market_symbol"]
        onchain_path = get_onchain_features_path(asset_symbol)

        if not onchain_path.exists():
            rows.append(
                {
                    "symbol": market_symbol,
                    "latest_onchain_snapshot_label": "unavailable",
                    "latest_onchain_snapshot_score": pd.NA,
                    "latest_onchain_snapshot_reason": pd.NA,
                    "latest_onchain_snapshot_date": pd.NA,
                    "latest_onchain_snapshot_age_days": pd.NA,
                    "latest_onchain_snapshot_status": "unavailable",
                }
            )
            continue

        onchain_df = pd.read_csv(onchain_path, parse_dates=["window_end_utc"])
        if onchain_df.empty:
            rows.append(
                {
                    "symbol": market_symbol,
                    "latest_onchain_snapshot_label": "unavailable",
                    "latest_onchain_snapshot_score": pd.NA,
                    "latest_onchain_snapshot_reason": pd.NA,
                    "latest_onchain_snapshot_date": pd.NA,
                    "latest_onchain_snapshot_age_days": pd.NA,
                    "latest_onchain_snapshot_status": "unavailable",
                }
            )
            continue

        if "onchain_data_available" in onchain_df.columns:
            availability_series = onchain_df["onchain_data_available"].astype(str).str.lower().eq("true")
            available_df = onchain_df[availability_series].copy()
        else:
            available_df = onchain_df.copy()

        if available_df.empty:
            rows.append(
                {
                    "symbol": market_symbol,
                    "latest_onchain_snapshot_label": "unavailable",
                    "latest_onchain_snapshot_score": pd.NA,
                    "latest_onchain_snapshot_reason": pd.NA,
                    "latest_onchain_snapshot_date": pd.NA,
                    "latest_onchain_snapshot_age_days": pd.NA,
                    "latest_onchain_snapshot_status": "unavailable",
                }
            )
            continue

        latest_row = available_df.sort_values("window_end_utc").iloc[-1]
        snapshot_timestamp = pd.Timestamp(latest_row["window_end_utc"])
        if snapshot_timestamp.tzinfo is None:
            snapshot_timestamp = snapshot_timestamp.tz_localize("UTC")
        else:
            snapshot_timestamp = snapshot_timestamp.tz_convert("UTC")
        age_days = max((reference_timestamp - snapshot_timestamp).days, 0)
        snapshot_status = "available" if age_days <= 7 else "stale"

        rows.append(
            {
                "symbol": market_symbol,
                "latest_onchain_snapshot_label": latest_row.get("onchain_regime_label", "unavailable") or "unavailable",
                "latest_onchain_snapshot_score": latest_row.get("onchain_regime_score"),
                "latest_onchain_snapshot_reason": latest_row.get("onchain_regime_reason"),
                "latest_onchain_snapshot_date": snapshot_timestamp.isoformat(),
                "latest_onchain_snapshot_age_days": age_days,
                "latest_onchain_snapshot_status": snapshot_status,
            }
        )

    return pd.DataFrame(rows)


def build_latest_gdelt_snapshot_df(reference_end_date, processed_dir):
    """Load the latest available asset-news sentiment snapshot for each supported asset."""
    rows = []
    reference_timestamp = pd.Timestamp(reference_end_date)
    if reference_timestamp.tzinfo is None:
        reference_timestamp = reference_timestamp.tz_localize("UTC")
    else:
        reference_timestamp = reference_timestamp.tz_convert("UTC")

    for asset_symbol, asset_config in ASSET_REGISTRY.items():
        market_symbol = asset_config["market_symbol"]
        candidates = sorted(processed_dir.glob(f"{asset_symbol}_gdelt_sentiment_summary_1d_*.csv"))
        if not candidates:
            rows.append(
                {
                    "symbol": market_symbol,
                    "latest_gdelt_snapshot_date": pd.NA,
                    "latest_gdelt_snapshot_age_days": pd.NA,
                    "latest_gdelt_snapshot_status": "unavailable",
                }
            )
            continue

        gdelt_df = pd.read_csv(candidates[-1])
        if gdelt_df.empty:
            rows.append(
                {
                    "symbol": market_symbol,
                    "latest_gdelt_snapshot_date": pd.NA,
                    "latest_gdelt_snapshot_age_days": pd.NA,
                    "latest_gdelt_snapshot_status": "unavailable",
                }
            )
            continue

        latest_row = gdelt_df.iloc[-1]
        timestamp_value = latest_row.get("latest_window_end")
        if not timestamp_value:
            rows.append(
                {
                    "symbol": market_symbol,
                    "latest_gdelt_snapshot_date": pd.NA,
                    "latest_gdelt_snapshot_age_days": pd.NA,
                    "latest_gdelt_snapshot_status": "unavailable",
                }
            )
            continue

        snapshot_timestamp = pd.Timestamp(timestamp_value)
        if snapshot_timestamp.tzinfo is None:
            snapshot_timestamp = snapshot_timestamp.tz_localize("UTC")
        else:
            snapshot_timestamp = snapshot_timestamp.tz_convert("UTC")
        age_days = max((reference_timestamp - snapshot_timestamp).days, 0)
        if age_days <= 5:
            status = "fresh"
        elif age_days <= 10:
            status = "aging"
        else:
            status = "stale"

        rows.append(
            {
                "symbol": market_symbol,
                "latest_gdelt_snapshot_date": snapshot_timestamp.isoformat(),
                "latest_gdelt_snapshot_age_days": age_days,
                "latest_gdelt_snapshot_status": status,
            }
        )

    return pd.DataFrame(rows)


def build_latest_defi_snapshot_df(reference_end_date):
    """Load the latest DeFiLlama ecosystem context row per asset."""
    summary_df = load_csv_or_empty(get_defillama_summary_path("1d"))
    if summary_df.empty:
        return pd.DataFrame(
            [
                {
                    "symbol": asset_config["market_symbol"],
                    "defi_snapshot_status": "unavailable",
                }
                for asset_config in ASSET_REGISTRY.values()
            ]
        )

    reference_timestamp = pd.Timestamp(reference_end_date)
    if reference_timestamp.tzinfo is None:
        reference_timestamp = reference_timestamp.tz_localize("UTC")
    else:
        reference_timestamp = reference_timestamp.tz_convert("UTC")

    rows = []
    for _, row in summary_df.iterrows():
        timestamp_value = row.get("latest_defi_window_end")
        if not timestamp_value or pd.isna(timestamp_value):
            status = "unavailable"
            age_days = pd.NA
        else:
            snapshot_timestamp = pd.Timestamp(timestamp_value)
            if snapshot_timestamp.tzinfo is None:
                snapshot_timestamp = snapshot_timestamp.tz_localize("UTC")
            else:
                snapshot_timestamp = snapshot_timestamp.tz_convert("UTC")
            age_days = max((reference_timestamp - snapshot_timestamp).days, 0)
            status = "fresh" if age_days <= 3 else ("aging" if age_days <= 10 else "stale")

        rows.append(
            {
                "symbol": row.get("symbol"),
                "defi_chain_name": row.get("chain_name"),
                "latest_defi_window_end": row.get("latest_defi_window_end"),
                "latest_defi_tvl_usd": row.get("latest_defi_tvl_usd"),
                "defi_tvl_change_pct_1d": row.get("defi_tvl_change_pct_1d"),
                "defi_tvl_change_pct_7d": row.get("defi_tvl_change_pct_7d"),
                "defi_tvl_change_pct_30d": row.get("defi_tvl_change_pct_30d"),
                "defi_tvl_zscore_30d": row.get("defi_tvl_zscore_30d"),
                "defi_tvl_drawdown_30d": row.get("defi_tvl_drawdown_30d"),
                "defi_regime_score": row.get("defi_regime_score"),
                "defi_regime_label": row.get("defi_regime_label"),
                "defi_context_available": row.get("defi_context_available"),
                "defi_summary": row.get("defi_summary"),
                "defi_snapshot_age_days": age_days,
                "defi_snapshot_status": status,
            }
        )
    return pd.DataFrame(rows)


def build_asset_spine_df(timeframe, start_date, end_date, source_frames):
    """Use the seven-asset registry as the app-facing overview spine."""
    merge_keys = ["symbol", "timeframe", "window_start", "window_end"]
    registry_rows = [
        {
            "symbol": asset_config["market_symbol"],
            "timeframe": timeframe,
            "window_start": start_date,
            "window_end": end_date,
        }
        for asset_config in ASSET_REGISTRY.values()
        if asset_config.get("support_flags", {}).get("market_supported", False)
    ]
    key_frames = [pd.DataFrame(registry_rows)]
    for frame in source_frames:
        if not frame.empty and all(key in frame.columns for key in merge_keys):
            key_frames.append(frame[merge_keys])

    return pd.concat(key_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def load_latest_broad_sentiment_row():
    """Use the broad market mood summary when asset news is missing or stale."""
    sentiment_summary_df = load_csv_or_empty(get_sentiment_summary_path("1d"))
    if sentiment_summary_df.empty:
        return {}
    return sentiment_summary_df.iloc[-1].to_dict()


def build_market_intelligence_overview(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Merge dated market context and dated backend signal selections into one app-facing table."""
    ensure_dirs()
    market_overview_path = get_market_overview_path(timeframe, start_date, end_date)
    signal_summary_path = get_market_futures_signal_summary_path(timeframe, start_date, end_date)
    backtest_summary_path = get_market_futures_backtest_summary_path(timeframe, start_date, end_date)
    walkforward_summary_path = get_market_futures_walkforward_summary_path(timeframe, start_date, end_date)
    multimodal_summary_path = get_market_multimodal_strategy_summary_path(timeframe, start_date, end_date)
    context_ablation_summary_path = get_market_context_ablation_summary_path(timeframe, start_date, end_date)

    market_df = load_csv_or_empty(market_overview_path)
    signal_df = load_csv_or_empty(signal_summary_path)
    backtest_df = load_csv_or_empty(backtest_summary_path)
    walkforward_df = load_csv_or_empty(walkforward_summary_path)
    multimodal_df = load_csv_or_empty(multimodal_summary_path)
    context_ablation_df = load_csv_or_empty(context_ablation_summary_path)
    onchain_snapshot_df = build_latest_onchain_snapshot_df(end_date)
    gdelt_snapshot_df = build_latest_gdelt_snapshot_df(end_date, multimodal_summary_path.parent)
    defi_snapshot_df = build_latest_defi_snapshot_df(end_date)
    broad_sentiment_row = load_latest_broad_sentiment_row()

    source_frames = [market_df, signal_df, backtest_df, walkforward_df, multimodal_df, context_ablation_df]
    if all(frame.empty for frame in source_frames):
        raise FileNotFoundError(
            f"No market intelligence inputs found for {timeframe} {start_date} to {end_date}."
        )

    merge_keys = ["symbol", "timeframe", "window_start", "window_end"]
    merged_df = build_asset_spine_df(timeframe, start_date, end_date, source_frames)

    for source_df in [market_df, signal_df]:
        if not source_df.empty and all(key in source_df.columns for key in merge_keys):
            merged_df = merged_df.merge(source_df, on=merge_keys, how="left")

    if not backtest_df.empty:
        backtest_columns = [
            "symbol",
            "timeframe",
            "window_start",
            "window_end",
            "policy_name",
            "probability_mode",
            "calibration_temperature",
            "buy_threshold",
            "exit_threshold",
            "latest_action",
            "latest_position",
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
            "backtest_summary",
        ]
        merged_df = merged_df.merge(
            backtest_df[backtest_columns],
            on=["symbol", "timeframe", "window_start", "window_end"],
            how="left",
        )

    if not walkforward_df.empty:
        walkforward_columns = [
            "symbol",
            "timeframe",
            "window_start",
            "window_end",
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
            "walkforward_selected_policy",
            "walkforward_selected_probability_mode",
            "futures_feature_completeness_score",
            "futures_completeness_label",
            "futures_context_resilience_score",
            "futures_context_resilience_label",
            "futures_basis_reliance_score",
            "basis_feature_available",
            "walkforward_summary",
        ]
        merged_df = merged_df.merge(
            walkforward_df[walkforward_columns],
            on=["symbol", "timeframe", "window_start", "window_end"],
            how="left",
        )

    if not multimodal_df.empty:
        multimodal_columns = [
            "symbol",
            "timeframe",
            "window_start",
            "window_end",
            "strategy_name",
            "latest_signal",
            "latest_signal_confidence",
            "latest_multimodal_context_label",
            "latest_sentiment_value",
            "latest_gdelt_sentiment_mean",
            "latest_gdelt_article_count",
            "latest_gdelt_regime_label",
            "latest_gdelt_attention_label",
            "latest_gdelt_coverage_quality_score",
            "latest_gdelt_dominant_event_theme",
            "latest_gdelt_risk_event_theme",
            "latest_gdelt_supportive_event_theme",
            "latest_effective_sentiment_source",
            "latest_effective_sentiment_label",
            "latest_onchain_regime_label",
            "context_selection_method",
            "selected_context_variant",
            "selected_context_feature_count",
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "test_accuracy",
            "test_macro_f1",
            "test_balanced_accuracy",
            "multimodal_summary",
        ]
        renamed_multimodal_df = multimodal_df.reindex(columns=multimodal_columns).rename(
            columns={
                "strategy_name": "multimodal_best_strategy",
                "latest_signal": "multimodal_latest_signal",
                "latest_signal_confidence": "multimodal_latest_signal_confidence",
                "context_selection_method": "multimodal_context_selection_method",
                "selected_context_variant": "multimodal_selected_context_variant",
                "selected_context_feature_count": "multimodal_selected_context_feature_count",
                "validation_macro_f1": "multimodal_validation_macro_f1",
                "validation_balanced_accuracy": "multimodal_validation_balanced_accuracy",
                "test_accuracy": "multimodal_test_accuracy",
                "test_macro_f1": "multimodal_test_macro_f1",
                "test_balanced_accuracy": "multimodal_test_balanced_accuracy",
            }
        )
        merged_df = merged_df.merge(
            renamed_multimodal_df,
            on=["symbol", "timeframe", "window_start", "window_end"],
            how="left",
        )

    if not context_ablation_df.empty:
        ranked_ablation_df = context_ablation_df.sort_values(
            ["symbol", "test_macro_f1", "test_balanced_accuracy", "test_accuracy", "latest_signal_confidence"],
            ascending=[True, False, False, False, False],
        )
        best_ablation_df = ranked_ablation_df.groupby("symbol", as_index=False).head(1).reset_index(drop=True)
        baseline_ablation_df = context_ablation_df.loc[
            context_ablation_df["variant_name"] == "market_futures_only",
            ["symbol", "timeframe", "window_start", "window_end", "test_macro_f1", "test_balanced_accuracy"],
        ].rename(
            columns={
                "test_macro_f1": "ablation_market_futures_macro_f1",
                "test_balanced_accuracy": "ablation_market_futures_balanced_accuracy",
            }
        )
        best_ablation_df = best_ablation_df.rename(
            columns={
                "variant_name": "ablation_best_variant",
                "latest_signal": "ablation_latest_signal",
                "latest_signal_confidence": "ablation_latest_signal_confidence",
                "test_accuracy": "ablation_best_accuracy",
                "test_macro_f1": "ablation_best_macro_f1",
                "test_balanced_accuracy": "ablation_best_balanced_accuracy",
            }
        )
        best_ablation_df["ablation_summary"] = (
            best_ablation_df["symbol"] + " performs best with " +
            best_ablation_df["ablation_best_variant"].astype(str).str.replace("_", " ") +
            ", changing macro-F1 by " +
            best_ablation_df["delta_macro_f1_vs_market_futures"].astype(float).map(lambda value: f"{value:+.3f}") +
            " versus market + futures only."
        )
        best_ablation_df = best_ablation_df.merge(
            baseline_ablation_df,
            on=["symbol", "timeframe", "window_start", "window_end"],
            how="left",
        )
        ablation_columns = [
            "symbol",
            "timeframe",
            "window_start",
            "window_end",
            "ablation_best_variant",
            "feature_count",
            "ablation_latest_signal",
            "ablation_latest_signal_confidence",
            "ablation_best_accuracy",
            "ablation_best_macro_f1",
            "ablation_best_balanced_accuracy",
            "ablation_market_futures_macro_f1",
            "ablation_market_futures_balanced_accuracy",
            "delta_macro_f1_vs_market_futures",
            "delta_balanced_accuracy_vs_market_futures",
            "ablation_summary",
        ]
        merged_df = merged_df.merge(
            best_ablation_df.reindex(columns=ablation_columns),
            on=["symbol", "timeframe", "window_start", "window_end"],
            how="left",
        )

    if not onchain_snapshot_df.empty:
        merged_df = merged_df.merge(
            onchain_snapshot_df,
            on="symbol",
            how="left",
        )
    if not gdelt_snapshot_df.empty:
        merged_df = merged_df.merge(
            gdelt_snapshot_df,
            on="symbol",
            how="left",
        )
    if not defi_snapshot_df.empty:
        merged_df = merged_df.merge(
            defi_snapshot_df,
            on="symbol",
            how="left",
        )

    for base_name in [
        "futures_feature_completeness_score",
        "futures_completeness_label",
        "futures_context_resilience_score",
        "futures_context_resilience_label",
        "futures_basis_reliance_score",
        "basis_feature_available",
    ]:
        merged_df = coalesce_suffix_columns(merged_df, base_name)

    for column_name in [
        "selected_backend_model",
        "selected_target_name",
        "latest_signal",
        "latest_signal_confidence",
        "backend_summary",
        "scaled_model_signal",
        "scaled_model_confidence",
        "analysis_summary",
        "backtest_summary",
        "policy_name",
        "probability_mode",
        "calibration_temperature",
        "latest_action",
        "walkforward_summary",
        "walkforward_selected_policy",
        "walkforward_selected_probability_mode",
        "multimodal_best_strategy",
        "multimodal_latest_signal",
        "multimodal_latest_signal_confidence",
        "latest_multimodal_context_label",
        "latest_sentiment_value",
        "latest_gdelt_sentiment_mean",
        "latest_gdelt_article_count",
        "latest_gdelt_regime_label",
        "latest_gdelt_attention_label",
        "latest_gdelt_coverage_quality_score",
        "latest_gdelt_dominant_event_theme",
        "latest_gdelt_risk_event_theme",
        "latest_gdelt_supportive_event_theme",
        "latest_gdelt_snapshot_date",
        "latest_gdelt_snapshot_age_days",
        "latest_gdelt_snapshot_status",
        "latest_effective_sentiment_source",
        "latest_effective_sentiment_label",
        "latest_onchain_regime_label",
        "multimodal_context_selection_method",
        "multimodal_selected_context_variant",
        "multimodal_selected_context_feature_count",
        "multimodal_validation_macro_f1",
        "multimodal_validation_balanced_accuracy",
        "latest_onchain_snapshot_label",
        "latest_onchain_snapshot_score",
        "latest_onchain_snapshot_reason",
        "latest_onchain_snapshot_date",
        "latest_onchain_snapshot_age_days",
        "latest_onchain_snapshot_status",
        "defi_chain_name",
        "latest_defi_window_end",
        "latest_defi_tvl_usd",
        "defi_tvl_change_pct_1d",
        "defi_tvl_change_pct_7d",
        "defi_tvl_change_pct_30d",
        "defi_tvl_zscore_30d",
        "defi_tvl_drawdown_30d",
        "defi_regime_score",
        "defi_regime_label",
        "defi_context_available",
        "defi_summary",
        "defi_snapshot_age_days",
        "defi_snapshot_status",
        "multimodal_test_accuracy",
        "multimodal_test_macro_f1",
        "multimodal_test_balanced_accuracy",
        "multimodal_summary",
        "ablation_best_variant",
        "feature_count",
        "ablation_latest_signal",
        "ablation_latest_signal_confidence",
        "ablation_best_accuracy",
        "ablation_best_macro_f1",
        "ablation_best_balanced_accuracy",
        "ablation_market_futures_macro_f1",
        "ablation_market_futures_balanced_accuracy",
        "delta_macro_f1_vs_market_futures",
        "delta_balanced_accuracy_vs_market_futures",
        "ablation_summary",
    ]:
        if column_name not in merged_df.columns:
            merged_df[column_name] = pd.NA

    if broad_sentiment_row and bool(broad_sentiment_row.get("sentiment_data_available", False)):
        sentiment_mask = merged_df["latest_sentiment_value"].apply(_is_blank_context)
        merged_df.loc[sentiment_mask, "latest_sentiment_value"] = broad_sentiment_row.get("sentiment_value")

        source_mask = merged_df["latest_effective_sentiment_source"].apply(_is_blank_context)
        merged_df.loc[source_mask, "latest_effective_sentiment_source"] = "fear_greed_market_fallback"

        label_mask = merged_df["latest_effective_sentiment_label"].apply(_is_blank_context)
        merged_df.loc[label_mask, "latest_effective_sentiment_label"] = broad_sentiment_row.get("market_mood_label")

    onchain_mask = merged_df["latest_onchain_regime_label"].apply(_is_blank_context)
    merged_df.loc[onchain_mask, "latest_onchain_regime_label"] = merged_df.loc[
        onchain_mask,
        "latest_onchain_snapshot_label",
    ]

    merged_df["current_pipeline_mode"] = merged_df["selected_backend_model"].apply(
        lambda value: "market_futures_backend" if isinstance(value, str) and value else "market_only"
    )
    merged_df["selected_primary_model"] = merged_df["selected_backend_model"].fillna("market_only_baseline")
    merged_df["selected_primary_signal"] = merged_df["latest_signal"].fillna(
        merged_df.get("scaled_model_signal", pd.Series(index=merged_df.index, dtype="object"))
    )
    merged_df["selected_primary_confidence"] = merged_df["latest_signal_confidence"].fillna(
        merged_df.get("scaled_model_confidence", pd.Series(index=merged_df.index, dtype="float64"))
    )
    merged_df["primary_summary"] = merged_df.apply(build_primary_summary, axis=1)

    output_path = get_market_intelligence_overview_path(timeframe, start_date, end_date)
    merged_df.to_csv(output_path, index=False)

    print("market intelligence overview generated")
    print(f"rows saved: {len(merged_df)}")
    print(f"overview saved to: {output_path}")
    return merged_df


if __name__ == "__main__":
    build_market_intelligence_overview()
