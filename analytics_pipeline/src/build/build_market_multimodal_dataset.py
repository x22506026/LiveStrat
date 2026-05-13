"""Build aligned market + futures + sentiment + on-chain datasets."""

import numpy as np
import pandas as pd

from src.config import (
    ASSET_REGISTRY,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    SENTIMENT_FREQUENCY,
    get_all_symbols,
    get_defillama_features_path,
    get_gdelt_sentiment_features_path,
    get_market_futures_dataset_path,
    get_market_multimodal_dataset_path,
    get_onchain_features_path,
    get_sentiment_features_path,
)
from src.io_paths import ensure_dirs


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
CONTEXT_TOLERANCE_DAYS = 7

SENTIMENT_SCORE_MAP = {
    "extreme_fear": -2.0,
    "fear": -1.0,
    "neutral": 0.0,
    "greed": 1.0,
    "extreme_greed": 2.0,
}


def classify_effective_sentiment_label(score):
    """Map a normalized effective sentiment score into a shared label set."""
    if pd.isna(score):
        return "unavailable"
    if score >= 0.15:
        return "supportive"
    if score <= -0.15:
        return "risk_off"
    return "mixed"


def compute_news_event_risk_score(df):
    """Turn asset-news into a cautious event overlay rather than a standalone alpha layer."""
    negative_sentiment = (-pd.to_numeric(df["gdelt_sentiment_mean"], errors="coerce")).clip(lower=0.0).fillna(0.0)
    positive_sentiment = pd.to_numeric(df["gdelt_sentiment_mean"], errors="coerce").clip(lower=0.0).fillna(0.0)
    negative_keywords = pd.to_numeric(df["gdelt_negative_keyword_share"], errors="coerce").fillna(0.0)
    positive_keywords = pd.to_numeric(df["gdelt_positive_keyword_share"], errors="coerce").fillna(0.0)
    event_intensity = pd.to_numeric(df["gdelt_event_intensity_score"], errors="coerce").fillna(0.0)
    coverage_quality = pd.to_numeric(df["gdelt_coverage_quality_score"], errors="coerce").fillna(0.0)
    source_count = pd.to_numeric(df["news_source_count"], errors="coerce").fillna(0.0)
    source_concentration = pd.to_numeric(df["gdelt_source_concentration"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    rss_share = pd.to_numeric(df["news_rss_share"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    attention_label = df["gdelt_attention_label"].astype(str).str.lower()
    dominant_theme = df["gdelt_dominant_event_theme"].astype(str).str.lower()
    risk_theme = df["gdelt_risk_event_theme"].astype(str).str.lower()
    supportive_theme = df["gdelt_supportive_event_theme"].astype(str).str.lower()
    availability = df["gdelt_context_available"].eq(True).astype(float)
    source_diversity = (1.0 - source_concentration).clip(lower=0.0, upper=1.0)
    source_count_factor = (source_count.clip(lower=0.0, upper=3.0) / 3.0)
    attention_weight = pd.Series(0.90, index=df.index)
    attention_weight = attention_weight.mask(attention_label.eq("active"), 1.00)
    attention_weight = attention_weight.mask(attention_label.eq("event_heavy"), 1.12)
    provenance_factor = 0.95 + 0.05 * rss_share
    risk_theme_factor = pd.Series(1.0, index=df.index)
    risk_theme_factor = risk_theme_factor.mask(risk_theme.eq("security"), 1.15)
    risk_theme_factor = risk_theme_factor.mask(risk_theme.eq("regulation"), 1.10)
    risk_theme_factor = risk_theme_factor.mask(risk_theme.eq("flows"), 1.08)
    risk_theme_factor = risk_theme_factor.mask(risk_theme.eq("macro_stress"), 1.16)
    risk_theme_factor = risk_theme_factor.mask(risk_theme.eq("capital_markets"), 1.05)
    supportive_theme_factor = pd.Series(1.0, index=df.index)
    supportive_theme_factor = supportive_theme_factor.mask(supportive_theme.eq("adoption"), 1.12)
    supportive_theme_factor = supportive_theme_factor.mask(supportive_theme.eq("network"), 1.05)
    supportive_theme_factor = supportive_theme_factor.mask(supportive_theme.eq("stablecoin_payments"), 1.10)
    watch_theme_factor = pd.Series(1.0, index=df.index)
    watch_theme_factor = watch_theme_factor.mask(dominant_theme.eq("network"), 1.03)
    watch_theme_factor = watch_theme_factor.mask(dominant_theme.eq("regulation"), 1.03)
    watch_theme_factor = watch_theme_factor.mask(dominant_theme.eq("capital_markets"), 1.04)
    watch_theme_factor = watch_theme_factor.mask(dominant_theme.eq("macro_stress"), 1.05)
    confidence_factor = (
        0.45
        + 0.20 * coverage_quality.clip(upper=1.0)
        + 0.20 * source_count_factor
        + 0.15 * source_diversity
    ).clip(lower=0.35, upper=1.0)

    raw_risk_score = (
        0.35 * (negative_sentiment.clip(upper=0.50) / 0.50)
        + 0.30 * event_intensity.clip(upper=1.0)
        + 0.20 * negative_keywords.clip(upper=1.0)
        + 0.15 * coverage_quality.clip(upper=1.0)
    )
    raw_supportive_score = (
        0.35 * (positive_sentiment.clip(upper=0.50) / 0.50)
        + 0.30 * event_intensity.clip(upper=1.0)
        + 0.20 * positive_keywords.clip(upper=1.0)
        + 0.15 * coverage_quality.clip(upper=1.0)
    )
    risk_score = raw_risk_score * confidence_factor * attention_weight * provenance_factor * risk_theme_factor * watch_theme_factor * availability
    supportive_score = raw_supportive_score * confidence_factor * attention_weight * provenance_factor * supportive_theme_factor * watch_theme_factor * availability
    return risk_score.clip(lower=0.0, upper=1.0), supportive_score.clip(lower=0.0, upper=1.0)


def get_asset_symbol_from_market_symbol(market_symbol):
    """Resolve a market symbol like BTCUSDT back to the asset registry key."""
    for asset_symbol, asset_config in ASSET_REGISTRY.items():
        if asset_config["market_symbol"] == market_symbol:
            return asset_symbol
    return market_symbol.replace("USDT", "")


def load_sentiment_context():
    """Load processed daily sentiment features as an aligned context table."""
    sentiment_path = get_sentiment_features_path(SENTIMENT_FREQUENCY)
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["window_end_utc"])
    sentiment_df = sentiment_df.sort_values("window_end_utc").reset_index(drop=True)
    sentiment_df["sentiment_regime_score"] = (
        sentiment_df["market_mood_label"].map(SENTIMENT_SCORE_MAP).fillna(0.0)
    )
    sentiment_df["sentiment_supportive_flag"] = sentiment_df["sentiment_regime_score"] > 0.0
    sentiment_df["sentiment_risk_off_flag"] = sentiment_df["sentiment_regime_score"] < 0.0
    sentiment_df = sentiment_df.rename(
        columns={
            "window_end_utc": "context_window_end_utc",
        }
    )
    return sentiment_df


def load_onchain_context(asset_symbol):
    """Load processed daily on-chain features for one asset if available."""
    onchain_path = get_onchain_features_path(asset_symbol)
    onchain_df = pd.read_csv(onchain_path, parse_dates=["window_end_utc"])
    onchain_df = onchain_df.sort_values("window_end_utc").reset_index(drop=True)
    onchain_df["onchain_supportive_flag"] = onchain_df["onchain_regime_label"] == "supportive"
    onchain_df["onchain_risk_off_flag"] = onchain_df["onchain_regime_label"] == "weakening"
    onchain_df = onchain_df.rename(
        columns={
            "window_end_utc": "onchain_window_end_utc",
        }
    )
    return onchain_df


def load_gdelt_context(asset_symbol, start_date, end_date):
    """Load processed asset-specific GDELT sentiment features when available."""
    gdelt_path = get_gdelt_sentiment_features_path(asset_symbol, SENTIMENT_FREQUENCY, start_date, end_date)
    if not gdelt_path.exists():
        fallback_candidates = sorted(
            gdelt_path.parent.glob(f"{asset_symbol}_gdelt_sentiment_features_{SENTIMENT_FREQUENCY}_*.csv")
        )
        if fallback_candidates:
            gdelt_path = fallback_candidates[-1]
        else:
            return pd.DataFrame(
                columns=[
                    "gdelt_window_end_utc",
                    "gdelt_article_count",
                    "gdelt_unique_domain_count",
                    "gdelt_sentiment_mean",
                    "gdelt_sentiment_std",
                    "gdelt_positive_share",
                    "gdelt_negative_share",
                    "gdelt_neutral_share",
                    "gdelt_sentiment_momentum_3d",
                    "gdelt_sentiment_momentum_7d",
                    "gdelt_article_count_zscore_30d",
                    "gdelt_article_count_3d_sum",
                    "gdelt_article_count_7d_sum",
                    "gdelt_positive_keyword_share",
                    "gdelt_negative_keyword_share",
                "gdelt_keyword_intensity_mean",
                "gdelt_source_concentration",
                "gdelt_coverage_quality_score",
                "gdelt_event_intensity_score",
                "gdelt_attention_label",
                    "news_source_count",
                    "news_primary_source_mode",
                    "news_rss_share",
                    "news_gdelt_share",
                    "gdelt_regulation_share",
                    "gdelt_security_share",
                    "gdelt_adoption_share",
                    "gdelt_network_share",
                    "gdelt_flows_share",
                    "gdelt_macro_stress_share",
                    "gdelt_capital_markets_share",
                    "gdelt_stablecoin_payments_share",
                    "gdelt_dominant_event_theme",
                    "gdelt_risk_event_theme",
                    "gdelt_supportive_event_theme",
                    "gdelt_regime_score",
                "gdelt_regime_label",
                "gdelt_supportive_flag",
                "gdelt_risk_off_flag",
                "gdelt_sentiment_data_available",
                    "gdelt_sentiment_model_used",
                ]
            )

    gdelt_df = pd.read_csv(gdelt_path, parse_dates=["window_end_utc"])
    gdelt_df = gdelt_df.sort_values("window_end_utc").reset_index(drop=True)
    gdelt_df = gdelt_df.rename(columns={"window_end_utc": "gdelt_window_end_utc"})
    return gdelt_df


def load_defi_context(asset_symbol):
    """Load processed DeFiLlama chain TVL context if available."""
    defi_path = get_defillama_features_path(asset_symbol)
    if not defi_path.exists():
        return pd.DataFrame()
    defi_df = pd.read_csv(defi_path, parse_dates=["window_end_utc"])
    if defi_df.empty:
        return pd.DataFrame()
    defi_df = defi_df.sort_values("window_end_utc").reset_index(drop=True)
    defi_df = defi_df.rename(columns={"window_end_utc": "defi_window_end_utc"})
    return defi_df


def merge_daily_context(base_df, context_df, left_time_col, right_time_col, tolerance_days=CONTEXT_TOLERANCE_DAYS):
    """Backward-align only completed daily context onto market rows.

    Daily sentiment/news/on-chain rows summarize a full UTC day, so they become
    eligible from the next UTC day. This avoids leaking later same-day context
    into intraday 1h/4h decisions.
    """
    tolerance = pd.Timedelta(days=tolerance_days)
    left_df = base_df.copy()
    right_df = context_df.copy()
    left_df[left_time_col] = pd.to_datetime(left_df[left_time_col], utc=True)
    right_df[right_time_col] = pd.to_datetime(right_df[right_time_col], utc=True) + pd.Timedelta(days=1)
    return pd.merge_asof(
        left_df.sort_values(left_time_col),
        right_df.sort_values(right_time_col),
        left_on=left_time_col,
        right_on=right_time_col,
        direction="backward",
        tolerance=tolerance,
    )


def ensure_gdelt_theme_columns(df):
    """Keep multimodal builds compatible with older cached GDELT feature files."""
    defaults = {
        "gdelt_regulation_share": 0.0,
        "gdelt_security_share": 0.0,
        "gdelt_adoption_share": 0.0,
        "gdelt_network_share": 0.0,
        "gdelt_flows_share": 0.0,
        "gdelt_macro_stress_share": 0.0,
        "gdelt_capital_markets_share": 0.0,
        "gdelt_stablecoin_payments_share": 0.0,
        "gdelt_dominant_event_theme": pd.NA,
        "gdelt_risk_event_theme": pd.NA,
        "gdelt_supportive_event_theme": pd.NA,
    }
    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value
    return df


def build_market_multimodal_dataset_for_symbol(symbol, timeframe=TIMEFRAME,
                                               start_date=START_DATE, end_date=END_DATE):
    """Merge the dated market + futures dataset with daily sentiment and on-chain context."""
    ensure_dirs()
    asset_symbol = get_asset_symbol_from_market_symbol(symbol)
    market_futures_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
    base_df = pd.read_csv(market_futures_path, parse_dates=["open_time", "close_time"])
    base_df = base_df.sort_values("close_time").reset_index(drop=True)
    base_df["context_time"] = pd.to_datetime(base_df["close_time"], utc=True)

    sentiment_df = load_sentiment_context()
    merged_df = merge_daily_context(base_df, sentiment_df, "context_time", "context_window_end_utc")

    gdelt_df = load_gdelt_context(asset_symbol, start_date, end_date)
    if not gdelt_df.empty:
        merged_df = merge_daily_context(merged_df, gdelt_df, "context_time", "gdelt_window_end_utc")
    else:
        merged_df["gdelt_window_end_utc"] = pd.NaT
        merged_df["gdelt_article_count"] = 0
        merged_df["gdelt_unique_domain_count"] = 0
        merged_df["gdelt_sentiment_mean"] = 0.0
        merged_df["gdelt_sentiment_std"] = 0.0
        merged_df["gdelt_positive_share"] = 0.0
        merged_df["gdelt_negative_share"] = 0.0
        merged_df["gdelt_neutral_share"] = 1.0
        merged_df["gdelt_sentiment_momentum_3d"] = 0.0
        merged_df["gdelt_sentiment_momentum_7d"] = 0.0
        merged_df["gdelt_article_count_zscore_30d"] = 0.0
        merged_df["gdelt_article_count_3d_sum"] = 0.0
        merged_df["gdelt_article_count_7d_sum"] = 0.0
        merged_df["gdelt_positive_keyword_share"] = 0.0
        merged_df["gdelt_negative_keyword_share"] = 0.0
        merged_df["gdelt_keyword_intensity_mean"] = 0.0
        merged_df["gdelt_source_concentration"] = 0.0
        merged_df["gdelt_coverage_quality_score"] = 0.0
        merged_df["gdelt_event_intensity_score"] = 0.0
        merged_df["gdelt_attention_label"] = pd.NA
        merged_df["news_source_count"] = 0
        merged_df["news_primary_source_mode"] = pd.NA
        merged_df["news_rss_share"] = 0.0
        merged_df["news_gdelt_share"] = 0.0
        merged_df["gdelt_regulation_share"] = 0.0
        merged_df["gdelt_security_share"] = 0.0
        merged_df["gdelt_adoption_share"] = 0.0
        merged_df["gdelt_network_share"] = 0.0
        merged_df["gdelt_flows_share"] = 0.0
        merged_df["gdelt_macro_stress_share"] = 0.0
        merged_df["gdelt_capital_markets_share"] = 0.0
        merged_df["gdelt_stablecoin_payments_share"] = 0.0
        merged_df["gdelt_dominant_event_theme"] = pd.NA
        merged_df["gdelt_risk_event_theme"] = pd.NA
        merged_df["gdelt_supportive_event_theme"] = pd.NA
        merged_df["gdelt_regime_score"] = 0.0
        merged_df["gdelt_regime_label"] = pd.NA
        merged_df["gdelt_supportive_flag"] = False
        merged_df["gdelt_risk_off_flag"] = False
        merged_df["gdelt_sentiment_data_available"] = False
        merged_df["gdelt_sentiment_model_used"] = pd.NA
    merged_df = ensure_gdelt_theme_columns(merged_df)

    onchain_supported = ASSET_REGISTRY.get(asset_symbol, {}).get("support_flags", {}).get("onchain_supported", False)
    if onchain_supported:
        onchain_df = load_onchain_context(asset_symbol)
        merged_df = merge_daily_context(merged_df, onchain_df, "context_time", "onchain_window_end_utc")
    else:
        merged_df["onchain_window_end_utc"] = pd.NaT
        merged_df["onchain_regime_score"] = np.nan
        merged_df["onchain_regime_label"] = pd.NA
        merged_df["onchain_supportive_flag"] = False
        merged_df["onchain_risk_off_flag"] = False
        merged_df["onchain_data_available"] = False

    defi_df = load_defi_context(asset_symbol)
    if not defi_df.empty:
        merged_df = merge_daily_context(merged_df, defi_df, "context_time", "defi_window_end_utc")
    else:
        merged_df["defi_window_end_utc"] = pd.NaT
        merged_df["defi_tvl_usd"] = np.nan
        merged_df["defi_tvl_change_1d"] = np.nan
        merged_df["defi_tvl_change_pct_1d"] = np.nan
        merged_df["defi_tvl_change_pct_7d"] = np.nan
        merged_df["defi_tvl_change_pct_30d"] = np.nan
        merged_df["defi_tvl_zscore_30d"] = np.nan
        merged_df["defi_tvl_drawdown_30d"] = np.nan
        merged_df["defi_regime_score"] = np.nan
        merged_df["defi_regime_label"] = pd.NA
        merged_df["defi_context_available"] = False

    merged_df["asset_symbol"] = asset_symbol
    merged_df["sentiment_context_available"] = merged_df["sentiment_data_available"].eq(True)
    merged_df["gdelt_context_available"] = merged_df["gdelt_sentiment_data_available"].eq(True)
    merged_df["onchain_context_available"] = merged_df["onchain_data_available"].eq(True)
    merged_df["defi_context_available"] = merged_df["defi_context_available"].eq(True)
    merged_df["sentiment_regime_score"] = merged_df["sentiment_regime_score"].fillna(0.0)
    merged_df["gdelt_regime_score"] = pd.to_numeric(
        merged_df["gdelt_regime_score"],
        errors="coerce",
    ).fillna(0.0)
    merged_df["onchain_regime_score"] = pd.to_numeric(
        merged_df["onchain_regime_score"],
        errors="coerce",
    ).fillna(0.0)
    merged_df["defi_regime_score"] = pd.to_numeric(
        merged_df["defi_regime_score"],
        errors="coerce",
    ).fillna(0.0)
    merged_df["sentiment_supportive_flag"] = merged_df["sentiment_supportive_flag"].eq(True)
    merged_df["sentiment_risk_off_flag"] = merged_df["sentiment_risk_off_flag"].eq(True)
    merged_df["gdelt_supportive_flag"] = merged_df["gdelt_supportive_flag"].eq(True)
    merged_df["gdelt_risk_off_flag"] = merged_df["gdelt_risk_off_flag"].eq(True)
    merged_df["onchain_supportive_flag"] = merged_df["onchain_supportive_flag"].eq(True)
    merged_df["onchain_risk_off_flag"] = merged_df["onchain_risk_off_flag"].eq(True)
    merged_df["defi_supportive_flag"] = merged_df["defi_regime_label"].eq("expanding")
    merged_df["defi_risk_off_flag"] = merged_df["defi_regime_label"].eq("contracting")

    merged_df["effective_sentiment_source"] = "unavailable"
    merged_df.loc[merged_df["sentiment_context_available"], "effective_sentiment_source"] = "fear_greed_market_fallback"
    merged_df.loc[merged_df["gdelt_context_available"], "effective_sentiment_source"] = "gdelt_asset_news"
    merged_df["effective_sentiment_score"] = merged_df["sentiment_regime_score"]
    merged_df.loc[merged_df["gdelt_context_available"], "effective_sentiment_score"] = merged_df.loc[
        merged_df["gdelt_context_available"], "gdelt_regime_score"
    ]
    merged_df["effective_sentiment_label"] = merged_df["effective_sentiment_score"].apply(classify_effective_sentiment_label)
    merged_df.loc[merged_df["effective_sentiment_source"] == "unavailable", "effective_sentiment_label"] = "unavailable"
    merged_df["effective_sentiment_supportive_flag"] = merged_df["effective_sentiment_label"].eq("supportive")
    merged_df["effective_sentiment_risk_off_flag"] = merged_df["effective_sentiment_label"].eq("risk_off")
    merged_df["effective_sentiment_available"] = merged_df["effective_sentiment_source"] != "unavailable"
    (
        merged_df["news_event_risk_score"],
        merged_df["news_event_supportive_score"],
    ) = compute_news_event_risk_score(merged_df)
    merged_df["news_event_risk_flag"] = (
        merged_df["gdelt_context_available"].eq(True) &
        (
            (merged_df["news_event_risk_score"] >= 0.40) |
            (
                merged_df["gdelt_attention_label"].astype(str).isin(["active", "event_heavy"]) &
                merged_df["gdelt_risk_off_flag"].eq(True)
            )
        )
    )
    merged_df["news_event_supportive_flag"] = (
        merged_df["gdelt_context_available"].eq(True) &
        (merged_df["news_event_supportive_score"] >= 0.40) &
        merged_df["news_event_risk_flag"].eq(False)
    )
    merged_df["news_event_regime_label"] = "quiet"
    merged_df.loc[
        (
            (merged_df["news_event_risk_score"] >= 0.25) |
            (merged_df["news_event_supportive_score"] >= 0.25)
        ) &
        merged_df["news_event_risk_flag"].eq(False) &
        merged_df["news_event_supportive_flag"].eq(False),
        "news_event_regime_label"
    ] = "watch_event"
    merged_df.loc[merged_df["news_event_supportive_flag"], "news_event_regime_label"] = "supportive_event"
    merged_df.loc[merged_df["news_event_risk_flag"], "news_event_regime_label"] = "risk_event"

    merged_df["multimodal_confirmation_count"] = (
        merged_df["effective_sentiment_supportive_flag"].astype(int) +
        merged_df["onchain_supportive_flag"].astype(int) +
        merged_df["news_event_supportive_flag"].astype(int) +
        merged_df["defi_supportive_flag"].astype(int)
    )
    merged_df["multimodal_risk_off_count"] = (
        merged_df["effective_sentiment_risk_off_flag"].astype(int) +
        merged_df["onchain_risk_off_flag"].astype(int) +
        merged_df["news_event_risk_flag"].astype(int) +
        merged_df["defi_risk_off_flag"].astype(int)
    )
    context_score_columns = [
        "effective_sentiment_score",
        "onchain_regime_score",
        "defi_regime_score",
        "news_event_supportive_score",
        "news_event_risk_score",
    ]
    context_available_columns = [
        "effective_sentiment_available",
        "onchain_context_available",
        "gdelt_context_available",
        "defi_context_available",
    ]
    context_score_df = merged_df[context_score_columns].copy()
    context_score_df["onchain_regime_score"] = context_score_df["onchain_regime_score"].clip(-2.0, 2.0)
    context_score_df["news_event_component"] = (
        context_score_df["news_event_supportive_score"].fillna(0.0) -
        context_score_df["news_event_risk_score"].fillna(0.0)
    )
    context_score_df = context_score_df.drop(columns=["news_event_supportive_score", "news_event_risk_score"])
    availability_counts = merged_df[context_available_columns].astype(int).sum(axis=1).replace(0, np.nan)
    merged_df["multimodal_context_score"] = (
        context_score_df.sum(axis=1) / availability_counts
    ).fillna(0.0)
    merged_df["multimodal_context_label"] = "mixed"
    merged_df.loc[merged_df["multimodal_context_score"] >= 0.5, "multimodal_context_label"] = "supportive"
    merged_df.loc[merged_df["multimodal_context_score"] <= -0.5, "multimodal_context_label"] = "risk_off"
    merged_df["multimodal_context_available"] = (
        merged_df["sentiment_context_available"] |
        merged_df["gdelt_context_available"] |
        merged_df["onchain_context_available"] |
        merged_df["defi_context_available"]
    )

    output_path = get_market_multimodal_dataset_path(symbol, timeframe, start_date, end_date)
    merged_df.to_csv(output_path, index=False)

    print(f"built market multimodal dataset for {symbol}")
    print(f"rows saved: {len(merged_df)}")
    print(f"combined dataset saved to: {output_path}")
    return merged_df


def build_market_multimodal_datasets(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE, symbols=None):
    """Build aligned multimodal datasets for all requested market symbols."""
    outputs = {}
    for symbol in symbols or get_all_symbols():
        outputs[symbol] = build_market_multimodal_dataset_for_symbol(
            symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
    return outputs


if __name__ == "__main__":
    build_market_multimodal_datasets()
