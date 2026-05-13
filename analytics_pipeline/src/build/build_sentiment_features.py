"""Build simple market mood features from Fear & Greed history."""

import pandas as pd

from src.config import (
    SENTIMENT_FREQUENCY,
    get_raw_sentiment_path,
    get_sentiment_features_path,
)
from src.io_paths import ensure_dirs


def classify_market_mood(value):
    """Translate the index value into a consistent project label."""
    if value <= 24:
        return "extreme_fear"
    if value <= 44:
        return "fear"
    if value <= 55:
        return "neutral"
    if value <= 74:
        return "greed"
    return "extreme_greed"


def build_sentiment_features(frequency=SENTIMENT_FREQUENCY):
    """Create processed market mood features from raw Fear & Greed data."""
    ensure_dirs()
    raw_path = get_raw_sentiment_path(frequency)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw sentiment data not found: {raw_path}")

    raw_df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    if raw_df.empty:
        features_df = pd.DataFrame(
            columns=[
                "window_end_utc",
                "sentiment_value",
                "sentiment_classification",
                "market_mood_label",
                "sentiment_change_1d",
                "sentiment_change_7d",
                "sentiment_rolling_mean_7d",
                "sentiment_zscore_30d",
                "sentiment_data_available",
            ]
        )
    else:
        features_df = raw_df[["timestamp", "value", "value_classification"]].copy()
        features_df = features_df.rename(
            columns={
                "timestamp": "window_end_utc",
                "value": "sentiment_value",
                "value_classification": "sentiment_classification",
            }
        )
        features_df = features_df.sort_values("window_end_utc").reset_index(drop=True)
        features_df["market_mood_label"] = features_df["sentiment_value"].apply(classify_market_mood)
        features_df["sentiment_change_1d"] = features_df["sentiment_value"].diff().fillna(0.0)
        features_df["sentiment_change_7d"] = features_df["sentiment_value"].diff(7).fillna(0.0)
        features_df["sentiment_rolling_mean_7d"] = (
            features_df["sentiment_value"].rolling(7, min_periods=1).mean()
        )
        rolling_mean = features_df["sentiment_value"].rolling(30, min_periods=5).mean()
        rolling_std = features_df["sentiment_value"].rolling(30, min_periods=5).std()
        features_df["sentiment_zscore_30d"] = (
            (features_df["sentiment_value"] - rolling_mean) / rolling_std.replace(0, pd.NA)
        ).fillna(0.0)
        features_df["sentiment_data_available"] = True
        features_df["window_end_utc"] = pd.to_datetime(features_df["window_end_utc"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    output_path = get_sentiment_features_path(frequency)
    features_df.to_csv(output_path, index=False)

    print("sentiment feature construction completed")
    print(f"rows saved: {len(features_df)}")
    print(f"processed features saved to: {output_path}")

    return features_df


if __name__ == "__main__":
    build_sentiment_features()
