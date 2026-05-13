"""Shared configuration for the LiveStrat analytics pipeline."""

from src.io_paths import (
    PROCESSED_DIR,
    RAW_BINANCE_DIR,
    RAW_FUTURES_DIR,
    RAW_GDELT_DIR,
    RAW_DEFI_LLAMA_DIR,
    RAW_ONCHAIN_DIR,
    RAW_SENTIMENT_DIR,
)


# primary market configuration
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "4h"
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2025-12-31"
CORE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
EXPANSION_SYMBOLS = ["BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
SUPPORTED_SYMBOLS = CORE_SYMBOLS + EXPANSION_SYMBOLS
SUPPORTED_TIMEFRAMES = ["1h", "4h", "1d"]


# modelling configuration
TRAIN_RATIO = 0.7
LABEL_HORIZON_STEPS = 6
BUY_THRESHOLD = 0.01
DONT_BUY_THRESHOLD = -0.01


# explicit leakage-safe market feature set
MARKET_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "return_4h",
    "return_24h",
    "return_3d",
    "sma_20",
    "sma_50",
    "price_sma_20_diff",
    "price_sma_50_diff",
    "volatility_20",
    "volatility_50",
    "volume_zscore",
    "high_low_range_pct",
    "candle_body_pct",
    "taker_buy_volume_ratio",
    "trade_count_zscore",
]


FUTURES_FEATURE_COLUMNS = [
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "mark_return",
    "mark_return_24h",
    "mark_return_3d",
    "funding_rate",
    "funding_rate_change",
    "funding_rate_rolling_mean_21",
    "funding_rate_zscore_21",
    "open_interest",
    "open_interest_value",
    "open_interest_change_pct",
    "open_interest_value_zscore_21",
    "open_interest_change_pct_zscore_21",
    "long_short_ratio",
    "long_account_share",
    "short_account_share",
    "long_short_ratio_zscore_21",
    "taker_buy_sell_ratio",
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_buy_sell_ratio_zscore_21",
    "basis_rate",
    "basis_value",
    "annualized_basis_rate",
    "basis_rate_zscore_21",
    "futures_crowding_score",
    "futures_activity_score",
]


# source-aware asset support registry
ASSET_REGISTRY = {
    "BTC": {
        "display_name": "Bitcoin",
        "market_symbol": "BTCUSDT",
        "coinmetrics_asset": "btc",
        "support_flags": {
            "market_supported": True,
            "sentiment_supported": True,
            "onchain_supported": True,
        },
        "validation_flags": {
            "market_validated": True,
            "sentiment_validated": False,
            "onchain_validated": False,
        },
        "strategy_modes": ["market_only", "market_sentiment", "market_onchain", "combined"],
        "prediction_modes": ["market_only", "multimodal"],
        "tier": "core",
        "notes": "Primary benchmark asset for the project. Sentiment currently uses market-wide mood data.",
    },
    "ETH": {
        "display_name": "Ethereum",
        "market_symbol": "ETHUSDT",
        "coinmetrics_asset": "eth",
        "support_flags": {
            "market_supported": True,
            "sentiment_supported": True,
            "onchain_supported": True,
        },
        "validation_flags": {
            "market_validated": True,
            "sentiment_validated": False,
            "onchain_validated": False,
        },
        "strategy_modes": ["market_only", "market_sentiment", "market_onchain", "combined"],
        "prediction_modes": ["market_only", "multimodal"],
        "tier": "core",
        "notes": "Second benchmark asset for generalization checks. Sentiment currently uses market-wide mood data.",
    },
    "SOL": {
        "display_name": "Solana",
        "market_symbol": "SOLUSDT",
        "coinmetrics_asset": "sol",
        "support_flags": {
            "market_supported": True,
            "sentiment_supported": True,
            "onchain_supported": False,
        },
        "validation_flags": {
            "market_validated": True,
            "sentiment_validated": False,
            "onchain_validated": False,
        },
        "strategy_modes": ["market_only", "market_sentiment"],
        "prediction_modes": ["market_only"],
        "tier": "core",
        "notes": "Core diversification asset with different market behaviour from BTC/ETH. Coin Metrics community on-chain candidate metrics are currently unavailable for SOL.",
    },
    "BNB": {
        "display_name": "BNB",
        "market_symbol": "BNBUSDT",
        "coinmetrics_asset": "bnb",
        "support_flags": {
            "market_supported": True,
            "sentiment_supported": True,
            "onchain_supported": False,
        },
        "validation_flags": {
            "market_validated": False,
            "sentiment_validated": False,
            "onchain_validated": False,
        },
        "strategy_modes": ["market_only", "market_sentiment"],
        "prediction_modes": ["market_only"],
        "tier": "expansion",
        "notes": "Expansion asset with market and news-sentiment support. Coin Metrics community on-chain candidate metrics are historical/stale for BNB, so the live on-chain layer stays disabled.",
    },
    "XRP": {
        "display_name": "XRP",
        "market_symbol": "XRPUSDT",
        "coinmetrics_asset": "xrp",
        "support_flags": {
            "market_supported": True,
            "sentiment_supported": True,
            "onchain_supported": True,
        },
        "validation_flags": {
            "market_validated": False,
            "sentiment_validated": False,
            "onchain_validated": False,
        },
        "strategy_modes": ["market_only", "market_sentiment", "market_onchain"],
        "prediction_modes": ["market_only"],
        "tier": "expansion",
        "notes": "Expansion asset with market, news-sentiment, and Coin Metrics community on-chain support.",
    },
    "ADA": {
        "display_name": "Cardano",
        "market_symbol": "ADAUSDT",
        "coinmetrics_asset": "ada",
        "support_flags": {
            "market_supported": True,
            "sentiment_supported": True,
            "onchain_supported": True,
        },
        "validation_flags": {
            "market_validated": False,
            "sentiment_validated": False,
            "onchain_validated": False,
        },
        "strategy_modes": ["market_only", "market_sentiment", "market_onchain"],
        "prediction_modes": ["market_only"],
        "tier": "expansion",
        "notes": "Expansion asset with market, news-sentiment, and Coin Metrics community on-chain support.",
    },
    "DOGE": {
        "display_name": "Dogecoin",
        "market_symbol": "DOGEUSDT",
        "coinmetrics_asset": "doge",
        "support_flags": {
            "market_supported": True,
            "sentiment_supported": True,
            "onchain_supported": True,
        },
        "validation_flags": {
            "market_validated": False,
            "sentiment_validated": False,
            "onchain_validated": False,
        },
        "strategy_modes": ["market_only", "market_sentiment", "market_onchain"],
        "prediction_modes": ["market_only"],
        "tier": "expansion",
        "notes": "Expansion asset with market, news-sentiment, and Coin Metrics community on-chain support.",
    },
}


# market-wide sentiment configuration
SENTIMENT_SOURCE_NAME = "alternative_me_fear_greed"
SENTIMENT_SOURCE_URL = "https://api.alternative.me/fng/"
SENTIMENT_FREQUENCY = "1d"
SENTIMENT_LOOKBACK_LIMIT = 0
SENTIMENT_MODEL_PRIMARY = "finbert"
SENTIMENT_MODEL_BASELINE = "gdelt_headline_lexicon_proxy"


# GDELT news sentiment configuration
GDELT_DOC_API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DEFAULT_MAX_RECORDS = 250
GDELT_DEFAULT_TIMESPAN = "1month"
GDELT_SUPPORTED_TIMESPANS = ["1day", "3days", "1week", "2weeks", "1month", "3month"]
GDELT_ASSET_QUERY_MAP = {
    "BTC": '("bitcoin" OR "bitcoin market" OR "bitcoin price" OR "spot bitcoin etf")',
    "ETH": '("ethereum" OR "ether" OR "ethereum market" OR "ethereum price" OR "ether etf")',
    "SOL": '("solana" OR "solana token" OR "solana market" OR "solana price")',
    "BNB": '("bnb" OR "binance coin" OR "bnb chain" OR "bnb token")',
    "XRP": '("xrp" OR "ripple" OR "xrp market" OR "xrp price")',
    "ADA": '("cardano" OR "ada token" OR "cardano ada" OR "ada price")',
    "DOGE": '("dogecoin" OR "doge" OR "dogecoin price" OR "doge market")',
}


# Coin Metrics community API configuration
COINMETRICS_COMMUNITY_API_BASE = "https://community-api.coinmetrics.io/v4"
CORE_ONCHAIN_ASSETS = ["BTC", "ETH", "XRP", "ADA", "DOGE"]
ONCHAIN_FREQUENCY = "1d"
ONCHAIN_LOOKBACK_DAYS = 365
COINMETRICS_CANDIDATE_METRICS = [
    "AdrActCnt",
    "TxCnt",
    "TxTfrValAdjUSD",
    "CapMrktCurUSD",
    "CapMVRVCur",
    "FlowInExUSD",
    "FlowOutExUSD",
]


# DeFiLlama chain TVL context configuration
DEFILLAMA_API_BASE = "https://api.llama.fi"
DEFILLAMA_FREQUENCY = "1d"
DEFILLAMA_CHAIN_MAP = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "BNB": "BSC",
    "XRP": "XRPL",
    "ADA": "Cardano",
    "DOGE": "Doge",
}


# Binance futures market structure configuration
FUTURES_BASE_URL = "https://fapi.binance.com"
FUTURES_CONTRACT_TYPE = "PERPETUAL"
FUTURES_RECENT_LOOKBACK_DAYS = 30
FUTURES_RATIO_LIMIT = 500


def _build_date_suffix(start_date=None, end_date=None):
    if start_date and end_date:
        return f"_{start_date}_{end_date}"
    return ""


def build_raw_binance_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                               start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    return f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"


def build_market_features_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                  start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_features{_build_date_suffix(start_date, end_date)}.csv"


def build_labeled_market_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                  start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_features_labeled{_build_date_suffix(start_date, end_date)}.csv"


def get_raw_binance_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                         start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    return RAW_BINANCE_DIR / build_raw_binance_filename(symbol, timeframe, start_date, end_date)


def get_market_features_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                             start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_features_filename(symbol, timeframe, start_date, end_date)


def get_labeled_market_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                            start_date=None, end_date=None):
    return PROCESSED_DIR / build_labeled_market_filename(symbol, timeframe, start_date, end_date)


def get_all_symbols():
    return SUPPORTED_SYMBOLS.copy()


def get_all_timeframes():
    return SUPPORTED_TIMEFRAMES.copy()


def get_defillama_supported_assets():
    return list(DEFILLAMA_CHAIN_MAP.keys())


def build_raw_defillama_chain_tvl_filename(asset_symbol, frequency=DEFILLAMA_FREQUENCY):
    return f"{asset_symbol}_defillama_chain_tvl_raw_{frequency}.csv"


def build_raw_defillama_chain_snapshot_filename():
    return "defillama_chains_snapshot.csv"


def build_defillama_features_filename(asset_symbol, frequency=DEFILLAMA_FREQUENCY):
    return f"{asset_symbol}_defillama_features_{frequency}.csv"


def build_defillama_summary_filename(frequency=DEFILLAMA_FREQUENCY):
    return f"defillama_summary_{frequency}.csv"


def get_raw_defillama_chain_tvl_path(asset_symbol, frequency=DEFILLAMA_FREQUENCY):
    return RAW_DEFI_LLAMA_DIR / build_raw_defillama_chain_tvl_filename(asset_symbol, frequency)


def get_raw_defillama_chain_snapshot_path():
    return RAW_DEFI_LLAMA_DIR / build_raw_defillama_chain_snapshot_filename()


def get_defillama_features_path(asset_symbol, frequency=DEFILLAMA_FREQUENCY):
    return PROCESSED_DIR / build_defillama_features_filename(asset_symbol, frequency)


def get_defillama_summary_path(frequency=DEFILLAMA_FREQUENCY):
    return PROCESSED_DIR / build_defillama_summary_filename(frequency)


def build_market_coefficients_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                      start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_baseline_coefficients{_build_date_suffix(start_date, end_date)}.csv"


def build_market_feature_importance_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                            start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_baseline_feature_importance{_build_date_suffix(start_date, end_date)}.csv"


def get_market_coefficients_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                 start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_coefficients_filename(symbol, timeframe, start_date, end_date)


def get_market_feature_importance_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                       start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_feature_importance_filename(symbol, timeframe, start_date, end_date)


def build_evaluation_metrics_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                      model_name="baseline", start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_{model_name}_evaluation_metrics{_build_date_suffix(start_date, end_date)}.csv"


def build_confusion_matrix_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                    model_name="baseline", start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_{model_name}_confusion_matrix{_build_date_suffix(start_date, end_date)}.csv"


def get_evaluation_metrics_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                model_name="baseline", start_date=None, end_date=None):
    return PROCESSED_DIR / build_evaluation_metrics_filename(
        symbol,
        timeframe,
        model_name,
        start_date,
        end_date,
    )


def get_confusion_matrix_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                              model_name="baseline", start_date=None, end_date=None):
    return PROCESSED_DIR / build_confusion_matrix_filename(
        symbol,
        timeframe,
        model_name,
        start_date,
        end_date,
    )


def build_market_summary_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                  start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_summary{_build_date_suffix(start_date, end_date)}.csv"


def build_market_overview_filename(timeframe=DEFAULT_TIMEFRAME, start_date=None, end_date=None):
    return f"market_overview_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_summary_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                            start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_summary_filename(symbol, timeframe, start_date, end_date)


def get_market_overview_path(timeframe=DEFAULT_TIMEFRAME, start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_overview_filename(timeframe, start_date, end_date)


def build_raw_futures_mark_price_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                          start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    return f"{symbol}_{timeframe}_{start_date}_{end_date}_futures_mark_price.csv"


def build_raw_futures_funding_rate_filename(symbol=DEFAULT_SYMBOL,
                                            start_date=DEFAULT_START_DATE,
                                            end_date=DEFAULT_END_DATE):
    return f"{symbol}_{start_date}_{end_date}_futures_funding_rate.csv"


def build_raw_futures_open_interest_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                             start_date=DEFAULT_START_DATE,
                                             end_date=DEFAULT_END_DATE):
    return f"{symbol}_{timeframe}_{start_date}_{end_date}_futures_open_interest.csv"


def build_raw_futures_long_short_ratio_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                                start_date=DEFAULT_START_DATE,
                                                end_date=DEFAULT_END_DATE):
    return f"{symbol}_{timeframe}_{start_date}_{end_date}_futures_long_short_ratio.csv"


def build_raw_futures_taker_volume_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                            start_date=DEFAULT_START_DATE,
                                            end_date=DEFAULT_END_DATE):
    return f"{symbol}_{timeframe}_{start_date}_{end_date}_futures_taker_volume.csv"


def build_raw_futures_basis_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                     start_date=DEFAULT_START_DATE,
                                     end_date=DEFAULT_END_DATE,
                                     contract_type=FUTURES_CONTRACT_TYPE):
    contract_label = contract_type.lower()
    return f"{symbol}_{timeframe}_{start_date}_{end_date}_futures_basis_{contract_label}.csv"


def get_raw_futures_mark_price_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                    start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    return RAW_FUTURES_DIR / build_raw_futures_mark_price_filename(symbol, timeframe, start_date, end_date)


def get_raw_futures_funding_rate_path(symbol=DEFAULT_SYMBOL,
                                      start_date=DEFAULT_START_DATE,
                                      end_date=DEFAULT_END_DATE):
    return RAW_FUTURES_DIR / build_raw_futures_funding_rate_filename(symbol, start_date, end_date)


def get_raw_futures_open_interest_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                       start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    return RAW_FUTURES_DIR / build_raw_futures_open_interest_filename(symbol, timeframe, start_date, end_date)


def get_raw_futures_long_short_ratio_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                          start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    return RAW_FUTURES_DIR / build_raw_futures_long_short_ratio_filename(symbol, timeframe, start_date, end_date)


def get_raw_futures_taker_volume_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                      start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    return RAW_FUTURES_DIR / build_raw_futures_taker_volume_filename(symbol, timeframe, start_date, end_date)


def get_raw_futures_basis_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                               start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                               contract_type=FUTURES_CONTRACT_TYPE):
    return RAW_FUTURES_DIR / build_raw_futures_basis_filename(
        symbol,
        timeframe,
        start_date,
        end_date,
        contract_type=contract_type,
    )


def build_futures_features_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                    start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_futures_features{_build_date_suffix(start_date, end_date)}.csv"


def build_market_futures_dataset_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                          start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_futures_dataset{_build_date_suffix(start_date, end_date)}.csv"


def get_futures_features_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                              start_date=None, end_date=None):
    return PROCESSED_DIR / build_futures_features_filename(symbol, timeframe, start_date, end_date)


def get_market_futures_dataset_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                    start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_dataset_filename(symbol, timeframe, start_date, end_date)


def build_market_futures_strategy_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                   start_date=None, end_date=None):
    return f"market_futures_strategy_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_strategy_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                             start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_strategy_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_target_variant_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                         start_date=None, end_date=None):
    return f"market_futures_target_variant_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_target_variant_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                   start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_target_variant_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_preferred_model_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                          start_date=None, end_date=None):
    return f"market_futures_preferred_model_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_preferred_model_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                    start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_preferred_model_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_signal_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                 start_date=None, end_date=None):
    return f"market_futures_signal_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_signal_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                           start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_signal_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_backtest_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                   start_date=None, end_date=None):
    return f"market_futures_backtest_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_backtest_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                             start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_backtest_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_policy_variant_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                         start_date=None, end_date=None):
    return f"market_futures_policy_variant_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_policy_variant_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                   start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_policy_variant_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_walkforward_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                      start_date=None, end_date=None):
    return f"market_futures_walkforward_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_walkforward_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_walkforward_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_walkforward_detail_filename(timeframe=DEFAULT_TIMEFRAME,
                                                     start_date=None, end_date=None):
    return f"market_futures_walkforward_detail_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_walkforward_detail_path(timeframe=DEFAULT_TIMEFRAME,
                                               start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_walkforward_detail_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_backtest_curve_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                                 start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_futures_backtest_curve{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_backtest_curve_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                           start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_backtest_curve_filename(
        symbol,
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_binary_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                 start_date=None, end_date=None):
    return f"market_futures_binary_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_binary_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                           start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_binary_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_binary_walkforward_detail_filename(timeframe=DEFAULT_TIMEFRAME,
                                                            start_date=None, end_date=None):
    return f"market_futures_binary_walkforward_detail_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_binary_walkforward_detail_path(timeframe=DEFAULT_TIMEFRAME,
                                                      start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_binary_walkforward_detail_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_binary_walkforward_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                             start_date=None, end_date=None):
    return f"market_futures_binary_walkforward_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_binary_walkforward_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                       start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_binary_walkforward_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_binary_backtest_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                          start_date=None, end_date=None):
    return f"market_futures_binary_backtest_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_binary_backtest_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                    start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_binary_backtest_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_binary_policy_variant_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                                start_date=None, end_date=None):
    return f"market_futures_binary_policy_variant_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_binary_policy_variant_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                          start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_binary_policy_variant_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_futures_binary_backtest_curve_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                                        start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_futures_binary_backtest_curve{_build_date_suffix(start_date, end_date)}.csv"


def get_market_futures_binary_backtest_curve_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                                  start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_futures_binary_backtest_curve_filename(
        symbol,
        timeframe,
        start_date,
        end_date,
    )


def build_market_intelligence_overview_filename(timeframe=DEFAULT_TIMEFRAME,
                                                start_date=None, end_date=None):
    return f"market_intelligence_overview_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_intelligence_overview_path(timeframe=DEFAULT_TIMEFRAME,
                                          start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_intelligence_overview_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_intelligence_refresh_manifest_filename():
    return "market_intelligence_refresh_manifest.json"


def get_market_intelligence_refresh_manifest_path():
    return PROCESSED_DIR / build_market_intelligence_refresh_manifest_filename()


def build_market_multimodal_dataset_filename(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                             start_date=None, end_date=None):
    return f"{symbol}_{timeframe}_market_multimodal_dataset{_build_date_suffix(start_date, end_date)}.csv"


def get_market_multimodal_dataset_path(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME,
                                       start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_multimodal_dataset_filename(
        symbol,
        timeframe,
        start_date,
        end_date,
    )


def build_market_multimodal_strategy_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                      start_date=None, end_date=None):
    return f"market_multimodal_strategy_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_multimodal_strategy_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_multimodal_strategy_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_context_ablation_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                   start_date=None, end_date=None):
    return f"market_context_ablation_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_context_ablation_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                             start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_context_ablation_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_trend_forecast_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                 start_date=None, end_date=None):
    return f"market_trend_forecast_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_trend_forecast_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                           start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_trend_forecast_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_trend_regression_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                   start_date=None, end_date=None):
    return f"market_trend_regression_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_trend_regression_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                             start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_trend_regression_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_trend_feature_review_filename(symbol=DEFAULT_SYMBOL,
                                               timeframe=DEFAULT_TIMEFRAME,
                                               start_date=None,
                                               end_date=None):
    return f"{symbol}_{timeframe}_market_trend_feature_review{_build_date_suffix(start_date, end_date)}.csv"


def get_market_trend_feature_review_path(symbol=DEFAULT_SYMBOL,
                                         timeframe=DEFAULT_TIMEFRAME,
                                         start_date=None,
                                         end_date=None):
    return PROCESSED_DIR / build_market_trend_feature_review_filename(
        symbol,
        timeframe,
        start_date,
        end_date,
    )


def build_market_trend_walkforward_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                    start_date=None, end_date=None):
    return f"market_trend_walkforward_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_trend_walkforward_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                              start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_trend_walkforward_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_market_trend_walkforward_detail_filename(timeframe=DEFAULT_TIMEFRAME,
                                                   start_date=None, end_date=None):
    return f"market_trend_walkforward_detail_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_market_trend_walkforward_detail_path(timeframe=DEFAULT_TIMEFRAME,
                                             start_date=None, end_date=None):
    return PROCESSED_DIR / build_market_trend_walkforward_detail_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_strategy_backbone_comparison_filename(timeframe=DEFAULT_TIMEFRAME,
                                                start_date=None, end_date=None):
    return f"strategy_backbone_comparison_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_strategy_backbone_comparison_path(timeframe=DEFAULT_TIMEFRAME,
                                          start_date=None, end_date=None):
    return PROCESSED_DIR / build_strategy_backbone_comparison_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_cross_asset_relative_strength_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                                         start_date=None, end_date=None):
    return f"cross_asset_relative_strength_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_cross_asset_relative_strength_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                                   start_date=None, end_date=None):
    return PROCESSED_DIR / build_cross_asset_relative_strength_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_structural_break_summary_filename(timeframe=DEFAULT_TIMEFRAME,
                                            start_date=None, end_date=None):
    return f"structural_break_summary_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_structural_break_summary_path(timeframe=DEFAULT_TIMEFRAME,
                                      start_date=None, end_date=None):
    return PROCESSED_DIR / build_structural_break_summary_filename(
        timeframe,
        start_date,
        end_date,
    )


def build_structural_break_detail_filename(timeframe=DEFAULT_TIMEFRAME,
                                           start_date=None, end_date=None):
    return f"structural_break_detail_{timeframe}{_build_date_suffix(start_date, end_date)}.csv"


def get_structural_break_detail_path(timeframe=DEFAULT_TIMEFRAME,
                                     start_date=None, end_date=None):
    return PROCESSED_DIR / build_structural_break_detail_filename(
        timeframe,
        start_date,
        end_date,
    )


def get_asset_config(asset_symbol):
    return ASSET_REGISTRY[asset_symbol]


def build_raw_sentiment_filename(frequency=SENTIMENT_FREQUENCY):
    return f"fear_greed_raw_{frequency}.csv"


def build_sentiment_features_filename(frequency=SENTIMENT_FREQUENCY):
    return f"sentiment_features_{frequency}.csv"


def build_sentiment_summary_filename(frequency=SENTIMENT_FREQUENCY):
    return f"sentiment_summary_{frequency}.csv"


def get_raw_sentiment_path(frequency=SENTIMENT_FREQUENCY):
    return RAW_SENTIMENT_DIR / build_raw_sentiment_filename(frequency)


def get_sentiment_features_path(frequency=SENTIMENT_FREQUENCY):
    return PROCESSED_DIR / build_sentiment_features_filename(frequency)


def get_sentiment_summary_path(frequency=SENTIMENT_FREQUENCY):
    return PROCESSED_DIR / build_sentiment_summary_filename(frequency)


def build_raw_gdelt_articles_filename(asset_symbol, start_date=None, end_date=None):
    return f"{asset_symbol}_gdelt_articles{_build_date_suffix(start_date, end_date)}.csv"


def build_gdelt_sentiment_features_filename(asset_symbol, frequency=SENTIMENT_FREQUENCY,
                                            start_date=None, end_date=None):
    return (
        f"{asset_symbol}_gdelt_sentiment_features_{frequency}"
        f"{_build_date_suffix(start_date, end_date)}.csv"
    )


def build_gdelt_sentiment_summary_filename(asset_symbol, frequency=SENTIMENT_FREQUENCY,
                                           start_date=None, end_date=None):
    return (
        f"{asset_symbol}_gdelt_sentiment_summary_{frequency}"
        f"{_build_date_suffix(start_date, end_date)}.csv"
    )


def get_raw_gdelt_articles_path(asset_symbol, start_date=None, end_date=None):
    return RAW_GDELT_DIR / build_raw_gdelt_articles_filename(asset_symbol, start_date, end_date)


def get_gdelt_sentiment_features_path(asset_symbol, frequency=SENTIMENT_FREQUENCY,
                                      start_date=None, end_date=None):
    return PROCESSED_DIR / build_gdelt_sentiment_features_filename(
        asset_symbol,
        frequency,
        start_date,
        end_date,
    )


def get_gdelt_sentiment_summary_path(asset_symbol, frequency=SENTIMENT_FREQUENCY,
                                     start_date=None, end_date=None):
    return PROCESSED_DIR / build_gdelt_sentiment_summary_filename(
        asset_symbol,
        frequency,
        start_date,
        end_date,
    )


def get_supported_onchain_assets():
    return [
        asset
        for asset in CORE_ONCHAIN_ASSETS
        if ASSET_REGISTRY[asset]["support_flags"]["onchain_supported"]
    ]


def build_raw_onchain_filename(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return f"{asset_symbol}_coinmetrics_raw_{frequency}.csv"


def build_onchain_features_filename(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return f"{asset_symbol}_onchain_features_{frequency}.csv"


def build_onchain_summary_filename(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return f"{asset_symbol}_onchain_summary_{frequency}.csv"


def build_onchain_overview_filename(frequency=ONCHAIN_FREQUENCY):
    return f"onchain_overview_{frequency}.csv"


def get_raw_onchain_path(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return RAW_ONCHAIN_DIR / build_raw_onchain_filename(asset_symbol, frequency)


def get_onchain_features_path(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return PROCESSED_DIR / build_onchain_features_filename(asset_symbol, frequency)


def get_onchain_summary_path(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return PROCESSED_DIR / build_onchain_summary_filename(asset_symbol, frequency)


def get_onchain_overview_path(frequency=ONCHAIN_FREQUENCY):
    return PROCESSED_DIR / build_onchain_overview_filename(frequency)


def build_market_onchain_dataset_filename(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return f"{asset_symbol}_market_onchain_dataset_{frequency}.csv"


def build_market_onchain_summary_filename(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return f"{asset_symbol}_market_onchain_summary_{frequency}.csv"


def build_market_onchain_overview_filename(frequency=ONCHAIN_FREQUENCY):
    return f"market_onchain_overview_{frequency}.csv"


def get_market_onchain_dataset_path(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return PROCESSED_DIR / build_market_onchain_dataset_filename(asset_symbol, frequency)


def get_market_onchain_summary_path(asset_symbol, frequency=ONCHAIN_FREQUENCY):
    return PROCESSED_DIR / build_market_onchain_summary_filename(asset_symbol, frequency)


def get_market_onchain_overview_path(frequency=ONCHAIN_FREQUENCY):
    return PROCESSED_DIR / build_market_onchain_overview_filename(frequency)


def build_strategy_summary_filename(strategy_group, frequency=ONCHAIN_FREQUENCY,
                                    start_date=None, end_date=None):
    return f"{strategy_group}_strategy_summary_{frequency}{_build_date_suffix(start_date, end_date)}.csv"


def get_strategy_summary_path(strategy_group, frequency=ONCHAIN_FREQUENCY,
                              start_date=None, end_date=None):
    return PROCESSED_DIR / build_strategy_summary_filename(
        strategy_group,
        frequency,
        start_date,
        end_date,
    )
