"""Build matched recent spot + futures windows for supported symbols."""

from src.audit.audit_binance import run_binance_audit_for_symbol
from src.audit.audit_binance_futures import run_binance_futures_audit_for_symbol
from src.build.build_futures_features import build_futures_features_for_symbol
from src.build.build_labels import build_labels_for_symbol
from src.build.build_market_features import build_market_features_for_symbol
from src.build.build_market_futures_dataset import build_market_futures_dataset_for_symbol
from src.config import DEFAULT_END_DATE, DEFAULT_START_DATE, DEFAULT_TIMEFRAME, get_all_symbols


START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
TIMEFRAME = DEFAULT_TIMEFRAME
RUN_ALL_SYMBOLS = True


def build_recent_market_futures_window_for_symbol(symbol, timeframe, start_date, end_date):
    """Run the full matched market + futures pipeline for one symbol."""
    run_binance_audit_for_symbol(symbol, timeframe, start_date, end_date, update_mode=False)
    build_market_features_for_symbol(symbol, timeframe, start_date, end_date)
    build_labels_for_symbol(symbol, timeframe, start_date, end_date)

    run_binance_futures_audit_for_symbol(symbol, timeframe, start_date, end_date)
    build_futures_features_for_symbol(symbol, timeframe, start_date, end_date)
    return build_market_futures_dataset_for_symbol(symbol, timeframe, start_date, end_date)


def build_recent_market_futures_windows(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE,
                                        symbols=None):
    """Run the matched market + futures window build for the configured symbol set."""
    if symbols is None:
        symbols = get_all_symbols() if RUN_ALL_SYMBOLS else [get_all_symbols()[0]]

    outputs = {}
    for symbol in symbols:
        outputs[symbol] = build_recent_market_futures_window_for_symbol(
            symbol,
            timeframe,
            start_date,
            end_date,
        )
    return outputs


if __name__ == "__main__":
    build_recent_market_futures_windows()
