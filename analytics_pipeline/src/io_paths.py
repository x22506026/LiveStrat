# this file defines where data and logs are stored for the analytics pipeline

from pathlib import Path

# project root directory (analytics_pipeline)
ROOT_DIR = Path(__file__).resolve().parents[1]

# data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_BINANCE_DIR = RAW_DIR / "binance"
RAW_FUTURES_DIR = RAW_DIR / "binance_futures"
RAW_SENTIMENT_DIR = RAW_DIR / "sentiment"
RAW_GDELT_DIR = RAW_DIR / "gdelt_news"
RAW_ONCHAIN_DIR = RAW_DIR / "coinmetrics"
RAW_DEFI_LLAMA_DIR = RAW_DIR / "defillama"
LOGS_DIR = DATA_DIR / "logs"
FUTURES_LOGS_DIR = LOGS_DIR / "binance_futures"
SENTIMENT_LOGS_DIR = LOGS_DIR / "sentiment"
GDELT_LOGS_DIR = LOGS_DIR / "gdelt_news"
ONCHAIN_LOGS_DIR = LOGS_DIR / "coinmetrics"
DEFI_LLAMA_LOGS_DIR = LOGS_DIR / "defillama"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_dirs():
    # create required directories if they do not exist
    RAW_BINANCE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FUTURES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_GDELT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ONCHAIN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DEFI_LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FUTURES_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SENTIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GDELT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ONCHAIN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DEFI_LLAMA_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
