"""Run the dated market intelligence pipeline used by the LiveStrat app."""

import argparse

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.audit.audit_defillama import run_defillama_audit_for_assets
from src.audit.audit_gdelt_news import run_gdelt_audit_for_assets
from src.audit.audit_rss_news import run_rss_audit_for_assets
from src.build.build_gdelt_sentiment_features import build_gdelt_sentiment_features_for_assets
from src.build.build_defillama_features import build_defillama_features_for_assets
from src.build.build_market_multimodal_dataset import build_market_multimodal_datasets
from src.build.build_onchain_features import build_onchain_features_for_all_supported_assets
from src.build.build_recent_market_futures_windows import build_recent_market_futures_windows
from src.build.build_sentiment_features import build_sentiment_features
from src.config import (
    DEFAULT_TIMEFRAME,
    GDELT_ASSET_QUERY_MAP,
    get_all_symbols,
    get_all_timeframes,
    get_defillama_features_path,
    get_defillama_summary_path,
    get_gdelt_sentiment_features_path,
    get_gdelt_sentiment_summary_path,
    get_market_context_ablation_summary_path,
    get_market_intelligence_overview_path,
    get_market_multimodal_dataset_path,
    get_market_multimodal_strategy_summary_path,
    get_onchain_features_path,
    get_raw_gdelt_articles_path,
    get_sentiment_features_path,
    get_supported_onchain_assets,
)
from src.io_paths import ensure_dirs
from src.models.evaluate_market_futures_backtests import evaluate_market_futures_backtests
from src.models.evaluate_market_futures_walkforward import evaluate_market_futures_walkforward
from src.models.evaluate_market_context_ablations import evaluate_market_context_ablations
from src.models.evaluate_market_multimodal_strategies import evaluate_market_multimodal_strategies
from src.models.evaluate_market_futures_preferred_models import evaluate_market_futures_preferred_models
from src.models.evaluate_market_futures_strategies import evaluate_market_futures_strategies
from src.models.pipeline_refresh import write_pipeline_refresh_manifest
from src.reports.build_gdelt_sentiment_summary import build_gdelt_sentiment_summary_for_assets
from src.reports.build_defillama_summary import build_defillama_summary
from src.reports.build_market_futures_signal_summary import build_market_futures_signal_summary
from src.reports.build_market_intelligence_overview import build_market_intelligence_overview
from src.reports.make_plots import generate_market_reports


TIMEFRAME = DEFAULT_TIMEFRAME
RECENT_LOOKBACK_DAYS = 30


def _all_paths_exist(paths):
    return all(Path(path).exists() for path in paths)


def _context_output_paths(timeframe, start_date, end_date, symbols, include_gdelt_context):
    paths = [
        get_sentiment_features_path(),
        get_market_multimodal_strategy_summary_path(timeframe, start_date, end_date),
        get_market_context_ablation_summary_path(timeframe, start_date, end_date),
        get_defillama_summary_path(),
    ]

    supported_onchain_assets = set(get_supported_onchain_assets())
    for asset_symbol in supported_onchain_assets:
        paths.append(get_onchain_features_path(asset_symbol))

    for symbol in symbols:
        paths.append(get_market_multimodal_dataset_path(symbol, timeframe, start_date, end_date))
        paths.append(get_defillama_features_path(symbol.replace("USDT", "")))

    if include_gdelt_context:
        gdelt_assets = [
            symbol.replace("USDT", "")
            for symbol in symbols
            if symbol.replace("USDT", "") in GDELT_ASSET_QUERY_MAP
        ]
        for asset_symbol in gdelt_assets:
            paths.append(get_gdelt_sentiment_features_path(asset_symbol, start_date=start_date, end_date=end_date))
            paths.append(get_gdelt_sentiment_summary_path(asset_symbol, start_date=start_date, end_date=end_date))

    return paths


def _run_context_suite(timeframe, start_date, end_date, symbols, include_gdelt_context, resume_existing):
    """Build slower multimodal/news/on-chain context artifacts."""
    if resume_existing and _all_paths_exist(
        _context_output_paths(timeframe, start_date, end_date, symbols, include_gdelt_context)
    ):
        print("context suite already available for this window; skipping expensive context rebuild")
        return

    sentiment_features_path = get_sentiment_features_path()
    if resume_existing and sentiment_features_path.exists():
        print("reusing existing broad sentiment features")
    else:
        build_sentiment_features()

    onchain_assets = sorted(set(get_supported_onchain_assets()))
    onchain_feature_paths = [get_onchain_features_path(asset_symbol) for asset_symbol in onchain_assets]
    if resume_existing and _all_paths_exist(onchain_feature_paths):
        print("reusing existing on-chain feature set")
    else:
        build_onchain_features_for_all_supported_assets()

    defi_feature_paths = [get_defillama_features_path(symbol.replace("USDT", "")) for symbol in symbols]
    defi_summary_path = get_defillama_summary_path()
    if resume_existing and _all_paths_exist(defi_feature_paths + [defi_summary_path]):
        print("reusing existing DeFiLlama ecosystem context")
    else:
        run_defillama_audit_for_assets([symbol.replace("USDT", "") for symbol in symbols])
        build_defillama_features_for_assets([symbol.replace("USDT", "") for symbol in symbols])
        build_defillama_summary()

    if include_gdelt_context:
        gdelt_assets = [
            symbol.replace("USDT", "")
            for symbol in symbols
            if symbol.replace("USDT", "") in GDELT_ASSET_QUERY_MAP
        ]
        gdelt_feature_paths = [
            get_gdelt_sentiment_features_path(asset_symbol, start_date=start_date, end_date=end_date)
            for asset_symbol in gdelt_assets
        ]
        gdelt_summary_paths = [
            get_gdelt_sentiment_summary_path(asset_symbol, start_date=start_date, end_date=end_date)
            for asset_symbol in gdelt_assets
        ]
        if resume_existing and _all_paths_exist(gdelt_feature_paths + gdelt_summary_paths):
            print("reusing existing asset-news sentiment outputs for this window")
        else:
            try:
                raw_paths = [
                    get_raw_gdelt_articles_path(asset_symbol, start_date, end_date)
                    for asset_symbol in gdelt_assets
                ]
                if not (resume_existing and _all_paths_exist(raw_paths)):
                    run_gdelt_audit_for_assets(gdelt_assets, start_date=start_date, end_date=end_date)

                missing_assets = [
                    asset_symbol
                    for asset_symbol in gdelt_assets
                    if not get_raw_gdelt_articles_path(asset_symbol, start_date, end_date).exists()
                ]
                if missing_assets:
                    print("Trying RSS news fallback for missing asset-news windows")
                    run_rss_audit_for_assets(missing_assets, start_date=start_date, end_date=end_date)
                build_gdelt_sentiment_features_for_assets(gdelt_assets, start_date=start_date, end_date=end_date)
                build_gdelt_sentiment_summary_for_assets(gdelt_assets, start_date=start_date, end_date=end_date)
            except Exception as exc:  # pragma: no cover - network/model availability is environment-dependent
                print("GDELT sentiment stage skipped")
                print(str(exc))
                print("Trying RSS news fallback")
                run_rss_audit_for_assets(gdelt_assets, start_date=start_date, end_date=end_date)
                build_gdelt_sentiment_features_for_assets(gdelt_assets, start_date=start_date, end_date=end_date)
                build_gdelt_sentiment_summary_for_assets(gdelt_assets, start_date=start_date, end_date=end_date)

    multimodal_paths = [
        get_market_multimodal_dataset_path(symbol, timeframe, start_date, end_date)
        for symbol in symbols
    ]
    if resume_existing and _all_paths_exist(multimodal_paths):
        print("reusing existing multimodal datasets for this window")
    else:
        build_market_multimodal_datasets(timeframe=timeframe, start_date=start_date, end_date=end_date, symbols=symbols)

    multimodal_summary_path = get_market_multimodal_strategy_summary_path(timeframe, start_date, end_date)
    if resume_existing and multimodal_summary_path.exists():
        print("reusing existing multimodal strategy summary")
    else:
        evaluate_market_multimodal_strategies(timeframe=timeframe, start_date=start_date, end_date=end_date)

    context_ablation_path = get_market_context_ablation_summary_path(timeframe, start_date, end_date)
    if resume_existing and context_ablation_path.exists():
        print("reusing existing context ablation summary")
    else:
        evaluate_market_context_ablations(timeframe=timeframe, start_date=start_date, end_date=end_date)


def derive_recent_window(end_date=None, lookback_days=RECENT_LOOKBACK_DAYS):
    """Build an inclusive recent window that avoids relying on very old fixed dates."""
    if end_date is None:
        window_end = datetime.now(timezone.utc).date() - timedelta(days=1)
    else:
        window_end = datetime.strptime(end_date, "%Y-%m-%d").date()

    window_start = window_end - timedelta(days=lookback_days)
    return window_start.isoformat(), window_end.isoformat()


def run_market_intelligence_pipeline(timeframe=TIMEFRAME, start_date=None, end_date=None,
                                     lookback_days=RECENT_LOOKBACK_DAYS,
                                     include_core_suite=True,
                                     include_strategy_suite=True,
                                     include_gdelt_context=True,
                                     include_context_suite=True,
                                     resume_existing=True):
    """Run the unified spot + futures backend and write one app-facing overview file."""
    ensure_dirs()
    if start_date is None or end_date is None:
        start_date, end_date = derive_recent_window(end_date=end_date, lookback_days=lookback_days)

    symbols = get_all_symbols()

    try:
        if include_core_suite:
            build_recent_market_futures_windows(
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
            )
            generate_market_reports(
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
            )

            if include_strategy_suite:
                evaluate_market_futures_strategies(timeframe=timeframe, start_date=start_date, end_date=end_date)

            evaluate_market_futures_preferred_models(timeframe=timeframe, start_date=start_date, end_date=end_date)
            build_market_futures_signal_summary(timeframe=timeframe, start_date=start_date, end_date=end_date)
            evaluate_market_futures_backtests(timeframe=timeframe, start_date=start_date, end_date=end_date)
            evaluate_market_futures_walkforward(timeframe=timeframe, start_date=start_date, end_date=end_date)

        if include_context_suite:
            _run_context_suite(
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                include_gdelt_context=include_gdelt_context,
                resume_existing=resume_existing,
            )

        overview_df = build_market_intelligence_overview(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        write_pipeline_refresh_manifest(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            status="completed",
            row_count=len(overview_df),
            include_strategy_suite=include_strategy_suite,
            include_gdelt_context=include_gdelt_context,
            extra={
                "symbols": symbols,
                "include_core_suite": include_core_suite,
                "include_context_suite": include_context_suite,
                "resume_existing": resume_existing,
            },
        )

        print("market intelligence pipeline completed")
        print(f"timeframe: {timeframe}")
        print(f"window: {start_date} to {end_date}")
        print(f"symbols: {', '.join(symbols)}")
        print(f"overview rows: {len(overview_df)}")
        return overview_df
    except Exception as exc:
        write_pipeline_refresh_manifest(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            status="failed",
            row_count=0,
            include_strategy_suite=include_strategy_suite,
            include_gdelt_context=include_gdelt_context,
            error=str(exc),
            extra={
                "symbols": symbols,
                "include_core_suite": include_core_suite,
                "include_context_suite": include_context_suite,
                "resume_existing": resume_existing,
            },
        )
        raise


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the LiveStrat market intelligence pipeline.")
    parser.add_argument("--timeframe", default=TIMEFRAME, choices=("1h", "4h", "1d"))
    parser.add_argument("--start-date", dest="start_date")
    parser.add_argument("--end-date", dest="end_date")
    parser.add_argument("--lookback-days", dest="lookback_days", type=int, default=RECENT_LOOKBACK_DAYS)
    parser.add_argument("--skip-strategy-suite", action="store_true")
    parser.add_argument("--skip-gdelt-context", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--context-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.core_only and args.context_only:
        parser.error("--core-only and --context-only cannot be used together.")
    run_market_intelligence_pipeline(
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        include_core_suite=not args.context_only,
        include_strategy_suite=not args.skip_strategy_suite,
        include_gdelt_context=not args.skip_gdelt_context,
        include_context_suite=not args.core_only,
        resume_existing=not args.no_resume,
    )


if __name__ == "__main__":
    main()
