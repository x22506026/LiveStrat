"""Run the LiveStrat market intelligence pipeline across multiple timeframes."""

import argparse

from src.build.run_market_intelligence_pipeline import derive_recent_window, run_market_intelligence_pipeline


DEFAULT_TIMEFRAMES = ("1h", "4h", "1d")
DEFAULT_LOOKBACKS = {
    "1h": 14,
    "4h": 30,
    "1d": 120,
}


def run_multi_timeframe_market_intelligence_pipeline(
    timeframes=DEFAULT_TIMEFRAMES,
    end_date=None,
    lookbacks=None,
    include_core_suite=True,
    include_strategy_suite=True,
    include_gdelt_context=True,
    include_context_suite=True,
    resume_existing=True,
    continue_on_error=True,
):
    """Generate app-facing outputs for each requested timeframe."""
    lookbacks = {**DEFAULT_LOOKBACKS, **(lookbacks or {})}
    outputs = {}

    for timeframe in timeframes:
        lookback_days = lookbacks.get(timeframe, 30)
        start_date, resolved_end_date = derive_recent_window(
            end_date=end_date,
            lookback_days=lookback_days,
        )
        print(f"running market intelligence pipeline for {timeframe}")
        print(f"window: {start_date} to {resolved_end_date}")
        try:
            overview_df = run_market_intelligence_pipeline(
                timeframe=timeframe,
                start_date=start_date,
                end_date=resolved_end_date,
                include_core_suite=include_core_suite,
                include_strategy_suite=include_strategy_suite,
                include_gdelt_context=include_gdelt_context,
                include_context_suite=include_context_suite,
                resume_existing=resume_existing,
            )
            outputs[timeframe] = {
                "status": "completed",
                "start_date": start_date,
                "end_date": resolved_end_date,
                "rows": len(overview_df),
            }
        except Exception as exc:
            outputs[timeframe] = {
                "status": "failed",
                "start_date": start_date,
                "end_date": resolved_end_date,
                "rows": 0,
                "error": str(exc),
            }
            print(f"pipeline failed for {timeframe}")
            print(str(exc))
            if not continue_on_error:
                raise

    print("multi-timeframe market intelligence pipeline completed")
    print(f"timeframes: {', '.join(timeframes)}")
    return outputs


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the LiveStrat market intelligence pipeline across multiple timeframes.")
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), choices=list(DEFAULT_TIMEFRAMES))
    parser.add_argument("--end-date", dest="end_date")
    parser.add_argument("--skip-strategy-suite", action="store_true")
    parser.add_argument("--skip-gdelt-context", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--context-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.core_only and args.context_only:
        parser.error("--core-only and --context-only cannot be used together.")
    run_multi_timeframe_market_intelligence_pipeline(
        timeframes=tuple(args.timeframes),
        end_date=args.end_date,
        include_core_suite=not args.context_only,
        include_strategy_suite=not args.skip_strategy_suite,
        include_gdelt_context=not args.skip_gdelt_context,
        include_context_suite=not args.core_only,
        resume_existing=not args.no_resume,
        continue_on_error=not args.stop_on_error,
    )


if __name__ == "__main__":
    main()
