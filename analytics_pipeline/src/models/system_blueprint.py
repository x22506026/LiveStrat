"""System-level data and strategy blueprint for LiveStrat."""


SEVEN_ASSET_MARKET_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
SEVEN_ASSET_CODE_UNIVERSE = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"]


def build_data_layer_registry():
    """Describe each major data layer, its role, and how it should be used."""
    return [
        {
            "id": "market_spot",
            "label": "Binance Spot Market",
            "source": "Binance spot klines / OHLCV",
            "cadence": ["1h", "4h", "1d"],
            "coverage": SEVEN_ASSET_MARKET_UNIVERSE,
            "frontend_roles": [
                "market display",
                "dashboard cards",
                "price charts",
                "basic trend summaries",
            ],
            "strategy_roles": [
                "core predictive layer",
                "feature engineering base",
                "target labelling base",
            ],
            "status": "core",
            "notes": (
                "This is the cleanest and most dependable layer. It should power both user-facing "
                "charts and the baseline signal engine."
            ),
        },
        {
            "id": "futures_structure",
            "label": "Binance Futures Structure",
            "source": "Binance futures mark price, funding, open interest, ratios, taker flow, basis",
            "cadence": ["1h", "4h", "1d"],
            "coverage": SEVEN_ASSET_MARKET_UNIVERSE,
            "frontend_roles": [
                "futures context summaries",
                "risk warnings",
                "crowding / leverage status",
            ],
            "strategy_roles": [
                "confirmation layer",
                "crowding / reversal features",
                "market agreement scoring",
            ],
            "status": "core",
            "notes": (
                "This layer should not replace spot price as the primary signal base. It works best "
                "as a structure and confirmation layer on top of market data."
            ),
        },
        {
            "id": "sentiment_broad",
            "label": "Broad Market Sentiment",
            "source": "Fear & Greed style daily market mood",
            "cadence": ["1d"],
            "coverage": ["BTC proxy", "market-wide fallback"],
            "frontend_roles": [
                "dashboard mood context",
                "market commentary",
                "notification explanations",
            ],
            "strategy_roles": [
                "fallback confirmation layer",
                "risk regime modifier",
            ],
            "status": "supporting",
            "notes": (
                "Useful as background context and fallback confirmation, but too broad to be treated "
                "as a strong asset-specific predictive source by itself."
            ),
        },
        {
            "id": "sentiment_asset_news",
            "label": "Asset-Specific News Sentiment",
            "source": "GDELT-based daily news sentiment features",
            "cadence": ["1d"],
            "coverage": SEVEN_ASSET_CODE_UNIVERSE,
            "frontend_roles": [
                "asset news status",
                "news regime summaries",
                "context explanation",
            ],
            "strategy_roles": [
                "confirmation / veto layer",
                "specialist context model input",
            ],
            "status": "experimental",
            "notes": (
                "Configured across the seven-asset universe through public GDELT/RSS-style news context. "
                "It should be treated as conditional context rather than a guaranteed predictive layer."
            ),
        },
        {
            "id": "onchain_daily",
            "label": "On-Chain Daily Context",
            "source": "Coin Metrics community-style daily on-chain features",
            "cadence": ["1d"],
            "coverage": ["BTC", "ETH", "XRP", "ADA", "DOGE", "SOL source-limited", "BNB stale-source-limited"],
            "frontend_roles": [
                "regime explanation",
                "holder / activity context",
                "risk context summaries",
            ],
            "strategy_roles": [
                "confirmation / veto layer",
                "daily context specialist strategies",
            ],
            "status": "experimental",
            "notes": (
                "On-chain is strong as a slower structural context layer. SOL has no current candidate "
                "metrics in the free Coin Metrics community set, and BNB's candidate metrics are historical/stale, "
                "so both should stay marked unavailable."
            ),
        },
        {
            "id": "defi_ecosystem",
            "label": "DeFi Ecosystem Context",
            "source": "DeFiLlama public chain TVL history",
            "cadence": ["1d"],
            "coverage": SEVEN_ASSET_CODE_UNIVERSE,
            "frontend_roles": [
                "ecosystem TVL cards",
                "chain activity trend charts",
                "SOL and BNB structural context",
                "context explanation",
            ],
            "strategy_roles": [
                "confirmation / caution layer",
                "ecosystem activity context",
                "multimodal context input",
            ],
            "status": "supporting",
            "notes": (
                "This fills a practical gap where Coin Metrics wallet-level support is unavailable or stale. "
                "It should be described as chain/ecosystem context, not wallet-level on-chain fundamentals."
            ),
        },
    ]


def build_model_family_audit():
    """Audit the current model families and recommend how to treat them."""
    return [
        {
            "id": "market_only_baselines",
            "scope": "spot market features only",
            "examples": [
                "scaled market baseline",
                "unscaled market baseline",
                "rule-based market benchmark",
            ],
            "prediction_target": "future return bucket -> buy / hold / dont_buy",
            "timeframe_support": ["1h", "4h", "1d"],
            "strengths": [
                "cleanest data foundation",
                "most explainable baseline",
                "best starting point for charts + signals separation",
            ],
            "weaknesses": [
                "can miss leverage-driven context",
                "currently not exposed cleanly by timeframe in the app",
            ],
            "recommendation": "keep_and_promote",
        },
        {
            "id": "market_futures_models",
            "scope": "spot + futures structure",
            "examples": [
                "market_futures_logistic_*",
                "market_futures_random_forest",
                "regime filter",
                "crowding reversal",
            ],
            "prediction_target": "future return bucket with futures confirmation",
            "timeframe_support": ["1h", "4h", "1d"],
            "strengths": [
                "good practical extension beyond price-only logic",
                "suits confirmation and reversal-style strategies",
            ],
            "weaknesses": [
                "family is larger than current UI presentation suggests",
                "needs clearer target/horizon selection per strategy",
            ],
            "recommendation": "keep_and_restructure",
        },
        {
            "id": "market_onchain_rules",
            "scope": "daily market + on-chain",
            "examples": [
                "market_onchain_regime_filter",
                "market_onchain_confirmation",
            ],
            "prediction_target": "next-day buy / hold / dont_buy",
            "timeframe_support": ["1d"],
            "strengths": [
                "good for slow structural context",
                "easy to explain as regime gating",
            ],
            "weaknesses": [
                "currently too narrow in asset coverage",
                "not suitable as the main intraday engine",
            ],
            "recommendation": "keep_as_specialist_only",
        },
        {
            "id": "multimodal_context_models",
            "scope": "spot + futures + sentiment + on-chain",
            "examples": [
                "market_multimodal_confirmation_gate",
                "market_multimodal_context_veto",
                "market_multimodal_logistic_*",
                "validation-selected specialists",
            ],
            "prediction_target": "future return bucket with context-aware confirmation",
            "timeframe_support": ["4h currently"],
            "strengths": [
                "closest to the long-term project vision",
                "lets context act as confirmation, veto, or specialist input",
            ],
            "weaknesses": [
                "coverage instability for sentiment/on-chain",
                "currently too experimental to present as the default live engine",
            ],
            "recommendation": "keep_experimental_and_compare_against_core",
        },
    ]


def build_strategy_architecture():
    """Define the intended long-term strategy stack for the product."""
    return {
        "display_lane": {
            "purpose": "Readable market and context information for users.",
            "uses": [
                "latest price and return",
                "market charts",
                "volume and volatility",
                "market mood and on-chain notes",
                "asset status summaries",
            ],
            "must_not_claim": [
                "this is not direct ML prediction output",
                "this is not a trade recommendation by itself",
            ],
        },
        "decision_lane": {
            "purpose": "Validated strategy and model outputs used for signals and saved profiles.",
            "uses": [
                "strategy presets",
                "custom builder resolution",
                "buy / hold / dont_buy signals",
                "confidence / caution states",
                "backtest and walk-forward summaries",
            ],
            "must_expose": [
                "asset",
                "timeframe",
                "strategy family",
                "data layers used",
                "evaluation basis",
                "refresh date",
            ],
        },
        "strategy_tiers": [
            {
                "id": "core_market",
                "label": "Core Market",
                "purpose": "Spot-market baseline strategies with the clearest interpretability.",
                "default_data_layers": ["market_spot"],
            },
            {
                "id": "enhanced_market_futures",
                "label": "Enhanced Market + Futures",
                "purpose": "Primary production-facing strategy tier for this project.",
                "default_data_layers": ["market_spot", "futures_structure"],
            },
            {
                "id": "context_aware_confirmations",
                "label": "Context-Aware Confirmations",
                "purpose": "Use sentiment and on-chain as confirmation, veto, and risk modifiers.",
                "default_data_layers": [
                    "market_spot",
                    "futures_structure",
                    "sentiment_broad",
                    "sentiment_asset_news",
                    "onchain_daily",
                ],
            },
            {
                "id": "specialist_experimental",
                "label": "Specialist Experimental",
                "purpose": "Research-only variants that test whether multimodal context materially helps.",
                "default_data_layers": [
                    "market_spot",
                    "futures_structure",
                    "sentiment_asset_news",
                    "onchain_daily",
                ],
            },
        ],
        "next_actions": [
            "separate Markets endpoints from strategy/model endpoints",
            "make timeframe explicit in every strategy response",
            "treat sentiment and on-chain as confirmation layers before promoting them to primary predictors",
            "keep multimodal models benchmarked against market-only and market+futures baselines",
        ],
    }


def build_strategy_research_program():
    """Define the concrete modelling program that should drive LiveStrat next."""
    return {
        "principles": [
            "Build a small number of strategy families with clear purpose rather than many disconnected models.",
            "Treat market + futures as the main predictive backbone until context layers prove durable uplift.",
            "Use sentiment and on-chain first as confirmation, veto, or regime context, not mandatory alpha sources.",
            "Handle structural breaks explicitly so the system can react when market behaviour changes sharply.",
            "Promote only strategies that survive held-out, walk-forward, and audit checks.",
        ],
        "families": [
            {
                "id": "market_trend_forecast",
                "label": "Market Trend Forecast",
                "role": "clean interpretable benchmark family",
                "core_question": "Can spot-market features alone forecast directional market posture at 1h and 4h?",
                "data_layers": ["market_spot"],
                "primary_targets": [
                    "buy_hold_dont_buy classification",
                    "future return regression",
                ],
                "candidate_methods": [
                    "logistic regression",
                    "random forest",
                    "gradient boosting",
                    "quantile or mean return regression",
                ],
                "must_have_outputs": [
                    "benchmark leaderboard",
                    "feature importance or coefficient inspection",
                    "walk-forward summary",
                    "asset-wise stability notes",
                ],
                "delivery_status": "build_first",
            },
            {
                "id": "market_futures_structure",
                "label": "Market + Futures Structure",
                "role": "primary LiveStrat backbone",
                "core_question": "Do leverage, crowding, and futures alignment materially improve directional forecasting and execution posture?",
                "data_layers": ["market_spot", "futures_structure"],
                "primary_targets": [
                    "buy_hold_dont_buy classification",
                    "regime-aware execution policy",
                ],
                "candidate_methods": [
                    "logistic regression",
                    "random forest",
                    "gradient boosting",
                    "rule-based regime filter",
                    "crowding reversal detector",
                ],
                "must_have_outputs": [
                    "policy comparison",
                    "walk-forward evaluation",
                    "target-horizon comparison",
                    "futures audit and feature reliability notes",
                ],
                "delivery_status": "build_first",
            },
            {
                "id": "context_confirmation",
                "label": "Context Confirmation",
                "role": "secondary multimodal improvement family",
                "core_question": "When market + futures is already active, do sentiment or on-chain layers improve confidence, veto bad entries, or explain regime shifts?",
                "data_layers": [
                    "market_spot",
                    "futures_structure",
                    "sentiment_broad",
                    "sentiment_asset_news",
                    "onchain_daily",
                ],
                "primary_targets": [
                    "confirmation gate",
                    "context veto",
                    "uplift versus core backbone",
                ],
                "candidate_methods": [
                    "specialist logistic comparisons",
                    "ablation framework",
                    "gated multimodal rules",
                    "stacked confirmation scores",
                ],
                "must_have_outputs": [
                    "delta versus market_futures_only",
                    "coverage-aware readiness labels",
                    "context-specific uplift summary",
                    "fallback handling when sentiment or on-chain is missing",
                ],
                "delivery_status": "build_second",
            },
            {
                "id": "structural_break_and_regime_change",
                "label": "Structural Break and Regime Change",
                "role": "stability and robustness family",
                "core_question": "Can LiveStrat detect when the recent market regime has shifted enough that older relationships become unreliable?",
                "data_layers": [
                    "market_spot",
                    "futures_structure",
                    "sentiment_broad",
                    "sentiment_asset_news",
                    "onchain_daily",
                ],
                "primary_targets": [
                    "regime_shift detection",
                    "stability warning states",
                    "model reliability downgrade flags",
                ],
                "candidate_methods": [
                    "rolling distribution shift tests",
                    "change-point detection",
                    "CUSUM-style break tests",
                    "volatility state classifiers",
                    "crowding anomaly detection",
                ],
                "must_have_outputs": [
                    "break alerts and diagnostics",
                    "reduced-trust mode triggers",
                    "asset-wise change summaries",
                    "recent-window audit reports",
                ],
                "delivery_status": "build_second",
            },
            {
                "id": "specialist_research_models",
                "label": "Specialist Research Models",
                "role": "academic breadth without pretending production readiness",
                "core_question": "Which additional techniques are worth comparing once the core backbone and stability family are already in place?",
                "data_layers": [
                    "market_spot",
                    "futures_structure",
                    "sentiment_asset_news",
                    "onchain_daily",
                ],
                "primary_targets": [
                    "specialist forecasting comparison",
                    "research-only variants",
                ],
                "candidate_methods": [
                    "sequence models",
                    "hidden-state or clustering regimes",
                    "anomaly detection for whale or crowding events",
                    "advanced return forecasting",
                ],
                "must_have_outputs": [
                    "comparison against simpler baselines",
                    "clear experimental labelling",
                    "discard criteria for underperforming complexity",
                ],
                "delivery_status": "build_later",
            },
        ],
        "implementation_order": [
            "Finalize strategy family definitions and website-facing labels.",
            "Complete the Market Trend Forecast family with strong baselines and regression comparison.",
            "Complete the Market + Futures Structure family as the main deployment candidate.",
            "Add the Context Confirmation family and benchmark every variant against market_futures_only.",
            "Build Structural Break and Regime Change handling so LiveStrat can warn when behaviour shifts sharply.",
            "Only then add Specialist Research Models that justify their complexity.",
        ],
        "evaluation_contract": [
            "Every strategy family must define its target, timeframe, asset scope, and data layers.",
            "Every promoted strategy must have held-out metrics, walk-forward results, and an audit summary.",
            "Every saved strategy shown to users must map to a real evaluated family, not only a UI idea.",
            "Every experimental strategy must clearly state why it is not yet a production candidate.",
        ],
        "data_expansion_notes": [
            "Free and ethical sources are acceptable if they materially improve one clearly defined family.",
            "Recent-window datasets should be retained because structural-break detection depends on fresh behaviour monitoring.",
            "Whale-style event proxies can be explored, but they should be framed as anomaly or regime features rather than guaranteed predictors.",
        ],
    }


def build_system_blueprint():
    """Return the full product-level data and strategy blueprint."""
    return {
        "data_layers": build_data_layer_registry(),
        "model_audit": build_model_family_audit(),
        "strategy_architecture": build_strategy_architecture(),
        "strategy_research_program": build_strategy_research_program(),
    }
