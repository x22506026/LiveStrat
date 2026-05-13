/* Analytics pulls structured strategy summaries from the backend. */

const marketSummaryScript = document.getElementById('market-summaries-data');
const marketSummaries = marketSummaryScript ? JSON.parse(marketSummaryScript.textContent) : {};

const analyticsAsset = document.getElementById('analytics-asset');
const analyticsTimeframe = document.getElementById('analytics-timeframe');
const analyticsRun = document.getElementById('analytics-run');

const analyticsSummary = document.getElementById('analytics-summary');
const analyticsRefreshNote = document.getElementById('analytics-refresh-note');
const analyticsLaneBadge = document.getElementById('analytics-lane-badge');
const analyticsActiveAsset = document.getElementById('analytics-active-asset');
const analyticsCurrentSignal = document.getElementById('analytics-current-signal');
const analyticsContextPosture = document.getElementById('analytics-context-posture');
const analyticsTargetHours = document.getElementById('analytics-target-hours');
const analyticsScaledAccuracy = document.getElementById('analytics-scaled-accuracy');
const analyticsRuleAccuracy = document.getElementById('analytics-rule-accuracy');
const analyticsReadiness = document.getElementById('analytics-readiness');
const analyticsEvaluationStrength = document.getElementById('analytics-evaluation-strength');
const analyticsSentimentRole = document.getElementById('analytics-sentiment-role');
const analyticsOnchainRole = document.getElementById('analytics-onchain-role');
const analyticsDefiRole = document.getElementById('analytics-defi-role');
const analyticsSourcePolicyTitle = document.getElementById('analytics-source-policy-title');
const analyticsSourcePolicyRationale = document.getElementById('analytics-source-policy-rationale');
const analyticsPolicyReturn = document.getElementById('analytics-policy-return');
const analyticsSignalExplainer = document.getElementById('analytics-signal-explainer');
const analyticsBuyHoldReturn = document.getElementById('analytics-buy-hold-return');
const analyticsSharpe = document.getElementById('analytics-sharpe');
const analyticsChart = document.getElementById('analytics-chart');
const analyticsChartCaption = document.getElementById('analytics-chart-caption');
const analyticsPulseTrend = document.getElementById('analytics-pulse-trend');
const analyticsPulseTrendBar = document.getElementById('analytics-pulse-trend-bar');
const analyticsPulseRange = document.getElementById('analytics-pulse-range');
const analyticsPulseRangeBar = document.getElementById('analytics-pulse-range-bar');
const analyticsPulseVolume = document.getElementById('analytics-pulse-volume');
const analyticsPulseVolumeBar = document.getElementById('analytics-pulse-volume-bar');
const analyticsWalkforwardFolds = document.getElementById('analytics-walkforward-folds');
const analyticsWalkforwardAcc = document.getElementById('analytics-walkforward-acc');
const analyticsWalkforwardExcess = document.getElementById('analytics-walkforward-excess');
const analyticsWhatNow = document.getElementById('analytics-what-now');
const analyticsWhyTrust = document.getElementById('analytics-why-trust');
const analyticsMethodSummary = document.getElementById('analytics-method-summary');
const analyticsPrimaryModel = document.getElementById('analytics-primary-model');
const analyticsPolicyName = document.getElementById('analytics-policy-name');
const analyticsProbabilityMode = document.getElementById('analytics-probability-mode');
const analyticsCalibrationTemperature = document.getElementById('analytics-calibration-temperature');
const analyticsBacktestSummary = document.getElementById('analytics-backtest-summary');
const analyticsMultimodalSummary = document.getElementById('analytics-multimodal-summary');
const analyticsMultimodalStrategy = document.getElementById('analytics-multimodal-strategy');
const analyticsMultimodalContext = document.getElementById('analytics-multimodal-context');
const analyticsMultimodalSignal = document.getElementById('analytics-multimodal-signal');
const analyticsMultimodalContextVariant = document.getElementById('analytics-multimodal-context-variant');
const analyticsMultimodalMacroF1 = document.getElementById('analytics-multimodal-macro-f1');
const analyticsMultimodalValidationMacroF1 = document.getElementById('analytics-multimodal-validation-macro-f1');
const analyticsEffectiveSentimentSource = document.getElementById('analytics-effective-sentiment-source');
const analyticsEffectiveSentimentLabel = document.getElementById('analytics-effective-sentiment-label');
const analyticsGdeltRegime = document.getElementById('analytics-gdelt-regime');
const analyticsGdeltCount = document.getElementById('analytics-gdelt-count');
const analyticsDefiChain = document.getElementById('analytics-defi-chain');
const analyticsDefiRegime = document.getElementById('analytics-defi-regime');
const analyticsMultimodalDetail = document.getElementById('analytics-multimodal-detail');
const analyticsContextGovernance = document.getElementById('analytics-context-governance');
const analyticsAblationSummary = document.getElementById('analytics-ablation-summary');
const analyticsAblationBestVariant = document.getElementById('analytics-ablation-best-variant');
const analyticsAblationBestMacroF1 = document.getElementById('analytics-ablation-best-macro-f1');
const analyticsAblationBaseMacroF1 = document.getElementById('analytics-ablation-base-macro-f1');
const analyticsAblationDelta = document.getElementById('analytics-ablation-delta');
const analyticsAblationDetail = document.getElementById('analytics-ablation-detail');
const analyticsCoverageSummary = document.getElementById('analytics-coverage-summary');
const analyticsCoverageMarketFutures = document.getElementById('analytics-coverage-market-futures');
const analyticsCoverageMultimodal = document.getElementById('analytics-coverage-multimodal');
const analyticsCoverageOnchain = document.getElementById('analytics-coverage-onchain');
const analyticsCoverageBaselines = document.getElementById('analytics-coverage-baselines');
const analyticsCoverageDetail = document.getElementById('analytics-coverage-detail');
const analyticsFamilyScopeSummary = document.getElementById('analytics-family-scope-summary');
const analyticsFamilyScopeDetail = document.getElementById('analytics-family-scope-detail');
const analyticsDefensibleComparison = document.getElementById('analytics-defensible-comparison');
const analyticsTimeframeReadinessSummary = document.getElementById('analytics-timeframe-readiness-summary');
const analyticsTimeframe1h = document.getElementById('analytics-timeframe-1h');
const analyticsTimeframe4h = document.getElementById('analytics-timeframe-4h');
const analyticsTimeframe1d = document.getElementById('analytics-timeframe-1d');
const analyticsTimeframeReadinessDetail = document.getElementById('analytics-timeframe-readiness-detail');
const analyticsCompareSummary = document.getElementById('analytics-compare-summary');
const analyticsCompareLeftTitle = document.getElementById('analytics-compare-left-title');
const analyticsCompareLeftSummary = document.getElementById('analytics-compare-left-summary');
const analyticsCompareRightTitle = document.getElementById('analytics-compare-right-title');
const analyticsCompareRightSummary = document.getElementById('analytics-compare-right-summary');
const analyticsCompareAccuracyWinner = document.getElementById('analytics-compare-accuracy-winner');
const analyticsCompareReturnWinner = document.getElementById('analytics-compare-return-winner');
const analyticsCompareInterpretabilityWinner = document.getElementById('analytics-compare-interpretability-winner');
const analyticsCompareDepthWinner = document.getElementById('analytics-compare-depth-winner');
const analyticsEvidenceSummary = document.getElementById('analytics-evidence-summary');
const analyticsEvidenceExport = document.getElementById('analytics-evidence-export');
const analyticsStrategyEvidenceBody = document.getElementById('analytics-strategy-evidence-body');
const analyticsDeepLearningBody = document.getElementById('analytics-deep-learning-body');
const analyticsStatusSignalBox = document.getElementById('analytics-status-signal-box');
const analyticsStatusReadinessBox = document.getElementById('analytics-status-readiness-box');
const analyticsStoryContextBox = document.getElementById('analytics-story-context-box');
const analyticsStoryTargetBox = document.getElementById('analytics-story-target-box');
const analyticsAccuracyBox = document.getElementById('analytics-accuracy-box');
const analyticsBalancedBox = document.getElementById('analytics-balanced-box');
const analyticsPolicyBox = document.getElementById('analytics-policy-box');
const analyticsCoverageBox = document.getElementById('analytics-coverage-box');
const analyticsSentimentBox = document.getElementById('analytics-sentiment-box');
const analyticsOnchainBox = document.getElementById('analytics-onchain-box');
const analyticsMultimodalBox = document.getElementById('analytics-multimodal-box');
const analyticsContextLabelBox = document.getElementById('analytics-context-label-box');
const analyticsMarketCoverageBox = document.getElementById('analytics-market-coverage-box');
const analyticsMultimodalCoverageBox = document.getElementById('analytics-multimodal-coverage-box');
const analyticsOnchainCoverageBox = document.getElementById('analytics-onchain-coverage-box');
const analyticsBaselineCoverageBox = document.getElementById('analytics-baseline-coverage-box');
const analyticsTimeframe1hBox = document.getElementById('analytics-timeframe-1h-box');
const analyticsTimeframe4hBox = document.getElementById('analytics-timeframe-4h-box');
const analyticsTimeframe1dBox = document.getElementById('analytics-timeframe-1d-box');

const STRATEGY_COMPARE_STORAGE_KEY = 'livestrat_strategy_compare_slots';

function formatSignal(value) {
    return (value || 'n/a').replaceAll('_', ' ');
}

function formatTitle(value) {
    const text = formatSignal(value);
    return text.charAt(0).toUpperCase() + text.slice(1);
}

function simplifyCopy(value) {
    return String(value || '')
        .replaceAll('decision lane', 'strategy view')
        .replaceAll('display lane', 'market view')
        .replaceAll('Decision analytics', 'Evaluation')
        .replaceAll('decision analytics', 'evaluation')
        .replaceAll('veto', 'warning')
        .replaceAll('fallback', 'backup')
        .replaceAll('Fallback', 'Backup')
        .replaceAll('slower structural', 'longer-term')
        .replaceAll('intraday timing', 'short-term timing')
        .replaceAll('intraday', 'short-term')
        .replaceAll('governance', 'rules')
        .replaceAll('Governance', 'Rules')
        .replaceAll('posture', 'status')
        .replaceAll('Posture', 'Status')
        .replaceAll('regime', 'state')
        .replaceAll('Regime', 'State')
        .replaceAll('prototype complexity', 'implementation depth')
        .replaceAll('genuinely available', 'available');
}

function formatDecisionLabel(value) {
    const normalized = String(value || '').toLowerCase();
    if (normalized === 'buy') {
        return 'Buy';
    }
    if (normalized === 'hold') {
        return 'Hold';
    }
    if (normalized === 'dont_buy' || normalized === 'do_not_buy' || normalized === 'sell') {
        return 'Avoid';
    }
    return normalized && normalized !== 'n/a' ? formatTitle(normalized) : 'n/a';
}

function formatAnalyticsStrategyLabel(value) {
    const raw = String(value || '').trim();
    if (!raw || raw.toLowerCase() === 'n/a') {
        return 'n/a';
    }

    const mappedLabels = {
        market_multimodal_specialist_market_futures_plus_onchain: 'Market + futures + on-chain',
        market_multimodal_specialist_market_futures_plus_gdelt: 'Market + futures + news sentiment',
        market_multimodal_specialist_market_futures_plus_fear_greed: 'Market + futures + broad sentiment',
        market_multimodal_specialist_full_multimodal: 'Full multimodal specialist',
        market_multimodal_validation_selected: 'Validation-selected multimodal',
        market_multimodal_confirmation_gate: 'Confirmation gate',
        market_multimodal_context_veto: 'Multimodal caution',
        market_futures_only: 'Market + futures only',
        market_futures_plus_onchain: 'Market + futures + on-chain',
        market_futures_plus_fear_greed: 'Market + futures + broad sentiment',
        market_futures_plus_gdelt: 'Market + futures + news sentiment',
        full_multimodal: 'Full multimodal',
    };

    if (mappedLabels[raw]) {
        return mappedLabels[raw];
    }

    if (raw.startsWith('market_multimodal_')) {
        return formatTitle(raw.replace('market_multimodal_', ''));
    }

    return formatTitle(raw);
}

function formatSentimentSource(value) {
    const raw = String(value || '').trim().toLowerCase();
    if (!raw || raw === 'n/a' || raw === 'unavailable') {
        return 'Not available';
    }
    const map = {
        gdelt_news: 'GDELT news',
        gdelt: 'GDELT news',
        fear_greed: 'Fear & Greed Index',
        alternative_me_fear_greed: 'Fear & Greed Index',
        fear_greed_market_fallback: 'Fear & Greed (backup)',
        broad_market_mood: 'Broad market mood',
    };
    return map[raw] || formatTitle(raw);
}

function formatSentimentLabel(value) {
    const raw = String(value || '').trim().toLowerCase();
    if (!raw || raw === 'n/a' || raw === 'unavailable') {
        return 'Not available';
    }
    const map = {
        supportive: 'Supportive',
        risk_on: 'Risk on',
        risk_off: 'Risk off',
        neutral: 'Neutral',
        mixed: 'Mixed',
        positive: 'Positive',
        negative: 'Negative',
    };
    return map[raw] || formatTitle(raw);
}

function formatPolicyLabel(value) {
    const raw = String(value || '').trim();
    if (!raw || raw.toLowerCase() === 'n/a') {
        return 'n/a';
    }

    const mappedLabels = {
        confidence_gated_long_flat: 'Confidence-gated long / flat',
        regime_adaptive_long_flat: 'Adaptive long / flat',
        conviction_weighted_long_only: 'Conviction-weighted long only',
        transparent_market_rule_set: 'Transparent market rule set',
        classification_only: 'Classification only',
    };

    return mappedLabels[raw] || formatTitle(raw);
}

function formatProbabilityLabel(value) {
    const raw = String(value || '').trim();
    if (!raw || raw.toLowerCase() === 'n/a') {
        return 'n/a';
    }

    const mappedLabels = {
        raw: 'Raw probabilities',
        temperature_scaled: 'Temperature-scaled',
        selected_probability_mode: 'Selected probability mode',
    };

    return mappedLabels[raw] || formatTitle(raw);
}

function formatPercent(value) {
    return `${((Number(value) || 0) * 100).toFixed(1)}%`;
}

function hasWalkforwardEvidence(performance) {
    return Number(performance?.walkforward_fold_count || 0) > 0;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function formatTimeframe(value) {
    return ({
        '1h': '1 hour',
        '4h': '4 hours',
        '1d': '1 day',
    }[value] || value || 'n/a');
}

function formatTimestamp(value) {
    if (!value) {
        return 'unknown refresh time';
    }
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'unknown refresh time' : date.toLocaleString();
}

function formatTimeframeList(values) {
    return Array.isArray(values) && values.length
        ? values.map((value) => formatTimeframe(value)).join(', ')
        : 'none';
}

function formatTargetSemantics(targetSemantics) {
    if (!targetSemantics || !targetSemantics.target_name) {
        return 'Target horizon semantics are not available yet.';
    }
    const hours = Number(targetSemantics.effective_horizon_hours || 0);
    const steps = Number(targetSemantics.horizon_steps || 0);
    const hoursText = hours ? `${hours}h` : 'unknown horizon';
    const stepText = steps ? `${steps} candle${steps === 1 ? '' : 's'}` : 'unknown steps';
    return `${formatTitle(targetSemantics.target_name)} resolves to ${stepText} (${hoursText}) on ${formatTimeframe(targetSemantics.requested_timeframe || targetSemantics.resolved_timeframe)}. ${targetSemantics.horizon_resolution_note || ''}`.trim();
}

function setToneClass(element, tone) {
    if (!element) {
        return;
    }
    element.classList.remove(
        'analytics-tone-positive',
        'analytics-tone-caution',
        'analytics-tone-negative',
        'analytics-tone-neutral',
    );
    element.classList.add(`analytics-tone-${tone}`);
}

function toneFromSignal(value) {
    const normalized = String(value || '').toLowerCase();
    if (normalized.includes('buy') && !normalized.includes('dont')) {
        return 'positive';
    }
    if (normalized.includes('avoid') || normalized.includes('dont') || normalized.includes('sell')) {
        return 'negative';
    }
    if (normalized.includes('hold') || normalized.includes('mixed')) {
        return 'caution';
    }
    return 'neutral';
}

function toneFromReadiness(value) {
    const normalized = String(value || '').toLowerCase();
    if (normalized === 'ready') {
        return 'positive';
    }
    if (normalized.includes('partial') || normalized.includes('fallback') || normalized.includes('reduced') || normalized.includes('moderate')) {
        return 'caution';
    }
    if (normalized.includes('experimental') || normalized.includes('missing') || normalized.includes('weak')) {
        return 'negative';
    }
    return 'neutral';
}

function toneFromContext(value) {
    const normalized = String(value || '').toLowerCase();
    if (normalized.includes('support') || normalized.includes('bullish') || normalized.includes('improv')) {
        return 'positive';
    }
    if (normalized.includes('mixed') || normalized.includes('neutral') || normalized.includes('moderate')) {
        return 'caution';
    }
    if (normalized.includes('unavailable') || normalized.includes('weak') || normalized.includes('bearish')) {
        return 'negative';
    }
    return 'neutral';
}

function toneFromNumericPercent(value) {
    const numeric = Number(value || 0);
    if (numeric > 0.45) {
        return 'positive';
    }
    if (numeric > 0.25) {
        return 'caution';
    }
    return 'negative';
}

function safeParseStorage(key, fallbackValue) {
    try {
        const rawValue = window.localStorage.getItem(key);
        return rawValue ? JSON.parse(rawValue) : fallbackValue;
    } catch (error) {
        return fallbackValue;
    }
}

function resolveWinnerLabel(leftConfig, rightConfig, metricAccessor) {
    if (!leftConfig || !rightConfig) {
        return 'n/a';
    }

    const leftValue = metricAccessor(leftConfig);
    const rightValue = metricAccessor(rightConfig);

    if (leftValue === rightValue) {
        return 'Tie';
    }

    return leftValue > rightValue
        ? `Left: ${leftConfig.strategy_name || 'Strategy'}`
        : `Right: ${rightConfig.strategy_name || 'Strategy'}`;
}

function renderAnalyticsComparison() {
    const slots = safeParseStorage(STRATEGY_COMPARE_STORAGE_KEY, { left: null, right: null });
    const leftConfig = slots.left;
    const rightConfig = slots.right;

    if (analyticsCompareLeftTitle) {
        analyticsCompareLeftTitle.textContent = leftConfig
            ? `${leftConfig.strategy_name || 'Strategy'} | ${leftConfig.asset || 'n/a'}`
            : 'No strategy loaded.';
    }

    if (analyticsCompareLeftSummary) {
        analyticsCompareLeftSummary.textContent = leftConfig
            ? leftConfig.scorecard?.scorecard_summary || 'Left comparison strategy loaded.'
            : 'Use the strategies page to send a profile into the left slot.';
    }

    if (analyticsCompareRightTitle) {
        analyticsCompareRightTitle.textContent = rightConfig
            ? `${rightConfig.strategy_name || 'Strategy'} | ${rightConfig.asset || 'n/a'}`
            : 'No strategy loaded.';
    }

    if (analyticsCompareRightSummary) {
        analyticsCompareRightSummary.textContent = rightConfig
            ? rightConfig.scorecard?.scorecard_summary || 'Right comparison strategy loaded.'
            : 'Use the strategies page to send a profile into the right slot.';
    }

    if (analyticsCompareAccuracyWinner) {
        analyticsCompareAccuracyWinner.textContent = resolveWinnerLabel(
            leftConfig,
            rightConfig,
            (config) => Number(config.scorecard?.reference_accuracy || 0),
        );
    }

    if (analyticsCompareReturnWinner) {
        analyticsCompareReturnWinner.textContent = resolveWinnerLabel(
            leftConfig,
            rightConfig,
            (config) => Number(config.scorecard?.reference_excess_return || 0),
        );
    }

    if (analyticsCompareInterpretabilityWinner) {
        analyticsCompareInterpretabilityWinner.textContent = resolveWinnerLabel(
            leftConfig,
            rightConfig,
            (config) => Number(config.scorecard?.interpretability_score || 0),
        );
    }

    if (analyticsCompareDepthWinner) {
        analyticsCompareDepthWinner.textContent = resolveWinnerLabel(
            leftConfig,
            rightConfig,
            (config) => Number(config.scorecard?.academic_depth_score || 0),
        );
    }

    if (analyticsCompareSummary) {
        analyticsCompareSummary.textContent = (!leftConfig && !rightConfig)
            ? 'Any side-by-side strategy comparison built on the strategies page will appear here as an evaluation carry-over.'
            : 'The analytics page is carrying over the current strategy comparison from the strategies page so evaluation and product decisions stay connected.';
    }
}

function renderAnalyticsChart(chartPayload) {
    if (!analyticsChart) {
        return;
    }

    const points = chartPayload?.points || [];
    if (!points.length) {
        analyticsChart.innerHTML = '<div class="analytics-chart-empty">No recent market chart is available for this asset and timeframe yet.</div>';
        if (analyticsChartCaption) {
            analyticsChartCaption.textContent = 'Live chart unavailable for the selected view.';
        }
        if (analyticsPulseTrend) {
            analyticsPulseTrend.textContent = 'n/a';
        }
        if (analyticsPulseRange) {
            analyticsPulseRange.textContent = 'n/a';
        }
        if (analyticsPulseVolume) {
            analyticsPulseVolume.textContent = 'n/a';
        }
        if (analyticsPulseTrendBar) {
            analyticsPulseTrendBar.style.width = '0%';
        }
        if (analyticsPulseRangeBar) {
            analyticsPulseRangeBar.style.width = '0%';
        }
        if (analyticsPulseVolumeBar) {
            analyticsPulseVolumeBar.style.width = '0%';
        }
        return;
    }

    const closes = points.map((point) => Number(point.close || 0));
    const volumes = points.map((point) => Number(point.volume || 0));
    const minClose = Math.min(...closes);
    const maxClose = Math.max(...closes);
    const firstClose = closes[0] || 0;
    const lastClose = closes[closes.length - 1] || 0;
    const totalMovePct = firstClose ? ((lastClose - firstClose) / firstClose) * 100 : 0;
    const priceRangePct = minClose ? ((maxClose - minClose) / minClose) * 100 : 0;
    const avgVolume = volumes.reduce((sum, value) => sum + value, 0) / Math.max(volumes.length, 1);
    const maxVolume = Math.max(...volumes, 1);
    const isPositive = lastClose >= firstClose;
    const lineColor = isPositive ? '#008954' : '#c94a4a';
    const fillColor = isPositive ? 'rgba(0, 137, 87, 0.12)' : 'rgba(201, 74, 74, 0.10)';
    const width = 720;
    const height = 260;
    const padX = 38;
    const padY = 24;
    const usableWidth = width - (padX * 2);
    const usableHeight = height - (padY * 2);
    const yRange = maxClose - minClose || 1;

    const pathPoints = closes.map((close, index) => {
        const x = padX + ((usableWidth * index) / Math.max(closes.length - 1, 1));
        const y = height - padY - (((close - minClose) / yRange) * usableHeight);
        return `${x},${y}`;
    }).join(' ');

    const areaPoints = `${padX},${height - padY} ${pathPoints} ${width - padX},${height - padY}`;
    const latestX = padX + usableWidth;
    const latestY = height - padY - (((lastClose - minClose) / yRange) * usableHeight);
    const highY = height - padY - (((maxClose - minClose) / yRange) * usableHeight);
    const lowY = height - padY;
    const volumeBars = points.map((point, index) => {
        const barWidth = Math.max(2, usableWidth / Math.max(points.length, 1) - 2);
        const x = padX + ((usableWidth * index) / Math.max(points.length - 1, 1)) - (barWidth / 2);
        const barHeight = Math.max((Number(point.volume || 0) / maxVolume) * 34, 1);
        return `<rect x="${x.toFixed(1)}" y="${(height - padY - barHeight).toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="1.5" fill="rgba(71, 85, 105, 0.18)"></rect>`;
    }).join('');

    analyticsChart.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Recent closing price chart">
            <line x1="${padX}" y1="${highY.toFixed(1)}" x2="${width - padX}" y2="${highY.toFixed(1)}" class="analytics-chart-guide"></line>
            <line x1="${padX}" y1="${lowY.toFixed(1)}" x2="${width - padX}" y2="${lowY.toFixed(1)}" class="analytics-chart-guide"></line>
            ${volumeBars}
            <polygon points="${areaPoints}" fill="${fillColor}"></polygon>
            <polyline
                points="${pathPoints}"
                fill="none"
                stroke="${lineColor}"
                stroke-width="3"
                stroke-linecap="round"
                stroke-linejoin="round"
            ></polyline>
            <circle cx="${latestX.toFixed(1)}" cy="${latestY.toFixed(1)}" r="5" fill="${lineColor}"></circle>
            <text x="${padX}" y="${Math.max(highY - 7, 13).toFixed(1)}" class="analytics-chart-label">High ${maxClose.toLocaleString(undefined, { maximumFractionDigits: 2 })}</text>
            <text x="${padX}" y="${Math.min(lowY + 14, height - 4).toFixed(1)}" class="analytics-chart-label">Low ${minClose.toLocaleString(undefined, { maximumFractionDigits: 2 })}</text>
            <text x="${width - padX}" y="${Math.max(latestY - 9, 13).toFixed(1)}" text-anchor="end" class="analytics-chart-label">Latest ${lastClose.toLocaleString(undefined, { maximumFractionDigits: 2 })}</text>
        </svg>
    `;

    if (analyticsChartCaption) {
        analyticsChartCaption.textContent =
            `${chartPayload.asset.replace('USDT', ' / USDT')} moved ${totalMovePct >= 0 ? '+' : ''}${totalMovePct.toFixed(2)}% across ${points.length} ${formatTimeframe(chartPayload.resolved_timeframe)} candles.`;
    }

    if (analyticsPulseTrend) {
        analyticsPulseTrend.textContent = `${totalMovePct >= 0 ? '+' : ''}${totalMovePct.toFixed(2)}%`;
    }
    if (analyticsPulseTrendBar) {
        analyticsPulseTrendBar.style.width = `${Math.min(Math.abs(totalMovePct) * 6, 100)}%`;
    }

    if (analyticsPulseRange) {
        analyticsPulseRange.textContent = `${priceRangePct.toFixed(2)}%`;
    }
    if (analyticsPulseRangeBar) {
        analyticsPulseRangeBar.style.width = `${Math.min(priceRangePct * 5, 100)}%`;
    }

    if (analyticsPulseVolume) {
        analyticsPulseVolume.textContent = avgVolume >= 1_000_000
            ? `${(avgVolume / 1_000_000).toFixed(1)}M`
            : avgVolume.toFixed(0);
    }
    if (analyticsPulseVolumeBar) {
        analyticsPulseVolumeBar.style.width = `${Math.min((avgVolume / maxVolume) * 100, 100)}%`;
    }
}

function buildFallbackAnalyticsPayload(asset, timeframe) {
    const summary = marketSummaries[asset] || {};
    const gdeltUnavailable = !summary.latest_gdelt_regime_label || summary.latest_gdelt_regime_label === 'unavailable';

    return {
        asset,
        requested_timeframe: timeframe,
        resolved_timeframe: timeframe,
        refreshed_at: null,
        data_lane: 'decision_analytics',
        decision: {
            current_signal: summary.selected_primary_signal || summary.latest_signal || summary.scaled_model_signal || 'n/a',
            confidence: Number(summary.selected_primary_confidence ?? summary.latest_signal_confidence ?? summary.scaled_model_confidence ?? 0),
            model_name: summary.selected_primary_model || summary.selected_backend_model || 'n/a',
            strategy_display_name: 'Balanced Default',
            engine_display_name: 'Market + futures engine',
            policy_name: summary.policy_name || 'n/a',
            probability_mode: summary.probability_mode || 'n/a',
        },
        target_semantics: null,
        timeframe_readiness: null,
        performance_overview: {
            summary_text: summary.walkforward_summary || summary.primary_summary || summary.backend_summary || summary.analysis_summary || 'No generated analytics summary is available yet.',
            accuracy: Number(summary.test_accuracy ?? summary.baseline_scaled_test_accuracy ?? 0),
            balanced_accuracy: Number(summary.test_balanced_accuracy ?? summary.rule_based_test_accuracy ?? 0),
            macro_f1: Number(summary.test_macro_f1 || 0),
            target_name: summary.selected_target_name || summary.top_feature_name || 'n/a',
            policy_return: Number(summary.strategy_total_return || 0),
            buy_hold_return: Number(summary.buy_hold_total_return || 0),
            sharpe_ratio: Number(summary.sharpe_ratio || 0),
            walkforward_fold_count: Number(summary.walkforward_fold_count || 0),
            walkforward_avg_accuracy: Number(summary.walkforward_avg_accuracy || 0),
            walkforward_avg_excess_return: Number(summary.walkforward_avg_excess_return || 0),
        },
        engine_summary: {
            primary_model: summary.selected_primary_model || summary.selected_backend_model || 'n/a',
            primary_model_display_name: 'Balanced Default',
            engine_label: 'Market + futures engine',
            policy_name: summary.policy_name || 'n/a',
            probability_mode: summary.probability_mode || 'n/a',
            calibration_temperature: Number(summary.calibration_temperature || 0),
            backtest_summary: summary.backtest_summary || summary.backend_summary || 'No backtest summary is available yet.',
        },
        multimodal_context: {
            best_strategy: summary.multimodal_best_strategy || 'n/a',
            context_label: summary.latest_multimodal_context_label || 'n/a',
            latest_signal: summary.multimodal_latest_signal || 'n/a',
            selected_context_variant: summary.multimodal_selected_context_variant || 'n/a',
            test_macro_f1: Number(summary.multimodal_test_macro_f1 || 0),
            validation_macro_f1: Number(summary.multimodal_validation_macro_f1 || 0),
            effective_sentiment_source: summary.latest_effective_sentiment_source || 'n/a',
            effective_sentiment_label: summary.latest_effective_sentiment_label || 'n/a',
            gdelt_regime: gdeltUnavailable ? 'Not Available' : summary.latest_gdelt_regime_label || 'n/a',
            gdelt_article_count: gdeltUnavailable ? null : Number(summary.latest_gdelt_article_count || 0),
            detail: summary.multimodal_summary || 'Multimodal evaluation summary not generated yet.',
        },
        ablation_study: {
            best_variant: summary.ablation_best_variant || 'n/a',
            best_macro_f1: Number(summary.ablation_best_macro_f1 || 0),
            market_futures_macro_f1: Number(summary.ablation_market_futures_macro_f1 || 0),
            delta_macro_f1: Number(summary.delta_macro_f1_vs_market_futures || 0),
            detail: summary.ablation_summary || 'Ablation summary not generated yet.',
        },
    };
}

function renderAnalyticsPayload(payload) {
    const summary = marketSummaries[payload.asset] || {};
    const performance = payload.performance_overview || {};
    const engine = payload.engine_summary || {};
    const multimodal = payload.multimodal_context || {};
    const ablation = payload.ablation_study || {};
    const governance = payload.decision?.governance || {};
    const contextAssessment = payload.context_assessment || {};
    const multimodalAssessment = multimodal.assessment || {};
    const evaluationCoverage = payload.evaluation_coverage || {};
    const targetSemantics = payload.target_semantics || payload.decision?.target_semantics || {};
    const timeframeReadiness = payload.timeframe_readiness || {};
    const defiContext = payload.defi_context || multimodal.defi_context || {};
    const sourcePolicy = payload.source_policy_decision || {};

    if (analyticsLaneBadge) {
        analyticsLaneBadge.textContent = `Evaluation - ${formatTimeframe(payload.resolved_timeframe)}`;
    }

    if (analyticsRefreshNote) {
        const exactNote = payload.exact_timeframe_match
            ? ''
            : ` Exact ${formatTimeframe(payload.requested_timeframe)} analytics outputs are not available yet. Current decision analytics exist for ${(payload.available_timeframes || []).map(formatTimeframe).join(', ') || 'no timeframes'}.`;
        analyticsRefreshNote.textContent = simplifyCopy(
            `Requested timeframe: ${formatTimeframe(payload.requested_timeframe)}. ` +
            `Resolved timeframe: ${formatTimeframe(payload.resolved_timeframe)}. ` +
            `Analytics refreshed: ${formatTimestamp(payload.refreshed_at)}.` +
            exactNote
        );
    }

    if (analyticsSummary) {
        analyticsSummary.textContent = simplifyCopy(
            `${payload.asset.replace('USDT', ' / USDT')} uses ${engine.primary_model_display_name || payload.decision?.strategy_display_name || formatTitle(engine.primary_model || 'n/a')} on ${formatTimeframe(payload.resolved_timeframe)}. ` +
            `Current signal: ${formatDecisionLabel(payload.decision?.current_signal || 'n/a')}. Headline accuracy: ${formatPercent(performance.accuracy)}.`
        );
    }

    if (analyticsActiveAsset) {
        analyticsActiveAsset.textContent = payload.asset.replace('USDT', ' / USDT');
    }

    if (analyticsCurrentSignal) {
        analyticsCurrentSignal.textContent = formatDecisionLabel(payload.decision?.current_signal || 'n/a');
    }

    if (analyticsContextPosture) {
        analyticsContextPosture.textContent = formatTitle(multimodal.context_label || payload.decision?.context_mode || 'n/a');
    }

    if (analyticsTargetHours) {
        analyticsTargetHours.textContent = targetSemantics?.effective_horizon_hours
            ? `${Number(targetSemantics.effective_horizon_hours)}h`
            : performance.target_name || 'n/a';
    }

    if (analyticsScaledAccuracy) {
        analyticsScaledAccuracy.textContent = formatPercent(performance.accuracy);
    }

    if (analyticsRuleAccuracy) {
        analyticsRuleAccuracy.textContent = formatPercent(performance.balanced_accuracy);
    }

    if (analyticsReadiness) {
        analyticsReadiness.textContent = governance.readiness_label || 'n/a';
    }

    if (analyticsEvaluationStrength) {
        analyticsEvaluationStrength.textContent = formatTitle(governance.evaluation_strength || 'n/a');
    }

    if (analyticsSentimentRole) {
        analyticsSentimentRole.textContent = formatTitle(contextAssessment.asset_news_sentiment?.role || contextAssessment.broad_sentiment?.role || 'n/a');
    }

    if (analyticsOnchainRole) {
        analyticsOnchainRole.textContent = formatTitle(contextAssessment.onchain_daily?.role || 'n/a');
    }

    if (analyticsDefiRole) {
        analyticsDefiRole.textContent = defiContext.available ? 'Ecosystem confirmation' : 'Unavailable';
    }

    if (analyticsSourcePolicyTitle) {
        analyticsSourcePolicyTitle.textContent = sourcePolicy.headline || 'Source policy unavailable';
    }

    if (analyticsSourcePolicyRationale) {
        analyticsSourcePolicyRationale.textContent = sourcePolicy.rationale
            || 'LiveStrat separates direct market inputs from context and confirmation layers.';
    }

    if (analyticsPolicyReturn) {
        analyticsPolicyReturn.textContent = formatPercent(performance.policy_return);
    }

    if (analyticsBuyHoldReturn) {
        analyticsBuyHoldReturn.textContent = formatPercent(performance.buy_hold_return);
    }

    if (analyticsSharpe) {
        analyticsSharpe.textContent = Number(performance.sharpe_ratio || 0).toFixed(2);
    }

    if (analyticsWalkforwardFolds) {
        analyticsWalkforwardFolds.textContent = hasWalkforwardEvidence(performance)
            ? Number(performance.walkforward_fold_count || 0).toFixed(0)
            : 'n/a';
    }

    if (analyticsWalkforwardAcc) {
        analyticsWalkforwardAcc.textContent = hasWalkforwardEvidence(performance)
            ? formatPercent(performance.walkforward_avg_accuracy)
            : 'n/a';
    }

    if (analyticsWalkforwardExcess) {
        analyticsWalkforwardExcess.textContent = hasWalkforwardEvidence(performance)
            ? formatPercent(performance.walkforward_avg_excess_return)
            : 'n/a';
    }

    if (analyticsSignalExplainer) {
        analyticsSignalExplainer.textContent = window.LiveStratSignalExplainer
            ? window.LiveStratSignalExplainer.buildSignalExplanation(summary, payload.asset)
            : `${payload.asset} analytics explain the active signal using the current strategy outputs.`;
    }

    if (analyticsWhatNow) {
        const walkforwardSummary = hasWalkforwardEvidence(performance)
            ? `Policy return is ${formatPercent(performance.policy_return)} versus ${formatPercent(performance.buy_hold_return)} buy-and-hold. Rolling validation excess return is ${formatPercent(performance.walkforward_avg_excess_return)}.`
            : `Policy return is ${formatPercent(performance.policy_return)} versus ${formatPercent(performance.buy_hold_return)} buy-and-hold. Rolling validation is not shown because this asset and timeframe do not have enough usable folds.`;
        analyticsWhatNow.textContent =
            `${formatDecisionLabel(payload.decision?.current_signal || 'n/a')} is the latest strategy signal. ` +
            walkforwardSummary;
    }

    if (analyticsWhyTrust) {
        const exactTimeframe = payload.exact_timeframe_match ? 'exactly matches' : 'falls back from';
        analyticsWhyTrust.textContent = simplifyCopy(
            `${governance.readiness_label || 'Unknown'} readiness with ${formatTitle(governance.evaluation_strength || 'n/a')} evaluation strength. ` +
            `The current ${formatTimeframe(payload.resolved_timeframe)} decision ${exactTimeframe} the requested ${formatTimeframe(payload.requested_timeframe)} view, and generated coverage currently exists for market+futures on ${formatTimeframeList((payload.evaluation_coverage?.market_futures || {}).available_timeframes)}.`
        );
    }

    if (analyticsMethodSummary) {
        analyticsMethodSummary.textContent = simplifyCopy(
            `For ${payload.asset}, LiveStrat currently uses ${engine.primary_model_display_name || formatTitle(engine.primary_model || 'n/a')} ` +
            `with ${formatPolicyLabel(engine.policy_name || 'n/a')} and ${formatProbabilityLabel(engine.probability_mode || 'n/a')}. ` +
            `${formatTargetSemantics(targetSemantics)}`
        );
    }

    if (analyticsPrimaryModel) {
        analyticsPrimaryModel.textContent = engine.engine_label || engine.primary_model_display_name || engine.primary_model || 'n/a';
    }

    if (analyticsPolicyName) {
        analyticsPolicyName.textContent = formatPolicyLabel(engine.policy_name || 'n/a');
    }

    if (analyticsProbabilityMode) {
        analyticsProbabilityMode.textContent = formatProbabilityLabel(engine.probability_mode || 'n/a');
    }

    if (analyticsCalibrationTemperature) {
        analyticsCalibrationTemperature.textContent = Number(engine.calibration_temperature || 0).toFixed(2);
    }

    if (analyticsBacktestSummary) {
        analyticsBacktestSummary.textContent = simplifyCopy(engine.backtest_summary || 'No backtest summary is available yet.');
    }

    if (analyticsMultimodalStrategy) {
        analyticsMultimodalStrategy.textContent = formatAnalyticsStrategyLabel(multimodal.best_strategy || 'n/a');
    }

    if (analyticsMultimodalContext) {
        analyticsMultimodalContext.textContent = formatTitle(multimodal.context_label || 'n/a');
    }

    if (analyticsMultimodalSignal) {
        analyticsMultimodalSignal.textContent = formatDecisionLabel(multimodal.latest_signal || 'n/a');
    }

    if (analyticsMultimodalContextVariant) {
        analyticsMultimodalContextVariant.textContent = multimodal.selected_context_variant
            ? formatAnalyticsStrategyLabel(multimodal.selected_context_variant)
            : 'n/a';
    }

    if (analyticsMultimodalMacroF1) {
        analyticsMultimodalMacroF1.textContent = formatPercent(multimodal.test_macro_f1);
    }

    if (analyticsMultimodalValidationMacroF1) {
        analyticsMultimodalValidationMacroF1.textContent = multimodal.validation_macro_f1
            ? formatPercent(multimodal.validation_macro_f1)
            : 'n/a';
    }

    if (analyticsEffectiveSentimentSource) {
        analyticsEffectiveSentimentSource.textContent = formatSentimentSource(multimodal.effective_sentiment_source);
    }

    if (analyticsEffectiveSentimentLabel) {
        analyticsEffectiveSentimentLabel.textContent = formatSentimentLabel(multimodal.effective_sentiment_label);
    }

    if (analyticsGdeltRegime) {
        analyticsGdeltRegime.textContent = multimodal.gdelt_regime || 'Not Available';
    }

    if (analyticsGdeltCount) {
        analyticsGdeltCount.textContent = multimodal.gdelt_article_count == null
            ? 'n/a'
            : Number(multimodal.gdelt_article_count).toFixed(0);
    }

    if (analyticsDefiChain) {
        analyticsDefiChain.textContent = defiContext.available
            ? defiContext.chain_name || 'Available'
            : 'Not Available';
    }

    if (analyticsDefiRegime) {
        analyticsDefiRegime.textContent = defiContext.available
            ? formatTitle(defiContext.regime_label || 'available')
            : 'Not Available';
    }

    if (analyticsMultimodalDetail) {
        analyticsMultimodalDetail.textContent = simplifyCopy(
            multimodal.detail ||
            'Multimodal evaluation summary not generated yet.'
        );
    }

    if (analyticsContextGovernance) {
        analyticsContextGovernance.textContent = simplifyCopy(
            `${multimodalAssessment.recommendation || 'Context-layer policy is not available yet.'} ` +
            `${contextAssessment.asset_news_sentiment?.headline || ''} ` +
            `${contextAssessment.onchain_daily?.headline || ''}`.trim()
        );
    }

    if (analyticsMultimodalSummary) {
        analyticsMultimodalSummary.textContent =
            `The current best multimodal route is ${formatAnalyticsStrategyLabel(multimodal.best_strategy || 'n/a')}. ` +
            `It is showing ${formatPercent(multimodal.test_macro_f1)} macro-F1 with ${formatTitle(multimodalAssessment.uplift_label || 'n/a')} uplift against market+futures only.`;
    }

    if (analyticsAblationBestVariant) {
        analyticsAblationBestVariant.textContent = formatAnalyticsStrategyLabel(ablation.best_variant || 'n/a');
    }

    if (analyticsAblationBestMacroF1) {
        analyticsAblationBestMacroF1.textContent = formatPercent(ablation.best_macro_f1);
    }

    if (analyticsAblationBaseMacroF1) {
        analyticsAblationBaseMacroF1.textContent = formatPercent(ablation.market_futures_macro_f1);
    }

    if (analyticsAblationDelta) {
        const delta = Number(ablation.delta_macro_f1 || 0) * 100;
        analyticsAblationDelta.textContent = `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`;
    }

    if (analyticsAblationDetail) {
        analyticsAblationDetail.textContent = simplifyCopy(ablation.detail || 'Ablation summary not generated yet.');
    }

    if (analyticsAblationSummary) {
        analyticsAblationSummary.textContent =
            `Adding context layers is currently ${Number(ablation.delta_macro_f1 || 0) >= 0 ? 'helping' : 'hurting'} by ` +
            `${Number(ablation.delta_macro_f1 || 0) >= 0 ? '+' : ''}${(Number(ablation.delta_macro_f1 || 0) * 100).toFixed(1)}% macro-F1 versus market+futures only. ` +
            `The best current variant is ${formatAnalyticsStrategyLabel(ablation.best_variant || 'n/a')}.`;
    }

    const marketFuturesCoverage = evaluationCoverage.market_futures || {};
    const marketTrendCoverage = evaluationCoverage.market_trend_benchmark || {};
    const crossAssetCoverage = evaluationCoverage.cross_asset_relative_strength || {};
    const multimodalCoverage = evaluationCoverage.multimodal || {};
    const onchainCoverage = evaluationCoverage.onchain_specialist || {};
    const scaledCoverage = evaluationCoverage.market_baseline_scaled || {};
    const unscaledCoverage = evaluationCoverage.market_baseline_unscaled || {};

    if (analyticsCoverageMarketFutures) {
        analyticsCoverageMarketFutures.textContent = [
            `Futures ${formatTimeframeList(marketFuturesCoverage.available_timeframes)}`,
            `Trend ${formatTimeframeList(marketTrendCoverage.available_timeframes)}`,
        ].join(' | ');
    }

    if (analyticsCoverageMultimodal) {
        analyticsCoverageMultimodal.textContent = formatTimeframeList(multimodalCoverage.available_timeframes);
    }

    if (analyticsCoverageOnchain) {
        analyticsCoverageOnchain.textContent = formatTimeframeList(onchainCoverage.available_timeframes);
    }

    if (analyticsCoverageBaselines) {
        analyticsCoverageBaselines.textContent = [
            `Cross ${formatTimeframeList(crossAssetCoverage.available_timeframes)}`,
            `Scaled ${formatTimeframeList(scaledCoverage.available_timeframes)}`,
            `Unscaled ${formatTimeframeList(unscaledCoverage.available_timeframes)}`,
        ].join(' | ');
    }

    if (analyticsCoverageSummary) {
        analyticsCoverageSummary.textContent = simplifyCopy(
            `${payload.asset} currently has market-trend benchmark coverage on ${formatTimeframeList(marketTrendCoverage.available_timeframes)}, ` +
            `cross-asset ranking on ${formatTimeframeList(crossAssetCoverage.available_timeframes)}, ` +
            `market+futures on ${formatTimeframeList(marketFuturesCoverage.available_timeframes)}, ` +
            `multimodal on ${formatTimeframeList(multimodalCoverage.available_timeframes)}, and on-chain specialist models on ${formatTimeframeList(onchainCoverage.available_timeframes)}.`
        );
    }

    if (analyticsCoverageDetail) {
        analyticsCoverageDetail.textContent = simplifyCopy(
            `Best market-trend benchmark: ${formatTitle(marketTrendCoverage.best_model_display_name || marketTrendCoverage.best_model_name || 'n/a')} ` +
            `(${formatPercent(marketTrendCoverage.best_macro_f1 || 0)} macro-F1). ` +
            `Best cross-asset ranker: ${formatTitle(crossAssetCoverage.best_model_display_name || crossAssetCoverage.best_model_name || 'n/a')} ` +
            `(${formatPercent(crossAssetCoverage.best_macro_f1 || 0)} hit-rate proxy). ` +
            `Best market+futures model: ${formatTitle(marketFuturesCoverage.best_model_display_name || marketFuturesCoverage.best_model_name || 'n/a')} ` +
            `(${formatPercent(marketFuturesCoverage.best_macro_f1 || 0)} macro-F1). ` +
            `Best multimodal model: ${formatTitle(multimodalCoverage.best_model_display_name || multimodalCoverage.best_model_name || 'n/a')} ` +
            `(${formatPercent(multimodalCoverage.best_macro_f1 || 0)} macro-F1). ` +
            `Best on-chain specialist: ${formatTitle(onchainCoverage.best_model_display_name || onchainCoverage.best_model_name || 'n/a')} ` +
            `(${formatPercent(onchainCoverage.best_macro_f1 || 0)} macro-F1).`
        );
    }

    const familyScope = payload.strategy_family_scope || {};
    const selectedScopeAsset = familyScope.selected_asset || {};
    const familyScopeMap = familyScope.families || {};
    const marketTrendScope = familyScopeMap.market_trend_benchmark || {};
    const futuresScope = familyScopeMap.market_futures_core || {};
    const binaryScope = familyScopeMap.market_futures_binary || {};
    const contextScope = familyScopeMap.multimodal_context || {};
    const onchainScope = familyScopeMap.daily_structural_confirmation || {};
    const crossAssetScope = familyScopeMap.cross_asset_relative_strength || {};
    if (analyticsFamilyScopeSummary) {
        analyticsFamilyScopeSummary.textContent = simplifyCopy(
            `${payload.asset} is currently a ${formatTitle(selectedScopeAsset.tier || 'unclassified')} asset. ` +
            `${selectedScopeAsset.recommended_scope || 'Scope classification is not available yet.'}`
        );
    }

    if (analyticsFamilyScopeDetail) {
        analyticsFamilyScopeDetail.textContent = simplifyCopy(
            `Market benchmark: ${formatTitle(marketTrendScope.defensibility_label || 'unknown')}. ` +
            `Cross-asset ranking: ${formatTitle(crossAssetScope.defensibility_label || 'unknown')}. ` +
            `Market+futures: ${formatTitle(futuresScope.defensibility_label || 'unknown')}. ` +
            `Binary backup: ${formatTitle(binaryScope.defensibility_label || 'unknown')}.`
        );
    }

    if (analyticsDefensibleComparison) {
        const framing = Array.isArray(familyScope.recommended_demo_framing)
            ? familyScope.recommended_demo_framing.join(' ')
            : '';
        analyticsDefensibleComparison.textContent = simplifyCopy(
            `${familyScope.defensibility_summary || 'Defensibility summary is unavailable.'} ${framing}`.trim()
        );
    }

    const readiness1h = timeframeReadiness.timeframes?.['1h'] || {};
    const readiness4h = timeframeReadiness.timeframes?.['4h'] || {};
    const readiness1d = timeframeReadiness.timeframes?.['1d'] || {};

    if (analyticsTimeframe1h) {
        analyticsTimeframe1h.textContent = formatTitle(readiness1h.readiness_label || 'unknown');
    }

    if (analyticsTimeframe4h) {
        analyticsTimeframe4h.textContent = formatTitle(readiness4h.readiness_label || 'unknown');
    }

    if (analyticsTimeframe1d) {
        analyticsTimeframe1d.textContent = formatTitle(readiness1d.readiness_label || 'unknown');
    }

    if (analyticsTimeframeReadinessSummary) {
        analyticsTimeframeReadinessSummary.textContent = simplifyCopy(
            `LiveStrat currently labels 1h as ${formatTitle(readiness1h.readiness_label || 'unknown')}, ` +
            `4h as ${formatTitle(readiness4h.readiness_label || 'unknown')}, and ` +
            `1d as ${formatTitle(readiness1d.readiness_label || 'unknown')} based on generated files.`
        );
    }

    if (analyticsTimeframeReadinessDetail) {
        const recommendations = Array.isArray(timeframeReadiness.recommended_next_runs)
            ? timeframeReadiness.recommended_next_runs.map((item) => `${formatTimeframe(item.timeframe)}: ${item.reason}`).join(' ')
            : '';
        analyticsTimeframeReadinessDetail.textContent = simplifyCopy(
            `${readiness1h.summary || ''} ${readiness4h.summary || ''} ${readiness1d.summary || ''} ${recommendations}`.trim()
        );
    }

    setToneClass(analyticsStatusSignalBox, toneFromSignal(payload.decision?.current_signal));
    setToneClass(analyticsStatusReadinessBox, toneFromReadiness(governance.readiness_label));
    setToneClass(analyticsStoryContextBox, toneFromContext(multimodal.context_label || payload.decision?.context_mode));
    setToneClass(analyticsStoryTargetBox, payload.exact_timeframe_match ? 'positive' : 'caution');
    setToneClass(analyticsAccuracyBox, toneFromNumericPercent(performance.accuracy));
    setToneClass(analyticsBalancedBox, toneFromNumericPercent(performance.balanced_accuracy));
    setToneClass(analyticsPolicyBox, Number(performance.policy_return || 0) >= 0 ? 'positive' : 'negative');
    setToneClass(analyticsCoverageBox, toneFromReadiness(governance.evaluation_strength));
    setToneClass(analyticsSentimentBox, toneFromContext(contextAssessment.asset_news_sentiment?.headline || contextAssessment.asset_news_sentiment?.role));
    setToneClass(analyticsOnchainBox, toneFromContext(contextAssessment.onchain_daily?.headline || contextAssessment.onchain_daily?.role));
    setToneClass(analyticsMultimodalBox, toneFromContext(multimodalAssessment.uplift_label || multimodal.best_strategy));
    setToneClass(analyticsContextLabelBox, toneFromContext(multimodal.context_label));
    setToneClass(
        analyticsMarketCoverageBox,
        ((marketFuturesCoverage.available_timeframes || []).length || (marketTrendCoverage.available_timeframes || []).length)
            ? 'positive'
            : 'negative'
    );
    setToneClass(analyticsMultimodalCoverageBox, (multimodalCoverage.available_timeframes || []).length ? 'positive' : 'negative');
    setToneClass(analyticsOnchainCoverageBox, (onchainCoverage.available_timeframes || []).length ? 'positive' : 'negative');
    setToneClass(
        analyticsBaselineCoverageBox,
        (
            (crossAssetCoverage.available_timeframes || []).length ||
            (scaledCoverage.available_timeframes || []).length ||
            (unscaledCoverage.available_timeframes || []).length
        ) ? 'positive' : 'negative'
    );
    setToneClass(analyticsTimeframe1hBox, toneFromReadiness(readiness1h.readiness_label));
    setToneClass(analyticsTimeframe4hBox, toneFromReadiness(readiness4h.readiness_label));
    setToneClass(analyticsTimeframe1dBox, toneFromReadiness(readiness1d.readiness_label));
}

function renderEvaluationEvidence(payload) {
    const strategyRows = payload?.strategy_rows || [];
    const deepRowsByAsset = payload?.best_deep_learning_by_asset || {};
    const selectedStrategies = strategyRows.filter((row) => {
        return ['recommended', 'conservative_trend', 'momentum_breakout', 'futures_crowd_reversal', 'multimodal_balanced', 'daily_structural_confirmation'].includes(row.strategy_key);
    });

    if (analyticsEvidenceSummary) {
        analyticsEvidenceSummary.textContent = payload?.summary || 'Evaluation evidence is unavailable for this timeframe.';
    }

    if (analyticsEvidenceExport) {
        const timeframe = payload?.timeframe || (analyticsTimeframe ? analyticsTimeframe.value : '4h');
        analyticsEvidenceExport.href = `/api/evaluation-evidence?timeframe=${encodeURIComponent(timeframe)}&format=csv`;
    }

    if (analyticsStrategyEvidenceBody) {
        analyticsStrategyEvidenceBody.innerHTML = selectedStrategies.length
            ? selectedStrategies.map((row) => `
                <tr>
                    <td>${escapeHtml(row.symbol || 'n/a')}</td>
                    <td>${escapeHtml(row.strategy_name || row.strategy_key || 'n/a')}</td>
                    <td>${escapeHtml(formatDecisionLabel(row.signal || 'n/a'))}</td>
                    <td>${Number(row.score || 0).toFixed(2)}</td>
                    <td>${formatPercent(row.predicted_return || 0)}</td>
                    <td>${formatPercent(row.accuracy || 0)}</td>
                </tr>
            `).join('')
            : '<tr><td colspan="6">No strategy evidence rows are available for this timeframe.</td></tr>';
    }

    if (analyticsDeepLearningBody) {
        const rows = Object.values(deepRowsByAsset).sort((left, right) => String(left.symbol || '').localeCompare(String(right.symbol || '')));
        analyticsDeepLearningBody.innerHTML = rows.length
            ? rows.map((row) => `
                <tr>
                    <td>${escapeHtml(row.symbol || 'n/a')}</td>
                    <td>${escapeHtml(formatTitle(row.model_name || 'LSTM'))}</td>
                    <td>${formatPercent(row.accuracy || 0)}</td>
                    <td>${formatPercent(row.macro_f1 || 0)}</td>
                    <td>${formatPercent(row.balanced_accuracy || 0)}</td>
                </tr>
            `).join('')
            : '<tr><td colspan="5">No LSTM/deep-learning evidence rows are available for this timeframe.</td></tr>';
    }
}

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
    }
    return response.json();
}

async function renderAnalyticsLane() {
    const asset = analyticsAsset ? analyticsAsset.value : 'BTCUSDT';
    const timeframe = analyticsTimeframe ? analyticsTimeframe.value : '4h';
    const originalButtonLabel = analyticsRun ? analyticsRun.textContent : '';

    if (analyticsRun) {
        analyticsRun.disabled = true;
        analyticsRun.textContent = 'Refreshing...';
    }

    try {
        const [payload, chartPayload, evidencePayload] = await Promise.all([
            fetchJson(`/api/analytics-summary?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}`),
            fetchJson(`/api/market-chart?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&points=48`),
            fetchJson(`/api/evaluation-evidence?timeframe=${encodeURIComponent(timeframe)}`),
        ]);
        renderAnalyticsPayload(payload);
        renderAnalyticsChart(chartPayload);
        renderEvaluationEvidence(evidencePayload);
    } catch (error) {
        renderAnalyticsPayload(buildFallbackAnalyticsPayload(asset, timeframe));
        renderAnalyticsChart({ asset, resolved_timeframe: timeframe, points: [] });
        if (analyticsRefreshNote) {
            analyticsRefreshNote.textContent = `Analytics endpoint unavailable. ${error.message}`;
        }
        if (analyticsLaneBadge) {
            analyticsLaneBadge.textContent = 'Evaluation - backup';
        }
    } finally {
        if (analyticsRun) {
            analyticsRun.disabled = false;
            analyticsRun.textContent = originalButtonLabel || 'Refresh view';
        }
    }

    renderAnalyticsComparison();
}

if (analyticsRun) {
    analyticsRun.addEventListener('click', renderAnalyticsLane);
}

if (analyticsAsset) {
    analyticsAsset.addEventListener('change', renderAnalyticsLane);
}

if (analyticsTimeframe) {
    analyticsTimeframe.addEventListener('change', renderAnalyticsLane);
    analyticsTimeframe.value = '4h';
}

renderAnalyticsLane();

// ============================================================
// Limitations panel + transaction cost sensitivity (added T2.5/T2.6)
// ============================================================


function escapeHtmlSafe(value) {
    return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

async function loadCostSensitivity() {
    const grid = document.getElementById('analytics-cost-grid');
    if (!grid) return;
    const timeframe = analyticsTimeframe ? analyticsTimeframe.value : '4h';
    const asset = analyticsAsset ? analyticsAsset.value : '';
    try {
        const url = `/api/transaction-cost-sensitivity?timeframe=${encodeURIComponent(timeframe)}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const rows = Array.isArray(data.rows) ? data.rows : [];
        if (!rows.length) {
            grid.innerHTML = '<p class="note">No cost sensitivity output available for this timeframe yet.</p>';
            return;
        }

        const grouped = {};
        rows.forEach((r) => {
            const sym = String(r.symbol || '').replace('USDT', '');
            (grouped[sym] = grouped[sym] || []).push(r);
        });

        const order = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE'];
        const cards = order
            .filter((s) => grouped[s])
            .map((sym) => {
                const groupRows = grouped[sym].sort((a, b) => Number(a.cost_multiplier) - Number(b.cost_multiplier));
                const baseHold = Number(groupRows[0].total_buy_hold_return || 0) * 100;
                const trs = groupRows.map((r) => {
                    const mult = Number(r.cost_multiplier);
                    const stratPct = Number(r.total_strategy_return) * 100;
                    const excessPct = Number(r.excess_return) * 100;
                    const sharpe = Number(r.sharpe_ratio);
                    const cls = excessPct > 0 ? 'cost-row-positive' : (excessPct < 0 ? 'cost-row-negative' : '');
                    return `<tr><td>${mult.toFixed(0)}x</td><td>${stratPct.toFixed(2)}%</td><td class="${cls}">${excessPct >= 0 ? '+' : ''}${excessPct.toFixed(2)}%</td><td>${sharpe.toFixed(2)}</td></tr>`;
                }).join('');
                const isSelected = asset && String(asset).startsWith(sym);
                const border = isSelected ? '; border-color: #008B5A;' : '';
                return `<article class="cost-card" style="background:#ffffff${border}">
                    <h4>${sym}/USDT</h4>
                    <p class="note" style="margin:0;font-size:0.82rem;">Buy &amp; hold over window: ${baseHold >= 0 ? '+' : ''}${baseHold.toFixed(2)}%</p>
                    <table class="cost-table">
                        <thead><tr><th>Fee</th><th>Strategy</th><th>Excess</th><th>Sharpe</th></tr></thead>
                        <tbody>${trs}</tbody>
                    </table>
                </article>`;
            }).join('');
        grid.innerHTML = cards || '<p class="note">No data.</p>';
    } catch (err) {
        grid.innerHTML = `<p class="note">Could not load cost sensitivity: ${escapeHtmlSafe(err.message)}</p>`;
    }
}

async function loadLimitations() {
    const grid = document.getElementById('analytics-limitations-grid');
    if (!grid) return;
    try {
        const resp = await fetch('/api/limitations');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const sections = Array.isArray(data.sections) ? data.sections : [];
        if (!sections.length) {
            grid.innerHTML = '<p class="note">Limitations document not yet available.</p>';
            return;
        }
        const cards = sections.map((s) => {
            const cleanBody = String(s.body || '').replace(/\*\*/g, '').replace(/^\* /gm, '- ');
            return `<article class="limitation-card">
                <p class="limitation-category">${escapeHtmlSafe(s.category || '')}</p>
                <h4>${escapeHtmlSafe(s.title || '')}</h4>
                <p>${escapeHtmlSafe(cleanBody)}</p>
                <details><summary>Show full text</summary><p>${escapeHtmlSafe(cleanBody)}</p></details>
            </article>`;
        }).join('');
        grid.innerHTML = cards;
    } catch (err) {
        grid.innerHTML = `<p class="note">Could not load limitations: ${escapeHtmlSafe(err.message)}</p>`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadCostSensitivity();
    loadLimitations();
});

if (analyticsAsset) analyticsAsset.addEventListener('change', loadCostSensitivity);
if (analyticsTimeframe) analyticsTimeframe.addEventListener('change', loadCostSensitivity);
if (analyticsRun) analyticsRun.addEventListener('click', () => {
    loadCostSensitivity();
    loadLimitations();
});
