/* The markets page uses market-view endpoints and keeps strategy decisions separate. */

const marketSummaryScript = document.getElementById('market-summaries-data');
const marketSummaries = marketSummaryScript ? JSON.parse(marketSummaryScript.textContent) : {};

const marketAsset = document.getElementById('market-asset');
const marketTimeframe = document.getElementById('market-timeframe');
const marketButton = document.getElementById('market-generate');

const latestClose = document.getElementById('market-close');
const return24h = document.getElementById('market-return-24h');
const volatility = document.getElementById('market-volatility');
const marketActivity = document.getElementById('market-activity');
const marketRuleSignal = document.getElementById('market-rule-signal');
const marketSummary = document.getElementById('market-summary');
const marketPolicyReturn = document.getElementById('market-policy-return');
const marketSharpe = document.getElementById('market-sharpe');
const marketDrawdown = document.getElementById('market-drawdown');
const marketSignalExplainer = document.getElementById('market-signal-explainer');
const marketSignalMode = document.getElementById('market-signal-mode');
const marketEffectiveSentiment = document.getElementById('market-effective-sentiment');
const marketGdeltStatus = document.getElementById('market-gdelt-status');
const marketOnchainStatus = document.getElementById('market-onchain-status');
const marketDefiStatus = document.getElementById('market-defi-status');
const marketCapabilityNote = document.getElementById('market-capability-note');
const marketMultimodalSelection = document.getElementById('market-multimodal-selection');
const marketLaneBadge = document.getElementById('market-lane-badge');
const marketRefreshNote = document.getElementById('market-refresh-note');
const marketChart = document.getElementById('market-chart');
const marketChartStats = document.getElementById('market-chart-stats');
const marketChartNote = document.getElementById('market-chart-note');
const marketFuturesSupport = document.getElementById('market-futures-support');
const marketBasisMode = document.getElementById('market-basis-mode');
const marketNewsTheme = document.getElementById('market-news-theme');
const marketOnchainDriver = document.getElementById('market-onchain-driver');
const marketDefiChain = document.getElementById('market-defi-chain');
const marketContextNote = document.getElementById('market-context-note');
const marketSourcePolicyNote = document.getElementById('market-source-policy-note');
const marketPriceRead = document.getElementById('market-price-read');
const marketReturnMeaning = document.getElementById('market-return-meaning');
const marketVolatilityMeaning = document.getElementById('market-volatility-meaning');
const marketActivityMeaning = document.getElementById('market-activity-meaning');
const marketFuturesNote = document.getElementById('market-futures-note');
const marketSentimentNote = document.getElementById('market-sentiment-note');
const marketNewsNote = document.getElementById('market-news-note');
const marketOnchainNote = document.getElementById('market-onchain-note');
const marketDefiNote = document.getElementById('market-defi-note');

function formatSignal(value) {
    return (value || 'n/a').replaceAll('_', ' ');
}

function toTitleCase(value) {
    const text = formatSignal(value);
    return text.charAt(0).toUpperCase() + text.slice(1);
}

function simplifyCopy(value) {
    return String(value || '')
        .replaceAll('display lane', 'market view')
        .replaceAll('decision lane', 'strategy view')
        .replaceAll('fallback', 'backup')
        .replaceAll('Fallback', 'Backup')
        .replaceAll('veto', 'warning')
        .replaceAll('governance', 'rules')
        .replaceAll('posture', 'status')
        .replaceAll('regime', 'state')
        .replaceAll('Regime', 'State')
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
    return normalized && normalized !== 'n/a' ? toTitleCase(normalized) : 'n/a';
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

function formatMoney(value) {
    const number = Number(value || 0);
    if (!number) {
        return 'n/a';
    }
    if (number >= 1_000_000_000) {
        return `$${(number / 1_000_000_000).toFixed(2)}B`;
    }
    if (number >= 1_000_000) {
        return `$${(number / 1_000_000).toFixed(1)}M`;
    }
    return `$${number.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function formatAsset(value) {
    return String(value || 'BTCUSDT').replace('USDT', ' / USDT');
}

function formatSignedPercent(value) {
    const number = Number(value || 0);
    return `${number >= 0 ? '+' : ''}${number.toFixed(2)}%`;
}

function describeMove(value) {
    const number = Number(value || 0);
    if (number >= 3) {
        return 'A strong positive move over the latest 24h window.';
    }
    if (number >= 0.5) {
        return 'A positive 24h move, but still worth checking volatility and volume.';
    }
    if (number <= -3) {
        return 'A sharp negative move over the latest 24h window.';
    }
    if (number <= -0.5) {
        return 'A negative 24h move, so avoid treating the market as automatically strong.';
    }
    return 'A small 24h move. The market is not showing a major directional change here.';
}

function describeVolatility(value) {
    const normalized = String(value || '').toLowerCase();
    if (normalized.includes('high')) {
        return 'Price is moving sharply, so timing and risk control matter more.';
    }
    if (normalized.includes('medium')) {
        return 'Price movement is noticeable but not extreme.';
    }
    if (normalized.includes('low')) {
        return 'Price movement is calmer than usual.';
    }
    return 'Volatility describes how jumpy the recent market window looks.';
}

function describeActivity(value) {
    const normalized = String(value || '').toLowerCase();
    if (normalized.includes('elevated') || normalized.includes('high')) {
        return 'Trading activity is higher than usual, so recent moves may carry more weight.';
    }
    if (normalized.includes('quiet') || normalized.includes('low')) {
        return 'Trading activity is quiet, so recent moves may be less convincing.';
    }
    return 'Trading activity looks normal for this recent window.';
}

function contextAvailabilityLabel(value, fallback = 'Not Available') {
    const raw = String(value || '').trim();
    if (!raw || raw.toLowerCase() === 'unavailable' || raw.toLowerCase() === 'n/a') {
        return fallback;
    }
    return toTitleCase(raw);
}

function buildFallbackSnapshot(asset, timeframe) {
    const summary = marketSummaries[asset] || {};
    return {
        asset,
        requested_timeframe: timeframe,
        resolved_timeframe: timeframe,
        latest_close: Number(summary.latest_close || 0),
        latest_return_24h_pct: Number(summary.latest_return_24h_pct || 0),
        volatility_status: summary.volatility_status || 'n/a',
        activity_status: 'Normal',
        summary_text: summary.primary_summary || summary.analysis_summary || 'No generated market snapshot is available yet.',
        context_mode: 'Ready',
        rule_signal: summary.rule_signal || 'n/a',
        policy_return: Number(summary.strategy_total_return || 0),
        sharpe_ratio: Number(summary.sharpe_ratio || 0),
        max_drawdown: Number(summary.max_drawdown || 0),
        effective_sentiment_source: summary.latest_effective_sentiment_source || 'unavailable',
        gdelt_status: summary.latest_gdelt_regime_label || 'unavailable',
        gdelt_article_count: Number(summary.latest_gdelt_article_count || 0),
        onchain_status: summary.latest_onchain_regime_label || 'unavailable',
        onchain_snapshot_status: summary.latest_onchain_snapshot_status || 'unavailable',
        onchain_snapshot_label: summary.latest_onchain_snapshot_label || 'unavailable',
        onchain_snapshot_age_days: Number(summary.latest_onchain_snapshot_age_days || 0),
        capability_notes: ['Market view is using saved summary data because the API is unavailable.'],
        multimodal_selected_context_variant: summary.multimodal_selected_context_variant || null,
        refreshed_at: null,
    };
}

function renderMarketChart(chartPayload) {
    if (!marketChart) {
        return;
    }

    const points = chartPayload?.points || [];
    if (!points.length) {
        marketChart.innerHTML = '<div class="market-chart-empty">No recent chart points are available for this asset and timeframe yet.</div>';
        if (marketChartNote) {
            marketChartNote.textContent = 'Chart data unavailable.';
        }
        return;
    }

    const closes = points.map((point) => Number(point.close || 0));
    const volumes = points.map((point) => Number(point.volume || 0));
    const minClose = Math.min(...closes);
    const maxClose = Math.max(...closes);
    const firstClose = closes[0] || 0;
    const lastClose = closes[closes.length - 1] || 0;
    const movePct = firstClose ? ((lastClose - firstClose) / firstClose) * 100 : 0;
    const isPositive = lastClose >= firstClose;
    const lineColor = isPositive ? '#008954' : '#c94a4a';
    const fillColor = isPositive ? 'rgba(0, 137, 87, 0.12)' : 'rgba(201, 74, 74, 0.10)';
    const maxVolume = Math.max(...volumes, 1);
    const width = 720;
    const height = 320;
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
        const barHeight = Math.max((Number(point.volume || 0) / maxVolume) * 36, 1);
        return `<rect x="${x.toFixed(1)}" y="${(height - padY - barHeight).toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="1.5" fill="rgba(71, 85, 105, 0.20)"></rect>`;
    }).join('');

    marketChart.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Recent closing price chart">
            <line x1="${padX}" y1="${highY.toFixed(1)}" x2="${width - padX}" y2="${highY.toFixed(1)}" class="market-chart-guide"></line>
            <line x1="${padX}" y1="${lowY.toFixed(1)}" x2="${width - padX}" y2="${lowY.toFixed(1)}" class="market-chart-guide"></line>
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
        </svg>
    `;

    const latestPoint = points[points.length - 1];
    const firstPoint = points[0];
    if (marketChartStats) {
        marketChartStats.innerHTML = `
            <div><span>Window move</span><strong>${formatSignedPercent(movePct)}</strong></div>
            <div><span>High</span><strong>${maxClose.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong></div>
            <div><span>Low</span><strong>${minClose.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong></div>
            <div><span>Latest</span><strong>${lastClose.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong></div>
        `;
    }
    if (marketChartNote) {
        marketChartNote.textContent =
            `${formatAsset(chartPayload.asset)} chart uses ${points.length} saved ${formatTimeframe(chartPayload.resolved_timeframe)} candles from ${new Date(firstPoint.time).toLocaleDateString()} to ${new Date(latestPoint.time).toLocaleDateString()}.`;
    }
}

function deriveMarketBehaviour(points) {
    // computed from the same chart bars shown above so every asset has values
    const closes = (points || []).map((p) => Number(p.close)).filter((v) => Number.isFinite(v) && v > 0);
    const volumes = (points || []).map((p) => Number(p.volume || 0)).filter(Number.isFinite);
    const result = { volatilityLabel: null, activityLabel: null, indicatorLabel: null };

    if (closes.length >= 5) {
        const returns = [];
        for (let i = 1; i < closes.length; i += 1) {
            returns.push(Math.log(closes[i] / closes[i - 1]));
        }
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
        const stdev = Math.sqrt(variance);
        result.volatilityLabel = stdev > 0.025 ? 'High' : (stdev > 0.012 ? 'Medium' : 'Low');

        const window = Math.min(20, closes.length);
        const tail = closes.slice(-window);
        const ma = tail.reduce((a, b) => a + b, 0) / window;
        const last = closes[closes.length - 1];
        if (last > ma * 1.005) result.indicatorLabel = 'Buy';
        else if (last < ma * 0.995) result.indicatorLabel = 'Avoid';
        else result.indicatorLabel = 'Hold';
    }

    if (volumes.length >= 5) {
        const vMean = volumes.reduce((a, b) => a + b, 0) / volumes.length;
        const vVar = volumes.reduce((a, b) => a + (b - vMean) ** 2, 0) / volumes.length;
        const vStd = Math.sqrt(vVar) || 1;
        const z = (volumes[volumes.length - 1] - vMean) / vStd;
        result.activityLabel = z >= 1.0 ? 'Elevated' : (z <= -0.5 ? 'Quiet' : 'Normal');
    }

    return result;
}

function renderMarketSnapshot(snapshot, derived = {}) {
    if (!snapshot) {
        return;
    }

    const defiContext = snapshot.defi_context || {};

    const returnValue = Number(snapshot.latest_return_24h_pct || 0);
    const assetLabel = formatAsset(snapshot.asset);
    latestClose.textContent = Number(snapshot.latest_close || 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
    return24h.textContent = formatSignedPercent(returnValue);
    const rawVolatility = contextAvailabilityLabel(snapshot.volatility_status, '');
    const rawActivity = snapshot.activity_status || '';
    const rawRule = formatDecisionLabel(snapshot.rule_signal);
    const isMissing = (v) => !v || ['n/a', 'unknown', 'not available', ''].includes(String(v).toLowerCase());

    volatility.textContent = isMissing(rawVolatility) ? (derived.volatilityLabel || 'n/a') : rawVolatility;
    marketActivity.textContent = isMissing(rawActivity) ? (derived.activityLabel || 'n/a') : rawActivity;
    marketRuleSignal.textContent = isMissing(rawRule) ? (derived.indicatorLabel || 'n/a') : rawRule;
    marketSummary.textContent = `${assetLabel} is ${returnValue >= 0 ? 'up' : 'down'} ${Math.abs(returnValue).toFixed(2)}% over the latest 24h window. Volatility is ${contextAvailabilityLabel(snapshot.volatility_status, 'unknown').toLowerCase()} and activity is ${String(snapshot.activity_status || 'normal').toLowerCase()}.`;

    if (marketPriceRead) {
        marketPriceRead.textContent = `Last saved Binance spot close for ${assetLabel}.`;
    }

    if (marketReturnMeaning) {
        marketReturnMeaning.textContent = describeMove(returnValue);
    }

    if (marketVolatilityMeaning) {
        marketVolatilityMeaning.textContent = describeVolatility(snapshot.volatility_status);
    }

    if (marketActivityMeaning) {
        marketActivityMeaning.textContent = describeActivity(snapshot.activity_status);
    }

    if (marketPolicyReturn) {
        marketPolicyReturn.textContent = `${(Number(snapshot.policy_return || 0) * 100).toFixed(1)}%`;
    }

    if (marketSharpe) {
        marketSharpe.textContent = Number(snapshot.sharpe_ratio || 0).toFixed(2);
    }

    if (marketDrawdown) {
        marketDrawdown.textContent = `${(Number(snapshot.max_drawdown || 0) * 100).toFixed(1)}%`;
    }

    if (marketSignalMode) {
        const rawMode = snapshot.context_mode === 'Ready'
            ? 'Market data ready'
            : contextAvailabilityLabel(snapshot.context_mode, 'Available');
        marketSignalMode.textContent = String(rawMode).replace(/Fallback/gi, 'Backup');
    }

    if (marketEffectiveSentiment) {
        marketEffectiveSentiment.textContent = snapshot.effective_sentiment_source === 'unavailable'
            ? 'Not Available'
            : toTitleCase(snapshot.effective_sentiment_source);
    }
    if (marketSentimentNote) {
        marketSentimentNote.textContent = snapshot.effective_sentiment_source === 'unavailable'
            ? 'No usable sentiment source for this asset.'
            : snapshot.effective_sentiment_source === 'fear_greed_market_fallback'
                ? 'Broad market mood (asset news not available).'
                : 'Available sentiment source in use.';
    }

    if (marketGdeltStatus) {
        marketGdeltStatus.textContent = snapshot.gdelt_status && snapshot.gdelt_status !== 'unavailable'
            ? `${toTitleCase(snapshot.gdelt_status)} (${Number(snapshot.gdelt_article_count || 0).toFixed(0)})`
            : 'Not Available';
    }
    if (marketNewsNote) {
        marketNewsNote.textContent = snapshot.gdelt_status && snapshot.gdelt_status !== 'unavailable'
            ? `${Number(snapshot.gdelt_article_count || 0).toFixed(0)} recent articles for this asset.`
            : 'Asset news not available right now.';
    }

    if (marketOnchainStatus) {
        marketOnchainStatus.textContent = snapshot.onchain_status && snapshot.onchain_status !== 'unavailable'
            ? toTitleCase(snapshot.onchain_status)
            : snapshot.onchain_snapshot_status === 'stale'
                ? `Stale ${toTitleCase(snapshot.onchain_snapshot_label)} (${Number(snapshot.onchain_snapshot_age_days || 0).toFixed(0)}d)`
                : 'Not Available';
    }
    if (marketOnchainNote) {
        marketOnchainNote.textContent = snapshot.onchain_status && snapshot.onchain_status !== 'unavailable'
            ? 'Network data available as confirmation.'
            : snapshot.onchain_snapshot_status === 'stale'
                ? `Older snapshot only (${Number(snapshot.onchain_snapshot_age_days || 0).toFixed(0)}d old).`
                : 'Network on-chain data not available.';
    }

    if (marketDefiStatus) {
        marketDefiStatus.textContent = defiContext.available
            ? toTitleCase(defiContext.regime_label || 'available')
            : 'Not Available';
    }
    if (marketDefiNote) {
        marketDefiNote.textContent = defiContext.available
            ? `${defiContext.chain_name || 'Chain'} TVL: ${formatMoney(defiContext.latest_tvl_usd)}.`
            : 'TVL context not available.';
    }

    if (marketCapabilityNote) {
        marketCapabilityNote.textContent = Array.isArray(snapshot.capability_notes)
            ? simplifyCopy(snapshot.capability_notes.join(' '))
            : 'LiveStrat shows the extra context available for each asset.';
    }

    if (marketMultimodalSelection) {
        marketMultimodalSelection.textContent = snapshot.multimodal_selected_context_variant
            ? `A multimodal context family is available in the research layer: ${toTitleCase(snapshot.multimodal_selected_context_variant)}. Its actual decision use belongs on the Strategies and Analytics pages.`
            : 'No validation-selected multimodal context family is active for this asset right now.';
    }

    if (marketSignalExplainer) {
        marketSignalExplainer.textContent =
            `Use this market read to understand recent price behaviour for ${assetLabel}. The basic rule read is ${formatDecisionLabel(snapshot.rule_signal)}, but trade decisions should be checked on the Strategies tab.`;
    }

    if (marketFuturesSupport) {
        marketFuturesSupport.textContent = toTitleCase(
            snapshot.futures_context_resilience_label || snapshot.futures_completeness_label || 'n/a'
        );
    }
    if (marketFuturesNote) {
        marketFuturesNote.textContent = snapshot.futures_completeness_label
            ? `Futures coverage: ${toTitleCase(snapshot.futures_completeness_label)}.`
            : 'Futures coverage not available.';
    }

    if (marketBasisMode) {
        marketBasisMode.textContent = snapshot.basis_proxy_active
            ? `Proxy ${toTitleCase(snapshot.basis_feature_mode || 'basis')}`
            : toTitleCase(snapshot.basis_feature_mode || 'n/a');
    }

    if (marketNewsTheme) {
        marketNewsTheme.textContent = snapshot.gdelt_status && snapshot.gdelt_status !== 'unavailable'
            ? toTitleCase(snapshot.gdelt_dominant_event_theme || 'none')
            : 'Unavailable';
    }

    if (marketOnchainDriver) {
        marketOnchainDriver.textContent = snapshot.onchain_primary_support_driver
            ? toTitleCase(snapshot.onchain_primary_support_driver)
            : 'Unavailable';
    }

    if (marketDefiChain) {
        marketDefiChain.textContent = defiContext.available
            ? `${defiContext.chain_name || 'Chain'} ${formatMoney(defiContext.latest_tvl_usd)}`
            : 'Unavailable';
    }

    if (marketContextNote) {
        marketContextNote.textContent = simplifyCopy(
            `Summary: market and futures are the main usable layers here. Sentiment, on-chain, and TVL are shown only when available, and should be treated as supporting context.`
        );
    }

    if (marketSourcePolicyNote) {
        marketSourcePolicyNote.textContent =
            'How to use this: price and futures are the main market inputs. Sentiment, on-chain, and TVL are extra checks when available.';
    }

    if (marketLaneBadge) {
        marketLaneBadge.textContent = `Market view - ${formatTimeframe(snapshot.resolved_timeframe)}`;
    }

    if (marketRefreshNote) {
        const availabilityNote = snapshot.exact_timeframe_match
            ? ''
            : ` Using ${formatTimeframe(snapshot.resolved_timeframe)} data because exact ${formatTimeframe(snapshot.requested_timeframe)} summaries are not available.`;
        marketRefreshNote.textContent =
            `${assetLabel} | ${formatTimeframe(snapshot.resolved_timeframe)} view | Updated ${formatTimestamp(snapshot.refreshed_at)}.` +
            availabilityNote;
    }

    updateHeroAsset(snapshot, assetLabel);
    updateContextDots(snapshot, defiContext);
}

const ASSET_NAME_MAP = {
    BTCUSDT: 'Bitcoin', ETHUSDT: 'Ethereum', SOLUSDT: 'Solana',
    BNBUSDT: 'BNB', XRPUSDT: 'XRP', ADAUSDT: 'Cardano', DOGEUSDT: 'Dogecoin',
};

function updateHeroAsset(snapshot, assetLabel) {
    const symbolEl = document.getElementById('market-hero-symbol');
    const nameEl = document.getElementById('market-hero-name');
    if (symbolEl) symbolEl.textContent = assetLabel;
    if (nameEl) {
        const asset = snapshot.asset || (marketAsset ? marketAsset.value : 'BTCUSDT');
        nameEl.textContent = ASSET_NAME_MAP[asset] || asset;
    }
}

function setContextDot(key, dotClass) {
    const dot = document.getElementById(`context-${key}-dot`);
    if (!dot) return;
    dot.classList.remove(
        'evidence-dot-good',
        'evidence-dot-warn',
        'evidence-dot-bad',
        'evidence-dot-neutral',
        'evidence-dot-na'
    );
    dot.classList.add(`evidence-dot-${dotClass}`);
    const card = dot.closest('.context-card');
    if (card) card.classList.toggle('is-unavailable', dotClass === 'na');
}

function updateContextDots(snapshot, defiContext) {
    // Futures
    const futuresLabel = String(snapshot.futures_context_resilience_label || snapshot.futures_completeness_label || '').toLowerCase();
    if (!futuresLabel || futuresLabel === 'unavailable' || futuresLabel === 'n/a') {
        setContextDot('futures', 'na');
    } else if (futuresLabel.includes('robust') || futuresLabel.includes('full')) {
        setContextDot('futures', 'good');
    } else if (futuresLabel.includes('partial') || futuresLabel.includes('proxy') || futuresLabel.includes('fragile')) {
        setContextDot('futures', 'warn');
    } else {
        setContextDot('futures', 'neutral');
    }

    // Sentiment source
    const sentSource = String(snapshot.effective_sentiment_source || '').toLowerCase();
    if (!sentSource || sentSource === 'unavailable') {
        setContextDot('sentiment', 'na');
    } else if (sentSource === 'fear_greed_market_fallback') {
        setContextDot('sentiment', 'warn');
    } else {
        setContextDot('sentiment', 'good');
    }

    // News (GDELT)
    const gdeltStatus = String(snapshot.gdelt_status || '').toLowerCase();
    if (!gdeltStatus || gdeltStatus === 'unavailable') {
        setContextDot('news', 'na');
    } else if (gdeltStatus === 'supportive') {
        setContextDot('news', 'good');
    } else if (gdeltStatus === 'risk_off') {
        setContextDot('news', 'bad');
    } else {
        setContextDot('news', 'neutral');
    }

    // On-chain
    const onchainStatus = String(snapshot.onchain_status || '').toLowerCase();
    const onchainSnapshotStatus = String(snapshot.onchain_snapshot_status || '').toLowerCase();
    if (!onchainStatus || onchainStatus === 'unavailable') {
        if (onchainSnapshotStatus === 'stale') {
            setContextDot('onchain', 'warn');
        } else {
            setContextDot('onchain', 'na');
        }
    } else if (onchainStatus.includes('supportive') || onchainStatus.includes('bullish')) {
        setContextDot('onchain', 'good');
    } else if (onchainStatus.includes('risk') || onchainStatus.includes('bearish')) {
        setContextDot('onchain', 'bad');
    } else {
        setContextDot('onchain', 'neutral');
    }

    // DeFi TVL
    if (!defiContext || !defiContext.available) {
        setContextDot('defi', 'na');
    } else {
        const regime = String(defiContext.regime_label || '').toLowerCase();
        if (regime.includes('expanding') || regime.includes('growing')) {
            setContextDot('defi', 'good');
        } else if (regime.includes('contracting') || regime.includes('shrinking')) {
            setContextDot('defi', 'warn');
        } else {
            setContextDot('defi', 'neutral');
        }
    }
}

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
    }
    return response.json();
}

async function renderMarketLane() {
    const asset = marketAsset ? marketAsset.value : 'BTCUSDT';
    const timeframe = marketTimeframe ? marketTimeframe.value : '4h';

    try {
        const [snapshot, chartPayload] = await Promise.all([
            fetchJson(`/api/market-snapshot?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}`),
            fetchJson(`/api/market-chart?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&points=48`),
        ]);
        const derived = deriveMarketBehaviour(chartPayload.points);
        renderMarketSnapshot(snapshot, derived);
        renderMarketChart(chartPayload);
    } catch (error) {
        renderMarketSnapshot(buildFallbackSnapshot(asset, timeframe));
        renderMarketChart({ points: [] });
        if (marketRefreshNote) {
            marketRefreshNote.textContent = `Market display API unavailable. ${error.message}`;
        }
    }
}

if (marketButton) {
    marketButton.addEventListener('click', renderMarketLane);
}

if (marketAsset) {
    marketAsset.addEventListener('change', renderMarketLane);
}

if (marketTimeframe) {
    marketTimeframe.addEventListener('change', renderMarketLane);
    marketTimeframe.value = '4h';
}

renderMarketLane();
