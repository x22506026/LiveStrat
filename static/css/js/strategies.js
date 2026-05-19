/* ============================================================
   LiveStrat - Strategies page
   Renders: hero verdict card, five evidence cards,
   track record, strategy grid, technical details.
   ============================================================ */

const marketSummaryScript = document.getElementById('market-summaries-data');
const marketSummaries = marketSummaryScript ? JSON.parse(marketSummaryScript.textContent) : {};

const strategySelect = document.getElementById('strategy-select');
const strategyAsset = document.getElementById('strategy-asset');
const strategyTimeframe = document.getElementById('strategy-timeframe');

const ASSET_NAMES = {
    BTCUSDT: 'Bitcoin',
    ETHUSDT: 'Ethereum',
    SOLUSDT: 'Solana',
    BNBUSDT: 'BNB',
    XRPUSDT: 'XRP',
    ADAUSDT: 'Cardano',
    DOGEUSDT: 'Dogecoin',
};

const STRATEGY_LABELS = {
    recommended: 'Balanced Default',
    conservative_trend: 'Trend Confirmation',
    momentum_breakout: 'Momentum Breakout',
    futures_crowd_reversal: 'Crowd Reversal',
    rule_based: 'Transparent Rule Benchmark',
};

const ENGINE_FRIENDLY_NAMES = {
    market_futures_lagged_h24: 'Market + Futures (24h horizon)',
    market_futures_lagged_h72: 'Market + Futures (72h horizon)',
    market_futures_lagged_h8: 'Market + Futures (8h horizon)',
    market_futures_voladj_h24: 'Market + Futures (volatility-adjusted, 24h)',
    market_futures_voladj_h72: 'Market + Futures (volatility-adjusted, 72h)',
    market_futures_lstm_preferred_fixed_h24: 'LSTM on Market + Futures (24h)',
    market_futures_lstm_preferred_voladj_h24: 'LSTM on Market + Futures (vol-adj, 24h)',
    scaled_market_baseline: 'Scaled Market Baseline',
    unscaled_market_baseline: 'Unscaled Market Baseline',
    rule_based: 'Transparent rule-based',
    market_baseline: 'Market Baseline (Logistic)',
};

const POLICY_FRIENDLY_NAMES = {
    confidence_gated_long_flat: 'Confidence-gated long / flat',
    regime_adaptive_long_flat: 'Regime-adaptive long / flat',
    conviction_weighted_long_only: 'Conviction-weighted long-only',
    transparent_market_rule_set: 'Transparent rule set',
    classification_only: 'Classification-only (no policy)',
};

function friendlyEngineName(raw) {
    if (!raw || raw === 'n/a') return 'n/a';
    if (ENGINE_FRIENDLY_NAMES[raw]) return ENGINE_FRIENDLY_NAMES[raw];
    // Best-effort beautification for unknown engine codes
    return String(raw)
        .replace(/_/g, ' ')
        .replace(/\bh(\d+)\b/g, '($1h)')
        .replace(/\b\w/g, (c) => c.toUpperCase());
}

function friendlyPolicyName(raw) {
    if (!raw || raw === 'n/a') return 'n/a';
    if (POLICY_FRIENDLY_NAMES[raw]) return POLICY_FRIENDLY_NAMES[raw];
    return String(raw).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

const SIGNAL_TO_VERDICT = {
    buy: { label: 'BUY', class: 'verdict-badge-buy' },
    long: { label: 'BUY', class: 'verdict-badge-buy' },
    hold: { label: 'HOLD', class: 'verdict-badge-hold' },
    flat: { label: 'HOLD', class: 'verdict-badge-hold' },
    dont_buy: { label: 'AVOID', class: 'verdict-badge-avoid' },
    do_not_buy: { label: 'AVOID', class: 'verdict-badge-avoid' },
    sell: { label: 'AVOID', class: 'verdict-badge-avoid' },
    short: { label: 'AVOID', class: 'verdict-badge-avoid' },
};

/* ----- utilities ----- */

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function formatPriceUSD(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric === 0) return 'n/a';
    if (numeric >= 1000) {
        return `$${numeric.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    if (numeric >= 1) {
        return `$${numeric.toFixed(2)}`;
    }
    return `$${numeric.toFixed(4)}`;
}

function formatPercent(value, withSign = false) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 'n/a';
    const formatted = (numeric * 100).toFixed(1);
    if (withSign) {
        return `${numeric >= 0 ? '+' : ''}${formatted}%`;
    }
    return `${formatted}%`;
}

function daysAgo(isoString) {
    if (!isoString) return null;
    const then = new Date(isoString);
    if (Number.isNaN(then.getTime())) return null;
    const now = new Date();
    return Math.max(0, Math.floor((now - then) / (1000 * 60 * 60 * 24)));
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

/* ----- hero card ----- */

function renderHero(asset, summary, signal, confidence, refreshDays, timeframe) {
    setText('hero-symbol', asset.replace('USDT', ' / USDT'));
    setText('hero-asset-name', ASSET_NAMES[asset] || asset);

    const latestClose = Number(summary.latest_close || 0);
    setText('hero-price', formatPriceUSD(latestClose));

    const change24h = Number(summary.latest_return_24h_pct || 0);
    const changeEl = document.getElementById('hero-change');
    if (changeEl) {
        const hasData = latestClose > 0;
        changeEl.classList.remove('is-positive', 'is-negative');
        if (!hasData) {
            changeEl.textContent = 'No recent data';
        } else {
            const sign = change24h >= 0 ? '+' : '';
            changeEl.textContent = `${sign}${change24h.toFixed(2)}% over 24h`;
            changeEl.classList.add(change24h >= 0 ? 'is-positive' : 'is-negative');
        }
    }

    // Verdict badge
    const verdictBadge = document.getElementById('hero-verdict-badge');
    const verdictText = document.getElementById('hero-verdict-text');
    if (verdictBadge && verdictText) {
        verdictBadge.classList.remove(
            'verdict-badge-buy',
            'verdict-badge-hold',
            'verdict-badge-avoid',
            'verdict-badge-unavailable'
        );
        const normalized = String(signal || '').toLowerCase();
        const verdict = SIGNAL_TO_VERDICT[normalized];
        if (verdict && latestClose > 0) {
            verdictText.textContent = verdict.label;
            verdictBadge.classList.add(verdict.class);
        } else {
            verdictText.textContent = 'NOT AVAILABLE';
            verdictBadge.classList.add('verdict-badge-unavailable');
        }
    }

    // Confidence
    const confidencePct = Math.round(clamp(Number(confidence) || 0, 0, 1) * 100);
    setText('hero-confidence-value', `${confidencePct}%`);
    const fill = document.getElementById('hero-confidence-fill');
    if (fill) fill.style.width = `${confidencePct}%`;

    // Freshness
    const freshnessEl = document.getElementById('hero-freshness');
    const dot = document.getElementById('hero-freshness-dot');
    if (freshnessEl && dot) {
        dot.classList.remove('freshness-dot-green', 'freshness-dot-amber', 'freshness-dot-red');
        let text;
        if (refreshDays === null || !Number.isFinite(refreshDays)) {
            text = 'Refresh date unavailable.';
            dot.classList.add('freshness-dot-amber');
        } else if (refreshDays === 0) {
            text = 'Refreshed today.';
            dot.classList.add('freshness-dot-green');
        } else if (refreshDays <= 2) {
            text = `Refreshed ${refreshDays}d ago.`;
            dot.classList.add('freshness-dot-green');
        } else if (refreshDays <= 7) {
            text = `Refreshed ${refreshDays}d ago. Treat with caution.`;
            dot.classList.add('freshness-dot-amber');
        } else {
            text = `Refreshed ${refreshDays}d ago. Stale.`;
            dot.classList.add('freshness-dot-red');
        }
        const strategyLabel = STRATEGY_LABELS[strategySelect?.value] || 'Strategy';
        const timeframeLabel = strategyTimeframe?.value === '1h' ? '1h' : '4h';
        freshnessEl.innerHTML = '';
        freshnessEl.appendChild(dot);
        freshnessEl.append(` ${text} · ${strategyLabel} · ${timeframeLabel}`);
    }

    // Strategy note in verdict card
    const strategyLabel = STRATEGY_LABELS[strategySelect?.value] || 'Strategy';
    setText('hero-strategy-note', `${strategyLabel} strategy.`);

    // Sparkline (real recent price data)
    renderSparkline(summary, asset, timeframe || '4h');
}

async function renderSparkline(summary, asset, timeframe) {
    const container = document.getElementById('hero-spark');
    if (!container) return;
    container.innerHTML = '';
    container.style.alignItems = '';
    container.style.justifyContent = '';

    let points = [];
    try {
        const response = await fetch(`/api/market-chart?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&points=60`);
        if (response.ok) {
            const payload = await response.json();
            points = (payload.points || []).map((p) => Number(p.close)).filter((v) => Number.isFinite(v) && v > 0);
        }
    } catch (err) {
        // fall through to empty state
    }

    if (points.length < 2) {
        const empty = document.createElement('p');
        empty.style.cssText = 'margin: auto; color: var(--text-muted); font-size: 0.9rem; font-style: italic;';
        empty.textContent = 'No recent price data.';
        container.style.alignItems = 'center';
        container.style.justifyContent = 'center';
        container.appendChild(empty);
        return;
    }

    const min = Math.min(...points);
    const max = Math.max(...points);
    const range = Math.max(max - min, 0.0001);
    const width = 100;
    const height = 30;
    const padY = 2;

    const path = points.map((value, idx) => {
        const x = (idx / (points.length - 1)) * width;
        const y = (height - padY) - ((value - min) / range) * (height - padY * 2);
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(' ');

    const totalReturn = (points[points.length - 1] - points[0]) / points[0];
    const strokeColor = totalReturn >= 0 ? '#0d7c4e' : '#b84b4b';
    const fillColor = totalReturn >= 0 ? 'rgba(0, 139, 90, 0.12)' : 'rgba(184, 75, 75, 0.10)';

    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'none');

    const fillPath = document.createElementNS(svgNS, 'path');
    fillPath.setAttribute('d', `${path} L ${width} ${height} L 0 ${height} Z`);
    fillPath.setAttribute('fill', fillColor);
    svg.appendChild(fillPath);

    const stroke = document.createElementNS(svgNS, 'path');
    stroke.setAttribute('d', path);
    stroke.setAttribute('fill', 'none');
    stroke.setAttribute('stroke', strokeColor);
    stroke.setAttribute('stroke-width', '1.4');
    stroke.setAttribute('stroke-linecap', 'round');
    stroke.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(stroke);

    container.appendChild(svg);
}

/* ----- evidence cards ----- */

function setEvidence(key, status, dotClass, detail) {
    const card = document.querySelector(`[data-evidence="${key}"]`);
    if (!card) return;
    card.classList.remove('is-unavailable');
    if (dotClass === 'na') {
        card.classList.add('is-unavailable');
    }
    setText(`evidence-${key}-status`, status);
    setText(`evidence-${key}-detail`, detail);
    const dot = document.getElementById(`evidence-${key}-dot`);
    if (dot) {
        dot.classList.remove(
            'evidence-dot-good',
            'evidence-dot-warn',
            'evidence-dot-bad',
            'evidence-dot-neutral',
            'evidence-dot-na'
        );
        dot.classList.add(`evidence-dot-${dotClass}`);
    }
}

function readPriceTrend(summary) {
    const close = Number(summary.latest_close || 0);
    if (close <= 0) {
        return setEvidence('trend', 'Not available', 'na', 'Run the pipeline refresh to populate price trend data for this asset.');
    }
    const r24 = Number(summary.latest_return_24h_pct || 0);
    const r3d = Number(summary.latest_return_3d_pct || 0);
    const smaDiff = Number(summary.latest_price_sma_20_diff || 0);

    let status, dot, detail;
    if (smaDiff > 0.01 && r24 > 0) {
        status = 'Bullish';
        dot = 'good';
        detail = `Up ${r24.toFixed(2)}% over 24h, trading above the 20-period moving average.`;
    } else if (smaDiff > 0.01 && r24 <= 0) {
        status = 'Mildly bullish';
        dot = 'warn';
        detail = `Above 20-period moving average, but down ${Math.abs(r24).toFixed(2)}% in the last 24h.`;
    } else if (smaDiff < -0.01 && r24 < 0) {
        status = 'Bearish';
        dot = 'bad';
        detail = `Down ${Math.abs(r24).toFixed(2)}% over 24h, trading below the 20-period moving average.`;
    } else if (smaDiff < -0.01) {
        status = 'Mildly bearish';
        dot = 'warn';
        detail = `Below 20-period moving average, recent 24h move ${r24 >= 0 ? '+' : ''}${r24.toFixed(2)}%.`;
    } else {
        status = 'Sideways';
        dot = 'neutral';
        detail = `Price near the 20-period average. 24h move ${r24 >= 0 ? '+' : ''}${r24.toFixed(2)}%, 3-day ${r3d >= 0 ? '+' : ''}${r3d.toFixed(2)}%.`;
    }
    setEvidence('trend', status, dot, detail);
}

function readActivity(summary) {
    const close = Number(summary.latest_close || 0);
    if (close <= 0) {
        return setEvidence('activity', 'Not available', 'na', 'Volume and participation data not loaded for this asset.');
    }
    const volZ = Number(summary.latest_volume_zscore || 0);
    const volatility = Number(summary.latest_volatility_20 || 0);

    let status, dot, detail;
    if (volZ > 1.5) {
        status = 'Elevated';
        dot = 'warn';
        detail = `Trading volume well above its 20-period average (z-score ${volZ.toFixed(2)}). Activity is unusually high.`;
    } else if (volZ < -1) {
        status = 'Quiet';
        dot = 'warn';
        detail = `Trading volume below its 20-period average (z-score ${volZ.toFixed(2)}). Low participation reduces signal reliability.`;
    } else {
        status = 'Normal';
        dot = 'good';
        detail = `Volume in line with recent average (z-score ${volZ.toFixed(2)}). Volatility reading ${(volatility * 100).toFixed(2)}%.`;
    }
    setEvidence('activity', status, dot, detail);
}

function readFutures(summary) {
    const crowdScore = summary.latest_futures_crowding_score;
    const fundingZ = summary.latest_funding_rate_zscore_21;
    const futuresAvailable = crowdScore !== undefined && crowdScore !== null && crowdScore !== '';

    if (!futuresAvailable) {
        return setEvidence('futures', 'Not available', 'na', 'Futures positioning data has not been generated for this asset/timeframe yet.');
    }

    const crowd = Number(crowdScore);
    const funding = Number(fundingZ || 0);

    let status, dot, detail;
    if (crowd > 0.5) {
        status = 'Crowded long';
        dot = 'warn';
        detail = `Positioning leans heavily long (crowding score ${crowd.toFixed(2)}). Reversal risk elevated if sentiment turns.`;
    } else if (crowd < -0.5) {
        status = 'Crowded short';
        dot = 'warn';
        detail = `Positioning leans heavily short (crowding score ${crowd.toFixed(2)}). Squeeze potential if price stabilises.`;
    } else if (Math.abs(funding) > 1.5) {
        status = 'Funding stretched';
        dot = 'warn';
        detail = `Funding rate z-score ${funding.toFixed(2)} which is well outside the recent range. Watch for normalisation.`;
    } else {
        status = 'Balanced';
        dot = 'good';
        detail = `Crowding score ${crowd.toFixed(2)}, funding z-score ${funding.toFixed(2)}. Positioning looks healthy.`;
    }
    setEvidence('futures', status, dot, detail);
}

function readNews(summary) {
    const gdeltLabel = String(summary.latest_gdelt_regime_label || '').toLowerCase();
    const articleCount = Number(summary.latest_gdelt_article_count || 0);

    if (!gdeltLabel || gdeltLabel === 'unavailable' || gdeltLabel === 'n/a') {
        // Fall back to broad market mood via Fear & Greed if available
        const fgLabel = String(summary.latest_effective_sentiment_label || '').toLowerCase();
        if (fgLabel && fgLabel !== 'unavailable' && fgLabel !== 'n/a') {
            const mapped = fgLabel.replace(/_/g, ' ');
            return setEvidence(
                'news',
                mapped.charAt(0).toUpperCase() + mapped.slice(1),
                fgLabel === 'risk_off' ? 'bad' : (fgLabel === 'risk_on' || fgLabel === 'supportive' ? 'good' : 'neutral'),
                'No recent asset-specific news. Showing broad market mood from the Fear & Greed Index as a backup.'
            );
        }
        return setEvidence('news', 'No coverage', 'na', 'No recent asset-specific news coverage detected for this asset.');
    }

    let status, dot, detail;
    if (gdeltLabel === 'supportive') {
        status = 'Supportive';
        dot = 'good';
        detail = `${articleCount} recent asset news articles leaning positive.`;
    } else if (gdeltLabel === 'risk_off') {
        status = 'Risk-off';
        dot = 'bad';
        detail = `${articleCount} recent asset news articles leaning negative. Caution warranted.`;
    } else {
        status = 'Mixed';
        dot = 'neutral';
        detail = `${articleCount} recent asset news articles with mixed sentiment. No strong directional bias.`;
    }
    setEvidence('news', status, dot, detail);
}

function readOnchain(summary) {
    const onchainLabel = String(summary.latest_onchain_regime_label || '').toLowerCase();

    if (!onchainLabel || onchainLabel === 'unavailable' || onchainLabel === 'n/a') {
        return setEvidence('onchain', 'Not available', 'na', 'Network on-chain telemetry is not currently covered for this asset.');
    }

    const onchainScore = Number(summary.latest_onchain_snapshot_score || 0);

    let status, dot, detail;
    if (onchainLabel.includes('supportive') || onchainLabel === 'bullish_alignment') {
        status = 'Supportive';
        dot = 'good';
        detail = `On-chain regime label "${onchainLabel.replace(/_/g, ' ')}", structural score ${onchainScore.toFixed(2)}. Network activity supports the market read.`;
    } else if (onchainLabel.includes('risk') || onchainLabel === 'bearish_alignment') {
        status = 'Weakening';
        dot = 'bad';
        detail = `On-chain regime label "${onchainLabel.replace(/_/g, ' ')}", structural score ${onchainScore.toFixed(2)}. Network telemetry weakening.`;
    } else {
        status = 'Neutral';
        dot = 'neutral';
        detail = `On-chain regime label "${onchainLabel.replace(/_/g, ' ')}", structural score ${onchainScore.toFixed(2)}. No strong directional bias.`;
    }
    setEvidence('onchain', status, dot, detail);
}

/* ----- track record ----- */

function renderTrackRecord(summary, asset) {
    // Prefer walk-forward stats over single-window in-sample stats for the track record.
    // Walk-forward numbers are more honest because they aggregate across multiple
    // retrained folds rather than reporting a single optimistic backtest pass.
    const foldCount = Number(summary.walkforward_fold_count ?? 0);
    const useWalkForward = foldCount > 0;

    const accuracy = useWalkForward
        ? Number(summary.walkforward_avg_accuracy ?? 0)
        : Number(summary.test_accuracy ?? summary.baseline_scaled_test_accuracy ?? 0);

    const stratReturn = useWalkForward
        ? Number(summary.walkforward_avg_strategy_total_return ?? summary.strategy_total_return ?? 0)
        : Number(summary.strategy_total_return ?? 0);

    const buyHold = useWalkForward
        ? Number(summary.walkforward_avg_buy_hold_return ?? summary.buy_hold_total_return ?? 0)
        : Number(summary.buy_hold_total_return ?? 0);

    const excess = useWalkForward
        ? Number(summary.walkforward_avg_excess_return ?? (stratReturn - buyHold))
        : Number(summary.excess_return ?? (stratReturn - buyHold));

    const accuracyEl = document.getElementById('track-accuracy');
    if (accuracyEl) {
        accuracyEl.textContent = accuracy > 0 ? formatPercent(accuracy) : 'n/a';
        accuracyEl.classList.toggle('is-na', accuracy <= 0);
    }
    const accuracyFill = document.getElementById('track-accuracy-fill');
    if (accuracyFill) {
        accuracyFill.style.width = `${clamp(accuracy * 100, 0, 100)}%`;
    }

    const stratEl = document.getElementById('track-strategy-return');
    if (stratEl) {
        stratEl.textContent = stratReturn === 0 ? '0.0%' : formatPercent(stratReturn, true);
        stratEl.classList.remove('is-positive', 'is-negative', 'is-na');
        if (stratReturn > 0) stratEl.classList.add('is-positive');
        else if (stratReturn < 0) stratEl.classList.add('is-negative');
    }

    const buyHoldEl = document.getElementById('track-buy-hold');
    if (buyHoldEl) {
        buyHoldEl.textContent = buyHold === 0 ? 'n/a' : formatPercent(buyHold, true);
        buyHoldEl.classList.remove('is-positive', 'is-negative', 'is-na');
        if (buyHold > 0) buyHoldEl.classList.add('is-positive');
        else if (buyHold < 0) buyHoldEl.classList.add('is-negative');
        if (buyHold === 0) buyHoldEl.classList.add('is-na');
    }

    const excessEl = document.getElementById('track-excess');
    const excessNote = document.getElementById('track-excess-note');
    if (excessEl) {
        excessEl.textContent = formatPercent(excess, true);
        excessEl.classList.remove('is-positive', 'is-negative', 'is-na');
        if (excess > 0) excessEl.classList.add('is-positive');
        else if (excess < 0) excessEl.classList.add('is-negative');
    }
    if (excessNote) {
        if (excess > 0.005) {
            excessNote.textContent = 'Beat a passive hold.';
        } else if (excess < -0.005) {
            excessNote.textContent = 'Underperformed a passive hold.';
        } else {
            excessNote.textContent = 'Roughly matches a passive hold.';
        }
    }

    // Short read paragraph
    const honesty = document.getElementById('track-honesty');
    if (honesty) {
        honesty.classList.remove('is-positive', 'is-warn', 'is-negative');
        const assetName = ASSET_NAMES[asset] || asset;
        const basis = useWalkForward
            ? `${foldCount}-fold walk-forward`
            : 'Latest single-window';
        let message;
        let toneClass;
        if (accuracy <= 0) {
            message = `${basis} data for ${assetName} not generated yet.`;
            toneClass = 'is-warn';
        } else if (excess > 0.02) {
            message = `${basis}: beat hold of ${assetName} by ${formatPercent(excess, true)}. Direction calls ${formatPercent(accuracy)}.`;
            toneClass = 'is-positive';
        } else if (excess > 0) {
            message = `${basis}: small edge of ${formatPercent(excess, true)} over hold of ${assetName}. Direction calls ${formatPercent(accuracy)}.`;
            toneClass = 'is-positive';
        } else if (excess > -0.02) {
            message = `${basis}: roughly in line with hold of ${assetName} (${formatPercent(excess, true)}). Direction calls ${formatPercent(accuracy)}.`;
            toneClass = 'is-warn';
        } else {
            message = `${basis}: hold of ${assetName} returned more. Direction calls ${formatPercent(accuracy)}, but excess is ${formatPercent(excess, true)}.`;
            toneClass = 'is-negative';
        }
        honesty.classList.add(toneClass);
        honesty.innerHTML = `<p>${message}</p>`;
    }
}

/* ----- technical details ----- */

function renderTechnicalDetails(summary) {
    const confidence = Number(summary.latest_signal_confidence ?? summary.selected_primary_confidence ?? 0);
    setText('tech-confidence', confidence ? confidence.toFixed(3) : 'n/a');
    setText('tech-folds', summary.walkforward_fold_count ? Number(summary.walkforward_fold_count).toFixed(0) : 'n/a');

    // Prefer walk-forward metrics; fall back to in-sample only if walk-forward fields are missing.
    const macroF1 = Number(summary.walkforward_avg_macro_f1 ?? summary.test_macro_f1 ?? 0);
    setText('tech-macro-f1', macroF1 ? formatPercent(macroF1) : 'n/a');

    const sharpe = Number(summary.walkforward_avg_sharpe ?? summary.sharpe_ratio ?? 0);
    setText('tech-sharpe', sharpe ? sharpe.toFixed(2) : 'n/a');

    const drawdown = Number(summary.walkforward_avg_max_drawdown ?? summary.max_drawdown ?? 0);
    setText('tech-drawdown', drawdown ? formatPercent(drawdown, true) : 'n/a');

    const trades = Number(summary.trade_count ?? 0);
    setText('tech-trade-count', trades ? trades.toFixed(0) : 'n/a');

    setText('tech-engine', friendlyEngineName(summary.selected_primary_model || summary.selected_target_name));
    setText('tech-policy', friendlyPolicyName(summary.policy_name));
}

/* ----- strategy card active state ----- */

function highlightActiveStrategy(strategyId) {
    document.querySelectorAll('[data-strategy-id]').forEach((card) => {
        card.classList.toggle('is-active', card.dataset.strategyId === strategyId);
    });
}

/* ----- main render ----- */

async function renderStrategyView() {
    const asset = strategyAsset?.value || 'BTCUSDT';
    const strategy = strategySelect?.value || 'recommended';
    const timeframe = strategyTimeframe?.value || '4h';

    highlightActiveStrategy(strategy);

    let summary = marketSummaries[asset] || {};
    let signal = summary.selected_primary_signal || summary.latest_signal || 'hold';
    let confidence = summary.selected_primary_confidence ?? summary.latest_signal_confidence ?? 0;
    let refreshDays = daysAgo(summary.evaluation_window_end || summary.window_end || null);

    // Pull the per-strategy decision from the API if available
    try {
        const response = await fetch(
            `/api/strategy-decision?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&strategy=${encodeURIComponent(strategy)}`
        );
        if (response.ok) {
            const decision = await response.json();
            // Decision payload may have a more specific signal/confidence
            if (decision?.signal) signal = decision.signal;
            if (decision?.confidence !== undefined) confidence = decision.confidence;
            // Merge any decision-specific summary keys into the local summary
            if (decision?.summary) {
                summary = { ...summary, ...decision.summary };
            }
        }
    } catch (err) {
        // Silently fall back to the embedded market summary - the page still renders.
    }

    renderHero(asset, summary, signal, confidence, refreshDays, timeframe);
    readPriceTrend(summary);
    readActivity(summary);
    readFutures(summary);
    readNews(summary);
    readOnchain(summary);
    renderTrackRecord(summary, asset);
    renderTechnicalDetails(summary);
    renderBacktestViewer(asset, timeframe);
}

/* ----- Backtest viewer ----- */

function setBacktestStat(id, valuePct, signed = true) {
    const el = document.getElementById(id);
    if (!el) return;
    if (valuePct === null || valuePct === undefined || Number.isNaN(valuePct)) {
        el.textContent = 'n/a';
        el.classList.remove('is-positive', 'is-negative');
        return;
    }
    el.textContent = formatPercent(valuePct / 100, signed);
    el.classList.toggle('is-positive', valuePct > 0);
    el.classList.toggle('is-negative', valuePct < 0);
}

function drawEquityCurve(container, points) {
    container.innerHTML = '';
    if (!points || points.length < 2) return;

    const width = 800;
    const height = 240;
    const padX = 40;
    const padY = 20;

    const strategyValues = points.map((p) => p.strategy_equity);
    const holdValues = points.map((p) => p.buy_hold_equity);
    const allValues = strategyValues.concat(holdValues);
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const range = maxVal - minVal || 1;

    const xStep = (width - padX * 2) / (points.length - 1);

    function buildPath(values) {
        let d = '';
        values.forEach((v, i) => {
            const x = padX + i * xStep;
            const y = height - padY - ((v - minVal) / range) * (height - padY * 2);
            d += (i === 0 ? 'M' : 'L') + x.toFixed(2) + ',' + y.toFixed(2);
        });
        return d;
    }

    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'none');

    // baseline at 1.0 if it is inside the visible range
    if (minVal <= 1 && maxVal >= 1) {
        const yBase = height - padY - ((1 - minVal) / range) * (height - padY * 2);
        const baseline = document.createElementNS(svgNS, 'line');
        baseline.setAttribute('x1', padX);
        baseline.setAttribute('x2', width - padX);
        baseline.setAttribute('y1', yBase);
        baseline.setAttribute('y2', yBase);
        baseline.setAttribute('stroke', '#cdd8d3');
        baseline.setAttribute('stroke-dasharray', '3,4');
        baseline.setAttribute('stroke-width', '1');
        svg.appendChild(baseline);
    }

    const holdPath = document.createElementNS(svgNS, 'path');
    holdPath.setAttribute('d', buildPath(holdValues));
    holdPath.setAttribute('fill', 'none');
    holdPath.setAttribute('stroke', '#274c77');
    holdPath.setAttribute('stroke-width', '1.6');
    holdPath.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(holdPath);

    const stratPath = document.createElementNS(svgNS, 'path');
    stratPath.setAttribute('d', buildPath(strategyValues));
    stratPath.setAttribute('fill', 'none');
    stratPath.setAttribute('stroke', '#008b5a');
    stratPath.setAttribute('stroke-width', '2.1');
    stratPath.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(stratPath);

    container.appendChild(svg);
}

function renderFoldRows(folds) {
    const tbody = document.getElementById('backtest-folds-body');
    if (!tbody) return;
    if (!folds || folds.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9">No folds available for this asset and timeframe.</td></tr>';
        return;
    }
    const rows = folds.map((fold) => {
        const window = (fold.test_start || '').slice(0, 10) + ' to ' + (fold.test_end || '').slice(0, 10);
        const excessClass = fold.excess_return > 0 ? 'is-positive' : (fold.excess_return < 0 ? 'is-negative' : '');
        const stratClass = fold.strategy_return > 0 ? 'is-positive' : (fold.strategy_return < 0 ? 'is-negative' : '');
        return `<tr>
            <td>${fold.fold_number}</td>
            <td>${window}</td>
            <td>${(fold.accuracy * 100).toFixed(1)}%</td>
            <td>${(fold.macro_f1 * 100).toFixed(1)}%</td>
            <td class="${stratClass}">${(fold.strategy_return * 100).toFixed(2)}%</td>
            <td>${(fold.buy_hold_return * 100).toFixed(2)}%</td>
            <td class="${excessClass}">${(fold.excess_return * 100).toFixed(2)}%</td>
            <td>${Number.isFinite(fold.sharpe) ? fold.sharpe.toFixed(2) : 'n/a'}</td>
            <td>${fold.trade_count}</td>
        </tr>`;
    });
    tbody.innerHTML = rows.join('');
}

async function renderBacktestViewer(asset, timeframe) {
    const chartHost = document.getElementById('backtest-equity-chart');
    const caption = document.getElementById('backtest-chart-caption');
    const exportLink = document.getElementById('backtest-folds-export');
    if (exportLink) {
        exportLink.href = `/api/backtest-curve?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}`;
    }
    try {
        const response = await fetch(
            `/api/backtest-curve?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}`
        );
        if (!response.ok) throw new Error('request failed');
        const payload = await response.json();
        const curve = payload.curve || {};
        const folds = payload.folds || {};

        if (curve.available) {
            setBacktestStat('backtest-strategy-return', curve.strategy_total_return_pct);
            setBacktestStat('backtest-buy-hold-return', curve.buy_hold_total_return_pct);
            setBacktestStat('backtest-excess-return', curve.excess_return_pct);
            setBacktestStat('backtest-max-drawdown', curve.max_drawdown_pct, false);
            drawEquityCurve(chartHost, curve.points || []);
            const start = (curve.window?.start || '').slice(0, 10);
            const end = (curve.window?.end || '').slice(0, 10);
            const badge = document.getElementById('backtest-window-badge');
            if (badge) badge.textContent = start && end ? `${start} to ${end}` : 'Window';
            if (caption) caption.textContent = `${curve.point_count} bars on the ${timeframe} timeframe.`;
        } else {
            setBacktestStat('backtest-strategy-return', null);
            setBacktestStat('backtest-buy-hold-return', null);
            setBacktestStat('backtest-excess-return', null);
            setBacktestStat('backtest-max-drawdown', null, false);
            if (chartHost) chartHost.innerHTML = '';
            if (caption) caption.textContent = 'No backtest curve saved for this asset and timeframe.';
            const badge = document.getElementById('backtest-window-badge');
            if (badge) badge.textContent = 'No data';
        }

        const foldCountEl = document.getElementById('backtest-fold-count');
        if (foldCountEl) foldCountEl.textContent = folds.fold_count || 0;
        renderFoldRows(folds.folds || []);
    } catch (err) {
        if (caption) caption.textContent = 'Could not load backtest data.';
    }
}

/* ----- Strategy builder ----- */

const strategyRegistryScript = document.getElementById('strategy-registry-data');
const strategyRegistry = strategyRegistryScript ? JSON.parse(strategyRegistryScript.textContent) : {};
let builderResolvedConfig = null;
let builderResolvedAsset = null;

const SINGLE_SELECT_SECTIONS = new Set(['core_signal', 'timeframe_scope', 'decision_rules', 'risk_profile']);
const MULTI_SELECT_SECTIONS = new Set(['data_sources', 'confirmation_filters']);

function getBuilderDefaults() {
    const def = strategyRegistry?.default_custom_selection || {};
    return {
        core_signal: def.core_signal || 'trend_following',
        timeframe_scope: (def.timeframes && def.timeframes[0]) || '4h',
        data_sources: def.data_sources || ['market'],
        confirmation_filters: def.confirmation_filters || [],
        decision_rules: def.decision_rules || 'double_confirmation',
        risk_profile: def.risk_profile || 'balanced_risk',
    };
}

function renderBuilderForm() {
    const host = document.getElementById('builder-questions');
    if (!host) return;
    const sections = strategyRegistry?.custom_builder?.sections || [];
    if (sections.length === 0) {
        host.innerHTML = '<p class="note">Builder schema is not available.</p>';
        return;
    }
    const defaults = getBuilderDefaults();
    const blocks = sections.map((section) => {
        const isMulti = MULTI_SELECT_SECTIONS.has(section.id);
        const inputType = isMulti ? 'checkbox' : 'radio';
        const currentValue = defaults[section.id];
        const options = section.options.map((opt) => {
            let checked = false;
            if (isMulti) checked = Array.isArray(currentValue) && currentValue.includes(opt.id);
            else checked = currentValue === opt.id;
            return `<label class="builder-option ${checked ? 'is-selected' : ''}" data-section="${section.id}" data-value="${opt.id}">
                <input type="${inputType}" name="builder-${section.id}" value="${opt.id}" ${checked ? 'checked' : ''}>
                <span>
                    <span class="builder-option-label">${opt.label}</span>
                    <span class="builder-option-desc">${opt.description || ''}</span>
                </span>
            </label>`;
        }).join('');
        return `<div class="builder-question" data-section="${section.id}">
            <h4>${section.label}</h4>
            <p class="builder-question-desc">${section.description || ''}</p>
            <div class="builder-options">${options}</div>
        </div>`;
    });
    host.innerHTML = blocks.join('');

    host.querySelectorAll('.builder-option input').forEach((input) => {
        input.addEventListener('change', () => {
            const wrap = input.closest('.builder-question');
            if (!wrap) return;
            wrap.querySelectorAll('.builder-option').forEach((opt) => {
                const opInput = opt.querySelector('input');
                opt.classList.toggle('is-selected', !!opInput && opInput.checked);
            });
        });
    });
}

function readBuilderSelection() {
    const selection = {
        data_sources: [],
        confirmation_filters: [],
    };
    document.querySelectorAll('.builder-option input').forEach((input) => {
        if (!input.checked) return;
        const wrap = input.closest('.builder-question');
        if (!wrap) return;
        const sectionId = wrap.dataset.section;
        if (MULTI_SELECT_SECTIONS.has(sectionId)) {
            if (!selection[sectionId]) selection[sectionId] = [];
            selection[sectionId].push(input.value);
        } else if (SINGLE_SELECT_SECTIONS.has(sectionId)) {
            selection[sectionId] = input.value;
        }
    });
    if (!selection.data_sources.includes('market')) selection.data_sources.unshift('market');
    return selection;
}

function pickTimeframeForApi(timeframeScope) {
    if (timeframeScope === '1h_4h_stack' || timeframeScope === '4h_1d_stack') return '4h';
    if (['1h', '4h', '1d'].includes(timeframeScope)) return timeframeScope;
    return '4h';
}

function joinLayers(layers) {
    if (!layers || layers.length === 0) return 'none';
    return layers.map((layer) => String(layer).replace(/_/g, ' ')).join(', ');
}

function renderBuilderPreview(config) {
    if (!config) return;
    const required = config.required_layers || [];
    const optional = config.optional_layers || [];
    const unavailable = config.unavailable_layers || [];
    const governance = config.governance || {};
    const timeframePolicy = config.timeframe_policy || {};

    document.getElementById('builder-summary').textContent = config.display_summary || 'Custom strategy';
    document.getElementById('builder-evaluation-note').textContent = config.evaluation_basis_note || 'Evaluation basis will appear here.';
    document.getElementById('builder-required-layers').textContent = joinLayers(required);
    document.getElementById('builder-optional-layers').textContent = joinLayers(optional);
    document.getElementById('builder-unavailable-layers').textContent = joinLayers(unavailable);
    document.getElementById('builder-sentiment-role').textContent = (config.sentiment_role || 'n/a').replace(/_/g, ' ');
    document.getElementById('builder-onchain-role').textContent = (config.onchain_role || 'n/a').replace(/_/g, ' ');
    document.getElementById('builder-timeframe').textContent = timeframePolicy.resolved_timeframe || 'n/a';
    document.getElementById('builder-readiness').textContent = governance.readiness_label || 'Resolved';

    document.getElementById('builder-save').disabled = false;
}

async function previewBuilder() {
    const asset = document.getElementById('builder-asset')?.value || 'BTCUSDT';
    const selection = readBuilderSelection();
    const timeframe = pickTimeframeForApi(selection.timeframe_scope);
    try {
        const response = await fetch('/api/strategy-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'custom', asset, timeframe, selection }),
        });
        if (!response.ok) throw new Error('preview failed');
        const config = await response.json();
        builderResolvedConfig = config;
        builderResolvedAsset = asset;
        renderBuilderPreview(config);
    } catch (err) {
        const status = document.getElementById('builder-save-status');
        if (status) status.textContent = 'Could not generate preview.';
    }
}

function resetBuilder() {
    renderBuilderForm();
    builderResolvedConfig = null;
    builderResolvedAsset = null;
    document.getElementById('builder-save').disabled = true;
    document.getElementById('builder-summary').textContent = 'Pick options and press Preview.';
    document.getElementById('builder-evaluation-note').textContent = 'Evaluation basis will appear here.';
    ['required', 'optional', 'unavailable', 'sentiment-role', 'onchain-role', 'timeframe'].forEach((suffix) => {
        const id = suffix.includes('-') ? `builder-${suffix}` : `builder-${suffix}-layers`;
        const el = document.getElementById(id);
        if (el) el.textContent = 'n/a';
    });
    document.getElementById('builder-readiness').textContent = 'Pending';
    document.getElementById('builder-save-status').textContent = '';
    document.getElementById('builder-name').value = '';
}

async function saveBuilderProfile() {
    if (!builderResolvedConfig) return;
    const name = document.getElementById('builder-name')?.value?.trim() || 'Custom strategy';
    const asset = builderResolvedAsset || 'BTCUSDT';
    const status = document.getElementById('builder-save-status');
    try {
        const response = await fetch('/api/strategy-profiles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, asset, tag: 'custom', config: builderResolvedConfig }),
        });
        if (!response.ok) throw new Error('save failed');
        const out = await response.json();
        if (status) status.textContent = `Saved as "${out.profile?.name || name}". Open Account to view it.`;
    } catch (err) {
        if (status) status.textContent = 'Could not save profile.';
    }
}

function applyBuilderSelection(selection) {
    if (!selection) return;
    document.querySelectorAll('.builder-question').forEach((wrap) => {
        const sectionId = wrap.dataset.section;
        const value = selection[sectionId];
        wrap.querySelectorAll('.builder-option').forEach((opt) => {
            const input = opt.querySelector('input');
            if (!input) return;
            let shouldCheck = false;
            if (MULTI_SELECT_SECTIONS.has(sectionId)) {
                shouldCheck = Array.isArray(value) && value.includes(input.value);
            } else if (SINGLE_SELECT_SECTIONS.has(sectionId)) {
                shouldCheck = value === input.value;
            }
            input.checked = shouldCheck;
            opt.classList.toggle('is-selected', shouldCheck);
        });
    });
}

function hydrateBuilderFromStorage() {
    const PROFILE_KEY = 'livestrat:loadProfile';
    let payload = null;
    try {
        const raw = localStorage.getItem(PROFILE_KEY);
        if (!raw) return;
        payload = JSON.parse(raw);
    } catch (err) {
        return;
    }
    localStorage.removeItem(PROFILE_KEY);
    if (!payload) return;

    const builderAsset = document.getElementById('builder-asset');
    if (builderAsset && payload.asset) {
        if ([...builderAsset.options].some((opt) => opt.value === payload.asset)) {
            builderAsset.value = payload.asset;
        }
    }
    applyBuilderSelection(payload.selection);

    const nameField = document.getElementById('builder-name');
    if (nameField && payload.name) nameField.value = payload.name;

    document.getElementById('builder-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    previewBuilder();
}

function deriveTimeframeScope(selection) {
    const timeframes = selection?.timeframes || [];
    if (selection?.timeframe_scope) return selection.timeframe_scope;
    if (timeframes.includes('1h') && timeframes.includes('4h')) return '1h_4h_stack';
    if (timeframes.includes('4h') && timeframes.includes('1d')) return '4h_1d_stack';
    return timeframes[0] || '4h';
}

function loadProfileIntoBuilder(profile) {
    if (!profile) return;
    const config = profile.config || {};
    const selection = config.selection || {};
    const builderSelection = {
        core_signal: selection.core_signal || 'trend_following',
        timeframe_scope: deriveTimeframeScope(selection),
        data_sources: selection.data_sources || ['market'],
        confirmation_filters: selection.confirmation_filters || [],
        decision_rules: selection.decision_rules || 'double_confirmation',
        risk_profile: selection.risk_profile || 'balanced_risk',
    };
    const builderAsset = document.getElementById('builder-asset');
    if (builderAsset && profile.asset) {
        if ([...builderAsset.options].some((opt) => opt.value === profile.asset)) {
            builderAsset.value = profile.asset;
        }
    }
    applyBuilderSelection(builderSelection);
    const nameField = document.getElementById('builder-name');
    if (nameField && profile.name) nameField.value = profile.name;
    document.getElementById('builder-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    previewBuilder();
}

async function deleteProfileFromBuilder(profileId) {
    if (!profileId) return;
    const confirmed = window.confirm('Delete this saved profile?');
    if (!confirmed) return;
    try {
        const response = await fetch(`/api/strategy-profiles/${profileId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('delete failed');
        await refreshBuilderProfiles();
    } catch (err) {
        const status = document.getElementById('builder-save-status');
        if (status) status.textContent = 'Could not delete profile.';
    }
}

function renderProfileChips(profiles) {
    const host = document.getElementById('builder-profiles-list');
    if (!host) return;
    if (!profiles || profiles.length === 0) {
        host.innerHTML = '<p class="note">No saved profiles yet. Use Save below to keep one for later.</p>';
        return;
    }
    host.innerHTML = '';
    profiles.forEach((profile) => {
        const chip = document.createElement('div');
        chip.className = 'builder-profile-chip';
        chip.dataset.profileId = profile.id;
        const tfPolicy = profile.config?.timeframe_policy || {};
        const tf = tfPolicy.resolved_timeframe || (profile.selected_timeframes || []).join('+') || '4h';
        chip.innerHTML = `
            <span class="chip-name">${profile.name}</span>
            <span class="chip-meta">${(profile.asset || '').replace('USDT', '')} / ${tf}</span>
            <button type="button" class="chip-load">Load</button>
            <button type="button" class="chip-delete" aria-label="Delete">&times;</button>
        `;
        chip.querySelector('.chip-load').addEventListener('click', () => loadProfileIntoBuilder(profile));
        chip.querySelector('.chip-delete').addEventListener('click', () => deleteProfileFromBuilder(profile.id));
        host.appendChild(chip);
    });
}

async function refreshBuilderProfiles() {
    const host = document.getElementById('builder-profiles-list');
    if (!host) return;
    try {
        const response = await fetch('/api/strategy-profiles');
        if (!response.ok) throw new Error('list failed');
        const payload = await response.json();
        renderProfileChips(payload.profiles || []);
    } catch (err) {
        host.innerHTML = '<p class="note">Could not load saved profiles.</p>';
    }
}

renderBuilderForm();
hydrateBuilderFromStorage();
refreshBuilderProfiles();
document.getElementById('builder-preview')?.addEventListener('click', previewBuilder);
document.getElementById('builder-reset')?.addEventListener('click', resetBuilder);
document.getElementById('builder-save')?.addEventListener('click', async () => {
    await saveBuilderProfile();
    refreshBuilderProfiles();
});
document.getElementById('builder-profiles-refresh')?.addEventListener('click', refreshBuilderProfiles);

/* ----- event handlers ----- */

[strategySelect, strategyAsset, strategyTimeframe].forEach((el) => {
    if (el) el.addEventListener('change', () => renderStrategyView());
});

document.querySelectorAll('[data-strategy-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
        const id = btn.dataset.strategyAction;
        if (strategySelect) {
            strategySelect.value = id;
        }
        renderStrategyView();
        // smooth scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});

renderStrategyView();
