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

function renderHero(asset, summary, signal, confidence, refreshDays) {
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
            text = 'Data refreshed today. Signal is current.';
            dot.classList.add('freshness-dot-green');
        } else if (refreshDays <= 2) {
            text = `Data refreshed ${refreshDays} day${refreshDays === 1 ? '' : 's'} ago, signal is current.`;
            dot.classList.add('freshness-dot-green');
        } else if (refreshDays <= 7) {
            text = `Data refreshed ${refreshDays} days ago. Treat the signal with caution.`;
            dot.classList.add('freshness-dot-amber');
        } else {
            text = `Data refreshed ${refreshDays} days ago. Signal is stale; refresh the pipeline.`;
            dot.classList.add('freshness-dot-red');
        }
        const strategyLabel = STRATEGY_LABELS[strategySelect?.value] || 'Strategy';
        const timeframeLabel = strategyTimeframe?.value === '1h' ? '1 hour view' : '4 hour view';
        freshnessEl.innerHTML = '';
        freshnessEl.appendChild(dot);
        freshnessEl.append(` ${text} · ${strategyLabel} · ${timeframeLabel}`);
    }

    // Strategy note in verdict card
    const strategyLabel = STRATEGY_LABELS[strategySelect?.value] || 'Strategy';
    setText('hero-strategy-note', `Signal produced by the ${strategyLabel} strategy.`);

    // Sparkline
    renderSparkline(summary);
}

function renderSparkline(summary) {
    const container = document.getElementById('hero-spark');
    if (!container) return;
    container.innerHTML = '';

    // We don't have an inline series, so render a synthetic line based on the
    // latest_close, 4h, 24h, and 3d returns to show recent price direction.
    const close = Number(summary.latest_close || 0);
    if (close <= 0) {
        const empty = document.createElement('p');
        empty.style.cssText = 'margin: auto; color: var(--text-muted); font-size: 0.9rem; font-style: italic;';
        empty.textContent = 'No recent price data. Run a pipeline refresh.';
        container.style.alignItems = 'center';
        container.style.justifyContent = 'center';
        container.appendChild(empty);
        return;
    }

    const r4 = Number(summary.latest_return_4h_pct || 0) / 100;
    const r24 = Number(summary.latest_return_24h_pct || 0) / 100;
    const r3d = Number(summary.latest_return_3d_pct || 0) / 100;

    // Synthesize a smooth path back from the current price
    const pts = [];
    const steps = 60;
    const totalReturn = r3d || r24 || r4;
    for (let i = 0; i <= steps; i += 1) {
        const t = i / steps;
        // ease price from (1 - totalReturn) at t=0 to 1.0 at t=1
        const base = (1 - totalReturn) + totalReturn * t;
        // gentle wobble
        const wobble = Math.sin(t * Math.PI * 3) * Math.abs(totalReturn) * 0.18;
        pts.push(base + wobble);
    }
    const min = Math.min(...pts);
    const max = Math.max(...pts);
    const range = Math.max(max - min, 0.0001);

    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', '0 0 100 30');
    svg.setAttribute('preserveAspectRatio', 'none');

    const pathPoints = pts.map((p, i) => {
        const x = (i / steps) * 100;
        const y = 28 - ((p - min) / range) * 26;
        return `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(' ');

    const fillPath = `${pathPoints} L 100 30 L 0 30 Z`;

    const fill = document.createElementNS(svgNS, 'path');
    fill.setAttribute('d', fillPath);
    fill.setAttribute('fill', 'rgba(0, 139, 90, 0.12)');
    svg.appendChild(fill);

    const stroke = document.createElementNS(svgNS, 'path');
    stroke.setAttribute('d', pathPoints);
    stroke.setAttribute('fill', 'none');
    stroke.setAttribute('stroke', totalReturn >= 0 ? '#0d7c4e' : '#b84b4b');
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
            excessNote.textContent = 'Strategy outperformed simply holding the asset.';
        } else if (excess < -0.005) {
            excessNote.textContent = 'Strategy currently underperforms simply holding the asset.';
        } else {
            excessNote.textContent = 'Strategy roughly matches a passive hold.';
        }
    }

    // Honest read paragraph
    const honesty = document.getElementById('track-honesty');
    if (honesty) {
        honesty.classList.remove('is-positive', 'is-warn', 'is-negative');
        const assetName = ASSET_NAMES[asset] || asset;
        const evidenceBasis = useWalkForward
            ? `Averaged across ${foldCount} walk-forward fold${foldCount === 1 ? '' : 's'} (retrained each fold)`
            : 'On the latest single-window evaluation pass';
        let message;
        let toneClass;
        if (accuracy <= 0) {
            message = `Track record data for ${assetName} on this strategy and timeframe has not been generated yet. Refresh the pipeline to populate the evaluation results.`;
            toneClass = 'is-warn';
        } else if (excess > 0.02) {
            message = `${evidenceBasis}, this strategy beat a simple buy-and-hold of ${assetName} by ${formatPercent(excess, true)}, with ${formatPercent(accuracy)} directional accuracy on unseen data.`;
            toneClass = 'is-positive';
        } else if (excess > 0) {
            message = `${evidenceBasis}, this strategy outperformed a buy-and-hold of ${assetName} by a small margin (${formatPercent(excess, true)}). Directional accuracy was ${formatPercent(accuracy)}.`;
            toneClass = 'is-positive';
        } else if (excess > -0.02) {
            message = `${evidenceBasis}, the strategy performed roughly in line with a buy-and-hold of ${assetName} (${formatPercent(excess, true)} difference). Directional accuracy ${formatPercent(accuracy)}.`;
            toneClass = 'is-warn';
        } else {
            message = `${evidenceBasis}, holding ${assetName} returned more than running the strategy. The model called direction correctly ${formatPercent(accuracy)} of the time but did not turn that into a return advantage of ${formatPercent(Math.abs(excess))}.`;
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

    renderHero(asset, summary, signal, confidence, refreshDays);
    readPriceTrend(summary);
    readActivity(summary);
    readFutures(summary);
    readNews(summary);
    readOnchain(summary);
    renderTrackRecord(summary, asset);
    renderTechnicalDetails(summary);
}

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
