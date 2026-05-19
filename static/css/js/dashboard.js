/* ============================================================
   LiveStrat - Dashboard
   Renders: asset strip sparklines, today's market read,
   featured asset card, data freshness section.
   ============================================================ */

const marketSummaryScript = document.getElementById('market-summaries-data');
const marketSummaries = marketSummaryScript ? JSON.parse(marketSummaryScript.textContent) : {};

const ASSET_NAMES = {
    BTCUSDT: 'Bitcoin',
    ETHUSDT: 'Ethereum',
    SOLUSDT: 'Solana',
    BNBUSDT: 'BNB',
    XRPUSDT: 'XRP',
    ADAUSDT: 'Cardano',
    DOGEUSDT: 'Dogecoin',
};

const SIGNAL_TO_VERDICT = {
    buy: 'BUY', long: 'BUY',
    hold: 'HOLD', flat: 'HOLD',
    dont_buy: 'AVOID', do_not_buy: 'AVOID', sell: 'AVOID', short: 'AVOID',
};

/* ----- utilities ----- */

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
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

function buildSparkPathFromPoints(points, width, height) {
    if (!points || points.length < 2) return null;
    const min = Math.min(...points);
    const max = Math.max(...points);
    const range = Math.max(max - min, 0.0001);
    const path = points.map((p, i) => {
        const x = (i / (points.length - 1)) * width;
        const y = height - 4 - ((p - min) / range) * (height - 8);
        return `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(' ');
    const totalReturn = (points[points.length - 1] - points[0]) / points[0];
    return { path, totalReturn, width, height };
}

async function fetchSparkPoints(symbol, timeframe = '4h', count = 60) {
    try {
        const response = await fetch(`/api/market-chart?asset=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}&points=${count}`);
        if (!response.ok) return [];
        const payload = await response.json();
        return (payload.points || [])
            .map((p) => Number(p.close))
            .filter((v) => Number.isFinite(v) && v > 0);
    } catch (err) {
        return [];
    }
}

function renderSparkSvg(container, sparkData, options = {}) {
    if (!container) return;
    container.innerHTML = '';
    if (!sparkData) {
        const empty = document.createElement('p');
        empty.style.cssText = 'margin: auto; padding: 12px; color: var(--text-muted); font-size: 0.85rem; font-style: italic; text-align: center;';
        empty.textContent = options.emptyText || 'No recent data';
        container.style.alignItems = 'center';
        container.style.justifyContent = 'center';
        container.appendChild(empty);
        return;
    }

    const { path, totalReturn, width, height } = sparkData;
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'none');

    const fillPath = `${path} L ${width} ${height} L 0 ${height} Z`;
    const fill = document.createElementNS(svgNS, 'path');
    fill.setAttribute('d', fillPath);
    fill.setAttribute('fill', 'rgba(0, 139, 90, 0.12)');
    svg.appendChild(fill);

    const stroke = document.createElementNS(svgNS, 'path');
    stroke.setAttribute('d', path);
    stroke.setAttribute('fill', 'none');
    stroke.setAttribute('stroke', totalReturn >= 0 ? '#0d7c4e' : '#b84b4b');
    stroke.setAttribute('stroke-width', '1.6');
    stroke.setAttribute('stroke-linecap', 'round');
    stroke.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(stroke);

    container.appendChild(svg);
}

/* ----- asset strip sparklines ----- */

async function renderAssetStripSparks() {
    const entries = Object.entries(marketSummaries);
    await Promise.all(entries.map(async ([symbol]) => {
        const container = document.getElementById(`asset-chart-${symbol}`);
        if (!container) return;
        const points = await fetchSparkPoints(symbol, '4h', 60);
        const spark = buildSparkPathFromPoints(points, 200, 112);
        renderSparkSvg(container, spark, { emptyText: 'No recent price data' });
    }));
}

/* ----- today's market read ----- */

function getSummarySignal(summary) {
    const raw = String(
        summary.selected_primary_signal || summary.latest_signal || ''
    ).toLowerCase();
    return SIGNAL_TO_VERDICT[raw] || null;
}

function renderTodayMarketRead() {
    const symbols = Object.keys(marketSummaries);
    let buyCount = 0;
    let holdCount = 0;
    let avoidCount = 0;
    let coveredCount = 0;

    let leaderSymbol = null;
    let leaderChange = -Infinity;

    let confSymbol = null;
    let confValue = -1;

    symbols.forEach((symbol) => {
        const s = marketSummaries[symbol] || {};
        const close = Number(s.latest_close || 0);
        if (close <= 0) return;
        coveredCount += 1;

        const sig = getSummarySignal(s);
        if (sig === 'BUY') buyCount += 1;
        else if (sig === 'AVOID') avoidCount += 1;
        else if (sig === 'HOLD') holdCount += 1;

        const change = Number(s.latest_return_24h_pct || 0);
        if (Number.isFinite(change) && change > leaderChange) {
            leaderChange = change;
            leaderSymbol = symbol;
        }

        const conf = Number(s.selected_primary_confidence ?? s.latest_signal_confidence ?? 0);
        if (Number.isFinite(conf) && conf > confValue && sig) {
            confValue = conf;
            confSymbol = symbol;
        }
    });

    setText('today-buy-count', String(buyCount));
    setText('today-hold-count', String(holdCount));
    setText('today-avoid-count', String(avoidCount));

    const tallyNote = document.getElementById('today-signal-note');
    if (tallyNote) {
        if (coveredCount === 0) {
            tallyNote.textContent = 'No signals available yet. Refresh the pipeline to populate.';
        } else {
            const missing = symbols.length - coveredCount;
            const missingNote = missing > 0 ? ` ${missing} asset${missing === 1 ? '' : 's'} pending refresh.` : '';
            tallyNote.textContent = `Across ${coveredCount} of ${symbols.length} tracked assets.${missingNote}`;
        }
    }

    // 24h leader
    if (leaderSymbol && Number.isFinite(leaderChange) && leaderChange > -Infinity) {
        setText('today-leader-asset', leaderSymbol.replace('USDT', ' / USDT'));
        const leaderEl = document.getElementById('today-leader-change');
        if (leaderEl) {
            leaderEl.textContent = `${leaderChange >= 0 ? '+' : ''}${leaderChange.toFixed(2)}%`;
            leaderEl.classList.remove('is-positive', 'is-negative');
            leaderEl.classList.add(leaderChange >= 0 ? 'is-positive' : 'is-negative');
        }
        setText('today-leader-note', `${ASSET_NAMES[leaderSymbol] || leaderSymbol} led the basket over the last 24 hours.`);
    } else {
        setText('today-leader-asset', 'Pending');
        setText('today-leader-change', 'n/a');
        setText('today-leader-note', 'No 24h price data available yet.');
    }

    // Strongest confidence
    if (confSymbol && confValue >= 0) {
        const sig = getSummarySignal(marketSummaries[confSymbol]) || 'Signal';
        setText('today-conf-asset', confSymbol.replace('USDT', ' / USDT'));
        setText('today-conf-value', `${Math.round(clamp(confValue, 0, 1) * 100)}% · ${sig}`);
        setText('today-conf-note', `${ASSET_NAMES[confSymbol] || confSymbol} carries the most conviction right now.`);
    } else {
        setText('today-conf-asset', 'Pending');
        setText('today-conf-value', 'n/a');
        setText('today-conf-note', 'Strategy confidence data not loaded yet.');
    }
}

/* ----- featured asset card ----- */

function getLeadAsset() {
    // Prefer the symbol with highest confidence and a real signal
    let best = null;
    let bestConf = -1;
    Object.entries(marketSummaries).forEach(([symbol, summary]) => {
        const close = Number(summary.latest_close || 0);
        if (close <= 0) return;
        const sig = getSummarySignal(summary);
        if (!sig) return;
        const conf = Number(summary.selected_primary_confidence ?? summary.latest_signal_confidence ?? 0);
        if (conf > bestConf) {
            bestConf = conf;
            best = symbol;
        }
    });
    return best || Object.keys(marketSummaries).find((sym) => Number(marketSummaries[sym]?.latest_close || 0) > 0) || 'BTCUSDT';
}

function renderFeaturedAsset(asset) {
    const summary = marketSummaries[asset] || {};
    const close = Number(summary.latest_close || 0);

    setText('featured-symbol', asset.replace('USDT', ' / USDT'));
    setText('featured-name', ASSET_NAMES[asset] || asset);
    setText('featured-price', formatPriceUSD(close));

    const change = Number(summary.latest_return_24h_pct || 0);
    const changeEl = document.getElementById('featured-change');
    if (changeEl) {
        changeEl.classList.remove('is-positive', 'is-negative');
        if (close <= 0) {
            changeEl.textContent = 'No recent data';
        } else {
            const sign = change >= 0 ? '+' : '';
            changeEl.textContent = `${sign}${change.toFixed(2)}% over 24h`;
            changeEl.classList.add(change >= 0 ? 'is-positive' : 'is-negative');
        }
    }

    const chart = document.getElementById('featured-chart');
    fetchSparkPoints(asset, '4h', 60).then((points) => {
        const spark = buildSparkPathFromPoints(points, 320, 160);
        renderSparkSvg(chart, spark, { emptyText: 'No recent price data.' });
    });

    // Read paragraph
    const readEl = document.getElementById('featured-read');
    if (readEl) {
        if (close <= 0) {
            readEl.textContent = 'No recent price data is available for this asset. Run the pipeline refresh to populate the chart.';
        } else {
            const r24 = Number(summary.latest_return_24h_pct || 0);
            const volStatus = String(summary.volatility_status || 'normal').toLowerCase();
            const direction = r24 > 0.5 ? 'up' : (r24 < -0.5 ? 'down' : 'roughly flat');
            const directionMagnitude = `${r24 >= 0 ? '+' : ''}${r24.toFixed(2)}%`;
            readEl.textContent = `${ASSET_NAMES[asset] || asset} is ${direction} ${directionMagnitude} over the last 24 hours. Volatility is ${volStatus}.`;
        }
    }

    // Verdict
    const sig = getSummarySignal(summary);
    const verdict = document.getElementById('featured-verdict');
    const verdictText = document.getElementById('featured-verdict-text');
    if (verdict && verdictText) {
        verdict.classList.remove('verdict-badge-buy', 'verdict-badge-hold', 'verdict-badge-avoid', 'verdict-badge-unavailable');
        if (!sig || close <= 0) {
            verdict.classList.add('verdict-badge-unavailable');
            verdictText.textContent = 'NO DATA';
        } else {
            verdict.classList.add(
                sig === 'BUY' ? 'verdict-badge-buy' :
                sig === 'AVOID' ? 'verdict-badge-avoid' :
                'verdict-badge-hold'
            );
            verdictText.textContent = sig;
        }
    }

    const conf = Number(summary.selected_primary_confidence ?? summary.latest_signal_confidence ?? 0);
    const confPct = Math.round(clamp(conf, 0, 1) * 100);
    setText('featured-conf-value', sig && close > 0 ? `${confPct}%` : 'n/a');
    const confFill = document.getElementById('featured-conf-fill');
    if (confFill) confFill.style.width = `${sig && close > 0 ? confPct : 0}%`;

    // Walk-forward audit strip
    const folds = Number(summary.walkforward_fold_count ?? 0);
    const wfSharpe = Number(summary.walkforward_avg_sharpe ?? 0);
    setText('featured-wf-folds', folds > 0 ? folds.toFixed(0) : 'n/a');
    setText('featured-wf-sharpe', folds > 0 ? wfSharpe.toFixed(2) : 'n/a');
}

function renderAssetToggle(activeAsset) {
    const container = document.getElementById('featured-asset-toggle');
    if (!container) return;
    container.innerHTML = '';
    Object.keys(marketSummaries).forEach((symbol) => {
        const close = Number(marketSummaries[symbol]?.latest_close || 0);
        if (close <= 0) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = symbol.replace('USDT', '');
        if (symbol === activeAsset) btn.classList.add('is-active');
        btn.addEventListener('click', () => {
            container.querySelectorAll('button').forEach((b) => b.classList.remove('is-active'));
            btn.classList.add('is-active');
            renderFeaturedAsset(symbol);
        });
        container.appendChild(btn);
    });
}

/* ----- data freshness section ----- */

async function loadPipelineRefresh() {
    try {
        const response = await fetch('/api/pipeline-refresh');
        if (!response.ok) throw new Error(`Status ${response.status}`);
        const payload = await response.json();
        renderPipelineRefresh(payload);
    } catch (err) {
        const badge = document.getElementById('dashboard-data-status-badge');
        if (badge) badge.textContent = 'Unavailable';
        setText('dashboard-generated-window', 'Unavailable');
        setText('dashboard-generated-note', 'Refresh manifest could not be loaded.');
        setText('today-freshness-value', 'Unknown');
        setText('today-freshness-window', 'n/a');
        const lanes = document.getElementById('dashboard-refresh-lanes');
        if (lanes) lanes.innerHTML = '<p class="note">Run the pipeline refresh to populate timeframe coverage.</p>';
    }
}

function renderPipelineRefresh(payload) {
    const badge = document.getElementById('dashboard-data-status-badge');
    const lanes = payload?.timeframes || {};
    const overall = payload?.overall_state || 'unknown';

    // Headline freshness label
    let freshnessLabel = 'Unknown';
    let freshnessWindow = 'n/a';
    let freshnessNote = 'How recent the saved evaluation runs are.';
    if (overall === 'current') freshnessLabel = 'Current';
    else if (overall === 'recent') freshnessLabel = 'Recent';
    else if (overall === 'stale') freshnessLabel = 'Stale';
    else if (overall === 'mixed') freshnessLabel = 'Mixed';
    else if (overall === 'failed' || overall === 'missing') freshnessLabel = 'Refresh needed';

    if (badge) badge.textContent = freshnessLabel;

    // Use the 4h lane window if available for the summary
    const fourH = lanes['4h'] || lanes['1h'] || lanes['1d'] || {};
    if (fourH.window_end) {
        freshnessWindow = `through ${fourH.window_end}`;
        const days = daysAgo(fourH.window_end);
        if (days !== null) {
            if (days === 0) freshnessNote = 'Refreshed today.';
            else if (days <= 2) freshnessNote = `Refreshed ${days} day${days === 1 ? '' : 's'} ago. current.`;
            else if (days <= 7) freshnessNote = `Refreshed ${days} days ago. use with caution.`;
            else freshnessNote = `Refreshed ${days} days ago. run pipeline refresh.`;
        }
    }

    setText('today-freshness-value', freshnessLabel);
    setText('today-freshness-window', freshnessWindow);
    setText('today-freshness-note', freshnessNote);

    setText('dashboard-generated-window', fourH.timeframe_label || (fourH.window_end ? `4 hours through ${fourH.window_end}` : 'Unavailable'));
    setText('dashboard-generated-note', fourH.state_summary || freshnessNote);

    const lanesEl = document.getElementById('dashboard-refresh-lanes');
    if (lanesEl) {
        lanesEl.innerHTML = '';
        ['1h', '4h', '1d'].forEach((tf) => {
            const lane = lanes[tf];
            if (!lane) return;
            const stateClass = (lane.state || 'unknown').toLowerCase();
            const wrap = document.createElement('div');
            wrap.className = 'refresh-lane';
            wrap.innerHTML = `
                <div class="refresh-lane-top">
                    <strong>${tf === '1h' ? '1 hour' : tf === '4h' ? '4 hours' : '1 day'}</strong>
                    <span class="refresh-lane-state refresh-lane-${stateClass}">${formatStateLabel(lane.state)}</span>
                </div>
                <p class="note">${escapeHtml(lane.state_summary || `${tf} pipeline state: ${stateClass}.`)}</p>
            `;
            lanesEl.appendChild(wrap);
        });
        if (!lanesEl.children.length) {
            lanesEl.innerHTML = '<p class="note">No timeframe coverage data available.</p>';
        }
    }
}

function formatStateLabel(state) {
    const raw = String(state || '').toLowerCase();
    const map = {
        current: 'Current',
        recent: 'Recent',
        stale: 'Needs refresh',
        failed: 'Refresh needed',
        missing: 'Not yet generated',
    };
    return map[raw] || (raw.charAt(0).toUpperCase() + raw.slice(1));
}

function escapeHtml(value) {
    return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function daysAgo(isoString) {
    if (!isoString) return null;
    const then = new Date(isoString);
    if (Number.isNaN(then.getTime())) return null;
    const now = new Date();
    return Math.max(0, Math.floor((now - then) / (1000 * 60 * 60 * 24)));
}

/* ----- live price check ----- */

const livePriceBtn = document.getElementById('dashboard-live-check');
if (livePriceBtn) {
    livePriceBtn.addEventListener('click', async () => {
        const liveEl = document.getElementById('dashboard-live-price');
        const noteEl = document.getElementById('dashboard-live-note');
        const leadAssetActive = document.querySelector('#featured-asset-toggle button.is-active');
        const targetAsset = leadAssetActive?.textContent ? `${leadAssetActive.textContent}USDT` : getLeadAsset();
        if (liveEl) liveEl.textContent = 'Checking…';
        if (noteEl) noteEl.textContent = `Fetching latest ${targetAsset.replace('USDT', ' / USDT')} price from Binance...`;
        try {
            const response = await fetch(`/api/live-market-check?asset=${encodeURIComponent(targetAsset)}`);
            if (!response.ok) throw new Error(`Status ${response.status}`);
            const data = await response.json();
            const price = Number(data.latest_price || 0);
            const change = Number(data.price_change_pct_24h || 0);
            if (liveEl) liveEl.textContent = formatPriceUSD(price);
            if (noteEl) {
                const sign = change >= 0 ? '+' : '';
                noteEl.textContent = `${targetAsset.replace('USDT', ' / USDT')} · ${sign}${change.toFixed(2)}% over 24h. Live ticker from Binance.`;
            }
        } catch (err) {
            if (liveEl) liveEl.textContent = 'Error';
            if (noteEl) noteEl.textContent = 'Could not fetch live price right now. Try again in a moment.';
        }
    });
}

/* ----- init ----- */

renderAssetStripSparks();
renderTodayMarketRead();
const leadAsset = getLeadAsset();
renderFeaturedAsset(leadAsset);
renderAssetToggle(leadAsset);
loadPipelineRefresh();
