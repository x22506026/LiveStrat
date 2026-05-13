/* uses the generated market summary data passed from Flask */

const marketSummaryScript = document.getElementById('market-summaries-data');
const marketSummaries = marketSummaryScript ? JSON.parse(marketSummaryScript.textContent) : {};
const sentimentSummaryScript = document.getElementById('sentiment-summary-data');
const sentimentSummary = sentimentSummaryScript ? JSON.parse(sentimentSummaryScript.textContent) : {};

/* gets references to the notification controls */
const notifAsset = document.getElementById('notif-asset');
const notifStrategySelect = document.getElementById('notif-strategy-select');
const notifTimeframe = document.getElementById('notif-timeframe');
const notifPrice = document.getElementById('notif-price');
const notifStrategy = document.getElementById('notif-strategy');
const notifSentiment = document.getElementById('notif-sentiment');
const notifOnchain = document.getElementById('notif-onchain');
const notifDefi = document.getElementById('notif-defi');
const notifVolatility = document.getElementById('notif-volatility');
const notifTest = document.getElementById('notif-test');
const notifSavePreferences = document.getElementById('notif-save-preferences');
const notifSaveEvents = document.getElementById('notif-save-events');
const notifSaveStatus = document.getElementById('notif-save-status');
const notifTelegramStatus = document.getElementById('notif-telegram-status');
const notifTelegramEnabled = document.getElementById('notif-telegram-enabled');
const notifTelegramChatId = document.getElementById('notif-telegram-chat-id');
const notifPreview = document.getElementById('notif-preview');
const notifLatestClose = document.getElementById('notif-latest-close');
const notifLatestReturn = document.getElementById('notif-latest-return');
const notifModelConfidence = document.getElementById('notif-model-confidence');
const notifSentimentSummary = document.getElementById('notif-sentiment-summary');
const notifLatestAction = document.getElementById('notif-latest-action');
const notifBacktestSharpe = document.getElementById('notif-backtest-sharpe');
const notifSignalExplainer = document.getElementById('notif-signal-explainer');
const notifSignalMode = document.getElementById('notif-signal-mode');
const notifEffectiveSentiment = document.getElementById('notif-effective-sentiment');
const notifGdeltStatus = document.getElementById('notif-gdelt-status');
const notifOnchainStatus = document.getElementById('notif-onchain-status');
const notifDefiStatus = document.getElementById('notif-defi-status');
const notifCapabilityNote = document.getElementById('notif-capability-note');
const notifSourcePolicyNote = document.getElementById('notif-source-policy-note');
const notifMultimodalSelection = document.getElementById('notif-multimodal-selection');
const notifAlertList = document.getElementById('notif-alert-list');
const notifAlertCount = document.getElementById('notif-alert-count');
const notifEventList = document.getElementById('notif-event-list');
const notifEventCount = document.getElementById('notif-event-count');
const notifPreviewPanel = document.getElementById('notif-preview-panel');
const notifPreviewHeading = document.getElementById('notif-preview-heading');
const notifPreviewMeta = document.getElementById('notif-preview-meta');
const notifReturnCard = document.getElementById('notif-return-card');
const notifConfidenceCard = document.getElementById('notif-confidence-card');
const notifSignalModeBox = document.getElementById('notif-signal-mode-box');
const notifEffectiveSentimentBox = document.getElementById('notif-effective-sentiment-box');
const notifGdeltStatusBox = document.getElementById('notif-gdelt-status-box');
const notifOnchainStatusBox = document.getElementById('notif-onchain-status-box');
const notifDefiStatusBox = document.getElementById('notif-defi-status-box');

function formatSignal(value) {
    return (value || 'n/a').replaceAll('_', ' ');
}

function formatTitle(value) {
    return (value || 'n/a')
        .replaceAll('_', ' ')
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function simplifyCopy(value) {
    return String(value || '')
        .replaceAll('fallback', 'backup')
        .replaceAll('Fallback', 'Backup')
        .replaceAll('regime', 'state')
        .replaceAll('Regime', 'State')
        .replaceAll('veto', 'warning')
        .replaceAll('governance', 'rules')
        .replaceAll('structural', 'longer-term')
        .replaceAll('Structural', 'Longer-term');
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

function escapeHtml(value) {
    return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function setToneClass(element, tone) {
    if (!element) {
        return;
    }
    element.classList.remove('tone-positive', 'tone-mixed', 'tone-negative', 'tone-neutral');
    element.classList.add(`tone-${tone || 'neutral'}`);
}

function getDirectionalTone(value) {
    const numeric = Number(value || 0);
    if (numeric > 0.05) {
        return 'positive';
    }
    if (numeric < -0.05) {
        return 'negative';
    }
    return 'mixed';
}

function getSignalToneFromSummary(summary) {
    const signal = String(summary.selected_primary_signal || summary.scaled_model_signal || '').toLowerCase();
    const trend = String(summary.trend_status || '').toLowerCase();
    const sentiment = String(summary.latest_effective_sentiment_label || summary.latest_gdelt_regime_label || '').toLowerCase();

    if (signal.includes('dont_buy') || signal.includes('sell') || signal.includes('avoid')) {
        return 'negative';
    }
    if (signal.includes('buy') || trend.includes('bull') || sentiment.includes('positive') || sentiment.includes('support')) {
        return 'positive';
    }
    if (signal.includes('hold') || sentiment.includes('mixed') || trend.includes('neutral')) {
        return 'mixed';
    }
    return 'neutral';
}

function getAlertTone(alert) {
    const text = `${alert.title || ''} ${alert.message || ''} ${alert.action || ''}`.toLowerCase();

    if (
        text.includes('avoid') ||
        text.includes('dont buy') ||
        text.includes("don't buy") ||
        text.includes('bear') ||
        text.includes('down') ||
        text.includes('negative') ||
        text.includes('risk-off')
    ) {
        return 'negative';
    }

    if (
        text.includes('mixed') ||
        text.includes('medium') ||
        text.includes('watch') ||
        text.includes('caution') ||
        text.includes('review')
    ) {
        return 'mixed';
    }

    if (
        text.includes('bull') ||
        text.includes('up') ||
        text.includes('supportive') ||
        text.includes('positive') ||
        text.includes('buy')
    ) {
        return 'positive';
    }

    return 'neutral';
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

function buildPreviewItems(summary, symbol, capabilityState, confidenceValue) {
    const items = [];

    if (notifPrice?.checked) {
        items.push({
            label: 'Price',
            tone: getDirectionalTone(summary.latest_return_24h_pct || 0),
            text: `${symbol} moved ${Number(summary.latest_return_24h_pct || 0).toFixed(2)}% over 24 hours, with latest close at ${Number(summary.latest_close || 0).toFixed(2)}.`,
        });
    }

    if (notifStrategy?.checked) {
        items.push({
            label: 'Strategy',
            tone: getSignalToneFromSummary(summary),
            text: `Primary signal is ${formatDecisionLabel(summary.selected_primary_signal || summary.scaled_model_signal)} at ${confidenceValue}% confidence, while policy action remains ${formatSignal(summary.latest_action || 'not available')}.`,
        });
    }

    if (notifSentiment?.checked) {
        items.push({
            label: 'Sentiment',
            tone: summary.latest_effective_sentiment_label === 'mixed' ? 'mixed' : getAlertTone({
                title: summary.latest_effective_sentiment_label,
                message: summary.latest_effective_sentiment_source,
            }),
            text: summary.latest_effective_sentiment_source === 'gdelt_asset_news'
                ? `Asset news sentiment is ${formatSignal(summary.latest_effective_sentiment_label)} from GDELT coverage.`
                : `Effective sentiment is currently sourced from ${formatTitle(summary.latest_effective_sentiment_source)}.`,
        });
    }

    if (notifOnchain?.checked) {
        items.push({
            label: 'Network on-chain',
            tone: getAlertTone({
                title: summary.latest_onchain_regime_label,
                message: summary.latest_onchain_snapshot_label,
            }),
            text: summary.latest_onchain_regime_label && summary.latest_onchain_regime_label !== 'unavailable'
                ? `Network on-chain context is ${formatSignal(summary.latest_onchain_regime_label)} and should be used as confirmation rather than a standalone trigger.`
                : 'Network on-chain context is not currently available for this asset.',
        });
    }

    if (notifDefi?.checked) {
        items.push({
            label: 'Ecosystem TVL',
            tone: getAlertTone({
                title: summary.defi_regime_label,
                message: summary.defi_summary,
            }),
            text: summary.defi_context_available === true || String(summary.defi_context_available || '').toLowerCase() === 'true'
                ? `${summary.defi_chain_name || symbol.replace('USDT', '')} ecosystem TVL is ${formatMoney(summary.latest_defi_tvl_usd)} and the current DeFiLlama state is ${formatSignal(summary.defi_regime_label || 'available')}.`
                : 'Ecosystem TVL context is not currently available for this asset.',
        });
    }

    if (notifVolatility?.checked) {
        items.push({
            label: 'Risk',
            tone: String(summary.volatility_state || '').toLowerCase().includes('high') ? 'negative' : String(summary.volatility_state || '').toLowerCase().includes('medium') ? 'mixed' : 'positive',
            text: `Volatility is ${formatSignal(summary.volatility_state || 'unknown')}, so entry timing and sizing should stay aligned with current risk.`,
        });
    }

    if (!items.length) {
        items.push({
            label: 'No sections selected',
            tone: 'neutral',
            text: 'No message types are selected right now.',
        });
    }

    return items;
}

function getNotificationCapabilityState(summary) {
    const sentimentSource = String(summary.latest_effective_sentiment_source || 'unavailable');
    const gdeltLabel = String(summary.latest_gdelt_regime_label || 'unavailable');
    const onchainLabel = String(summary.latest_onchain_regime_label || 'unavailable');
    const defiAvailable = summary.defi_context_available === true
        || String(summary.defi_context_available || '').toLowerCase() === 'true';

    const sentimentAvailable = sentimentSource !== 'unavailable';
    const gdeltAvailable = gdeltLabel !== 'unavailable';
    const onchainAvailable = onchainLabel !== 'unavailable';

    if (sentimentSource === 'fear_greed_market_fallback') {
        return {
            modeLabel: 'Backup context',
            note: 'This asset is using market-wide sentiment because asset-specific news sentiment is unavailable.',
            sentimentAvailable,
            gdeltAvailable,
            onchainAvailable,
            defiAvailable,
        };
    }

    if (!sentimentAvailable && !onchainAvailable && !defiAvailable) {
        return {
            modeLabel: 'Reduced',
            note: 'This asset has no usable sentiment, network on-chain, or ecosystem TVL context, so signals are driven mainly by market and futures layers.',
            sentimentAvailable,
            gdeltAvailable,
            onchainAvailable,
            defiAvailable,
        };
    }

    return {
        modeLabel: 'Ready',
        note: 'This asset can use its currently available context layers.',
        sentimentAvailable,
        gdeltAvailable,
        onchainAvailable,
        defiAvailable,
    };
}

function renderNotificationPreview(symbol) {
    const summary = marketSummaries[symbol];
    if (!summary) {
        return;
    }
    const capabilityState = getNotificationCapabilityState(summary);
    const confidenceValue = (((summary.selected_primary_confidence ?? summary.scaled_model_confidence) || 0) * 100).toFixed(1);
    const signalTone = getSignalToneFromSummary(summary);
    const previewItems = buildPreviewItems(summary, symbol, capabilityState, confidenceValue);

    notifLatestClose.textContent = Number(summary.latest_close || 0).toFixed(2);
    notifLatestReturn.textContent = `${Number(summary.latest_return_24h_pct || 0).toFixed(2)}%`;
    notifModelConfidence.textContent = `${confidenceValue}%`;

    if (notifLatestAction) {
        notifLatestAction.textContent = formatSignal(summary.latest_action || 'not available');
    }

    if (notifBacktestSharpe) {
        notifBacktestSharpe.textContent = Number(summary.sharpe_ratio || 0).toFixed(2);
    }

    if (notifSignalExplainer && window.LiveStratSignalExplainer) {
        notifSignalExplainer.textContent = window.LiveStratSignalExplainer.buildSignalExplanation(summary, symbol);
    }

    if (notifSignalMode) {
        notifSignalMode.textContent = capabilityState.modeLabel;
    }
    setToneClass(notifSignalModeBox, capabilityState.modeLabel === 'Ready' ? 'positive' : capabilityState.modeLabel === 'Backup context' || capabilityState.modeLabel === 'Reduced' ? 'mixed' : 'neutral');

    if (notifEffectiveSentiment) {
        notifEffectiveSentiment.textContent = capabilityState.sentimentAvailable
            ? formatTitle(summary.latest_effective_sentiment_source || 'n/a')
            : 'Not Available';
    }
    setToneClass(notifEffectiveSentimentBox, capabilityState.sentimentAvailable ? getAlertTone({
        title: summary.latest_effective_sentiment_label,
        message: summary.latest_effective_sentiment_source,
    }) : 'neutral');

    if (notifGdeltStatus) {
        notifGdeltStatus.textContent = capabilityState.gdeltAvailable
            ? `${formatTitle(summary.latest_gdelt_regime_label)} (${Number(summary.latest_gdelt_article_count || 0).toFixed(0)})`
            : 'Not Available';
    }
    setToneClass(notifGdeltStatusBox, capabilityState.gdeltAvailable ? getAlertTone({
        title: summary.latest_gdelt_regime_label,
        message: summary.latest_gdelt_article_count,
    }) : 'neutral');

    if (notifOnchainStatus) {
        notifOnchainStatus.textContent = capabilityState.onchainAvailable
            ? simplifyCopy(formatTitle(summary.latest_onchain_regime_label || 'n/a'))
            : summary.latest_onchain_snapshot_status === 'stale'
                ? `Older ${simplifyCopy(formatTitle(summary.latest_onchain_snapshot_label || 'snapshot'))} (${Number(summary.latest_onchain_snapshot_age_days || 0).toFixed(0)}d)`
                : 'Not Available';
    }
    setToneClass(notifOnchainStatusBox, capabilityState.onchainAvailable ? getAlertTone({
        title: summary.latest_onchain_regime_label,
        message: summary.latest_onchain_snapshot_label,
    }) : summary.latest_onchain_snapshot_status === 'stale' ? 'mixed' : 'neutral');

    if (notifDefiStatus) {
        notifDefiStatus.textContent = capabilityState.defiAvailable
            ? `${simplifyCopy(formatTitle(summary.defi_regime_label || 'available'))} (${summary.defi_chain_name || symbol.replace('USDT', '')})`
            : 'Not Available';
    }
    setToneClass(notifDefiStatusBox, capabilityState.defiAvailable ? getAlertTone({
        title: summary.defi_regime_label,
        message: summary.defi_summary,
    }) : 'neutral');

    if (notifCapabilityNote) {
        notifCapabilityNote.textContent = summary.latest_onchain_snapshot_status === 'stale' && !capabilityState.onchainAvailable
            ? `${capabilityState.note} The latest available network on-chain snapshot is ${formatSignal(summary.latest_onchain_snapshot_label)} from ${Number(summary.latest_onchain_snapshot_age_days || 0).toFixed(0)} days ago${summary.latest_onchain_snapshot_reason ? ` because ${summary.latest_onchain_snapshot_reason}` : ''}.`
            : capabilityState.note;
    }

    if (notifSourcePolicyNote) {
        notifSourcePolicyNote.textContent =
            'Source policy: strategy and price messages come from market/futures outputs. Network on-chain and DeFiLlama TVL are shown separately as supporting context.';
    }

    if (notifMultimodalSelection) {
        notifMultimodalSelection.textContent = summary.multimodal_selected_context_variant
            ? `Current context model: ${formatTitle(summary.multimodal_selected_context_variant)} for ${symbol}.`
            : `No context model is active for ${symbol} right now.`;
    }

    if (notifSentiment) {
        notifSentiment.disabled = !capabilityState.sentimentAvailable;
        if (!capabilityState.sentimentAvailable) {
            notifSentiment.checked = false;
        }
    }

    if (notifSentimentSummary) {
        if (!capabilityState.sentimentAvailable) {
            notifSentimentSummary.textContent =
                `${symbol} does not currently have a usable sentiment layer, so LiveStrat falls back to market, futures, and any other available context only.`;
        } else if (summary.latest_effective_sentiment_source === 'gdelt_asset_news') {
            notifSentimentSummary.textContent =
                `${symbol} currently uses asset-specific GDELT news sentiment with a ${formatSignal(summary.latest_effective_sentiment_label)} state${summary.multimodal_selected_context_variant ? `, and the current context model is ${formatSignal(summary.multimodal_selected_context_variant)}` : ''}.`;
        } else {
            notifSentimentSummary.textContent =
                simplifyCopy(`${symbol} is currently using ${formatTitle(summary.latest_effective_sentiment_source)} as its sentiment source. ${sentimentSummary.sentiment_summary || ''}`.trim());
        }
    }

    if (notifPreviewPanel) {
        setToneClass(notifPreviewPanel, signalTone);
    }
    if (notifReturnCard) {
        setToneClass(notifReturnCard, getDirectionalTone(summary.latest_return_24h_pct || 0));
    }
    if (notifConfidenceCard) {
        setToneClass(notifConfidenceCard, signalTone);
    }

    if (notifPreviewHeading) {
        const primarySignal = formatDecisionLabel(summary.selected_primary_signal || summary.scaled_model_signal || 'n/a');
        notifPreviewHeading.textContent = `${symbol} ${primarySignal} overview`;
    }

    if (notifPreviewMeta) {
        notifPreviewMeta.innerHTML = `
            <span class="alert-chip">${escapeHtml(symbol)}</span>
            <span class="alert-chip">${escapeHtml(formatTitle(summary.trend_status || 'unknown'))}</span>
            <span class="alert-chip">${escapeHtml(capabilityState.modeLabel)}</span>
        `;
    }

    if (notifPreview) {
        notifPreview.innerHTML = previewItems.map((item) => `
            <div class="notification-preview-item tone-${escapeHtml(item.tone)}">
                <span class="preview-item-label">${escapeHtml(item.label)}</span>
                <p>${escapeHtml(item.text)}</p>
            </div>
        `).join('');
    }
}

function getAlertPreferences() {
    return {
        price: Boolean(notifPrice?.checked),
        strategy: Boolean(notifStrategy?.checked),
        sentiment: Boolean(notifSentiment?.checked),
        volatility: Boolean(notifVolatility?.checked),
        onchain: Boolean(notifOnchain?.checked),
        telegram_enabled: Boolean(notifTelegramEnabled?.checked),
        telegram_chat_id: notifTelegramChatId?.value?.trim() || '',
    };
}

function renderGeneratedAlerts(alerts) {
    if (!notifAlertList) {
        return;
    }

    if (notifAlertCount) {
        notifAlertCount.textContent = `${alerts.length} ${alerts.length === 1 ? 'message' : 'messages'}`;
    }

    if (!alerts.length) {
        notifAlertList.innerHTML = `
            <article class="generated-alert empty">
                <h3>No messages generated</h3>
                <p>No selected message rules were triggered for the current asset.</p>
            </article>
        `;
        return;
    }

    notifAlertList.innerHTML = alerts.map((alert) => `
        <article class="generated-alert severity-${escapeHtml(alert.severity)} tone-${escapeHtml(getAlertTone(alert))}">
            <div class="generated-alert-header">
                <h3>${escapeHtml(alert.title)}</h3>
                <span class="generated-alert-tone">${escapeHtml(formatTitle(getAlertTone(alert)))}</span>
            </div>
            <div class="alert-meta">
                <span class="alert-chip">${escapeHtml(alert.category)}</span>
                <span class="alert-chip">${escapeHtml(alert.severity)}</span>
                <span class="alert-chip">${escapeHtml(alert.symbol)}</span>
            </div>
            <p class="generated-alert-message">${escapeHtml(alert.message)}</p>
            <p class="generated-alert-action"><strong>Suggested action:</strong> ${escapeHtml(alert.action)}</p>
        </article>
    `).join('');
}

function renderTelegramStatus(telegram) {
    if (!notifTelegramStatus) {
        return;
    }

    if (!telegram) {
        notifTelegramStatus.textContent = 'Telegram delivery status is unavailable right now.';
        return;
    }

    const botState = telegram.bot_configured
        ? `Bot configured${telegram.bot_username ? ` as @${telegram.bot_username}` : ''}`
        : 'Bot token not configured in environment';
    const userState = telegram.user_telegram_enabled
        ? telegram.user_chat_id_present
            ? 'Your account is ready for Telegram delivery.'
            : 'Telegram is enabled for your account, but no chat ID is saved yet.'
        : 'Telegram delivery is disabled for your account.';
    notifTelegramStatus.textContent = `${botState}. ${userState}`;

    // Mirror short state into the toolbar pill if present.
    const metaPill = document.getElementById('notif-telegram-meta');
    if (metaPill) {
        let pillText = 'Telegram: off';
        if (telegram.bot_configured && telegram.user_telegram_enabled && telegram.user_chat_id_present) {
            pillText = 'Telegram: ready';
        } else if (telegram.bot_configured) {
            pillText = 'Telegram: bot ready';
        }
        metaPill.textContent = pillText;
    }
}

function renderSavedEvents(events) {
    if (!notifEventList) {
        return;
    }

    if (notifEventCount) {
        notifEventCount.textContent = `${events.length} ${events.length === 1 ? 'message' : 'messages'}`;
    }

    if (!events.length) {
        notifEventList.innerHTML = `
            <article class="generated-alert empty">
                <h3>No saved messages yet</h3>
                <p>Use Save generated messages to create account-linked message history.</p>
            </article>
        `;
        return;
    }

    notifEventList.innerHTML = events.map((event) => `
        <article class="generated-alert severity-${escapeHtml(event.severity)} tone-${escapeHtml(getAlertTone(event))} ${event.read_at ? 'is-read' : ''}">
            <div class="generated-alert-header">
                <h3>${escapeHtml(event.title)}</h3>
                <span class="generated-alert-tone">${escapeHtml(formatTitle(getAlertTone(event)))}</span>
            </div>
            <div class="saved-event-meta">
                <span class="alert-chip">${escapeHtml(event.category)}</span>
                <span class="alert-chip">${escapeHtml(event.severity)}</span>
                <span class="alert-chip">${escapeHtml(event.symbol)}</span>
            </div>
            <p class="generated-alert-message">${escapeHtml(event.message)}</p>
            <p class="generated-alert-action"><strong>Suggested action:</strong> ${escapeHtml(event.action || 'Review in LiveStrat.')}</p>
            <div class="saved-event-actions">
                <p class="note">Created: ${escapeHtml(new Date(event.created_at).toLocaleString())}${event.read_at ? ` | Read: ${escapeHtml(new Date(event.read_at).toLocaleString())}` : ''}</p>
                ${event.read_at ? '' : `<button type="button" class="button-secondary mark-event-read" data-event-id="${escapeHtml(event.id)}">Mark as read</button>`}
            </div>
        </article>
    `).join('');

    for (const button of Array.from(document.querySelectorAll('.mark-event-read'))) {
        button.addEventListener('click', async () => {
            await markEventRead(button.dataset.eventId);
        });
    }
}

async function fetchGeneratedAlerts(symbol) {
    try {
        const response = await fetch('/api/alerts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                asset: symbol,
                strategy: notifStrategySelect ? notifStrategySelect.value : 'recommended',
                timeframe: notifTimeframe ? notifTimeframe.value : '4h',
                preferences: getAlertPreferences(),
            }),
        });

        if (!response.ok) {
            throw new Error(`Alert API failed with ${response.status}`);
        }

        const payload = await response.json();
        renderGeneratedAlerts(payload.alerts || []);
    } catch (error) {
        if (notifAlertList) {
            notifAlertList.innerHTML = `
                <article class="generated-alert severity-high">
                <h3>Message engine unavailable</h3>
                <p>The message rules could not be generated right now. Try refreshing the page.</p>
                </article>
            `;
        }
    }
}

async function saveAlertPreferences() {
    if (!notifSaveStatus) {
        return;
    }

    notifSaveStatus.textContent = 'Saving preferences...';

    try {
        const response = await fetch('/api/alert-preferences', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                preferences: getAlertPreferences(),
            }),
        });

        if (!response.ok) {
            throw new Error(`Preference save failed with ${response.status}`);
        }

        const payload = await response.json();
        notifSaveStatus.textContent = 'Preferences saved to your account.';
        renderTelegramStatus(payload.telegram);
        refreshNotifications(notifAsset ? notifAsset.value : 'BTCUSDT');
    } catch (error) {
        notifSaveStatus.textContent = 'Preferences could not be saved right now.';
    }
}

async function loadSavedEvents() {
    if (!notifEventList) {
        return;
    }

    try {
        const response = await fetch('/api/notification-events');
        if (!response.ok) {
            throw new Error(`Event load failed with ${response.status}`);
        }
        const payload = await response.json();
        renderSavedEvents(payload.events || []);
    } catch (error) {
        notifEventList.innerHTML = `
            <article class="generated-alert severity-high">
                <h3>Saved events unavailable</h3>
                <p>The notification event history could not be loaded right now.</p>
            </article>
        `;
    }
}

async function saveGeneratedAlerts() {
    if (!notifSaveStatus) {
        return;
    }

    notifSaveStatus.textContent = 'Saving generated messages...';

    try {
        const response = await fetch('/api/notification-events', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                asset: notifAsset ? notifAsset.value : 'BTCUSDT',
                strategy: notifStrategySelect ? notifStrategySelect.value : 'recommended',
                timeframe: notifTimeframe ? notifTimeframe.value : '4h',
                preferences: getAlertPreferences(),
            }),
        });

        if (!response.ok) {
            let detail = `Event save failed with ${response.status}`;
            try {
                const errorPayload = await response.json();
                if (errorPayload?.detail) {
                    detail = `${detail}: ${errorPayload.detail}`;
                }
            } catch (error) {
                // keep the generic message if the error response is not JSON
            }
            throw new Error(detail);
        }

        const payload = await response.json();
        const telegramCount = Array.isArray(payload.telegram_ready_messages) ? payload.telegram_ready_messages.length : 0;
        const sentCount = Array.isArray(payload.telegram_delivery_attempts)
            ? payload.telegram_delivery_attempts.filter((attempt) => attempt.ok).length
            : 0;
        notifSaveStatus.textContent =
            sentCount
                ? `Saved ${payload.saved_count} messages. ${sentCount} Telegram messages were sent.`
                : telegramCount
                    ? `Saved ${payload.saved_count} messages. ${telegramCount} Telegram-ready message payloads were prepared.`
                : `Saved ${payload.saved_count} messages to your account history.`;
        loadSavedEvents();
    } catch (error) {
        notifSaveStatus.textContent = `Generated messages could not be saved right now. ${error.message}`;
    }
}

async function markEventRead(eventId) {
    try {
        const response = await fetch(`/api/notification-events/${encodeURIComponent(eventId)}/read`, {
            method: 'POST',
        });
        if (!response.ok) {
            throw new Error(`Read update failed with ${response.status}`);
        }
        loadSavedEvents();
    } catch (error) {
        if (notifSaveStatus) {
            notifSaveStatus.textContent = 'Event read state could not be updated right now.';
        }
    }
}

async function loadTelegramStatus() {
    try {
        const response = await fetch('/api/telegram-status');
        if (!response.ok) {
            throw new Error(`Telegram status failed with ${response.status}`);
        }
        const payload = await response.json();
        renderTelegramStatus(payload.telegram);
    } catch (error) {
        renderTelegramStatus(null);
    }
}

function refreshNotifications(symbol) {
    renderNotificationPreview(symbol);
    fetchGeneratedAlerts(symbol);
}

if (notifAsset) {
    notifAsset.addEventListener('change', () => {
        refreshNotifications(notifAsset.value);
    });
}

for (const select of [notifStrategySelect, notifTimeframe]) {
    if (select) {
        select.addEventListener('change', () => {
            refreshNotifications(notifAsset ? notifAsset.value : 'BTCUSDT');
        });
    }
}

if (notifTest) {
    notifTest.addEventListener('click', () => {
        refreshNotifications(notifAsset ? notifAsset.value : 'BTCUSDT');
    });
}

if (notifSavePreferences) {
    notifSavePreferences.addEventListener('click', saveAlertPreferences);
}

if (notifSaveEvents) {
    notifSaveEvents.addEventListener('click', saveGeneratedAlerts);
}

for (const checkbox of [notifPrice, notifStrategy, notifSentiment, notifOnchain, notifDefi, notifVolatility]) {
    if (checkbox) {
        checkbox.addEventListener('change', () => {
            refreshNotifications(notifAsset ? notifAsset.value : 'BTCUSDT');
        });
    }
}

if (notifAsset) {
    refreshNotifications(notifAsset.value);
}

loadSavedEvents();
loadTelegramStatus();
