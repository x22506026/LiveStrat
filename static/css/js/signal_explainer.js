/* builds plain-English signal explanations from the current LiveStrat summary fields */

(function () {
    function formatSignal(value) {
        return (value || 'n/a').replaceAll('_', ' ');
    }

    function formatTitle(value) {
        const text = formatSignal(value);
        return text.charAt(0).toUpperCase() + text.slice(1);
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

    function simplifyCopy(value) {
        return String(value || '')
            .replaceAll('slower structural', 'longer-term')
            .replaceAll('intraday timing', 'short-term timing')
            .replaceAll('regime', 'state')
            .replaceAll('fallback', 'backup')
            .replaceAll('veto', 'warning');
    }

    function getContextMode(summary) {
        const sentimentSource = String(summary.latest_effective_sentiment_source || 'unavailable');
        const onchainUnavailable = !summary.latest_onchain_regime_label || summary.latest_onchain_regime_label === 'unavailable';

        if (sentimentSource === 'fear_greed_market_fallback') {
            return {
                modeLabel: 'Backup context',
                note: 'market-wide sentiment is being used because asset-specific news sentiment is unavailable',
            };
        }

        if (sentimentSource === 'unavailable' && onchainUnavailable) {
            return {
                modeLabel: 'Reduced',
                note: 'sentiment and on-chain context are both unavailable, so the signal relies mainly on market and futures layers',
            };
        }

        return {
            modeLabel: 'Ready',
            note: 'the currently available context layers can be used',
        };
    }

    function buildSignalExplanation(summary, asset) {
        const signal = String(summary.selected_primary_signal || summary.latest_signal || summary.scaled_model_signal || 'n/a');
        const trend = formatSignal(summary.trend_status || 'unknown');
        const volatility = formatSignal(summary.volatility_status || 'unknown');
        const confidence = Number(
            summary.selected_primary_confidence ?? summary.latest_signal_confidence ?? summary.scaled_model_confidence ?? 0
        );
        const sentimentLabel = String(summary.latest_effective_sentiment_label || 'unavailable');
        const sentimentSource = String(summary.latest_effective_sentiment_source || 'unavailable');
        const onchainLabel = String(summary.latest_onchain_regime_label || 'unavailable');
        const walkforwardExcess = Number(summary.walkforward_avg_excess_return || 0);
        const strategyReturn = Number(summary.strategy_total_return || 0);
        const buyHoldReturn = Number(summary.buy_hold_total_return || 0);
        const multimodalContextVariant = String(summary.multimodal_selected_context_variant || '');
        const multimodalValidationMacroF1 = Number(summary.multimodal_validation_macro_f1 || 0);
        const multimodalTestMacroF1 = Number(summary.multimodal_test_macro_f1 || 0);
        const contextMode = getContextMode(summary);

        let base;
        const conf = (confidence * 100).toFixed(1);
        if (signal === 'buy') {
            base = `${asset}: Buy at ${conf}% confidence. Trend ${trend}.`;
        } else if (signal === 'dont_buy') {
            base = `${asset}: Avoid at ${conf}% confidence.`;
        } else if (signal === 'hold') {
            base = `${asset}: Hold. Mixed picture.`;
        } else {
            base = `${asset}: no strong signal right now.`;
        }

        const reasonBits = [`Volatility ${volatility}.`];

        if (sentimentSource !== 'unavailable') {
            reasonBits.push(`Sentiment ${formatSignal(sentimentLabel)} (${formatTitle(sentimentSource)}).`);
        }

        if (onchainLabel !== 'unavailable') {
            reasonBits.push(`On-chain ${formatSignal(onchainLabel)}.`);
        }

        const walkforwardFolds = Number(summary.walkforward_fold_count || 0);
        if (walkforwardFolds > 0 && walkforwardExcess !== 0) {
            reasonBits.push(`Walk-forward excess ${(walkforwardExcess * 100).toFixed(1)}%.`);
        }

        if (strategyReturn !== 0 || buyHoldReturn !== 0) {
            reasonBits.push(`Strategy ${(strategyReturn * 100).toFixed(1)}% vs hold ${(buyHoldReturn * 100).toFixed(1)}%.`);
        }

        return simplifyCopy(`${base} ${reasonBits.join(' ')}`);
    }

    window.LiveStratSignalExplainer = {
        buildSignalExplanation,
        getContextMode,
        formatTitle,
        formatSignal,
        formatDecisionLabel,
    };
})();
