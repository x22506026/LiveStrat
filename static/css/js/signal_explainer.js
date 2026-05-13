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
        if (signal === 'buy') {
            base = `${asset} currently says Buy because the backend sees enough positive evidence. Confidence is ${(
                confidence * 100
            ).toFixed(1)}% under a ${trend} trend reading.`;
        } else if (signal === 'dont_buy') {
            base = `${asset} currently says Avoid because the backend does not see enough evidence to justify entry. Confidence is ${(
                confidence * 100
            ).toFixed(1)}%.`;
        } else if (signal === 'hold') {
            base = `${asset} currently says Hold because the backend sees a mixed market picture.`;
        } else {
            base = `${asset} does not currently have a strong actionable signal, so the engine is staying descriptive rather than directional.`;
        }

        const reasonBits = [`Volatility is ${volatility}.`];

        if (sentimentSource !== 'unavailable') {
            reasonBits.push(
                `Effective sentiment is ${formatSignal(sentimentLabel)} via ${formatTitle(sentimentSource)}.`
            );
        } else {
            reasonBits.push('No usable sentiment layer is currently available.');
        }

        if (onchainLabel !== 'unavailable') {
            const onchainReason = summary.latest_onchain_regime_reason || '';
            reasonBits.push(
                onchainReason
                    ? `On-chain context is ${formatSignal(onchainLabel)} because ${onchainReason}.`
                    : `On-chain context is ${formatSignal(onchainLabel)}.`
            );
        } else if (String(summary.latest_onchain_snapshot_status || 'unavailable') === 'stale') {
            reasonBits.push(
                `The latest available on-chain snapshot is ${formatSignal(summary.latest_onchain_snapshot_label || 'unknown')} from ${Number(summary.latest_onchain_snapshot_age_days || 0).toFixed(0)} days ago${summary.latest_onchain_snapshot_reason ? ` because ${summary.latest_onchain_snapshot_reason}` : ''}.`
            );
        } else {
            reasonBits.push('No on-chain confirmation is currently available.');
        }

        const walkforwardFolds = Number(summary.walkforward_fold_count || 0);
        if (walkforwardFolds > 0 && walkforwardExcess > 0) {
            reasonBits.push(`Rolling validation excess return is positive at ${(walkforwardExcess * 100).toFixed(1)}%.`);
        } else if (walkforwardFolds > 0 && walkforwardExcess < 0) {
            reasonBits.push(`Rolling validation excess return is negative at ${(walkforwardExcess * 100).toFixed(1)}%, so this signal should be treated cautiously.`);
        }

        if (strategyReturn !== 0 || buyHoldReturn !== 0) {
            reasonBits.push(
                `Current policy return is ${(strategyReturn * 100).toFixed(1)}% versus buy-and-hold ${(buyHoldReturn * 100).toFixed(1)}%.`
            );
        }

        if (multimodalContextVariant) {
            reasonBits.push(
                `The current multimodal extension selected ${formatSignal(multimodalContextVariant)} on validation, with validation macro-F1 ${(multimodalValidationMacroF1 * 100).toFixed(1)}% and held-out macro-F1 ${(multimodalTestMacroF1 * 100).toFixed(1)}%.`
            );
        }

        reasonBits.push(`Context mode is ${contextMode.modeLabel.toLowerCase()}, meaning ${contextMode.note}.`);

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
