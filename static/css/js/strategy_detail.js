const strategyDetailScript = document.getElementById('strategy-detail-data');
const initialStrategyDetail = strategyDetailScript ? JSON.parse(strategyDetailScript.textContent) : {};
const strategyDetailAsset = document.getElementById('strategy-detail-asset');
const strategyDetailTimeframe = document.getElementById('strategy-detail-timeframe-select');
const strategyDetailRefreshButton = document.getElementById('strategy-detail-refresh');

function detailTitle(value) {
    const text = String(value || 'n/a').replaceAll('_', ' ');
    return text.charAt(0).toUpperCase() + text.slice(1);
}

function simplifyDetailCopy(value) {
    return String(value || '')
        .replaceAll('fallback', 'backup')
        .replaceAll('Fallback', 'Backup')
        .replaceAll('governance', 'rules')
        .replaceAll('Governance', 'Rules')
        .replaceAll('posture', 'status')
        .replaceAll('regime', 'state')
        .replaceAll('Regime', 'State')
        .replaceAll('structural', 'longer-term')
        .replaceAll('Structural', 'Longer-term')
        .replace(/\s+/g, ' ')
        .trim();
}

function shortenDetailCopy(value, maxLength = 260) {
    const text = simplifyDetailCopy(value);
    if (text.length <= maxLength) {
        return text;
    }
    return `${text.slice(0, maxLength).replace(/\s+\S*$/, '')}...`;
}

function detailPercent(value) {
    if (value === null || value === undefined || value === '' || Number.isNaN(Number(value))) {
        return 'n/a';
    }
    return `${(Number(value) * 100).toFixed(1)}%`;
}

function detailScore(value) {
    if (value === null || value === undefined || value === '' || Number.isNaN(Number(value)) || Number(value) === 0) {
        return 'n/a / 5';
    }
    return `${Number(value)} / 5`;
}

function joinLayers(values) {
    return Array.isArray(values) && values.length ? values.join(', ') : 'none';
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function renderStrategyDetail(payload) {
    const preset = payload?.preset || {};
    const config = payload?.resolved_config || {};
    const governance = config.governance || {};
    const scorecard = config.scorecard || {};
    const contextAssessment = payload?.context_assessment || {};
    const overlay = payload?.context_overlay_entry || {};
    const fit = payload?.daily_structural_fit || {};

    setText('strategy-detail-title', preset.name || 'Strategy');
    setText('strategy-detail-name', preset.name || 'Strategy');
    setText('strategy-detail-tagline', payload?.strategy_page_brief?.tagline || preset.tagline || 'Strategy detail view.');
    setText('strategy-detail-brief', simplifyDetailCopy(config.evaluation_basis_note || preset.explanation || 'Evaluation basis will appear here.'));
    setText('strategy-detail-readiness', governance.readiness_label || 'Loading');
    setText('strategy-detail-timeframe', detailTitle(config.resolved_timeframe || payload?.resolved_timeframe || 'n/a'));
    setText('strategy-detail-best-for', 'Best for');
    setText('strategy-detail-best-for-copy', payload?.strategy_page_brief?.best_for || preset.best_for || 'Usage guidance will appear here.');
    setText('strategy-detail-risk', preset.risk_profile || 'n/a');
    setText('strategy-detail-engine', preset.core_engine || 'n/a');
    setText('strategy-detail-daily-fit', detailTitle(config.daily_confirmation_fit || 'n/a'));
    setText('strategy-detail-tier', simplifyDetailCopy(detailTitle(governance.deployment_tier || 'n/a')));
    setText('strategy-detail-family', payload?.family_governance?.lead_family_label || 'n/a');
    setText('strategy-detail-fit', simplifyDetailCopy(detailTitle(config.timeframe_fit_label || 'n/a')));
    setText('strategy-detail-context-role', simplifyDetailCopy(detailTitle(payload?.context_overlay_role || 'n/a')));
    setText('strategy-detail-governance-note', shortenDetailCopy(governance.evaluation_reason || 'Evidence notes will appear here.'));
    setText('strategy-detail-timeframe-note', shortenDetailCopy(payload?.strategy_page_brief?.timeframe_note || 'Timeframe note will appear here.'));
    setText('strategy-detail-accuracy', detailPercent(scorecard.reference_accuracy));
    setText('strategy-detail-excess', detailPercent(scorecard.reference_excess_return));
    setText('strategy-detail-depth', detailScore(scorecard.academic_depth_score));
    setText('strategy-detail-interpretability', detailScore(scorecard.interpretability_score));
    setText('strategy-detail-scorecard-summary', shortenDetailCopy(scorecard.scorecard_summary || 'Scorecard summary will appear here.'));
    setText('strategy-detail-reference-summary', shortenDetailCopy(scorecard.reference_summary || 'Reference summary will appear here.'));
    setText('strategy-detail-required', joinLayers(config.required_layers));
    setText('strategy-detail-optional', joinLayers(config.optional_layers));
    setText('strategy-detail-unavailable', joinLayers(config.unavailable_layers));
    setText('strategy-detail-target', config.target_family || 'n/a');
    setText('strategy-detail-model', config.model_family || 'n/a');
    setText('strategy-detail-policy', config.policy_family || 'n/a');
    setText('strategy-detail-news-theme', detailTitle(overlay.latest_news_event_mode || contextAssessment.asset_news_sentiment?.dominant_theme || 'none'));
    setText('strategy-detail-onchain-support', detailTitle(fit.primary_support_driver || contextAssessment.onchain_daily?.primary_support_driver || 'none'));
    setText('strategy-detail-futures-support', simplifyDetailCopy(detailTitle(payload?.family_governance?.futures_support_mode || governance.futures_support_mode || 'n/a')));
    setText('strategy-detail-refresh-state', simplifyDetailCopy(detailTitle(config.governance?.pipeline_freshness_label || 'n/a')));
    setText('strategy-detail-context-summary', shortenDetailCopy(contextAssessment.context_reliability_summary || 'Context summary will appear here.'));
    setText('strategy-detail-overlay-summary', shortenDetailCopy(overlay.context_overlay_summary || 'Overlay summary will appear here.'));
}

async function fetchStrategyDetail() {
    const presetId = initialStrategyDetail?.preset?.id || window.location.pathname.split('/').pop();
    const asset = strategyDetailAsset?.value || 'BTCUSDT';
    const timeframe = strategyDetailTimeframe?.value || '4h';
    const response = await fetch(`/api/strategy-catalog/${encodeURIComponent(presetId)}?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}`);
    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
    }
    return response.json();
}

async function refreshStrategyDetail() {
    try {
        const payload = await fetchStrategyDetail();
        renderStrategyDetail(payload);
    } catch (error) {
        setText('strategy-detail-governance-note', `Strategy detail could not be refreshed right now. ${error.message}`);
    }
}

renderStrategyDetail(initialStrategyDetail);

if (strategyDetailRefreshButton) {
    strategyDetailRefreshButton.addEventListener('click', refreshStrategyDetail);
}
if (strategyDetailAsset) {
    strategyDetailAsset.addEventListener('change', refreshStrategyDetail);
}
if (strategyDetailTimeframe) {
    strategyDetailTimeframe.addEventListener('change', refreshStrategyDetail);
}
