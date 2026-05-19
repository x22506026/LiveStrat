/* Account page actions for saved strategy profiles */

const STORAGE_KEY = 'livestrat:loadProfile';

function showStatus(message) {
    const el = document.getElementById('account-action-status');
    if (!el) return;
    el.textContent = message;
    el.hidden = !message;
}

function readProfileConfig(card) {
    const dataNode = card.querySelector('.profile-config-data');
    if (!dataNode) return null;
    try {
        return JSON.parse(dataNode.textContent);
    } catch (err) {
        return null;
    }
}

function handleUseProfile(card) {
    const config = readProfileConfig(card);
    const asset = card.dataset.profileAsset || 'BTCUSDT';
    const name = card.querySelector('h4')?.textContent || 'Custom strategy';
    if (!config) {
        showStatus('Profile data could not be read.');
        return;
    }
    const selection = config.selection || {};
    const timeframes = selection.timeframes || [];
    let timeframeScope = selection.timeframe_scope || timeframes[0] || '4h';
    if (timeframes.length > 1) {
        if (timeframes.includes('1h') && timeframes.includes('4h')) timeframeScope = '1h_4h_stack';
        else if (timeframes.includes('4h') && timeframes.includes('1d')) timeframeScope = '4h_1d_stack';
    }
    const payload = {
        name,
        asset,
        selection: {
            core_signal: selection.core_signal || 'trend_following',
            timeframe_scope: timeframeScope,
            data_sources: selection.data_sources || ['market'],
            confirmation_filters: selection.confirmation_filters || [],
            decision_rules: selection.decision_rules || 'double_confirmation',
            risk_profile: selection.risk_profile || 'balanced_risk',
        },
    };
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (err) {
        showStatus('Could not store profile for handoff.');
        return;
    }
    window.location.href = '/strategies#builder-section';
}

async function handleDeleteProfile(card) {
    const id = card.dataset.profileId;
    if (!id) return;
    const confirmed = window.confirm('Delete this saved profile?');
    if (!confirmed) return;
    try {
        const response = await fetch(`/api/strategy-profiles/${id}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('delete failed');
        card.remove();
        showStatus('Profile deleted.');
    } catch (err) {
        showStatus('Could not delete profile.');
    }
}

document.querySelectorAll('.account-strategy-card').forEach((card) => {
    card.querySelector('[data-action="use"]')?.addEventListener('click', () => handleUseProfile(card));
    card.querySelector('[data-action="delete"]')?.addEventListener('click', () => handleDeleteProfile(card));
});
