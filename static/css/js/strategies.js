/*gets references to the strategy selection controls*/
const strategySelect = document.getElementById('strategy-select');
const strategyAsset = document.getElementById('strategy-asset');
const strategyRun = document.getElementById('strategy-run');

/*gets references to the elements where results are displayed*/
const strategyWinrate = document.getElementById('strategy-winrate');
const strategyConfidence = document.getElementById('strategy-confidence');
const strategyDrawdown = document.getElementById('strategy-drawdown');
const strategySummary = document.getElementById('strategy-summary');
const strategyNote = document.getElementById('strategy-note');

/*checks that the run button exists before attaching the event*/
if (strategyRun) {
    /*runs when the user clicks the run strategy button*/
    strategyRun.addEventListener('click', () => {
        /*reads the selected strategy and asset*/
        const strat = strategySelect.value;
        const asset = strategyAsset.value;

        /*demo logic for different strategy types*/
        if (strat === 'rsi') {
            /*example outputs for RSI rebound strategy*/
            strategyWinrate.textContent = '54%';
            strategyConfidence.textContent = '0.58';
            strategyDrawdown.textContent = '-18%';
            strategySummary.textContent =
                `RSI Rebound strategy on ${asset}: prefers oversold bounces with neutral sentiment.`;
        } else if (strat === 'ma') {
            /*example outputs for moving average crossover strategy*/
            strategyWinrate.textContent = '61%';
            strategyConfidence.textContent = '0.66';
            strategyDrawdown.textContent = '-22%';
            strategySummary.textContent =
                `Moving Average Crossover on ${asset}: follows medium-term trends and ignores short noise.`;
        } else {
            /*example outputs for trend and sentiment strategy*/
            strategyWinrate.textContent = '64%';
            strategyConfidence.textContent = '0.71';
            strategyDrawdown.textContent = '-20%';
            strategySummary.textContent =
                `Trend + Sentiment model on ${asset}: combines trend features with Reddit/Twitter sentiment.`;
        }

        /*note explaining that values are placeholders at this stage*/
        strategyNote.textContent =
            'In the real system, these values will come from backtests ' +
            'performed on historical Binance data with sentiment features included.';
    });
}
