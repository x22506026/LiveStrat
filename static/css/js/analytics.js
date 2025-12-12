/*gets references to the analytics controls*/
const analyticsAsset = document.getElementById('analytics-asset');
const analyticsRun = document.getElementById('analytics-run');

/*gets references to the elements where analytics results are displayed*/
const analyticsSummary = document.getElementById('analytics-summary');
const analyticsMean = document.getElementById('analytics-mean');
const analyticsVol = document.getElementById('analytics-volatility');
const analyticsSentiment = document.getElementById('analytics-sentiment');

/*checks that the run button exists before attaching the event*/
if (analyticsRun) {
    /*runs when the user clicks the demo analytics button*/
    analyticsRun.addEventListener('click', () => {
        /*reads the selected asset*/
        const asset = analyticsAsset.value;

        /*demo analytics outputs based on the selected asset*/
        if (asset === 'BTC') {
            analyticsMean.textContent = '0.8%';
            analyticsVol.textContent = '4.3%';
            analyticsSentiment.textContent = 'Sentiment moderately supports up moves.';
            analyticsSummary.textContent =
                'Demo: BTC shows positive average daily return with elevated volatility. ' +
                'Reddit sentiment is supportive, Twitter data is limited but not negative.';
        } else {
            analyticsMean.textContent = '0.5%';
            analyticsVol.textContent = '3.1%';
            analyticsSentiment.textContent = 'Sentiment slightly positive but weaker than BTC.';
            analyticsSummary.textContent =
                'Demo: ETH has smoother volatility and lower mean returns. ' +
                'Sentiment is mixed but tends to follow BTC mood.';
        }
    });
}
