/*simple demo logic used on the dashboard page*/

/*gets references to the trend and sentiment controls*/
const trendAsset = document.getElementById('trend-asset');
const trendStatus = document.getElementById('trend-status');
const sentimentAsset = document.getElementById('sentiment-asset');
const sentimentScore = document.getElementById('sentiment-score');
const redditLine = document.getElementById('reddit-line');
const twitterLine = document.getElementById('twitter-line');
const modelVerdict = document.getElementById('model-verdict');

/*handles changes to the selected asset for trend demo*/
if (trendAsset) {
    trendAsset.addEventListener('change', () => {
        const value = trendAsset.value;

        /*updates trend and verdict based on selected asset*/
        if (value === 'BTCUSDT') {
            trendStatus.textContent = 'Moderately Bullish';
            modelVerdict.textContent = 'Buy (demo)';
        } else if (value === 'ETHUSDT') {
            trendStatus.textContent = 'Sideways / Neutral';
            modelVerdict.textContent = 'Hold / Don’t Buy';
        } else {
            trendStatus.textContent = 'High Volatility';
            modelVerdict.textContent = 'High Risk – Avoid';
        }
    });
}

/*handles changes to the selected asset for sentiment demo*/
if (sentimentAsset) {
    sentimentAsset.addEventListener('change', () => {
        /*updates sentiment score and source summaries*/
        if (sentimentAsset.value === 'BTC') {
            sentimentScore.textContent = '68 / 100';
            redditLine.textContent = 'Reddit: strong positive interest.';
            twitterLine.textContent = 'Twitter: limited sample, slightly positive.';
        } else {
            sentimentScore.textContent = '55 / 100';
            redditLine.textContent = 'Reddit: mixed but calm discussions.';
            twitterLine.textContent = 'Twitter: low activity, mostly neutral.';
        }
    });
}
