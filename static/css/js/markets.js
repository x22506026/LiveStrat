/*gets references to the market selection controls*/
const marketAsset = document.getElementById('market-asset');
const marketTimeframe = document.getElementById('market-timeframe');
const marketButton = document.getElementById('market-generate');

/*gets references to the elements where market metrics are displayed*/
const highLow = document.getElementById('market-high-low');
const avgPrice = document.getElementById('market-average');
const volatility = document.getElementById('market-volatility');
const volumeTrend = document.getElementById('market-volume-trend');
const marketSummary = document.getElementById('market-summary');

/*checks that the generate button exists before attaching the event*/
if (marketButton) {
    /*runs when the user clicks the market snapshot button*/
    marketButton.addEventListener('click', () => {
        /*reads the selected asset and timeframe*/
        const asset = marketAsset.value;
        const tf = marketTimeframe.value;

        /*rough demo values used to simulate market data*/
        let high = 72000, low = 69000, avg = 70500, vol = 'Medium', volTrend = 'Rising';

        /*adjusts demo values based on the selected asset*/
        if (asset === 'ETHUSDT') {
            high = 4200;
            low = 3800;
            avg = 4000;
            vol = 'Low';
            volTrend = 'Steady';
        } else if (asset === 'SOLUSDT') {
            high = 220;
            low = 190;
            avg = 205;
            vol = 'High';
            volTrend = 'Spike then cool-off';
        }

        /*updates the displayed market metrics*/
        highLow.textContent = `${high} / ${low}`;
        avgPrice.textContent = avg;
        volatility.textContent = vol;
        volumeTrend.textContent = volTrend;

        /*updates the summary text explaining the demo output*/
        marketSummary.textContent =
            `Demo snapshot for ${asset} on ${tf} timeframe. ` +
            `In the final project, these values will be calculated from Binance OHLCV data.`;
    });
}
