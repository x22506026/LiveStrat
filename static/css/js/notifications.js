/*gets references to the notification option checkboxes*/
const notifPrice = document.getElementById('notif-price');
const notifStrategy = document.getElementById('notif-strategy');
const notifSentiment = document.getElementById('notif-sentiment');

/*gets references to the test button and preview area*/
const notifTest = document.getElementById('notif-test');
const notifPreview = document.getElementById('notif-preview');

/*checks that the test button exists before attaching the event*/
if (notifTest) {
    /*runs when the user clicks the test notification button*/
    notifTest.addEventListener('click', () => {
        /*array used to build the notification message*/
        let parts = [];

        /*adds a price alert message if enabled*/
        if (notifPrice.checked) {
            parts.push('Price alert: BTC breaks key resistance.');
        }

        /*adds a strategy signal message if enabled*/
        if (notifStrategy.checked) {
            parts.push('Strategy signal: Model recommends BUY with confidence 0.67.');
        }

        /*adds a sentiment alert message if enabled*/
        if (notifSentiment.checked) {
            parts.push('Sentiment alert: Reddit sentiment turns sharply positive.');
        }

        /*handles the case where no options are selected*/
        if (parts.length === 0) {
            notifPreview.textContent = 'No notification types selected.';
        } else {
            /*shows an example combined notification message*/
            notifPreview.textContent =
                'Example notification: ' + parts.join(' ');
        }
    });
}
