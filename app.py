from flask import Flask, render_template, redirect, url_for, request

#this creates the flask app
app = Flask(__name__)

#this route shows the login page
@app.route('/')
def login():
    return render_template('login.html')

#this route shows the signup page
@app.route('/signup')
def signup():
    return render_template('signup.html')

#this handles the login form (no real auth yet)
@app.route('/login', methods=['POST'])
def login_user():
    return redirect(url_for('dashboard'))

#this handles the signup form (no real storage yet)
@app.route('/signup_user', methods=['POST'])
def signup_user():
    return redirect(url_for('dashboard'))

#this shows the main dashboard after "login"
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', active_page='dashboard')

#this shows the markets page
@app.route('/markets')
def markets():
    return render_template('markets.html', active_page='markets')

#this shows the strategies page
@app.route('/strategies')
def strategies():
    return render_template('strategies.html', active_page='strategies')

#this shows the analytics page
@app.route('/analytics')
def analytics():
    return render_template('analytics.html', active_page='analytics')

#this shows the notifications page
@app.route('/notifications')
def notifications():
    return render_template('notifications.html', active_page='notifications')

#this shows the sign out confirmation
@app.route('/logout')
def logout():
    return render_template('logout_confirm.html')

#this actually "logs out" by sending user back to login screen
@app.route('/logout_confirmed')
def logout_confirmed():
    return redirect(url_for('login'))

#this runs the app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
