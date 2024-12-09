from flask import Flask
from apps.fundamentalanalysis import fundamentalanalysis_bp
from apps.market_sentiment import market_sentiment_bp
from apps.option_trading import option_trading_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(fundamentalanalysis_bp, url_prefix="/fundamentalanalysis")
app.register_blueprint(market_sentiment_bp, url_prefix="/market_sentiment")
app.register_blueprint(option_trading_bp, url_prefix="/option_trading")

@app.route("/")
def home():
    return """
    <h1>Welcome to the Stock Analysis App</h1>
    <ul>
        <li><a href="/fundamentalanalysis/">Fundamental Analysis</a></li>
        <li><a href="/market_sentiment/">Market Sentiment</a></li>
        <li><a href="/option_trading/">Option Trading</a></li>
    </ul>
    """

if __name__ == "__main__":
    app.run(debug=True)
