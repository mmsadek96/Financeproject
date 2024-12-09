import pandas as pd
import requests
import logging
import json
import plotly.utils
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, Blueprint
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import openai
from markdown2 import markdown # For converting analysis markdown to HTML
import os
from dotenv import load_dotenv


# Configure Flask app
app = Flask(__name__)
load_dotenv()

fundamentalanalysis_bp = Blueprint(
    "fundamentalanalysis", __name__, template_folder="templates"
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# API Keys
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY or not OPENAI_API_KEY:
    logging.error("API keys are missing.")
    raise ValueError("Missing Alpha Vantage or OpenAI API key.")

openai.api_key = OPENAI_API_KEY

# Fetch stock time series data from Alpha Vantage
def get_time_series_daily(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={API_KEY}'
    logging.debug(f"Fetching time series data for symbol: {symbol}")
    try:
        response = requests.get(url)
        data = response.json()

        # Check for errors in the API response
        if "Error Message" in data:
            logging.error(f"Invalid symbol or API call error for {symbol}: {data['Error Message']}")
            return None

        if 'Time Series (Daily)' not in data:
            logging.error(f"API error for {symbol}: {data.get('Note', 'Unknown error')}")
            return None

        # Parse the time series data
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adjusted_close',
            '6. volume': 'volume',
            '7. dividend amount': 'dividend_amount',
            '8. split coefficient': 'split_coefficient'
        })
        df = df.apply(pd.to_numeric)
        df['date'] = df.index
        logging.debug(f"Time series DataFrame for {symbol} created with {len(df)} records.")
        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error processing time series data for {symbol}: {e}")
        return None

# Advanced analysis functions
def calculate_fibonacci_levels(df):
    try:
        high_price = df['high'].max()
        low_price = df['low'].min()
        diff = high_price - low_price
        levels = [high_price - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
        logging.debug(f"Calculated Fibonacci levels: {levels}")
        return {
            '0%': high_price,
            '23.6%': levels[0],
            '38.2%': levels[1],
            '50%': levels[2],
            '61.8%': levels[3],
            '78.6%': levels[4],
            '100%': low_price
        }
    except Exception as e:
        logging.error(f"Error calculating Fibonacci levels: {e}")
        return {}

def calculate_support_resistance(df):
    try:
        df['rolling_high'] = df['high'].rolling(window=20).max()
        df['rolling_low'] = df['low'].rolling(window=20).min()
        support = df['rolling_low'].dropna().unique().tolist()
        resistance = df['rolling_high'].dropna().unique().tolist()
        logging.debug(f"Support levels: {support}, Resistance levels: {resistance}")
        return {
            'support_levels': support,
            'resistance_levels': resistance
        }
    except Exception as e:
        logging.error(f"Error calculating support and resistance levels: {e}")
        return {}

def fetch_moving_average(symbol, interval='daily', time_period=20, series_type='close', ma_type='SMA'):
    url = f'https://www.alphavantage.co/query?function={ma_type}&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&apikey={API_KEY}'
    try:
        response = requests.get(url)
        data = response.json()
        key = f"Technical Analysis: {ma_type}"
        if key not in data:
            logging.error(f"API error fetching {ma_type} for {symbol}: {data.get('Error Message', 'Unknown error')}")
            return {}
        ma_data = pd.DataFrame.from_dict(data[key], orient='index')
        ma_data.index = pd.to_datetime(ma_data.index)
        ma_data = ma_data.rename(columns={f"{ma_type}": f"{ma_type}_{time_period}"})
        ma_data = ma_data.apply(pd.to_numeric)
        logging.debug(f"Fetched {ma_type} data for {symbol} with {len(ma_data)} records.")

        # Convert DataFrame to dictionary for easier processing in main code
        return {f"{ma_type}_{time_period}": ma_data[f"{ma_type}_{time_period}"].tolist()}
    except Exception as e:
        logging.error(f"Error fetching {ma_type} for {symbol}: {e}")
        return {}

def fetch_rsi(symbol, interval='daily', time_period=14, series_type='close'):
    url = f'https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&apikey={API_KEY}'
    try:
        response = requests.get(url)
        data = response.json()
        key = "Technical Analysis: RSI"
        if key not in data:
            logging.error(f"API error fetching RSI for {symbol}: {data.get('Error Message', 'Unknown error')}")
            return None
        rsi_data = pd.DataFrame.from_dict(data[key], orient='index')
        rsi_data.index = pd.to_datetime(rsi_data.index)
        rsi_data = rsi_data.rename(columns={"RSI": f"RSI_{time_period}"})
        rsi_data = rsi_data.apply(pd.to_numeric)
        logging.debug(f"Fetched RSI data for {symbol} with {len(rsi_data)} records.")
        return rsi_data[f"RSI_{time_period}"].tolist()  # Return as a list
    except Exception as e:
        logging.error(f"Error fetching RSI for {symbol}: {e}")
        return []

def fetch_bollinger_bands(symbol, interval='daily', time_period=20, series_type='close', nbdevup=2, nbdevdn=2):
    url = f'https://www.alphavantage.co/query?function=BBANDS&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&nbdevup={nbdevup}&nbdevdn={nbdevdn}&apikey={API_KEY}'
    try:
        response = requests.get(url)
        data = response.json()
        key = "Technical Analysis: BBANDS"
        if key not in data:
            logging.error(f"API error fetching Bollinger Bands for {symbol}: {data.get('Error Message', 'Unknown error')}")
            return None
        bb_data = pd.DataFrame.from_dict(data[key], orient='index')
        bb_data.index = pd.to_datetime(bb_data.index)
        bb_data = bb_data.rename(columns={
            "Real Upper Band": "upper_band",
            "Real Middle Band": "middle_band",
            "Real Lower Band": "lower_band"
        })
        bb_data = bb_data.apply(pd.to_numeric)
        logging.debug(f"Fetched Bollinger Bands data for {symbol} with {len(bb_data)} records.")
        return {
            'upper_band': bb_data['upper_band'].tolist(),
            'middle_band': bb_data['middle_band'].tolist(),
            'lower_band': bb_data['lower_band'].tolist()
        }
    except Exception as e:
        logging.error(f"Error fetching Bollinger Bands for {symbol}: {e}")
        return {
            'upper_band': [],
            'middle_band': [],
            'lower_band': []
        }

def detect_candlestick_patterns(df):
    try:
        patterns = {}
        df['doji'] = abs(df['close'] - df['open']) <= ((df['high'] - df['low']) * 0.1)
        doji_dates = df[df['doji']]['date'].dt.strftime('%Y-%m-%d').tolist()
        patterns['Doji'] = doji_dates

        df['hammer'] = (
            (df['close'] > df['open']) &
            (df['low'] < df['open']) &
            ((df['open'] - df['low']) >= 2 * abs(df['close'] - df['open']))
        )
        hammer_dates = df[df['hammer']]['date'].dt.strftime('%Y-%m-%d').tolist()
        patterns['Hammer'] = hammer_dates

        logging.debug(f"Detected candlestick patterns: {patterns}")
        return patterns
    except Exception as e:
        logging.error(f"Error detecting candlestick patterns: {e}")
        return {}

def predict_future_price(df):
    try:
        df = df[['date', 'close']].copy()
        df['timestamp'] = (df['date'] - df['date'].min()).dt.days

        X = df[['timestamp']]
        y = df['close']

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.array([df['timestamp'].max() + i for i in range(1, 6)]).reshape(-1, 1)
        predictions = model.predict(future_days)

        # Ensure predictions are converted to a list
        predictions_list = predictions.tolist()

        confidence_intervals = [(pred - 5, pred + 5) for pred in predictions_list]
        logging.debug(f"Predicted future prices: {predictions_list}, Confidence intervals: {confidence_intervals}")
        return predictions_list, confidence_intervals
    except Exception as e:
        logging.error(f"Error predicting future prices: {e}")
        return [], []

def call_openai_api(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use GPT-4 or the preferred model
            messages=[
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = response['choices'][0]['message']['content']
        return analysis
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return "Error generating analysis."

def generate_openai_analysis(symbol, fibonacci_levels, future_prices, moving_averages, rsi, bollinger_bands, patterns, support_resistance):
    try:
        # Extract and interpret RSI
        rsi_last_value = rsi[-1] if rsi else None
        rsi_status = (
            "Overbought" if rsi_last_value and rsi_last_value > 70 else
            "Oversold" if rsi_last_value and rsi_last_value < 30 else
            "Neutral" if rsi_last_value is not None else "Unavailable"
        )

        # Summarize Moving Averages
        ma_summary = {key: round(values[-1], 2) if values else "N/A" for key, values in moving_averages.items()}

        # Summarize Bollinger Bands
        bollinger_summary = {
            "Upper Band": round(bollinger_bands.get('upper_band', [None])[-1], 2) if bollinger_bands.get('upper_band') else "N/A",
            "Middle Band": round(bollinger_bands.get('middle_band', [None])[-1], 2) if bollinger_bands.get('middle_band') else "N/A",
            "Lower Band": round(bollinger_bands.get('lower_band', [None])[-1], 2) if bollinger_bands.get('lower_band') else "N/A"
        }

        # Key Fibonacci levels (rounded)
        key_fibonacci_levels = {level: round(fibonacci_levels.get(level, "N/A"), 2)
                                for level in ['0%', '100%', '23.6%', '38.2%', '50%', '61.8%', '78.6%']}

        # Candlestick patterns summary
        pattern_summary = {pattern: len(dates) for pattern, dates in patterns.items()}

        # Support and resistance
        support_levels = [round(level, 2) for level in support_resistance.get("support_levels", [])]
        resistance_levels = [round(level, 2) for level in support_resistance.get("resistance_levels", [])]

        # Rounded future prices
        future_prices_rounded = [round(p, 2) for p in future_prices]

        # Enhanced Prompt
        prompt = f"""
You are an elite financial strategist and an expert trading professional. Your goal is to provide a top-tier, authoritative, and highly actionable trading strategy based on the technical indicators provided for {symbol}. Assume that this analysis will be used to make real trades, so it must be explicit, confident, and carefully reasoned.

### Technical Data for {symbol} ###
- Fibonacci Levels: {key_fibonacci_levels}
- Future Price Predictions (approx.): {', '.join([f'${p:.2f}' for p in future_prices_rounded]) if future_prices_rounded else "Unavailable"}
- RSI Status: {rsi_status} (Latest Value: {round(rsi_last_value,2) if rsi_last_value else "N/A"})
- Moving Averages: {ma_summary}
- Bollinger Bands: {bollinger_summary}
- Candlestick Patterns: {pattern_summary}
- Support Levels: {support_levels if support_levels else "None detected"}
- Resistance Levels: {resistance_levels if resistance_levels else "None detected"}

### Instructions ###
1. Integrate all indicators (Fibonacci, RSI, Bollinger Bands, MAs, Candlestick Patterns) into a single cohesive analysis.
2. Provide a clear, step-by-step trading plan including:
   - Specific entry points (either exact prices or conditions like a break above/below a certain Fibonacci level).
   - Stop-loss levels (exact prices or conditions).
   - Take-profit targets with logical rationale.
   - Position sizing or risk management guidelines (e.g., risk as a percentage of capital, how to adjust position size).
   - A recommended time horizon for the trade (intraday, swing, short-term, or long-term).
3. Include scenario planning:
   - If price breaches a key Fibonacci or resistance level, how to adapt the strategy.
   - If RSI enters a clearly overbought/oversold range, how to respond.
   - If price interacts significantly with the Bollinger Bands (e.g., hitting upper band), what adjustments to make.
4. Justify each action using the technical data provided, and explain the logic in a professional manner.
5. Present the final recommendation as if advising a highly skilled trader who expects the best possible guidance. Provide a confident conclusion: Buy, Sell, or Hold, and under what conditions to exit or scale the position.

Your answer should read like a comprehensive, professional trading plan a seasoned hedge fund manager would deliver, complete with contingencies and confident directives.
"""

        # Call OpenAI API with the refined prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use gpt-4 or the best model you have access to.
            messages=[
                {"role": "system", "content": "You are a seasoned hedge fund manager and expert financial strategist."},
                {"role": "user", "content": prompt}
            ]
        )

        analysis = response['choices'][0]['message']['content'].strip()

        # Return results as before
        result = {
            "symbol": symbol,
            "fibonacci_levels": key_fibonacci_levels,
            "future_prices": future_prices_rounded,
            "rsi_status": rsi_status,
            "rsi_last_value": round(rsi_last_value, 2) if rsi_last_value else "N/A",
            "moving_averages": ma_summary,
            "bollinger_bands": bollinger_summary,
            "candlestick_patterns": pattern_summary,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "openai_analysis": analysis
        }

        return result

    except Exception as e:
        logging.error(f"Error generating OpenAI analysis: {e}")
        return {"error": "An error occurred while generating the analysis."}




# Flask route to display the homepage
@fundamentalanalysis_bp.route("/", methods=["GET", "POST"])
def index():
    symbol = None
    start_date_str = None
    end_date_str = None
    chart_json = None
    error = None

    # Default empty data structures
    fibonacci_levels = {}
    moving_averages = {}
    rsi = None
    future_prices = []
    confidence_intervals = []
    bollinger_bands = {}
    candlestick_patterns = {}
    support_resistance = {}
    openai_analysis = None

    try:
        if request.method == 'POST':
            # Fetch user inputs
            symbol = request.form.get('symbol', '').upper()
            start_date_str = request.form.get('start_date')
            end_date_str = request.form.get('end_date')

            # Validate required fields
            if not symbol or not start_date_str or not end_date_str:
                error = "All fields are required."
                return render_template('index.html', error=error)

            # Parse dates
            try:
                start_date = pd.to_datetime(start_date_str).date()
                end_date = pd.to_datetime(end_date_str).date()
            except Exception as e:
                logging.error(f"Date parsing error: {e}")
                error = "Invalid date format. Please use YYYY-MM-DD."
                return render_template('index.html', error=error)

            # Fetch data
            df = get_time_series_daily(symbol)
            if df is None or df.empty:
                error = f"Could not fetch data for symbol {symbol}. Please check the symbol or try again."
                return render_template('index.html', error=error)

            # Validate date range
            available_start = df['date'].min().date()
            available_end = df['date'].max().date()
            if start_date < available_start or end_date > available_end:
                error = f"Data is only available from {available_start} to {available_end}. Please adjust your date range."
                return render_template('index.html', error=error)

            # Filter data by date range
            df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
            if df.empty:
                error = f"No data available for the selected date range: {start_date} to {end_date}."
                return render_template('index.html', error=error)

            # Compute technical indicators and analysis
            fibonacci_levels = calculate_fibonacci_levels(df)
            moving_averages_full = fetch_moving_average(symbol, time_period=20, ma_type='SMA') or {}
            rsi = fetch_rsi(symbol)
            future_prices, confidence_intervals = predict_future_price(df)
            bollinger_bands_full = fetch_bollinger_bands(symbol) or {}
            candlestick_patterns = detect_candlestick_patterns(df)
            support_resistance = calculate_support_resistance(df)

            # Generate OpenAI analysis
            openai_analysis_result = generate_openai_analysis(
                symbol, fibonacci_levels, future_prices, moving_averages_full, rsi, bollinger_bands_full, candlestick_patterns,
                support_resistance
            )

            # Extract OpenAI analysis text
            openai_analysis_text = openai_analysis_result.get('openai_analysis', '')

            # Convert the OpenAI analysis (which contains markdown headings) to HTML
            # If you don't want markdown conversion, you can do:
            # openai_analysis = openai_analysis_text.replace('\n', '<br>')
            openai_analysis = markdown(openai_analysis_text)

            # Round and clean up values for display
            # Fibonacci levels
            fibonacci_levels = {level: round(value, 2) for level, value in fibonacci_levels.items()}

            # Moving Averages: The original data likely gives a series of MA values.
            # We'll assume moving_averages_full is something like {'SMA_20': [values...]}.
            # If you just want to show the latest value:
            moving_averages = {
                ma_name: (round(vals[-1], 4) if vals and len(vals) > 0 else "N/A")
                for ma_name, vals in moving_averages_full.items()
            }

            # RSI
            rsi_last_value = round(rsi[-1], 4) if rsi else None
            rsi_status = (
                "Overbought" if rsi_last_value and rsi_last_value > 70 else
                "Oversold" if rsi_last_value and rsi_last_value < 30 else
                "Neutral" if rsi_last_value is not None else "N/A"
            )

            # Bollinger Bands: round the last value for display
            # Ensure bollinger_bands_full has data (list of values for each band)
            if bollinger_bands_full.get('upper_band') and bollinger_bands_full.get('middle_band') and bollinger_bands_full.get('lower_band'):
                bollinger_bands = {
                    'Upper Band': round(bollinger_bands_full['upper_band'][-1], 4),
                    'Middle Band': round(bollinger_bands_full['middle_band'][-1], 4),
                    'Lower Band': round(bollinger_bands_full['lower_band'][-1], 4)
                }
            else:
                bollinger_bands = {
                    'Upper Band': "N/A",
                    'Middle Band': "N/A",
                    'Lower Band': "N/A"
                }

            # Support and Resistance levels
            support_levels = [round(lvl, 2) for lvl in support_resistance.get('support_levels', [])]
            resistance_levels = [round(lvl, 2) for lvl in support_resistance.get('resistance_levels', [])]

            # Create Plotly chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Close Price'), secondary_y=False)

            # Add Fibonacci lines as reference lines
            for level, price in fibonacci_levels.items():
                fig.add_hline(y=price, annotation_text=f"{level}: {price:.2f}", line_dash="dot")

            # If you want to plot Moving Averages and Bollinger Bands as lines on the chart:
            # Ensure you have full series available. For example:
            # if 'SMA_20' in moving_averages_full and len(moving_averages_full['SMA_20']) == len(df):
            #     fig.add_trace(go.Scatter(x=df['date'], y=moving_averages_full['SMA_20'], mode='lines', name='SMA_20'),
            #                   secondary_y=False)

            # Similarly, for Bollinger Bands if full series is available:
            # if bollinger_bands_full.get('upper_band') and len(bollinger_bands_full['upper_band']) == len(df):
            #     fig.add_trace(go.Scatter(x=df['date'], y=bollinger_bands_full['upper_band'], mode='lines', name='Upper BB'),
            #                   secondary_y=False)
            #     fig.add_trace(go.Scatter(x=df['date'], y=bollinger_bands_full['middle_band'], mode='lines', name='Middle BB'),
            #                   secondary_y=False)
            #     fig.add_trace(go.Scatter(x=df['date'], y=bollinger_bands_full['lower_band'], mode='lines', name='Lower BB'),
            #                   secondary_y=False)

            fig.update_layout(
                title=f'Stock Price and Analysis for {symbol}',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                xaxis_rangeslider_visible=False
            )

            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) if fig else None

            return render_template(
                'index.html',
                symbol=symbol,
                start_date=start_date_str,
                end_date=end_date_str,
                chart_json=chart_json,
                fibonacci_levels=fibonacci_levels,
                moving_averages=moving_averages,
                rsi_last_value=rsi_last_value,
                rsi_status=rsi_status,
                bollinger_bands=bollinger_bands,
                candlestick_patterns=candlestick_patterns,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                openai_analysis=openai_analysis
            )
    except Exception as e:
        logging.error(f"Unexpected error in index function: {e}")
        error = "An unexpected error occurred. Please try again."
        return render_template('index.html', error=error)

    # For GET requests, just render the form
    return render_template('index.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)



