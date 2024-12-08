import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from arch import arch_model
from flask import Blueprint, render_template
from scipy.stats import linregress
import asyncio



# Alpha Vantage API Key
API_KEY = "URZENY6PEJWF13MM"

# ---- Data Fetching ----
import aiohttp


from flask import Blueprint, render_template

option_trading_bp = Blueprint("option_trading", __name__, template_folder="templates")


@option_trading_bp.route("/")
def home():
    return render_template("option_trading/index.html")



# Async function with retries
async def fetch_data_with_retry(session, url, retries=3):
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Attempt {attempt + 1}/{retries} failed. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed due to {e}. Retrying...")
    print("Failed to fetch data after multiple attempts.")
    return None

# Modify fetch_options_data to use aiohttp
async def fetch_options_data(symbol):
    url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&apikey={API_KEY}'
    async with aiohttp.ClientSession() as session:
        data = await fetch_data_with_retry(session, url)
    if data and "data" in data and isinstance(data["data"], list) and data["data"]:
        return pd.DataFrame(data["data"])
    else:
        print("No options data found or data structure has changed.")
        return None

# Modify fetch_stock_data to use aiohttp
async def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'
    async with aiohttp.ClientSession() as session:
        data = await fetch_data_with_retry(session, url)
    if data:
        try:
            latest_data = list(data["Time Series (Daily)"].values())[0]
            stock_price = float(latest_data["4. close"])
            prices = [float(item["4. close"]) for item in data["Time Series (Daily)"].values()]
            returns = np.diff(np.log(prices))  # Log returns
            historical_volatility = np.std(returns) * np.sqrt(252)  # Annualize
            return stock_price, historical_volatility, returns
        except (KeyError, IndexError):
            print("Error fetching stock price and volatility data.")
    return None, None, None


# ---- Analytics Functions ----
def add_volatility_metrics(df):
    df['implied_volatility'] = pd.to_numeric(df['implied_volatility'], errors='coerce')
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['iv_ma_7'] = df['implied_volatility'].rolling(window=7).mean()
    df['iv_ma_30'] = df['implied_volatility'].rolling(window=30).mean()
    call_iv = df[df['type'] == 'call']['implied_volatility']
    put_iv = df[df['type'] == 'put']['implied_volatility']
    df['volatility_skew'] = call_iv.mean() - put_iv.mean()
    return df

# ---- Visualization ----
def plot_greek_surface(df, greek="delta"):
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
    df[greek] = pd.to_numeric(df[greek], errors='coerce')
    df = df.dropna(subset=['strike', 'expiration', greek])
    greek_data = df.pivot_table(index='strike', columns='expiration', values=greek, aggfunc='mean')
    X, Y = np.meshgrid(greek_data.columns, greek_data.index)
    Z = greek_data.values

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])
    fig.update_layout(title=f"{greek.title()} Surface Plot", scene=dict(
        xaxis_title="Expiration Date",
        yaxis_title="Strike Price",
        zaxis_title=f"{greek.title()} Value"))
    fig.show()

def plot_iv_heatmap(df):
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
    df['implied_volatility'] = pd.to_numeric(df['implied_volatility'], errors='coerce')
    iv_data = df.pivot_table(index='strike', columns='expiration', values='implied_volatility', aggfunc='mean')

    fig = go.Figure(data=go.Heatmap(
        z=iv_data.values,
        x=iv_data.columns,
        y=iv_data.index,
        colorscale="Viridis"
    ))
    fig.update_layout(title="Implied Volatility Heatmap", xaxis_title="Expiration Date", yaxis_title="Strike Price")
    fig.show()

# ---- Options Strategy Simulations ----
def simulate_covered_call(stock_price, strike_price, premium, stock_quantity=100):
    x = np.linspace(stock_price * 0.5, stock_price * 1.5, 100)
    payoff_stock = (x - stock_price) * stock_quantity
    payoff_call = np.where(x > strike_price, -((x - strike_price) * stock_quantity) + premium, premium)
    payoff_covered_call = payoff_stock + payoff_call

    plt.figure(figsize=(10, 6))
    plt.plot(x, payoff_stock, label="Stock Only", linestyle="--")
    plt.plot(x, payoff_call, label="Call Only", linestyle="--")
    plt.plot(x, payoff_covered_call, label="Covered Call Payoff")
    plt.title("Covered Call Payoff")
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Profit/Loss")
    plt.legend()
    plt.show()

# ---- Predictive Modeling for Volatility ----
def forecast_volatility(returns, horizon=5):
    model = arch_model(returns, vol='Garch', p=1, q=1)
    fitted_model = model.fit(disp="off")
    forecast = fitted_model.forecast(horizon=horizon)
    forecast_variance = forecast.variance.iloc[-1].values

    plt.figure(figsize=(10, 6))
    plt.plot(forecast_variance, label="Forecasted Volatility", marker="o")
    plt.title("GARCH Forecasted Volatility")
    plt.xlabel("Days")
    plt.ylabel("Volatility (Variance)")
    plt.legend()
    plt.show()
    return forecast_variance

# ---- Portfolio Tracking and Risk Analysis ----
def calculate_portfolio_greeks(portfolio):
    greek_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
    for col in greek_cols:
        portfolio[col] = pd.to_numeric(portfolio[col], errors='coerce').fillna(0)
    portfolio_summary = portfolio[greek_cols].sum()
    print("\nPortfolio-Level Greeks:")
    print(portfolio_summary)
    return portfolio_summary

def calculate_var(portfolio, confidence_level=0.95):
    if portfolio['implied_volatility'].isna().sum() > len(portfolio) * 0.2:
        print("Insufficient implied volatility data for reliable VaR calculation.")
        return None
    portfolio['iv_return'] = portfolio['implied_volatility'].pct_change().dropna()
    if portfolio['iv_return'].dropna().shape[0] < 2:
        print("Not enough data for VaR calculation.")
        return None

    var_value = np.percentile(portfolio['iv_return'].dropna(), (1 - confidence_level) * 100)
    print(f"\nValue at Risk (VaR) at {int(confidence_level * 100)}% confidence level: {var_value}")
    return var_value

# ---- Alerts ----
def check_alerts(df, iv_threshold=0.5, delta_threshold=0.5):
    """Check and alert based on implied volatility and delta thresholds."""
    df['implied_volatility'] = pd.to_numeric(df['implied_volatility'], errors='coerce')
    df['delta'] = pd.to_numeric(df['delta'], errors='coerce')

    iv_alerts = df[df['implied_volatility'] > iv_threshold]
    delta_alerts = df[df['delta'].abs() > delta_threshold]

    if not iv_alerts.empty:
        print(f"\nALERT: Implied Volatility exceeds {iv_threshold}")
        print(iv_alerts[['contractID', 'symbol', 'strike', 'expiration', 'implied_volatility']])

    if not delta_alerts.empty:
        print(f"\nALERT: Delta exceeds +/- {delta_threshold}")
        print(delta_alerts[['contractID', 'symbol', 'strike', 'expiration', 'delta']])

def monte_carlo_stochastic_volatility(stock_price, strike_price, risk_free_rate, time_to_expiration, base_volatility,
                                      vol_of_vol=0.1, option_type="call", num_simulations=10000):
    """Monte Carlo simulation with stochastic volatility for realistic option pricing."""
    dt = time_to_expiration / 252
    prices = np.full(num_simulations, stock_price)
    volatilities = np.full(num_simulations, base_volatility)

    for t in range(int(252 * time_to_expiration)):
        random_shocks = np.random.normal(size=num_simulations)
        volatilities += vol_of_vol * np.sqrt(dt) * random_shocks
        prices *= np.exp((risk_free_rate - 0.5 * volatilities**2) * dt + volatilities * np.sqrt(dt) * random_shocks)

    if option_type == "call":
        payoffs = np.maximum(prices - strike_price, 0)
    elif option_type == "put":
        payoffs = np.maximum(strike_price - prices, 0)

    option_price = np.exp(-risk_free_rate * time_to_expiration) * np.mean(payoffs)
    print(f"Estimated {option_type.capitalize()} Option Price with Stochastic Volatility: {option_price}")
    return option_price


# ---- Monte Carlo Simulation for Option Pricing ----
def monte_carlo_option_pricing(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, option_type="call", num_simulations=10000):
    """Run Monte Carlo simulations to estimate the price of a call or put option."""
    drift = (risk_free_rate - 0.5 * volatility ** 2) * time_to_expiration
    random_component = volatility * np.sqrt(time_to_expiration) * np.random.normal(size=num_simulations)
    simulated_prices = stock_price * np.exp(drift + random_component)

    if option_type == "call":
        payoffs = np.maximum(simulated_prices - strike_price, 0)
    elif option_type == "put":
        payoffs = np.maximum(strike_price - simulated_prices, 0)

    option_price = np.exp(-risk_free_rate * time_to_expiration) * np.mean(payoffs)
    print(f"Estimated {option_type.capitalize()} Option Price (Monte Carlo): {option_price}")
    return option_price

# ---- Hedging Recommendation ----
def hedging_recommendation(portfolio, target_delta=0):
    """Recommend delta-neutral hedging adjustments for the portfolio."""
    portfolio_delta = portfolio['delta'].sum()
    adjustment_needed = target_delta - portfolio_delta

    print("\nDelta Neutral Hedging Recommendation:")
    if adjustment_needed > 0:
        print(f"Consider buying {adjustment_needed} shares of the underlying asset to balance delta.")
    elif adjustment_needed < 0:
        print(f"Consider selling {-adjustment_needed} shares of the underlying asset to balance delta.")
    else:
        print("Portfolio is already delta-neutral.")

# ---- Financial Metrics ----
def calculate_max_drawdown(returns):
    """Calculate Maximum Drawdown and Calmar Ratio from returns."""
    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = peak - cumulative_returns
    max_drawdown = np.max(drawdown)
    calmar_ratio = np.mean(returns) / max_drawdown if max_drawdown != 0 else np.nan

    print(f"Maximum Drawdown: {max_drawdown}")
    print(f"Calmar Ratio: {calmar_ratio}")
    return max_drawdown, calmar_ratio

def calculate_alpha_beta(returns, market_returns):
    """Calculate Alpha and Beta for strategy returns relative to the market."""
    slope, intercept, _, _, _ = linregress(market_returns, returns)
    beta = slope
    alpha = intercept

    print(f"Alpha: {alpha}")
    print(f"Beta: {beta}")
    return alpha, beta

# ---- Expanded Monte Carlo Simulation ----
def monte_carlo_with_correlation(stock_price, strike_price, risk_free_rate, time_to_expiration, volatility, correlation=0, option_type="call", num_simulations=10000):
    """Monte Carlo simulation with stock-market correlation for realistic option pricing."""
    drift = (risk_free_rate - 0.5 * volatility ** 2) * time_to_expiration
    random_component = volatility * np.sqrt(time_to_expiration) * (np.random.normal(size=num_simulations) + correlation)
    simulated_prices = stock_price * np.exp(drift + random_component)

    if option_type == "call":
        payoffs = np.maximum(simulated_prices - strike_price, 0)
    elif option_type == "put":
        payoffs = np.maximum(strike_price - simulated_prices, 0)

    option_price = np.exp(-risk_free_rate * time_to_expiration) * np.mean(payoffs)
    print(f"Estimated {option_type.capitalize()} Option Price with Correlation (Monte Carlo): {option_price}")
    return option_price


# ---- Backtesting Function ----
async def backtest_option_strategy(symbol, strategy="covered_call"):
    """Backtest an option strategy using historical data with dynamic strike and premium."""
    stock_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'

    # Use aiohttp to fetch data asynchronously
    async with aiohttp.ClientSession() as session:
        data = await fetch_data_with_retry(session, stock_url)

    if data:
        try:
            # Prepare historical data
            prices = [float(value["4. close"]) for value in data["Time Series (Daily)"].values()]
            dates = [pd.to_datetime(date) for date in data["Time Series (Daily)"].keys()]
            historical_df = pd.DataFrame({"Date": dates, "Close": prices}).set_index("Date").sort_index()

            # Auto-detect start and end dates for backtesting
            start_date = historical_df.index.min()
            end_date = historical_df.index.max()
            historical_df = historical_df.loc[start_date:end_date]

            if historical_df.empty:
                print(f"No data available for {symbol} in the specified date range ({start_date} to {end_date}).")
                return []

            # Set dynamic strike price and premium
            option_strike_price = historical_df['Close'].iloc[0] * (1 + np.mean(historical_df['Close'].pct_change()) * 2)
            option_premium = historical_df['Close'].std()  # Using historical std dev as an estimate for premium
            stock_purchase_price = historical_df['Close'].iloc[0]

            # Initialize returns tracking
            strategy_returns = []
            for _, row in historical_df.iterrows():
                stock_price = row['Close']

                # Calculate dynamic covered call returns
                stock_profit = stock_price - stock_purchase_price
                call_payoff = option_premium if stock_price <= option_strike_price else option_premium - (
                            stock_price - option_strike_price)
                total_payoff = stock_profit + call_payoff
                strategy_returns.append(total_payoff)

            if strategy_returns:
                total_return = sum(strategy_returns)
                avg_return = total_return / len(strategy_returns)
                print(
                    f"Backtest Result for {strategy} Strategy: Total Return = {total_return}, Average Return = {avg_return}")
                calculate_sharpe_sortino(strategy_returns)
                plot_cumulative_returns(strategy_returns)

            return strategy_returns  # Ensure strategy_returns is returned
        except KeyError:
            print("Error processing historical data. Data structure might have changed.")
            return []
    else:
        print("Failed to fetch historical data.")
        return []


def simulate_protective_put(stock_price, strike_price, premium_put, stock_quantity=100):
    """Simulate a protective put payoff."""
    x = np.linspace(stock_price * 0.5, stock_price * 1.5, 100)
    payoff_stock = (x - stock_price) * stock_quantity
    payoff_put = np.where(x < strike_price, ((strike_price - x) * stock_quantity) - premium_put, -premium_put)
    payoff_protective_put = payoff_stock + payoff_put

    plt.figure(figsize=(10, 6))
    plt.plot(x, payoff_stock, label="Stock Only", linestyle="--")
    plt.plot(x, payoff_put, label="Put Only", linestyle="--")
    plt.plot(x, payoff_protective_put, label="Protective Put Payoff")
    plt.title("Protective Put Payoff")
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Profit/Loss")
    plt.legend()
    plt.show()

from scipy.stats import linregress

def calculate_treynor_alpha(returns, market_returns, risk_free_rate=0.05):
    """Calculate Treynor Ratio and Jensen's Alpha."""
    beta, alpha, _, _, _ = linregress(market_returns, returns)
    treynor_ratio = (np.mean(returns) - risk_free_rate) / beta if beta else np.nan

    print(f"Treynor Ratio: {treynor_ratio}")
    print(f"Jensen's Alpha: {alpha}")
    return treynor_ratio, alpha

# Additional helper functions for performance metrics
def calculate_sharpe_sortino(returns, risk_free_rate=0.05):
    """Calculate Sharpe and Sortino ratios based on strategy returns."""
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    downside_dev = np.std([x for x in returns if x < 0])

    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev else 0
    sortino_ratio = (mean_return - risk_free_rate) / downside_dev if downside_dev else 0

    print(f"\nSharpe Ratio: {sharpe_ratio}")
    print(f"Sortino Ratio: {sortino_ratio}")

    # Ensure the function returns both ratios
    return sharpe_ratio, sortino_ratio


def plot_cumulative_returns(strategy_returns):
    """Plot cumulative returns of the strategy over time."""
    cumulative_returns = np.cumsum(strategy_returns)
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Cumulative Returns")
    plt.title("Cumulative Returns of Strategy")
    plt.xlabel("Periods")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()
import openai

# Add your OpenAI API key here
openai.api_key = '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'  # Replace with your actual OpenAI API key

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
        print(f"Error calling OpenAI API: {e}")
        return "Error generating analysis."

# Function to prepare and call OpenAI API with financial data
def generate_financial_analysis(vr, max_drawdown, calmar_ratio, alpha, beta, delta_hedge, call_price, put_price, sharpe_ratio, sortino_ratio, total_return, avg_return):
    # Create a prompt for OpenAI to analyze the data and provide buy/sell scores and sentiment analysis.
    prompt = f"""
    You are a financial analyst. Analyze the following financial metrics for a stock and provide clear buy and sell scores on a scale from 1 to 5, where 5 represents a strong recommendation (e.g., a strong buy or strong sell) and 1 represents a weak recommendation (e.g., minimal buy or minimal sell). Also, provide a concise analysis of the current market sentiment based on options data.

    **Financial Metrics:**   
    - Maximum Drawdown: {max_drawdown}
    - Calmar Ratio: {calmar_ratio}
    - Alpha: {alpha}
    - Beta: {beta}
    - Monte Carlo Estimated Call Option Price with Market Correlation: {call_price}
    - Monte Carlo Estimated Put Option Price with Market Correlation: {put_price}
    - Sharpe Ratio: {sharpe_ratio}
    - Sortino Ratio: {sortino_ratio}
    - Backtest Covered Call Strategy: Total Return = {total_return}, Average Return = {avg_return}

    **Market Sentiment Analysis Based on Options Data:**
    - Interpret the relationship between call and put prices to gauge market sentiment. If the call option price is significantly higher, it may suggest bullish sentiment; if the put price is higher, it may suggest bearish sentiment.
    - Based on Greeks (e.g., Delta, Gamma), indicate whether there is significant market demand or hedging activity that could influence sentiment.

    **Score and Recommendations:**
    - Assign a score out of 5 for a buying recommendation based on these metrics, with an explanation for the score.
    - Assign a score out of 5 for a selling recommendation based on these metrics, with an explanation for the score.
    - Offer a final recommendation on whether buying or selling is advisable based on the overall analysis.

    Provide a concise interpretation with the buy and sell scores and an assessment of the market sentiment.
    """
    # Call the function
    return call_openai_api(prompt)



def calculate_option_investment_profit(stock_price, call_price, put_price, call_strike, put_strike, investment_amount=1000):
    """Calculate the number of call and put option contracts you can buy and potential profits."""
    contract_size = 100  # 1 option contract covers 100 shares

    # Calculate the number of call and put contracts that can be bought within the investment amount
    num_call_contracts = int(investment_amount / (call_price * contract_size))
    num_put_contracts = int(investment_amount / (put_price * contract_size))

    # If no contracts can be bought, calculate fractional returns instead
    call_fraction = investment_amount / (call_price * contract_size) if call_price > 0 else 0
    put_fraction = investment_amount / (put_price * contract_size) if put_price > 0 else 0

    # Define scenarios for price movement: 20% up, 10% up, no change, 10% down, and 20% down
    price_movements = [1.2, 1.1, 1.0, 0.9, 0.8]
    scenario_prices = stock_price * np.array(price_movements)

    # Calculate potential profits for each scenario
    call_profits = []
    put_profits = []
    for future_price in scenario_prices:
        call_profit = (max(future_price - call_strike, 0) * contract_size * call_fraction) - (call_price * contract_size * call_fraction)
        put_profit = (max(put_strike - future_price, 0) * contract_size * put_fraction) - (put_price * contract_size * put_fraction)
        call_profits.append(call_profit)
        put_profits.append(put_profit)

    # Output the results
    print("\nOption Investment Analysis:")
    print(f"Investment Amount: ${investment_amount}")
    print(f"Number of Call Contracts (strike {call_strike}): {num_call_contracts} (fraction: {call_fraction:.2f})")
    print(f"Number of Put Contracts (strike {put_strike}): {num_put_contracts} (fraction: {put_fraction:.2f})")
    print("\nPrice Movements and Potential Profits:")
    for i, future_price in enumerate(scenario_prices):
        print(f"Future Stock Price: ${future_price:.2f}")
        print(f"  Call Profit: ${call_profits[i]:.2f}")
        print(f"  Put Profit: ${put_profits[i]:.2f}")



# ---- Main Function ----
import asyncio  # Ensure asyncio is imported

# ---- Main Function ----
from flask import request, jsonify  # Ensure these are imported

@option_trading_bp.route("/run_main", methods=["POST"])
async def run_main():
    try:
        # Get symbols from the POST request
        symbols = request.form.get("symbols", "").split(",")
        symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]

        if not symbols:
            return jsonify({"error": "No stock symbols provided."}), 400

        results = []

        # Loop through each symbol and perform the analysis
        for symbol in symbols:
            print(f"\n--- Analyzing {symbol} ---")

            # Step 1: Fetch stock data
            stock_price, historical_volatility, returns = await fetch_stock_data(symbol)
            if stock_price is None or historical_volatility is None:
                results.append({"symbol": symbol, "error": "Failed to fetch stock data."})
                continue

            # Step 2: Fetch options data
            df = await fetch_options_data(symbol)
            if df is None or df.empty:
                results.append({"symbol": symbol, "error": "No options data found."})
                continue

            # Step 3: Add volatility metrics to options data
            df = add_volatility_metrics(df)

            # Step 4: Run options strategy simulations (e.g., Covered Call)
            simulate_covered_call(
                stock_price=stock_price,
                strike_price=stock_price * 1.1,
                premium=2
            )

            # Step 5: Forecast volatility using GARCH Model
            if returns is not None and len(returns) > 1:
                forecast_volatility(returns, horizon=5)
            else:
                print("Insufficient historical returns for volatility forecasting.")

            # Step 6: Visualization of Greeks and Implied Volatility
            plot_greek_surface(df, greek="delta")
            plot_iv_heatmap(df)

            # Step 7: Generate alerts for implied volatility and delta thresholds
            check_alerts(df, iv_threshold=0.6, delta_threshold=0.5)

            # Step 8: Portfolio Tracking & Risk Analysis
            portfolio_summary = calculate_portfolio_greeks(df)
            value_at_risk = calculate_var(df, confidence_level=0.95)

            # Step 9: Additional Financial Metrics
            max_drawdown, calmar_ratio = calculate_max_drawdown(returns)
            market_returns = np.random.normal(0, 0.01, len(returns))  # Placeholder for market returns
            alpha, beta = calculate_alpha_beta(returns, market_returns)

            # Step 10: Delta Neutral Hedging Recommendations
            hedging_recommendation(df, target_delta=0)

            # Step 11: Monte Carlo Simulation for Option Pricing
            risk_free_rate = 0.05  # Standard risk-free rate
            strike_price = stock_price * 1.1  # Strike price set to 10% above stock price
            time_to_expiration = 1  # Assume 1 year to expiration

            call_price = monte_carlo_with_correlation(
                stock_price, strike_price, risk_free_rate, time_to_expiration,
                historical_volatility, correlation=0.5, option_type="call"
            )
            put_price = monte_carlo_with_correlation(
                stock_price, strike_price, risk_free_rate, time_to_expiration,
                historical_volatility, correlation=0.5, option_type="put"
            )

            # Step 12: Evaluate Option Investment Potential
            calculate_option_investment_profit(
                stock_price, call_price, put_price, strike_price, strike_price, investment_amount=1000
            )

            # Step 13: Backtest Covered Call Strategy
            strategy_returns = await backtest_option_strategy(symbol, strategy="covered_call")
            if strategy_returns:
                sharpe_ratio, sortino_ratio = calculate_sharpe_sortino(strategy_returns)
                total_return = sum(strategy_returns)
                avg_return = total_return / len(strategy_returns)

                # Prepare OpenAI API analysis
                delta_hedge = portfolio_summary.get("delta", 0)
                analysis = generate_financial_analysis(
                    vr=value_at_risk,
                    max_drawdown=max_drawdown,
                    calmar_ratio=calmar_ratio,
                    alpha=alpha,
                    beta=beta,
                    delta_hedge=delta_hedge,
                    call_price=call_price,
                    put_price=put_price,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    total_return=total_return,
                    avg_return=avg_return
                )

                results.append({
                    "symbol": symbol,
                    "stock_price": stock_price,
                    "historical_volatility": historical_volatility,
                    "call_price": call_price,
                    "put_price": put_price,
                    "analysis": analysis
                })
            else:
                results.append({"symbol": symbol, "error": "Failed to generate strategy returns."})

        return jsonify({"results": results})

    except Exception as e:
        print(f"Error in run_main: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    asyncio.run(run_main())

