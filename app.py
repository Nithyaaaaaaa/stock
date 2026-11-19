"""
Predictive Insights – Next-Day Stock Price & Direction Forecasting
Production Flask Web Application & Serverless REST API (Vercel Entry Point).
Exports top-level WSGI instance: `app`.
"""

import os
import sys

# Ensure project root is in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Threading safety
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, render_template, request, send_from_directory

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data
from src.data.calendar import get_next_trading_day
from src.features.technical import compute_all_candidate_features
from src.models.trees import CatBoostEXP2Forecaster

# Top-level Flask application instance required by Vercel
app = Flask(__name__, static_folder="static", template_folder="templates")

POPULAR_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Alphabet / Google (GOOGL)": "GOOGL",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "AMD (AMD)": "AMD",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "Other / Custom Ticker...": "CUSTOM"
}

# Reverse mapping for asset names
ASSET_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NFLX": "Netflix Inc.",
    "AMD": "Advanced Micro Devices",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys Limited"
}

# In-memory TTL data cache
_DATA_CACHE = {}
_CACHE_TTL = 3600  # 1 hour


def load_clean_data(ticker: str):
    """Retrieves and cleans historical market data with TTL in-memory caching."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Please provide a valid, non-empty stock ticker.")
        
    now = time.time()
    if ticker in _DATA_CACHE:
        df_cached, bench_cached, timestamp = _DATA_CACHE[ticker]
        if now - timestamp < _CACHE_TTL:
            return df_cached, bench_cached
            
    benchmark_ticker = "^NSEI" if ticker.endswith((".NS", ".BO")) else "SPY"
    start_date = "2015-01-01" if ticker == "AAPL" else "2020-01-01"
    raw_target, raw_bench = fetch_market_data(
        ticker=ticker,
        benchmark_ticker=benchmark_ticker,
        start=start_date
    )
    df_clean = validate_and_clean_data(raw_target, raw_bench)
    _DATA_CACHE[ticker] = (df_clean, benchmark_ticker, now)
    return df_clean, benchmark_ticker


def run_next_day_forecast(df_clean: pd.DataFrame, ticker: str = "AAPL"):
    """
    Fits the EXP2 champion CatBoost model on historical data.
    - Preserves continuous log return regression for price forecasting.
    - Learns an optimal return threshold tau* for balanced directional prediction.
    - Forecasts strictly for the NEXT trading session (no multi-day rollout).
    """
    stationary_cols = [
        "Return_Lag_1", "Return_Lag_5", "Log_Return_Lag_1", "Log_Return_Lag_5",
        "Ratio_Close_SMA_5", "Ratio_Close_SMA_20", "Ratio_Close_SMA_50", "Ratio_SMA_5_20",
        "Ratio_Close_EMA_12", "Ratio_Close_EMA_26", "MACD_Normalized",
        "RSI_14_Centered", "Stoch_K_14", "High_Low_Ratio", "Close_Open_Ratio",
        "Vol_10d", "Vol_20d", "Vol_60d", "NATR_14", "Bollinger_Pct_B",
        "Log_Volume_Change", "Volume_Ratio_SMA_20",
        "Benchmark_Return_Lag_1", "Excess_Return_Over_Benchmark", "Rolling_Beta_60"
    ]
    
    # Feature computation on historical clean data
    all_feats = compute_all_candidate_features(df_clean)
    usable_feats = all_feats.iloc[65:-1]
    feature_cols = [c for c in stationary_cols if c in usable_feats.columns]
    
    # For AAPL (primary asset), train strictly on the development partition
    # (prior to diagnostic holdout start 2024-07-08), guaranteeing zero holdout leakage.
    holdout_start = pd.Timestamp("2024-07-08")
    if ticker.upper() == "AAPL" and usable_feats.index[0] < pd.Timestamp("2016-01-01"):
        dev_feats = usable_feats.loc[usable_feats.index < holdout_start].iloc[:-1]  # N=2,326 sessions
    else:
        dev_feats = usable_feats

    # Fit scaler strictly on training features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(dev_feats[feature_cols].values)
    y_train = dev_feats["Target_Next_Log_Return"].values
    
    # Fit EXP2 Champion Model (CatBoost with Train-Tuned Return Threshold)
    model = CatBoostEXP2Forecaster(iterations=120, depth=4, learning_rate=0.02, random_seed=42)
    model.fit(X_train, y_train)
    
    # Origin of forecast: latest observed session
    last_date = df_clean.index[-1]
    last_close = float(df_clean["Close"].iloc[-1])
    
    # Next trading session date using exchange calendar (skipping weekends & holidays)
    next_trading_day = get_next_trading_day(last_date, ticker=ticker)
    
    # Extract latest feature vector (observed at cutoff t)
    latest_vec = all_feats.iloc[-1:][feature_cols].values
    scaled_latest = scaler.transform(latest_vec)
    
    # 1. Continuous price forecasting (preserves regression precision)
    pred_log_ret = float(model.predict_log_returns(scaled_latest)[0])
    pred_price = float(last_close * np.exp(pred_log_ret))
    
    # Expected change
    price_change = pred_price - last_close
    pct_change = (price_change / last_close) * 100.0
    
    # 2. Directional classification via learned EXP2 threshold tau*
    learned_tau = float(model.learned_threshold)
    pred_dir_int = int(model.predict_direction(scaled_latest)[0])
    pred_direction = "UP" if pred_dir_int == 1 else "DOWN"

    # Explicit distinction between price movement and EXP2 classification
    price_movement = (
        "PRICE INCREASE" if price_change > 1e-6
        else "PRICE DECREASE" if price_change < -1e-6
        else "NO PRICE CHANGE"
    )
    exp2_classification = (
        "UP — Above threshold" if pred_log_ret > learned_tau
        else "Below UP threshold"
    )

    # Historical 60 sessions for chart
    hist_recent = df_clean.iloc[-60:]
    historical_dates = [d.strftime("%Y-%m-%d") for d in hist_recent.index]
    historical_prices = [round(float(p), 2) for p in hist_recent["Close"]]
    
    return {
        "ticker": ticker,
        "last_date": last_date,
        "last_close": last_close,
        "next_date": next_trading_day,
        "predicted_price": pred_price,
        "price_change": price_change,
        "pct_change": pct_change,
        "predicted_direction": pred_direction,
        "price_movement": price_movement,
        "exp2_classification": exp2_classification,
        "learned_threshold": learned_tau,
        "pred_log_return": pred_log_ret,
        "historical_dates": historical_dates,
        "historical_prices": historical_prices
    }


# ============================================================================
# Flask HTTP Routes
# ============================================================================

@app.route("/")
def index():
    """Serves the main HTML application."""
    return render_template("index.html", stocks=POPULAR_STOCKS)


@app.route("/api/stocks")
def get_stocks():
    """Returns supported popular stock tickers."""
    return jsonify(POPULAR_STOCKS)


@app.route("/api/health")
def health():
    """Vercel / system health check endpoint."""
    return jsonify({
        "status": "healthy",
        "application": "Predictive Insights",
        "model_architecture": "CatBoost EXP2 Regressor with Train-Tuned Hurdle"
    })


@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    """
    API endpoint executing EXP2 next-trading-day forecast.
    Accepts GET query param ?ticker=... or POST JSON payload {"ticker": "..."}.
    """
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        ticker = payload.get("ticker", "")
    else:
        ticker = request.args.get("ticker", "")

    ticker = str(ticker).strip().upper()
    if not ticker or ticker == "CUSTOM":
        return jsonify({
            "success": False,
            "error": "Please provide a valid stock ticker symbol (e.g. AAPL, NVDA, RELIANCE.NS)."
        }), 400

    try:
        df_clean, benchmark = load_clean_data(ticker)
        res = run_next_day_forecast(df_clean, ticker=ticker)

        curr_sym = "₹" if ticker.endswith((".NS", ".BO")) else "$"
        asset_name = ASSET_NAMES.get(ticker, ticker)

        return jsonify({
            "success": true_bool(True),
            "ticker": ticker,
            "asset_name": asset_name,
            "currency_symbol": curr_sym,
            "last_date": res["last_date"].strftime("%b %d, %Y"),
            "next_date": res["next_date"].strftime("%Y-%m-%d (%a)"),
            "last_close": round(res["last_close"], 2),
            "predicted_price": round(res["predicted_price"], 2),
            "price_change": round(res["price_change"], 2),
            "pct_change": round(res["pct_change"], 2),
            "predicted_direction": res["predicted_direction"],
            "price_movement": res["price_movement"],
            "exp2_classification": res["exp2_classification"],
            "learned_threshold": round(res["learned_threshold"], 6),
            "pred_log_return": round(res["pred_log_return"], 6),
            "historical_dates": res["historical_dates"],
            "historical_prices": res["historical_prices"]
        })
    except Exception as e:
        err_str = str(e)
        if "No market data returned" in err_str or "verify the ticker" in err_str:
            user_msg = f"Could not retrieve market data for '{ticker}'. Please ensure it is a valid Yahoo Finance ticker symbol."
        else:
            user_msg = f"Forecasting error: {err_str}"
        return jsonify({"success": False, "error": user_msg}), 400


def true_bool(val: bool) -> bool:
    return bool(val)


if __name__ == "__main__":
    # Local development server
    port = int(os.environ.get("PORT", 8501))
    print(f"Starting Predictive Insights Flask App on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
