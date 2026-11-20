"""
Predictive Insights – Next-Day Stock Price & Direction Forecasting
Streamlit Web Application: Next-Trading-Day Only (EXP2 Champion Architecture).
"""

import os
import sys

# Ensure project root is in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Safe OpenMP / threading configuration
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import streamlit as st

from src.data.fetcher import fetch_market_data
from src.data.validator import validate_and_clean_data
from src.data.calendar import get_next_trading_day
from src.features.technical import compute_all_candidate_features
from src.models.trees import CatBoostEXP2Forecaster

# Page Configuration
st.set_page_config(
    page_title="Predictive Insights | Next-Day Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal Custom Styling (High-contrast, dark-mode compatible)
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #38BDF8;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        text-align: center;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #CBD5E1;
        margin-bottom: 1.8rem;
        text-align: center;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.0rem 0.75rem;
        text-align: center;
        min-height: 120px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        letter-spacing: 0.03em;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .metric-delta-pos {
        color: #4ADE80;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .metric-delta-neg {
        color: #F87171;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .price-badge-up {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        font-weight: 800;
        font-size: 1.05rem;
        border: 1px solid rgba(74, 222, 128, 0.4);
        border-radius: 6px;
        padding: 0.25rem 0.55rem;
        display: inline-block;
    }
    .price-badge-down {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        font-weight: 800;
        font-size: 1.05rem;
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 6px;
        padding: 0.25rem 0.55rem;
        display: inline-block;
    }
    .price-badge-neutral {
        background-color: rgba(148, 163, 184, 0.2);
        color: #E2E8F0;
        font-weight: 800;
        font-size: 1.05rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 6px;
        padding: 0.25rem 0.55rem;
        display: inline-block;
    }
    .exp2-badge-up {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        font-weight: 800;
        font-size: 1.05rem;
        border: 1px solid rgba(74, 222, 128, 0.4);
        border-radius: 6px;
        padding: 0.25rem 0.55rem;
        display: inline-block;
    }
    .exp2-badge-below {
        background-color: rgba(100, 116, 139, 0.2);
        color: #CBD5E1;
        font-weight: 800;
        font-size: 1.05rem;
        border: 1px solid rgba(203, 213, 225, 0.3);
        border-radius: 6px;
        padding: 0.25rem 0.55rem;
        display: inline-block;
    }
    .info-footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.82rem;
        margin-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)


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


@st.cache_data(ttl=3600, show_spinner=False)
def load_clean_data(ticker: str):
    """Retrieves and cleans historical market data for the selected asset."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Please provide a valid, non-empty stock ticker.")
        
    # Choose benchmark with calendar synchronization
    benchmark_ticker = "^NSEI" if ticker.endswith((".NS", ".BO")) else "SPY"
    
    # Primary benchmark asset uses 2015-01-01 start to preserve full development history
    start_date = "2015-01-01" if ticker == "AAPL" else "2020-01-01"
    raw_target, raw_bench = fetch_market_data(
        ticker=ticker,
        benchmark_ticker=benchmark_ticker,
        start=start_date
    )
    df_clean = validate_and_clean_data(raw_target, raw_bench)
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
    # User-facing Directional Signal consistent with predicted price movement
    directional_signal = (
        "UP" if price_change > 1e-6
        else "DOWN" if price_change < -1e-6
        else "NO CHANGE"
    )
    
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
        "directional_signal": directional_signal,
        "learned_threshold": learned_tau,
        "pred_log_return": pred_log_ret
    }


def main():
    st.markdown('<div class="main-title">📈 Predictive Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Intelligent Next-Day Stock Price & Direction Forecasting</div>', unsafe_allow_html=True)

    # 1. Inputs Section: Stock Selection Only
    stock_selection = st.selectbox(
        "Select Stock Ticker",
        options=list(POPULAR_STOCKS.keys()),
        index=0
    )
    
    # If user chooses custom ticker
    if POPULAR_STOCKS[stock_selection] == "CUSTOM":
        custom_ticker = st.text_input("Enter Yahoo Finance Ticker Symbol:", value="").upper().strip()
        selected_ticker = custom_ticker
    else:
        selected_ticker = POPULAR_STOCKS[stock_selection]

    # Currency Symbol helper
    curr_sym = "₹" if selected_ticker.endswith((".NS", ".BO")) else "$"

    # 2. Single Prominent Predict Button
    predict_clicked = st.button("🚀 Predict Next Trading Day", use_container_width=True, type="primary")

    if predict_clicked:
        if not selected_ticker:
            st.error("Please enter or select a valid stock ticker symbol.")
            return

        with st.spinner(f"Analyzing {selected_ticker} and generating next-day forecast..."):
            try:
                df_clean, bench = load_clean_data(selected_ticker)
            except Exception as e:
                err_msg = str(e)
                if "No market data returned" in err_msg or "verify the ticker" in err_msg:
                    st.error(f"❌ Could not retrieve market data for **{selected_ticker}**. Please ensure it is a valid Yahoo Finance ticker symbol (e.g. 'AAPL', 'NVDA', 'RELIANCE.NS').")
                else:
                    st.error(f"Error fetching data for {selected_ticker}: {err_msg}")
                return

            try:
                result = run_next_day_forecast(df_clean, ticker=selected_ticker)
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                return

        # 3. Model & Asset Header
        st.markdown("---")
        is_primary = selected_ticker == "AAPL"
        badge_text = (
            "✅ **Primary Benchmark Asset** (Validated in 5-Fold Walk-Forward Cross-Validation & Purged Diagnostic Holdout)"
            if is_primary
            else f"🧪 **Dynamic Asset Transfer** (Trained on {selected_ticker} history using the CatBoost architecture)"
        )
        st.caption(badge_text)

        last_close = result["last_close"]
        pred_price = result["predicted_price"]
        price_diff = result["price_change"]
        pct_diff = result["pct_change"]
        tau = result["learned_threshold"]
        pred_log_ret = result["pred_log_return"]
        price_movement = result["price_movement"]
        exp2_classification = result["exp2_classification"]

        delta_color = "metric-delta-pos" if price_diff >= 0 else "metric-delta-neg"
        delta_sign = "+" if price_diff >= 0 else "-"

        # 1. Price Movement & Directional Signal Indicators
        if price_diff > 1e-6:
            price_badge_class = "price-badge-up"
            price_arrow = "↗"
            directional_signal = "UP"
            signal_badge_class = "price-badge-up"
        elif price_diff < -1e-6:
            price_badge_class = "price-badge-down"
            price_arrow = "↘"
            directional_signal = "DOWN"
            signal_badge_class = "price-badge-down"
        else:
            price_badge_class = "price-badge-neutral"
            price_arrow = "—"
            directional_signal = "NO CHANGE"
            signal_badge_class = "price-badge-neutral"

        # 4. Metric Cards (5 Columns: 3 Price Metrics + Price Movement + Directional Signal)
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1.25, 1.35])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest Close</div>
                <div class="metric-value">{curr_sym}{last_close:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Close</div>
                <div class="metric-value">{curr_sym}{pred_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Change</div>
                <div class="metric-value"><span class="{delta_color}">{delta_sign}{abs(pct_diff):.2f}%</span></div>
                <div class="{delta_color}">{delta_sign}{curr_sym}{abs(price_diff):.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Price Movement</div>
                <div style="margin-top:0.35rem;"><span class="{price_badge_class}">{price_movement} {price_arrow}</span></div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.35rem;">{delta_sign}{curr_sym}{abs(price_diff):.2f} ({delta_sign}{abs(pct_diff):.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Directional Signal</div>
                <div style="margin-top:0.35rem;"><span class="{signal_badge_class}">{directional_signal} {price_arrow}</span></div>
                <div style="font-size:0.72rem; color:#94A3B8; margin-top:0.35rem; font-weight:500;">
                    Pred Return: <b>{pred_log_ret:+.2%}</b> | Threshold: <b>{tau:+.2%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 5. Simple, Clean Interactive Chart
        st.subheader("Recent Price Action & Next-Day Prediction")
        
        # Display the most recent 60 trading days for context
        hist_recent = df_clean.iloc[-60:]
        
        fig = go.Figure()
        
        # Historical Close Line
        fig.add_trace(go.Scatter(
            x=hist_recent.index,
            y=hist_recent["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#38BDF8", width=2)
        ))
        
        # Forecasted Next Trading Day
        chart_line_color = "#4ADE80" if price_diff >= 0 else "#F87171"
        fig.add_trace(go.Scatter(
            x=[result["last_date"], result["next_date"]],
            y=[last_close, pred_price],
            mode="lines+markers",
            name=f"Predicted ({directional_signal})",
            line=dict(color=chart_line_color, width=2.5, dash="dash"),
            marker=dict(size=8, color=chart_line_color)
        ))
        
        fig.update_layout(
            xaxis_title="Trading Session Date",
            yaxis_title=f"Price ({curr_sym})",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
            xaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
            yaxis=dict(gridcolor="rgba(148, 163, 184, 0.15)"),
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # 6. Single Forecast Table Row
        st.subheader("Next-Trading-Day Forecast Details")
        
        table_data = [{
            "Asset": f"{selected_ticker} ({stock_selection.split(' (')[0]})",
            "Next Session Date": result["next_date"].strftime("%Y-%m-%d (%A)"),
            "Latest Close": f"{curr_sym}{last_close:.2f}",
            "Predicted Close": f"{curr_sym}{pred_price:.2f}",
            "Expected Change": f"{delta_sign}{curr_sym}{abs(price_diff):.2f} ({delta_sign}{abs(pct_diff):.2f}%)",
            "Price Movement": f"{price_movement} {price_arrow}",
            "Directional Signal": f"{directional_signal} {price_arrow} (Pred Return: {pred_log_ret:+.2%}, Threshold: {tau:+.2%})"
        }]
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # 7. Disclaimers & Methodology Footer
    st.markdown("""
    <div class="info-footer">
        Forecasts represent statistical projections for educational demonstration and do not constitute financial advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
