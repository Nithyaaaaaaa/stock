# Predictive Insights – Intelligent Stock Market Forecasting System

A scientifically rigorous, leak-free, production-ready machine learning framework for short-horizon stock market forecasting.

---

## 📌 Executive Summary & Key Research Findings

1. **Architecture Discovery:**
   - **Tree-based gradient boosting (CatBoost & XGBoost)** consistently outperforms deep learning architectures (Two-Layer LSTM, GRU, CNN-LSTM, TCN) across expanding-window walk-forward cross-validation.
   - Deep learning models suffer from severe variance and overfitting when trained on noisy financial returns, yielding higher out-of-sample RMSE ($2.70 - $2.80) compared to CatBoost ($2.62 - $2.64) and persistence ($2.65).
   - CatBoost trains in **0.34 seconds** across 5 folds with an inference latency of **0.002 ms per sample**, compared to 19.68 seconds and 0.033 ms for PyTorch LSTM.

2. **EXP2 Integration — CatBoost Regressor with Train-Tuned Return Threshold:**
   - **Champion Generalist Configuration:** Combines continuous log-return regression with an inner validation-tuned return threshold ($\tau^*$) learned strictly on historical training data to optimize balanced directional accuracy.
   - **Development Walk-Forward Cross-Validation Results (N = 1,164 out-of-sample validation pairs):**
     - Overall Directional Accuracy: **52.88%** (outperforming both default CatBoost 52.45% and Always-UP 52.62%)
     - Balanced Accuracy: **51.53%** (highest balanced score among all CatBoost configurations)
     - DOWN Class F1-Score: **34.13%** (Recall: **25.77%**, Precision: **50.53%**; an +80% relative improvement over baseline 14.34% recall)
     - UP Class F1-Score: **63.32%** (Recall: **77.29%**, Precision: **53.63%**; preserving strong upside capture)
     - Macro F1-Score: **48.73%** (up from 43.99% baseline)
   - **Price Forecasting Preservation:** The underlying model remains a continuous return regressor, preserving next-day price compounding $\hat{P}_{t+1} = P_t \cdot \exp(\hat{y}_{t+1})$ without compromise.
   - **Experimental Downside Specialist (EXP13):** A direct CatBoost classifier with `auto_class_weights='Balanced'` and 6 new stationary downside features achieved **41.92% DOWN Recall** and **44.55% DOWN F1**, but reduced UP recall to 58.33% and accuracy to 50.56%. It is preserved in the codebase as an experimental risk/hedging tool, not the default.

3. **Distinction Between Price Prediction and Direction Prediction:**
   - **Continuous Price Prediction (Regression):** Predicts expected dollar close $\hat{P}_{t+1}$ by minimizing squared error loss. Evaluated via RMSE and MAE against the random-walk persistence benchmark ($P_{t+1} = P_t$).
   - **Directional Movement Prediction (Classification):** Categorizes tomorrow's movement into binary classes (UP vs. DOWN). Because stock prices possess positive secular drift in bull markets, standard regression models predict positive returns on ~86–90% of sessions. EXP2 solves this by learning an optimal decision boundary $\tau^*$ to balance UP and DOWN detections without destroying upside predictive power.

4. **The Persistence Benchmark Reality (Diagnostic Holdout Evaluation):**
   - On the 596-session holdout test set (July 2024 to November 2026), with complete 1-observation boundary purging applied, CatBoost achieves **RMSE \$4.3009** vs. Random-Walk Persistence **RMSE \$4.3086**.
   - A formal **Diebold-Mariano test** with Newey-West HAC standard errors yields $DM = 0.1637, p = 0.8699$ ($p > 0.05$, 95% CI: $[-\$0.7277, +\$0.8604] \text{ USD}^2$), confirming statistical parity with the random walk price benchmark.

5. **Conformal Uncertainty Calibration:**
   - Split-conformal calibration on development split residuals achieves **94.63% empirical coverage** for nominal 95% intervals (coverage gap: $-0.37\%$, average width: \$18.50).
   - Nominal 90% intervals achieve **87.58% empirical coverage** (coverage gap: $-2.42\%$, average width: \$13.22).

6. **Evaluation Integrity & Disclosure:**
   - Development walk-forward cross-validation metrics are strictly distinguished from final diagnostic holdout results.
   - The final diagnostic holdout was **never used for model fitting, feature selection, or threshold tuning**.

---

## 🖥️ Streamlit Web Application Architecture

The production application is implemented as an interactive, responsive **Streamlit application (`app.py`)**, purpose-built for deployment on **Streamlit Community Cloud**:

- **Streamlit Entry Point:** `app.py`
- **Minimalist Workflow:** **Select Stock Ticker → Click "🚀 Predict Next Trading Day" → View Predicted Price, Movement, and Classification.**
- **Dual Indicator Cards:**
  1. **Predicted Price Movement:** Evaluates expected continuous price change $\Delta P = \hat{P}_{t+1} - P_t$ (`PRICE INCREASE ↗` green, `PRICE DECREASE ↘` red, or `NO PRICE CHANGE —` neutral).
  2. **EXP2 Classification:** Evaluates whether predicted return clears the calibrated decision hurdle $\hat{y}_{t+1} > \tau^*$ (`UP — Above threshold` green badge, or `Below UP threshold` neutral slate badge; never mislabeled as "DOWN").
- **Exchange-Calendar Next Session:** Automatically calculates authentic next trading days skipping weekends and official exchange holidays (NYSE/NASDAQ and NSE/BSE) with safe business-day fallbacks.
- **Interactive Visualization:** Plotly chart displaying recent 60-day price action and a dynamic colored dashed forecast line connecting to tomorrow's projected close.
- **Single-Row Forecast Details Table:** High-density summary with currency-accurate formatting (`$` for US equities, `₹` for Indian equities).
- **Strict Next-Day Horizon:** Displays exactly one next-trading-day forecast; recursive multi-day extrapolations are disabled to prevent error compounding.

---

## 🏗️ Architecture & Directory Layout

```
predictive_insights/
├── app.py                             # Primary Streamlit Web Application (Cloud Entry Point)
├── requirements.txt                   # Production dependencies for Streamlit Community Cloud
├── requirements-dev.txt               # Full development dependencies (includes PyTorch & research tools)
├── README.md                          # Comprehensive scientific & deployment documentation
├── implementation_report.md           # EXP2 integration, UI disambiguation & audit findings
├── directional_model_improvement_report.md # 15-experiment directional study findings
├── final_classification_metrics.md    # Scikit-learn classification reports & confusion matrices
├── final_consistency_report.md        # Metric reconciliation and audit consistency report
├── final_scientific_audit.md          # Independent scientific audit findings
├── forecasting_validity_review.md     # Multi-day and boundary alignment review
├── artifacts/                         # Benchmark tables, charts, and CSV logs
├── experiments/                       # Reproducible experimentation scripts
│   ├── run_directional_improvement_study.py # 15 walk-forward directional experiments
│   ├── compute_classification_report.py
│   ├── run_walk_forward_benchmark.py
│   └── audit_reconciliation.py
├── src/                               # Modular package library
│   ├── data/                          # Data retrieval, cleaning, and exchange calendar
│   ├── features/                      # Technical feature computation & stationarity testing
│   ├── models/                        # Unified BaseForecaster, CatBoostEXP2Forecaster
│   └── evaluation/                    # Metrics, HAC significance tests & conformal calibration
└── tests/                             # Automated pytest suite (24 unit & integration tests)
    ├── test_calendar.py               # NYSE/NSE exchange calendar & holiday tests
    ├── test_dashboard_forecast.py     # Single-day forecast & delta percentage tests
    ├── test_ui_classification_cases.py# Price movement vs EXP2 classification display tests
    ├── test_exp2_integration.py       # EXP2 threshold tuning & price preservation tests
    ├── test_data_leakage.py           # Strict train/holdout temporal boundary assertions
    ├── test_evaluation.py             # HAC covariance & loss function invariance tests
    └── test_models.py                 # Tree & baseline model contract tests
```

---

## ⚡ Quickstart & Local Execution

### 1. Installation
```bash
cd predictive_insights
pip install -r requirements.txt
```

### 2. Run Automated Pytest Suite
```bash
PYTHONPATH=. pytest tests/ -v
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
python app.py
# Opens at http://localhost:8501
```

---

## 📊 Summary of Walk-Forward Cross-Validation (5 Expanding Folds, Purged)

Evaluated strictly across 5 expanding folds on development sessions (2015 to 2024) with boundary embargo ($N = 1,164$ out-of-sample validation sessions):

| Model Architecture | Purged Pooled RMSE | Unweighted Mean Fold RMSE | Pooled Acc | Bal Acc | Macro F1 | UP F1 | DOWN F1 | DOWN Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost EXP2 (Deployed Champion)** | **\$2.6373** | **\$2.6214** | **52.88%** | **51.53%** | **48.73%** | **63.32%** | **34.13%** | **25.77%** |
| CatBoost Baseline ($\tau = 0.0$) | \$2.6373 | \$2.6214 | 52.45% | 50.55% | 43.99% | 65.76% | 22.22% | 14.34% |
| CatBoost EXP13 Downside Specialist (Exp) | — | — | 50.56% | 50.13% | 49.97% | 55.39% | 44.55% | 41.92% |
| Random Forest Classifier (Unweighted) | — | — | 53.83% | 52.36% | 49.05% | 64.65% | 33.46% | 24.50% |
| XGBoost Classifier (Unweighted) | — | — | 52.28% | 50.99% | 48.47% | 62.47% | 34.47% | 26.50% |
| Always-UP Baseline | \$2.9421 | \$2.9388 | 52.62% | 50.00% | 34.48% | 68.96% | 0.00% | 0.00% |
| Persistence Baseline | \$2.6479 | \$2.6462 | 47.38% | 50.00% | 32.15% | 0.00% | 64.29% | 100.00% |
| Two-Layer LSTM | \$2.8031 | \$2.7984 | 49.44% | 48.50% | 45.10% | 56.05% | 28.10% | 21.05% |

---

## ⚠️ Financial Risk & Scientific Limitations

1. **Market Efficiency & Random-Walk Parity:**
   At a 1-day horizon, equity price dynamics are dominated by stochastic news arrival. While CatBoost EXP2 significantly improves directional balance and achieves lower price RMSE than deep learning, it operates in statistical parity ($p = 0.8699$) with the random-walk persistence baseline on holdout price RMSE.
2. **Directional Realities:**
   In an asset with positive secular drift, negative days are shorter, rarer, and sharper than positive drift days. EXP2 nearly doubles DOWN detections while preserving overall accuracy, but it cannot guarantee directional profitability in all market regimes.
3. **Execution Realities:**
   The model does not model trading friction, commissions, bid-ask spreads, execution delays, or market impact.
4. **Educational Notice:**
   All forecasts and directional signals are strictly educational and scientific demonstrations and do not constitute financial advice.
