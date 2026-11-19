# Forecasting Validity Review — Predictive Insights

**Document Title:** Forecasting Validity, Boundary Alignment, and Multi-Asset Transfer Review  
**Project:** Predictive Insights – Intelligent Stock Market Forecasting System  
**Review Date:** November 19, 2026  
**Auditor Protocol:** Rigorous Independent Methodological Review  

---

## 1. Multi-Day Forecasting Methodology and Limitations

### 1.1 Recursive vs. Direct Forecasting Architecture
The application generates multi-day forecasts ($H \in \{1, 5, 7, 15, 30\}$ trading days) using an **iterative recursive auto-regressive rollout**, rather than a direct multi-horizon model ($f_h(X_t)$):
- **Step 1 ($t+1$):** Uses 100% genuine historical market features $X_t$ observed at or before trading close $t$. The fitted CatBoost model predicts continuous log return $\hat{y}_{t+1}$, and next-day price is compounded via $\hat{P}_{t+1} = P_t \cdot \exp(\hat{y}_{t+1})$.
- **Steps 2 to $H$ ($t+k$):** The predicted price $\hat{P}_{t+k-1}$ is appended to the historical price series as a simulated bar to recompute the rolling features for the next step:
  - $\text{Close}_{t+k-1} = \hat{P}_{t+k-1}$
  - $\text{Open}_{t+k-1} = \hat{P}_{t+k-2}$
  - $\text{High}_{t+k-1} = \max(\hat{P}_{t+k-2}, \hat{P}_{t+k-1}) \times 1.002$ (synthetic $+0.2\%$ upper bound)
  - $\text{Low}_{t+k-1} = \min(\hat{P}_{t+k-2}, \hat{P}_{t+k-1}) \times 0.998$ (synthetic $-0.2\%$ lower bound)
  - $\text{Volume}_{t+k-1} = \text{Volume}_t$ (held constant at last observed volume)
  - $\text{Benchmark Close}_{t+k-1} = \text{Benchmark Close}_t$ (held constant at last observed benchmark close)

### 1.2 Verification of Zero Future Information Leakage
- **Future Data Check:** Confirmed that no actual future trading prices ($P_{t+k}$ for future dates) enter the feature matrix. Because future dates have not yet occurred, all future inputs are strictly recursive model projections.

### 1.3 Statistical Limitations & Error Compounding over Longer Horizons
- **Feature Distribution Drift:** In reality, volume, volatility, and macroeconomic benchmarks fluctuate stochastically. Holding volume constant forces `Log_Volume_Change` toward zero; holding benchmark price constant forces benchmark returns and excess returns to zero; fixing the intraday range to $\pm 0.2\%$ artificially dampens `NATR_14` and `High_Low_Ratio`.
- **Variance Accumulation:** In multi-step recursive forecasting of financial returns, prediction errors compound at each step ($k$). Because daily returns have an empirical half-life of 2 to 5 days, predicting 15 to 30 days ahead with a recursive 1-day model inevitably suffers from mean-reversion collapse toward historical drift or unconstrained trajectory drift.
- **Scientific Conclusion:** **Only the Day 1 ($H=1$) forecast is an empirically validated machine learning prediction.** Forecasts for Days 2 through 30 are simulated scenario trajectories under synthetic constant-volatility assumptions, not validated out-of-sample statistical predictions.

---

## 2. Multi-Stock Model-Training Methodology and Limitations

### 2.1 Independent Training vs. Transfer Re-Use
- **Confirmation:** The dashboard **trains a completely fresh CatBoost model for each selected stock** on that specific asset's historical clean data. It does **NOT** reuse the AAPL-trained weights for MSFT, NVDA, or RELIANCE.NS.
- **Training Efficiency:** Because CatBoost trains on tabular features in $\sim 0.08$ to $0.15$ seconds, per-stock training on demand is fast and avoids cross-asset feature pollution.

### 2.2 Feature Compatibility & Benchmark Synchronization
- **Standardized Pipeline:** All 25 candidate features rely exclusively on Open, High, Low, Close, Volume, and Benchmark Close. This schema is 100% compatible with both US equities (`AAPL`, `NVDA`, `MSFT`) and Indian equities (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`).
- **Calendar-Synchronized Benchmarks:**
  - **US Stocks:** Benchmarked against `SPY` (S&P 500 ETF) on the NYSE/NASDAQ calendar.
  - **Indian Stocks (`.NS`, `.BO`):** Benchmarked against `^NSEI` (NIFTY 50 Index) on the National Stock Exchange of India (NSE) calendar.
  - This prevents artificial missing-data drops caused by disparate national market holidays (e.g., US Thanksgiving vs. Diwali).

### 2.3 Crucial Validation Distinction
- **Primary Validated Stock:** **AAPL** is the sole asset on which formal stationarity auditing, lookback window optimization ($L=20$), 5-fold expanding walk-forward cross-validation, Diebold-Mariano parity testing ($DM = 0.1637, p = 0.8699$), and the 596-session diagnostic holdout evaluation (with disclosed boundary audit and purged re-evaluation) were conducted.
- **Experimental Multi-Stock Transfer:** Support for other tickers (`NVDA`, `MSFT`, `RELIANCE.NS`, etc.) demonstrates **pipeline architectural generality**, not certified forecasting accuracy. Without running dedicated walk-forward benchmarks and holdout audits for each individual asset, no claim of predictive edge or persistence parity can be made for any stock other than AAPL.

---

## 3. Fold-Boundary Target Alignment & Leakage Findings

### 3.1 Target Definition & Realization Timestamps
In daily financial forecasting:
- At trading close on date $t$, features $X_t$ are observed.
- The target label is the forward continuous log return $y_t = \ln(P_{t+1} / P_t)$ and forward closing price $P_{t+1}$ generated via `.shift(-1)`.
- **Crucial Realization Fact:** Target $y_t$ is **NOT observable on date $t$**; it only realizes when the market closes on date $t+1$.

### 3.2 Discovery of Boundary Target Leakage in Standard Splits
Consider an expanding walk-forward fold where training data spans indices $[0 : \text{tr\_end}]$ and validation data spans $[\text{tr\_end} : \text{vl\_end}]$:
1. The final training row is at index $\text{tr\_end} - 1$ (feature date $t$).
2. Because of `.shift(-1)`, the target label assigned to this row is the closing price at index $\text{tr\_end}$ (target date $t+1$).
3. However, index $\text{tr\_end}$ is the **first session of the validation set** ($\text{vl\_start} = \text{tr\_end}$)!
4. **The Overlap:** The model was trained using the closing price of the first validation day as a target label! If the model is trained at the cutoff of training date $t$, session $t+1$ has not yet closed, making that target an unobservable future realization.

```
  UNPURGED BOUNDARY (1-Observation Target Leakage):
  [Train Features: t-2] -> Target: Close(t-1)  (Observable at cutoff)
  [Train Features: t-1] -> Target: Close(t)    (Observable at cutoff)
  [Train Features: t  ] -> Target: Close(t+1)  <-- Target belongs to Validation Day 1!
  --------------------------------- SPLIT BOUNDARY ---------------------------------
  [Val Day 1: Features t+1] -> Target: Close(t+2)
```

### 3.3 The Purging (Embargo) Solution
To achieve 100% airtight chronological separation without forward target overlap, the final observation of each training slice must be **purged** (embargoed):

```
  PURGED BOUNDARY (Air-Tight Embargo):
  [Train Features: t-2] -> Target: Close(t-1)  (Observable)
  [Train Features: t-1] -> Target: Close(t)    (Observable at cutoff t)
  [Train Features: t  ] -> PURGED / EMBARGOED  (Target Close(t+1) has not occurred)
  --------------------------------- SPLIT BOUNDARY ---------------------------------
  [Val Day 1: Features t+1] -> Target: Close(t+2) (Completely unseen in training)
```

### 3.4 Empirical Impact Analysis of Purging & Holdout Reconciliations
We executed paired experiments comparing unpurged vs. purged training:

1. **CatBoost Holdout RMSE Diagnostic Reconciliation:**
   - **Initial Exploratory Holdout Run:** **\$4.2896** (preliminary script on unpurged boundaries).
   - **Pre-Audit Holdout Run ($N=2,327$ dev sessions):** **\$4.2913** (where the final development row `2024-07-05` carried forward target `2024-07-08`).
   - **Intermediate Diagnostic Slice:** **\$4.3003** (evaluated during preliminary boundary audit checks).
   - **Purged Final Diagnostic Holdout ($N=2,326$ dev sessions):** **\$4.3009** (authoritative final diagnostic result on 596 holdout sessions after purging `2024-07-05`).
   - **Impact:** Total delta across all holdout configurations is only **+\$0.0113 USD (+0.26%)**, confirming boundary overlap did not drive experimental results.
2. **5-Fold Walk-Forward Cross-Validation (Purged):**
   - **Primary Cross-Validation Metric:** **Purged Pooled RMSE = \$2.6373 USD** (pooled across all 1,164 out-of-sample prediction pairs; Pooled MAE = \$1.9508 USD).
   - **Unweighted Mean of Fold RMSEs:** **\$2.6214 USD** ($\sigma = \$0.2890$; Fold 1: \$2.4735, Fold 2: \$2.1912, Fold 3: \$3.0645, Fold 4: \$2.7473, Fold 5: \$2.6307).
   - **Unpurged Mean Fold RMSE:** **\$2.6408 USD**.
3. **Conclusion:** Purging the single boundary row produces negligible metric differences, confirming that the model's performance was not driven by boundary leakage. However, applying the purge is essential for complete econometric rigor.

---

## 4. Corrections Made

1. **Purging Implemented in Walk-Forward Validator:**
   Updated `WalkForwardValidator` in [src/evaluation/walk_forward.py](file:///Users/nithyashreevsuresh/Downloads/st/predictive_insights/src/evaluation/walk_forward.py) with `purge_boundary: bool = True` by default, ensuring that `actual_tr_end = train_end - 1` so that no training target date ever overlaps into a validation fold.
2. **Transparent Multi-Day UI Disclosure:**
   Updated `app.py` to explicitly badge:
   - Day 1 as **"Validated Machine Learning Forecast"**
   - Days 2–30 as **"Causal Recursive Trajectory Simulation under Constant Volatility Assumptions"**
3. **Multi-Stock Status Clarification:**
   Added clear UI tags identifying **AAPL** as the *"Primary Validated Benchmark Stock"* and all other tickers as *"Experimental Multi-Stock Transfer"*.

---

## 5. Summary of What Is Honestly Validated

| Prediction Type | Asset | Horizon | Validation Status | Empirical Support |
| :--- | :---: | :---: | :---: | :--- |
| **Next-Day Price & Direction** | **AAPL** | **1 Day** | ✅ **Rigorously Validated** | 5-Fold Purged Walk-Forward CV (Pooled RMSE: \$2.6373, Mean Fold RMSE: \$2.6214) + 596-Session Purged Diagnostic Holdout (\$4.3009 RMSE, $p=0.8699$ DM test parity with persistence). *Disclosed: Holdout re-evaluated following boundary audit.* |
| **Next-Day Conformal Intervals** | **AAPL** | **1 Day** | ✅ **Rigorously Validated** | Calibrated split-conformal intervals achieving **94.63% empirical holdout coverage** for nominal 95%. |
| **Multi-Day Recursive Trajectory** | AAPL | 5, 7, 15, 30 Days | ⚠️ **Unvalidated Simulation** | Plausible causal rollout; subject to compounding error and feature drift. |
| **Multi-Stock Predictions** | Other Tickers | Any | ⚠️ **Experimental Transfer** | Verified code pipeline execution and benchmark alignment; out-of-sample accuracy not certified. |
