# Predictive Insights – Intelligent Stock Market Forecasting System
## Comprehensive Machine Learning Research & Engineering Report (Reconciled Edition)

**Project Title:** Predictive Insights – Intelligent Stock Market Forecasting System  
**Lead Investigator & Engineering Team:** Machine Learning Systems Engineering Group  
**Dataset Coverage:** AAPL Equity OHLCV (Benchmark: SPY S&P 500 ETF) | 2,989 Trading Sessions (Jan 2, 2015 – Nov 19, 2026)  
**Evaluation Protocol:** 5-Fold Expanding-Window Walk-Forward Cross-Validation with Boundary Purging (2,326 Sessions) + 596-Session Holdout (Re-evaluated with 1-Observation Embargo)  
**Selected Champion Model:** Regularized Symmetric Tree Gradient Boosting (`CatBoostForecaster`)  

---

## 1. Executive Summary & Key Empirical Findings

This report documents the end-to-end design, implementation, and rigorous experimental evaluation of **Predictive Insights – Intelligent Stock Market Forecasting System**, built from the ground up as a scientifically defensible machine learning system.

```
                                      PROJECT ARCHITECTURE OVERVIEW
                                      
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. Raw Market Data Retrieval (yfinance) -> Calendar Synchronization & Corporate Split Adjustment│
  └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                   │
                                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. Chronological Split (80% Dev: 2,392 sessions | 20% Holdout: 597 sessions)                     │
  └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                   │
                                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. Feature Computation & Stationarity Audit: 25 Verified I(0) Features (ADF p<0.05, KPSS p>0.05)│
  │    Trimming: 65 warmup rows (Vol_60d) + 1 terminal shift(-1) row = 66 dropped -> 2,326 usable dev│
  └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                   │
                                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 4. 5-Fold Expanding Walk-Forward Cross-Validation with Boundary Embargo/Purging (2,326 sessions) │
  │    Primary Result: Purged Pooled RMSE = $2.6373 | Unweighted Mean Fold RMSE = $2.6214           │
  └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                   │
                                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 5. Model Selection: CatBoost Champion (Lowest Out-of-Sample RMSE, 52.49% Dir Acc, 0.34s fit)    │
  └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                   │
                                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 6. Holdout Evaluation (596 Sessions) Re-evaluated under 1-Observation Purge:                    │
  │    CatBoost RMSE: $4.3009 vs. Persistence: $4.3086 | Diebold-Mariano DM = 0.1637, p = 0.8699    │
  │    Split-Conformal Calibration: 94.63% Empirical Coverage for Nominal 95% ($18.50 Avg Width)   │
  └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Core Conclusions from Experimental Evidence:

1. **CatBoost & Gradient Boosted Trees Outperform Deep Learning Architectures:**
   In expanding-window walk-forward cross-validation across 5 chronological market regimes with boundary purging, **CatBoost** achieved the lowest out-of-sample pooled RMSE (\$2.6373), unweighted mean fold RMSE (\$2.6214), and an UP F1 score of 65.80%. In stark contrast, deep learning models (Two-Layer LSTM, GRU, CNN-LSTM, TCN) exhibited severe variance and overfitting to financial noise, posting worse RMSE (\$2.70 – \$2.80) than the zero-return persistence baseline (\$2.6479). Furthermore, CatBoost trained in **0.34 seconds** with **0.002 ms** inference latency, compared to 19.68 seconds and 0.033 ms for PyTorch LSTM.

2. **Holdout Evaluation Protocol Disclosure & Persistence Benchmark Parity:**
   *Evaluation Protocol Disclosure:* The holdout dataset was initially evaluated as an out-of-sample block and subsequently re-evaluated during the boundary leakage audit and purging pass. To prevent forward target leakage at the development-holdout boundary, the final development row (Friday, July 5, 2024) was purged so that no target date extended into the holdout set (Monday, July 8, 2024). On the final 596-session purged holdout, CatBoost achieved an RMSE of **\$4.3009** vs. Random-Walk Persistence at **\$4.3086**. A formal **Diebold-Mariano test** with Newey-West HAC standard errors yielded $DM = 0.1637, p = 0.8699$ ($95\% \text{ CI: } [-\$0.7277, +\$0.8604] \text{ USD}^2$). Because $p \gg 0.05$, the null hypothesis of equal predictive accuracy cannot be rejected. **We make no claim that CatBoost statistically outperforms persistence on price RMSE.**

3. **Directional Accuracy & Baseline Disclosures:**
   In a strong secular bull market, the naive **Majority-Class (Always UP)** baseline achieves **53.86%** directional accuracy. The standard persistence baseline (predicting zero return and classifying all days as DOWN under standard strict inequality) achieves **46.14%**. CatBoost achieved **51.51%** directional accuracy with a **66.12% UP F1 score**. CatBoost beats the arbitrary 46.14% persistence tie-break, but **does not beat the 53.86% Majority-Class baseline**.

4. **Split-Conformal Prediction Interval Coverage:**
   Calibrated solely on development split residuals with boundary purging, nominal 95% conformal prediction intervals achieved **94.63% empirical coverage** on the holdout test set (coverage gap of only $-0.37\%$, average width \$18.50). Nominal 90% intervals achieved **87.58% empirical coverage** (average width \$13.22).

---

## 2. Dataset Partitioning, Traceability, and Boundary Audit

| Partition | Raw Sessions | Start Date | End Date | Role & Privacy Constraints |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Total** | 2,989 | 2015-01-02 | 2026-11-19 | Complete calendar-synchronized AAPL & SPY dataset |
| **Development Set** | 2,392 | 2015-01-02 | 2024-07-05 | 80% chronological partition: feature exploration, walk-forward CV, calibration |
| **Holdout Test Set** | 597 | 2024-07-08 | 2026-11-19 | 20% chronological partition: evaluated out-of-sample and verified under boundary purge |

### Reconciliation of Observation Counts:
1. **Warm-Up Trimming (65 rows, indices 0 to 64):** Multi-period rolling features (`Vol_60d`, `Rolling_Beta_60`, `Ratio_Close_SMA_50`) require up to 60 trading days of historical observations to compute valid rolling statistics without lookahead imputation.
2. **Terminal Target Trimming (1 row, index 2,988):** The forecasting target is next-day price and return ($y_{t+1}$ via `.shift(-1)`). For the final session in the dataset (Nov 19, 2026), tomorrow's price is unobserved.
3. **Usable Sessions:** 2,923 total usable sessions (2,327 raw usable dev sessions, 596 evaluated holdout sessions).
4. **Boundary Purge (1 row dropped from dev):** At the dev-holdout boundary, row `2024-07-05` (Friday) carried a target label of `2024-07-08` (Monday, the first day of holdout). Purging this row yields **2,326 clean development sessions**, guaranteeing zero target date overlap into the holdout.

---

## 3. Walk-Forward Cross-Validation Multi-Model Benchmark (Purged)

Using a 5-fold expanding-window cross-validation design across the 2,326 usable development sessions with a 1-observation boundary embargo:

| Fold | Training Dates | Train $N$ | Validation Dates | Val $N$ | CatBoost RMSE | CatBoost MAE | Dir Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2015-04-08 to 2019-11-14 | 1,162 | 2019-11-18 to 2020-10-19 | 232 | \$2.4688 | \$1.7097 | 57.33% |
| **Fold 2** | 2015-04-08 to 2020-10-16 | 1,394 | 2020-10-20 to 2021-09-21 | 232 | \$2.1948 | \$1.6839 | 49.57% |
| **Fold 3** | 2015-04-08 to 2021-09-20 | 1,626 | 2021-09-22 to 2022-08-23 | 232 | \$3.0647 | \$2.3673 | 53.02% |
| **Fold 4** | 2015-04-08 to 2022-08-22 | 1,858 | 2022-08-24 to 2023-07-27 | 232 | \$2.7466 | \$2.0665 | 49.14% |
| **Fold 5** | 2015-04-08 to 2023-07-26 | 2,090 | 2023-07-28 to 2024-07-05 | 236 | \$2.6319 | \$1.9269 | 53.39% |

### Primary Cross-Validation Metrics (Baseline CatBoost, tau=0.0):
- **Purged Pooled RMSE:** **\$2.6373 USD** (computed over all 1,164 concatenated out-of-sample predictions).
- **Purged Pooled MAE:** **\$1.9508 USD**.
- **Purged Unweighted Mean of Fold RMSEs:** **\$2.6214 USD** ($\sigma = \$0.2890$).
- **Purged Pooled Directional Accuracy:** **52.45%** (Balanced Accuracy: **50.55%**, UP F1: **65.76%**, DOWN F1: **22.22%**, DOWN Recall: **14.34%**).

---

## 3.1 Directional Model Optimization: EXP2 Deployed Configuration

### The Directional Asymmetry Challenge
While CatBoost excels at continuous log-return regression, the underlying positive drift of equity markets caused unweighted models to predict UP on 86.17% of validation sessions, producing a low DOWN recall of **14.34%** (catching only 79 / 551 down sessions).

### EXP2 Methodology: CatBoost Regressor with Train-Tuned Return Threshold
To rectify this imbalance without manipulating predictions or sacrificing continuous price forecasting, we developed **EXP2**:
1. **Model Architecture:** Continuous CatBoost Regressor (120 iterations, depth 4, learning rate 0.02, L2 leaf reg 3.0).
2. **Independent Inner-Fold Threshold Tuning:** Inside each training fold, an inner 80/20 chronological train/calibration split is formed. An inner model searches candidate return thresholds $\tau \in [-0.004, +0.004]$ (81 points) to find $\tau^*$ that maximizes Balanced Accuracy while enforcing an UP recall floor ($\ge 50\%$).
3. **Zero Future Information:** The threshold for each fold is learned strictly using that fold's historical training slice; no validation or future information is ever accessible.
4. **Deployed Development Threshold:** Derived on the full development set ($N=2,326$) via an inner 80/20 split ($N_{\text{tr}}=1,860, N_{\text{val}}=466$), yielding $\tau^* = +0.001000$ (+0.10% return cutoff).

### Development Walk-Forward CV Results (Pooled $N=1,164$ Validation Sessions):
- **Overall Directional Accuracy:** **52.88%** (outperforming both default CatBoost 52.45% and Always-UP 52.62%).
- **Balanced Accuracy:** **51.53%** (highest balanced score among all CatBoost configurations).
- **Macro F1-Score:** **48.73%** (up from 43.99% baseline).
- **DOWN Class Performance:** Precision = **50.53%**, Recall = **25.77%**, F1 = **34.13%** (catching 142 down days vs. 79 in baseline; +80% relative improvement).
- **UP Class Performance:** Precision = **53.63%**, Recall = **77.29%**, F1 = **63.32%** (robust upside preservation).

### Rationale for Selecting EXP2 over EXP13:
- **EXP13 (CatBoost Classifier + Downside Features, Balanced):** In our 15-model study, EXP13 achieved the highest raw DOWN recall (**41.92%**) and DOWN F1 (**44.55%**). However, it reduced UP recall to **58.33%** and overall accuracy to **50.56%**, and as a discrete classifier it cannot naturally output continuous dollar price trajectories.
- **EXP2 Advantages:** EXP2 preserves continuous return forecasting for price estimation and conformal prediction intervals, improves overall accuracy to 52.88%, and nearly doubles DOWN detections without destabilizing UP capture. EXP13 is preserved in the research codebase as an experimental downside-specialist model for hedging contexts.

> [!NOTE]
> **Strict Evaluation Boundary:**
> All EXP2 metrics reported above represent out-of-sample **development walk-forward cross-validation results**. The final diagnostic holdout (2024–2026) was strictly preserved untouched and was never used for model fitting, feature selection, or threshold tuning.

---

## 4. Reconciled Holdout Evaluation Results (596 Sessions)

### Full Model Comparison on Purged Holdout:

| Model Architecture | RMSE (USD) | MAE (USD) | MAPE (%) | Dir Acc (%) | UP F1 (%) | DOWN F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Corrected Champion)**| **\$4.3009** | **\$2.9796** | **1.18%** | **51.51%** | **66.12%** | **14.75%** |
| **Persistence Baseline** | \$4.3086 | \$2.9769 | 1.18% | 46.14% | 0.00% | 63.15% |
| **Majority-Class (Always UP)** | \$4.8584 | \$3.5804 | 1.42% | 53.86% | 70.01% | 0.00% |

### Reconciliation of Historical Reported Values:
- **\$4.2896:** Preliminary unpurged holdout run in initial exploratory test.
- **\$4.2913:** Pre-audit holdout run on full dev partition ($N=2,327$) where boundary row `2024-07-05` carried target label `2024-07-08`.
- **\$4.3009:** **Corrected final diagnostic result** with the 1-observation boundary purge applied ($N=2,326$). The delta is only **+\$0.0090 USD (0.2%)**, verifying that boundary leakage did not drive the performance findings.

### Statistical Significance:
- **Diebold-Mariano Test (CatBoost vs. Persistence):**
  - Mean squared loss differential: $\bar{d} = +0.0663 \text{ USD}^2$
  - Newey-West HAC standard error: $0.4051$
  - 95% Confidence Interval: $[-\$0.7277, +\$0.8604] \text{ USD}^2$ (crosses zero)
  - $DM = 0.1637, p = 0.8699$. Fail to reject $H_0$. No statistically significant difference.
- **Wilcoxon Signed-Rank Test:** $W = 87,762.0, p = 0.7770$.

---

## 5. Split-Conformal Prediction Interval Audit

- **Nominal 95% Interval:** Empirical Holdout Coverage = **94.63%** (Avg Width = **\$18.50**, Gap = **-0.37%**).
- **Nominal 90% Interval:** Empirical Holdout Coverage = **87.58%** (Avg Width = **\$13.22**, Gap = **-2.42%**).
- **Regime Breakdown:**
  - Low-Volatility Regime (Vol $\le 1.50\%, N=298$): Coverage = **92.62%**, Avg Width = **\$13.42**
  - High-Volatility Regime (Vol $> 1.50\%, N=298$): Coverage = **96.64%**, Avg Width = **\$23.18**
- **Limitation:** Conformal coverage assumes exchangeability, which financial time series violate due to volatility clustering. While empirical coverage is near-nominal, future coverage under unseen structural regime breaks is not guaranteed.

---

## 6. Real-World Limitations & Disclosures

1. **Market Efficiency & Persistence Parity:** Stock prices at a 1-day horizon closely approximate a random walk ($P_{t+1} = P_t + \epsilon$). Beating persistence on price RMSE in a statistically significant manner is virtually impossible using public technical indicators alone.
2. **Directional Realities:** CatBoost achieves 51.51% directional accuracy, which does not beat the 53.86% Majority-Class baseline on a bullish holdout period.
3. **Multi-Day Recursive Simulation:** Forecasts beyond Day 1 are simulated scenario rollouts under constant volatility assumptions; only Day 1 is an empirically validated ML prediction.
4. **Educational Notice:** All forecasts and intervals are strictly educational demonstrations and do not constitute financial advice.
