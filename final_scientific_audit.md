# Final Scientific Audit — Predictive Insights (Reconciled Edition)

**Project Title:** Predictive Insights – Intelligent Stock Market Forecasting System  
**Audit Scope:** Full Reconciled Independent Verification of Data Integrity, Boundary Embargo/Purging, Walk-Forward Validation, Directional Classification, Statistical Significance, and Conformal Uncertainty Calibration.  
**Auditor:** Independent Scientific Audit Protocol  
**Audit Date:** November 19, 2026  
**Final Champion Model:** CatBoost Gradient Boosted Trees (`CatBoostForecaster`)  

---

## 1. Verified Findings

### 1.1 Dataset and Split Reconciliation
- **Total Clean Dataset:** 2,989 calendar-synchronized daily trading sessions spanning **January 2, 2015 to November 19, 2026** for both the target asset (`AAPL`) and macroeconomic benchmark (`SPY`).
- **Raw Partitioning (80:20):**
  - Raw Development Set: **2,392 sessions** (2015-01-02 to 2024-07-05)
  - Raw Holdout Test Set: **597 sessions** (2024-07-08 to 2026-11-19)
  - Verification: $2,392 + 597 = 2,989$ sessions (Exact match).
- **Traceability of the 66 Dropped Preprocessing Rows:**
  - **Warm-Up Drop (65 rows, indices 0 to 64):** 2015-01-02 to 2015-04-07. Rolling indicators (`Vol_60d`, `Rolling_Beta_60`, `Ratio_Close_SMA_50`) require up to 60 historical trading days to compute valid statistics without lookahead bias.
  - **Terminal Row Drop (1 row, index 2,988):** 2026-11-19. Forward forecasting targets (`Target_Next_Close` and `Target_Next_Log_Return`) are generated via `.shift(-1)`. For the most recent trading session, tomorrow's price is unobserved ($t+1 = \text{NaN}$).
  - **Total Usable Sessions:** $2,989 - 65 - 1 = 2,923$ sessions (2015-04-08 to 2026-11-18).
- **Holdout Evaluation Protocol & Re-Evaluation Disclosure:**
  - *Disclosure:* The holdout dataset was initially evaluated as an out-of-sample block, and subsequently re-evaluated during the boundary leakage audit and purging pass.
  - Exactly **596 sessions** have verified forward targets (the terminal raw row has no forward target and is dropped).
  - To eliminate forward target leakage at the boundary, row `2024-07-05` was purged from the development set, yielding **2,326 clean development sessions** whose targets strictly precede the holdout start date (`2024-07-08`).

---

### 1.2 Data Leakage & Boundary Embargo Audit
- **Feature Computation:** Every one of the 25 selected technical and macroeconomic features is computed strictly causally using historical information available at or before trading close $t$. No forward shifts (`shift(-k)`) or future High/Low/Close/Volume were used in any predictor.
- **Boundary Embargo/Purging:** 
  - Standard `.shift(-1)` target assignment creates a 1-observation boundary overlap: the final development session (Friday, July 5, 2024) carries a target label realizing on Monday, July 8, 2024 (the first day of the holdout set).
  - Applying a 1-observation purge (dropping July 5, 2024 from training) guarantees that no holdout observation is ever used as a training label.
  - Purging shifts CatBoost holdout RMSE from **\$4.2913** to **\$4.3009** (a delta of only **+\$0.0090 USD** or **+0.2%**), proving that boundary leakage did not drive the original results.

---

### 1.3 Walk-Forward Cross-Validation Audit (Purged Primary Result)
The usable development dataset (2,326 sessions) was evaluated across 5 expanding folds with boundary purging applied to eliminate target overlap between training and validation:

| Fold | Training Dates | Train $N$ | Validation Dates | Val $N$ | CatBoost RMSE | CatBoost MAE | Dir Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2015-04-08 to 2019-11-14 | 1,162 | 2019-11-18 to 2020-10-19 | 232 | \$2.4688 | \$1.7097 | 57.33% |
| **Fold 2** | 2015-04-08 to 2020-10-16 | 1,394 | 2020-10-20 to 2021-09-21 | 232 | \$2.1948 | \$1.6839 | 49.57% |
| **Fold 3** | 2015-04-08 to 2021-09-20 | 1,626 | 2021-09-22 to 2022-08-23 | 232 | \$3.0647 | \$2.3673 | 53.02% |
| **Fold 4** | 2015-04-08 to 2022-08-22 | 1,858 | 2022-08-24 to 2023-07-27 | 232 | \$2.7466 | \$2.0665 | 49.14% |
| **Fold 5** | 2015-04-08 to 2023-07-26 | 2,090 | 2023-07-28 to 2024-07-05 | 236 | \$2.6319 | \$1.9269 | 53.39% |

- **Primary Cross-Validation Metrics:**
  - **Purged Pooled RMSE:** **\$2.6373 USD** (computed over all 1,164 concatenated out-of-sample predictions).
  - **Purged Pooled MAE:** **\$1.9508 USD**.
  - **Purged Unweighted Mean Fold RMSE:** **\$2.6214 USD** ($\sigma = \$0.2890$).
  - **Purged Pooled Directional Accuracy:** **52.49%** (Unweighted Mean Fold Dir Acc: **52.49%**, Pooled UP F1: **65.80%**).

---

### 1.4 Directional Classification & Baseline Comparison (Holdout 596 Sessions)
The holdout dataset contains **321 UP sessions (53.86%)** and **275 DOWN sessions (46.14%)**.

```
  Persistence Baseline Confusion Matrix:
                 Pred DOWN    Pred UP
    Actual DOWN     275          0       (100.0% Recall)
    Actual UP       321          0       (  0.0% Recall)
    Directional Accuracy: 46.14% | UP F1: 0.00% | DOWN F1: 63.15%

  Majority-Class (Always UP) Confusion Matrix:
                 Pred DOWN    Pred UP
    Actual DOWN       0        275       (  0.0% Recall)
    Actual UP         0        321       (100.0% Recall)
    Directional Accuracy: 53.86% | UP F1: 70.01% | DOWN F1: 0.00%

  CatBoost Forecaster (Purged Diagnostic Result):
                 Pred DOWN    Pred UP
    Actual DOWN      25        250       (  9.09% Recall)
    Actual UP        39        282       ( 87.85% Recall)
    Directional Accuracy: 51.51% | Balanced Accuracy: 48.47%
    UP Class:   Precision = 52.81% | Recall = 87.85% | F1 = 66.12%
    DOWN Class: Precision = 39.06% | Recall =  9.09% | F1 = 14.75%
```

> [!IMPORTANT]
> **Directional Evaluation Reality:**
> CatBoost achieves **51.51%** directional accuracy. While higher than the arbitrary **46.14%** persistence tie-break score, it **DOES NOT beat the naive Majority-Class (Always UP) baseline of 53.86%**.

---

### 1.5 Statistical Significance Testing (Recalculated on Purged Holdout)
- **Diebold-Mariano Test (CatBoost vs. Persistence):**
  - Loss Function: Squared price error $L(e) = e^2$.
  - Sample Size: $N = 596$ sessions.
  - Mean Loss Differential: $\bar{d} = +0.0663 \text{ USD}^2$.
  - Newey-West HAC Standard Error ($h=1$): $0.4051$.
  - 95% Confidence Interval for $\bar{d}$: $[-\$0.7277, +\$0.8604] \text{ USD}^2$ (crosses zero).
  - DM Statistic: $DM = 0.1637$.
  - Asymptotic $p$-value: $p = 0.8699$.
  - Scientific Interpretation: Fail to reject $H_0$ ($p \gg 0.05$). CatBoost performs on par with the random walk persistence benchmark; there is no statistically significant difference in price forecasting error.
- **Wilcoxon Signed-Rank Test:** $W = 87,762.0, p = 0.7770$.

---

### 1.6 Conformal Prediction Calibration Audit (Recalculated on Purged Development)
- **Calibration Split:** Sub-train $N=1,860$ (2015-04-08 to 2022-08-24), Calibration $N=466$ (2022-08-25 to 2024-07-03).
- **Holdout Empirical Coverage:**
  - Nominal 95% Interval: **94.63% Empirical Coverage** (Avg Width: **\$18.50**, Gap: **-0.37%**).
  - Nominal 90% Interval: **87.58% Empirical Coverage** (Avg Width: **\$13.22**, Gap: **-2.42%**).
- **Regime Breakdown:**
  - Low-Volatility Regime (Vol $\le 1.50\%, N=298$): Empirical Coverage = **92.62%**, Avg Width = **\$13.42**
  - High-Volatility Regime (Vol $> 1.50\%, N=298$): Empirical Coverage = **96.64%**, Avg Width = **\$23.18**

---

## 2. Reconciliation of CatBoost Holdout RMSE Values

| Value | Pipeline Source | Description & Methodological Context | Status |
| :---: | :---: | :--- | :---: |
| **\$4.2896** | Initial Holdout Script (`run_final_holdout_evaluation.py`) | Single-pass evaluation on unpurged dev partition with initial feature warm-up offsets. | Superseded |
| **\$4.2913** | Pre-Audit Script (`audit_reconciliation.py`) | Evaluated on full development partition ($N=2,327$) where final row (`2024-07-05`) had target label `2024-07-08`. | Superseded |
| **\$4.3009** | Purged Holdout Evaluation | **Corrected Final Diagnostic Result:** 1-observation boundary purge applied ($N=2,326$) so that no training target extends into holdout. | **Authoritative Final Result** |

The difference between \$4.2896 and \$4.3009 is **+\$0.0113 USD (0.26%)**, confirming that boundary leakage did not materially affect experimental findings.

---

## 3. Final Defensible Model-Selection Conclusion

1. **CatBoost is the Defensible Champion Model:**
   - Outperforms deep learning models (LSTM, GRU, CNN-LSTM, TCN) on purged walk-forward cross-validation (Pooled RMSE \$2.6373 vs \$2.70 – \$2.80) and latency (0.002 ms vs 0.033 ms).
   - Matches Random-Walk Persistence on holdout RMSE (\$4.3009 vs \$4.3086, $p = 0.8699$).
2. **Transparent Boundaries:**
   - CatBoost achieves **statistical parity** with persistence, not superiority.
   - CatBoost does **not beat** the 53.86% Majority-Class baseline on a bullish holdout period.
   - Multi-day forecasts ($H > 1$) are simulated scenario rollouts; only Day 1 is an empirically validated ML prediction.
