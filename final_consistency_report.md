# Final Consistency Report — Predictive Insights

**Document Title:** Final Consistency, Metric Reconciliation, and Scientific Integrity Report  
**Project:** Predictive Insights – Intelligent Stock Market Forecasting System  
**Date:** November 19, 2026  
**Auditor Protocol:** Rigorous Independent Quality Assurance & Econometric Audit  
**Champion Model:** Regularized Symmetric Tree Gradient Boosting (`CatBoostForecaster`)  

---

## 1. Executive Summary & Audit Mandate

This document serves as the final, authoritative reconciliation pass for the **Predictive Insights** machine learning stock forecasting system. It traces and resolves all historical metric discrepancies, discloses the complete holdout evaluation history, provides unambiguous definitions for pooled versus unweighted cross-validation metrics, recalculates all dependent statistical tests, and catalogs every remaining scientific limitation.

---

## 2. Comprehensive Metric Reconciliation Across Iterations

Over the lifecycle of this research project, iterative methodological audits identified minor configuration discrepancies, warm-up boundary nuances, and fold-boundary target overlaps. The table below provides complete traceability for every historical and final metric:

| Metric Category | Metric Description | Historical / Pre-Audit Value | Corrected Final Diagnostic Value | Exact Source & Methodological Explanation |
| :--- | :--- | :---: | :---: | :--- |
| **Holdout RMSE** | CatBoost Next-Day Price RMSE | **$4.2896** | — | Preliminary exploratory script (`run_final_holdout_evaluation.py`) before formal warm-up index alignment. |
| **Holdout RMSE** | CatBoost Next-Day Price RMSE | **$4.2913** | — | Pre-audit holdout run on full dev set ($N=2,327$), where final row (`2024-07-05`) carried forward target `2024-07-08`. |
| **Holdout RMSE** | CatBoost Next-Day Price RMSE | **$4.3003** | — | Intermediate diagnostic slice evaluated during preliminary boundary audit checks. |
| **Holdout RMSE** | CatBoost Next-Day Price RMSE | — | **$4.3009** | **Authoritative Final Diagnostic Result:** Full 1-observation boundary purge applied ($N=2,326$) on 596 holdout sessions. Total delta across all holdout configurations is only +$0.0113 USD (+0.26%). |
| **Baseline RMSE** | Persistence (Random Walk) | $4.3086 | **$4.3086** | Predicts tomorrow's price equals today's close: $\hat{P}_{t+1} = P_t$. Unchanged. |
| **Baseline RMSE** | Majority-Class (Always UP) | — | **$4.8584** | Predicts median historical positive return on every session. |
| **Cross-Validation** | Purged Pooled CV RMSE | — | **$2.6373** | **Primary Cross-Validation Metric:** Computed over all 1,164 concatenated out-of-sample predictions across 5 expanding folds. |
| **Cross-Validation** | Purged Pooled CV MAE | — | **$1.9508** | Mean absolute error across all 1,164 pooled cross-validation predictions. |
| **Cross-Validation** | Purged Mean Fold RMSE | $2.6223 | **$2.6214** | Unweighted arithmetic mean of individual fold RMSEs ($\sigma = \$0.2890$). |
| **Cross-Validation** | Unpurged Mean Fold RMSE | $2.6408 | — | Superseded by purged expanding-fold validation. |
| **Hypothesis Testing**| Diebold-Mariano Statistic | $DM = 0.370$ | **$DM = 0.1637$** | Recalculated on purged holdout ($N=596$) using Newey-West HAC standard error ($h=1, \text{SE}=0.4051$). |
| **Hypothesis Testing**| Diebold-Mariano $p$-value | $p = 0.7117$ | **$p = 0.8699$** | Asymptotic two-sided test against persistence ($H_0: \text{Equal Accuracy}$). Fail to reject null. |
| **Hypothesis Testing**| DM 95% Confidence Interval | — | **[-$0.7277, +$0.8604] USD²** | Mean loss differential $\bar{d} = +0.0663 \text{ USD}^2$; interval heavily crosses zero. |
| **Hypothesis Testing**| Wilcoxon Signed-Rank Test | $p = 0.6800$ | **$W = 87,762.0, p = 0.7770$** | Non-parametric paired absolute error test against persistence. Fail to reject null. |
| **Directional Acc** | CatBoost Directional Accuracy | 50.84% / 51.17% | **51.51%** | 307 correct directional calls out of 596 sessions (Balanced Accuracy: 48.47%). |
| **Directional Acc** | Persistence Directional Acc | 46.21% | **46.14%** | Arbitrary tie-breaking convention: predicted zero return ($\hat{y} = 0$) is classified as DOWN ($\hat{y} \le 0$). |
| **Directional Acc** | Majority-Class Directional Acc | — | **53.86%** | Constant bullish classification (predicts UP on all sessions). |
| **Directional F1** | CatBoost UP Class F1 | 65.50% | **66.12%** | Precision = 52.81%, Recall = 87.85% (282 / 321 actual UP sessions caught). |
| **Directional F1** | CatBoost DOWN Class F1 | 14.10% | **14.75%** | Precision = 39.06%, Recall = 9.09% (25 / 275 actual DOWN sessions caught). |
| **Uncertainty** | 95% Conformal Coverage | 89.38% (Old) | **94.63%** | Split-conformal calibration on purged sub-train/calibration partitions. Avg width: $18.50 (Gap: -0.37%). |
| **Uncertainty** | 90% Conformal Coverage | — | **87.58%** | Split-conformal calibration on purged sub-train/calibration partitions. Avg width: $13.22 (Gap: -2.42%). |

---

## 3. Disclosure of Holdout Evaluation History

In accordance with strict scientific disclosure standards, we state unequivocally:

> [!WARNING]
> **Holdout Dataset Re-Evaluation Disclosure:**
> The 596-session holdout partition (spanning July 8, 2024 to November 18, 2026) **must NOT be described as "untouched throughout development", "completely frozen", or "evaluated exactly once".**
> 
> **History of Evaluations:**
> 1. **Initial Out-of-Sample Evaluation:** The holdout was initially evaluated after development model selection as an out-of-sample test block, yielding preliminary RMSEs of $4.2896 and $4.2913.
> 2. **Boundary Leakage Audit:** During an in-depth chronological alignment audit, a 1-observation target overlap was uncovered: because forward targets were defined via `.shift(-1)`, the final development row (Friday, July 5, 2024) possessed a target realization timestamp corresponding to Monday, July 8, 2024 (the first holdout observation).
> 3. **Subsequent Purged Re-Evaluation:** To establish 100% airtight econometric boundaries, row `2024-07-05` was purged from training ($N=2,326$), and the model was re-evaluated against the holdout, yielding the authoritative diagnostic result of **$4.3009 USD**.
> 
> Because the holdout was evaluated multiple times during audit and diagnostic reconciliation, it is accurately designated as a **diagnostic holdout verification set**, not a virgin, single-look test set.

---

## 4. Pooled vs. Unweighted Mean Cross-Validation Metrics

To prevent ambiguity in cross-validation reporting, we distinguish between **Pooled Cross-Validation RMSE** and **Unweighted Mean Fold RMSE**:

### 4.1 Purged Pooled Cross-Validation RMSE: **$2.6373 USD** (Primary Metric)
- **Definition:** The pooled RMSE evaluates all 1,164 out-of-sample prediction pairs $(P_{i}, \hat{P}_{i})$ generated across all 5 expanding folds as a single continuous vector:
  $$\text{Pooled RMSE} = \sqrt{\frac{1}{N_{\text{total}}} \sum_{k=1}^{K} \sum_{i \in \text{Val}_k} (P_i - \hat{P}_i)^2}$$
  where $N_{\text{total}} = 232 + 232 + 232 + 232 + 236 = 1,164$ sessions.
- **Why it is Primary:** In financial time series with expanding training windows, pooling observations weights each historical test day equally, reflecting the system's aggregate expected prediction error across all out-of-sample regimes.

### 4.2 Purged Unweighted Mean Fold RMSE: **$2.6214 USD** ($\sigma = \$0.2890$)
- **Definition:** The simple arithmetic mean of the individual RMSE values computed separately for each fold:
  $$\text{Mean Fold RMSE} = \frac{1}{K} \sum_{k=1}^{K} \text{RMSE}_k$$
- **Fold Breakdown:**
  - Fold 1 (2019-11-18 to 2020-10-19, $N=232$): **$2.4688**
  - Fold 2 (2020-10-20 to 2021-09-21, $N=232$): **$2.1948**
  - Fold 3 (2021-09-22 to 2022-08-23, $N=232$): **$3.0647**
  - Fold 4 (2022-08-24 to 2023-07-27, $N=232$): **$2.7466**
  - Fold 5 (2023-07-28 to 2024-07-05, $N=236$): **$2.6319**
- **Mathematical Distinction:** Because RMSE is a non-linear concave operator (Jensen's inequality) and fold sizes differ slightly ($N_5 = 236$ vs $232$), $\text{Pooled RMSE} \neq \text{Mean Fold RMSE}$. Both metrics are mathematically valid and now explicitly reported.

---

## 5. Recalculated Statistical Comparisons

All inferential statistics have been re-estimated on the purged holdout observations ($N=596$):

### 5.1 Diebold-Mariano Test (Price Forecasting Parity)
- **Null Hypothesis ($H_0$):** The CatBoost model and Random-Walk Persistence baseline have equal forecast accuracy under squared-error loss.
- **Loss Differential Series:** $d_t = (P_t - \hat{P}_{t, \text{Persistence}})^2 - (P_t - \hat{P}_{t, \text{CatBoost}})^2$
- **Sample Mean:** $\bar{d} = +0.0663 \text{ USD}^2$
- **Newey-West HAC Variance ($h=1$):** $0.1641 \implies \text{HAC SE} = 0.4051$
- **95% Confidence Interval:** $[-\$0.7277, +\$0.8604] \text{ USD}^2$
- **Test Statistic:** $DM = \frac{0.0663}{0.4051} = 0.1637$
- **$p$-value:** $p = 2 \cdot (1 - \Phi(|0.1637|)) = 0.8699$
- **Inference:** $p \gg 0.05$. There is zero statistical evidence of superior predictive accuracy over a random walk. CatBoost and Persistence perform in statistical parity on next-day price RMSE.

### 5.2 Directional Accuracy vs. Baselines
- **Holdout Composition:** 321 UP days (53.86%), 275 DOWN days (46.14%).
- **CatBoost Forecaster:** Directional Accuracy = **51.51%** (Balanced Accuracy: 48.47%).
- **Persistence (Zero-Return Tie):** Directional Accuracy = **46.14%** (Classifies all zero-return forecasts as DOWN).
- **Majority-Class (Always UP):** Directional Accuracy = **53.86%** (Matches the underlying upward drift of the bull market).
- **Inference:** While CatBoost outperforms the artificial 46.14% persistence score created by the zero-tie convention, it **fails to beat the naive 53.86% Majority-Class baseline**. The model exhibits a persistent positive bias (predicting UP on 89.26% of days), which yields an UP recall of 87.85% but a DOWN recall of only 9.09%.

### 5.3 Conformal Uncertainty Calibration
- **Nominal 95% Prediction Interval:** Empirical Holdout Coverage = **94.63%** (Average Width: **$18.50**, Coverage Gap: **-0.37%**).
- **Nominal 90% Prediction Interval:** Empirical Holdout Coverage = **87.58%** (Average Width: **$13.22**, Coverage Gap: **-2.42%**).
- **Regime Robustness:**
  - Low-Volatility Periods ($\text{Vol} \le 1.50\%, N=298$): Coverage = **92.62%**, Avg Width = **$13.42**
  - High-Volatility Periods ($\text{Vol} > 1.50\%, N=298$): Coverage = **96.64%**, Avg Width = **$23.18**

---

## 6. Comprehensive Catalog of Remaining Scientific Limitations

To ensure absolute transparency and prevent unwarranted claims, we explicitly catalog all remaining limitations of the forecasting system:

1. **Market Efficiency & Random-Walk Parity:**
   At a 1-day horizon, equity price dynamics are dominated by stochastic news arrival. CatBoost achieves statistical parity ($p = 0.8699$), not superiority, compared to the random-walk persistence baseline. No model demonstrated statistically significant alpha on price prediction using publicly available daily OHLCV features alone.

2. **No Directional Alpha over Trend Following:**
   The model's 51.51% directional accuracy is inferior to the naive Majority-Class baseline (53.86%). The model primarily reflects the underlying equity drift during the 2024–2026 bull market, rather than genuine symmetric market-timing capability.

3. **Multi-Day Recursive Forecast Degradation:**
   Forecasts for horizons $H \in \{5, 7, 15, 30\}$ are generated via recursive auto-regressive rollout. Because future volume, benchmark prices, and intraday ranges cannot be known in advance, they are held constant or bounded synthetically. Errors compound rapidly over multi-step horizons, causing the trajectory to damp toward mean drift. Only the Day 1 forecast is an empirically validated ML prediction.

4. **Asset-Specific Validation Scope:**
   Rigorous stationarity auditing, lookback window optimization ($L=20$), expanding walk-forward cross-validation, and holdout benchmarking were conducted exclusively on Apple Inc. (`AAPL`). While the application trains fresh per-stock models for other tickers (`NVDA`, `MSFT`, `RELIANCE.NS`), multi-stock support represents architectural pipeline generality, not certified out-of-sample accuracy for those assets.

5. **Violation of Exchangeability in Conformal Prediction:**
   Split-conformal prediction guarantees valid marginal coverage under the assumption that calibration and test errors are exchangeable. Financial time series violate exchangeability due to volatility clustering, ARCH/GARCH dynamics, and macro regime shifts. Although empirical coverage on the holdout was 94.63%, coverage may degrade during sudden, unprecedented market volatility shocks.

6. **Absence of Execution Realities:**
   The model does not account for transaction fees, bid-ask spreads, market impact, short-borrow fees, or order execution latency. A model with 51.51% directional accuracy and parity RMSE cannot generate positive trading profit after accounting for friction costs.

---

## 7. Final Verification Checklist

- [x] CatBoost holdout RMSE values ($4.2896, $4.2913, $4.3009) traced to exact scripts and reconciled.
- [x] All holdout descriptions updated to disclose boundary audit and re-evaluation.
- [x] Purged walk-forward results ($2.6373 pooled, $2.6214 mean fold) established as primary CV metrics.
- [x] Diebold-Mariano test recalculated ($DM = 0.1637, p = 0.8699, \text{CI}: [-\$0.7277, +\$0.8604]$).
- [x] Directional classification compared fairly against both Persistence (46.14%) and Majority-Class (53.86%).
- [x] Multi-day recursive forecasting transparently badged in the UI as scenario simulations.
- [x] Research report, scientific audit, validity review, README, and dashboard fully synchronized.
- [x] Complete automated test suite passing with zero regressions.
