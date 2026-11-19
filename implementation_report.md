# Implementation Report — EXP2 Integration & Dashboard Simplification

**Project:** Predictive Insights – Intelligent Stock Market Forecasting System  
**Document:** System Integration and User Interface Simplification Report  
**Date:** November 19, 2026  
**Status:** Successfully Implemented, Tested, and Deployed  

---

## 1. Executive Summary

This report documents the end-to-end integration of the champion **EXP2 configuration (CatBoost Regressor + Train-Tuned Return Threshold)** into the core forecasting pipeline, alongside the complete redesign and simplification of the user-facing Streamlit dashboard.

The application has been refined into a clean, presentation-ready, next-day-only forecasting tool. All multi-day recursive rollout logic and synthetic future OHLCV generation have been removed from the user interface. Continuous log-return regression is fully preserved for accurate next-day closing price estimation, while directional movement (UP/DOWN) is classified using an optimal return threshold learned strictly from historical training data.

---

## 2. Summary of Changes Made

1. **EXP2 Forecaster Integration:**
   - Enhanced `src/models/trees.py` to equip `CatBoostForecaster` with train-time threshold tuning (`tune_threshold=True`).
   - Created explicit aliases `CatBoostEXP2Forecaster` (champion generalist) and `CatBoostBaselineForecaster` (unweighted $\tau=0.0$ baseline comparison).
   - Implemented `CatBoostDownsideSpecialist` (EXP13: direct `CatBoostClassifier` with `auto_class_weights='Balanced'`) as an experimental downside-specialist model.
   - Updated `BaseForecaster.predict_direction` in `src/models/base.py` to use `self.learned_threshold` by default.

2. **Evaluation Pipeline Updates:**
   - Updated `src/evaluation/metrics.py` to support precomputed binary prediction vectors (`pred_directions`) and return comprehensive Macro F1, Weighted F1, Macro Precision, and Macro Recall.
   - Updated `src/evaluation/walk_forward.py` so that during cross-validation, each expanding fold independently tunes its threshold using only its own historical training slice, preventing any out-of-fold leakage.

3. **Dashboard Simplification (`app.py`):**
   - Completely removed the prediction horizon dropdown (no 5, 7, 15, or 30-day options).
   - Removed recursive auto-regressive trajectory rollout and synthetic future bar construction from the production app.
   - Simplified workflow: **Select Stock Ticker → Click "🚀 Predict Next Trading Day" → View Predicted Price and Direction.**
   - Implemented calendar-aware next-session dating (`pd.bdate_range`) that properly skips weekends and non-trading days.
   - Displayed 4 clean, focused metric cards: Latest Actual Close, Predicted Next-Day Close, Expected Dollar & Percentage Change, and Predicted Direction (UP ↗ / DOWN ↘) with the EXP2 decision rule.
   - Added a clean interactive Plotly chart showing recent historical prices and the single connected next-day predicted point.
   - Reduced the forecast table to a single clear row detailing the next trading session.

---

## 3. Files Modified and Created

| File Path | Action | Description |
| :--- | :---: | :--- |
| `src/models/base.py` | Modified | Updated `predict_direction` to use `self.learned_threshold` when threshold is not explicitly passed. |
| `src/models/trees.py` | Modified | Added train-tuned return threshold logic to `CatBoostForecaster`, added `CatBoostEXP2Forecaster`, `CatBoostBaselineForecaster`, and `CatBoostDownsideSpecialist`. |
| `src/evaluation/metrics.py` | Modified | Extended `compute_directional_metrics` to support `pred_directions` and compute Macro/Weighted F1. |
| `src/evaluation/walk_forward.py` | Modified | Enabled independent fold-by-fold threshold tuning and pooled direction metric tracking. |
| `app.py` | Modified | Completely redesigned Streamlit dashboard for next-day-only predictions using EXP2. |
| `tests/test_exp2_integration.py` | Created | Added 4 automated unit tests verifying EXP2 threshold tuning, continuous price preservation, independent fold tuning, and baseline availability. |
| `README.md` | Modified | Documented EXP2 architecture, next-day dashboard, regression vs classification distinction, and limitations. |
| `intelligent_market_forecasting_research_report.md` | Modified | Added EXP2 walk-forward CV results, independent fold tuning methodology, and selection rationale over EXP13. |
| `implementation_report.md` | Created | Comprehensive report documenting integration, testing, and integrity confirmation. |

---

## 4. EXP2 Integration & Threshold-Learning Methodology

### 4.1 Architecture
- **Base Model:** Symmetric oblivious tree gradient boosting via `CatBoostRegressor` (120 iterations, depth 4, learning rate 0.02, L2 leaf reg 3.0).
- **Target Representation:** Daily forward log return $y_t = \ln(P_{t+1} / P_t)$ computed on 25 verified stationary technical and macroeconomic features.
- **Continuous Price Compounding:** Next-day closing price is reconstructed via the exact continuous exponential formula:
  $$\hat{P}_{t+1} = P_t \cdot \exp(\hat{y}_{t+1})$$

### 4.2 Threshold-Learning Protocol
1. **Inner Chronological Split:**
   Within any training set $X_{\text{train}}$ of length $N$, an inner chronological cutoff is established at index $\lfloor 0.80 \cdot N \rfloor$. The first 80% is used for inner training, and the final 20% serves as an inner validation slice.
2. **Inner Model Fitting:**
   An inner CatBoost regressor is fitted on the 80% slice to generate out-of-sample predicted log returns on the 20% validation slice.
3. **Grid Optimization:**
   The algorithm searches candidate return thresholds $\tau \in [-0.004, +0.004]$ (81 candidate values) to find $\tau^*$ that maximizes Balanced Accuracy subject to an UP recall constraint:
   $$\tau^* = \arg\max_{\tau} \text{Balanced Accuracy}(\tau) \quad \text{s.t.} \quad \text{UP Recall}(\tau) \ge 50\%$$
4. **Final Fit:**
   The learned threshold $\tau^*$ is stored as `self.learned_threshold`, and the main model is fitted on the full $100\%$ training set.
5. **Directional Inference:**
   $$\text{Predicted Direction} = \begin{cases} \text{UP (1)} & \text{if } \hat{y}_{t+1} > \tau^* \\ \text{DOWN (0)} & \text{if } \hat{y}_{t+1} \le \tau^* \end{cases}$$

### 4.3 Deployed Parameter Values
- **Full Development Set ($N = 2,326$ sessions):** Inner train spans April 2015 to August 2022 ($N = 1,860$), inner validation spans August 2022 to July 2024 ($N = 466$).
- **Learned Deployed Threshold:** $\tau^* = +0.001000$ (+0.10% return hurdle).
- **Inner Validation Results:** Balanced Accuracy = **51.29%**, UP Recall = **51.23%**, DOWN Recall = **51.35%**.

---

## 5. Development Walk-Forward Results (EXP2 vs. Baselines)

Evaluated across 5 expanding folds with boundary purging on the development dataset ($N = 1,164$ out-of-sample validation pairs):

| Model Configuration | Accuracy | Balanced Acc | Macro F1 | UP F1 (Recall) | DOWN F1 (Recall) | DOWN Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost EXP2 (Deployed Champion)** | **52.88%** | **51.53%** | **48.73%** | **63.32%** (77.29%) | **34.13%** (**25.77%**) | **50.53%** |
| CatBoost Baseline ($\tau = 0.0$) | 52.45% | 50.55% | 43.99% | 65.76% (86.76%) | 22.22% (14.34%) | 49.38% |
| CatBoost EXP13 Downside Specialist (Exp) | 50.56% | 50.13% | 49.97% | 55.39% (58.33%) | 44.55% (41.92%) | 47.53% |
| Always-UP Baseline | 52.62% | 50.00% | 34.48% | 68.96% (100.0%) | 0.00% (0.00%) | 0.00% |

**Key Takeaways:**
- EXP2 outperforms the Always-UP baseline in overall accuracy (**52.88% vs 52.62%**).
- EXP2 increases DOWN recall from **14.34% to 25.77%** (+80% relative increase), catching 142 down days compared to 79.
- EXP2 achieves **50.53% DOWN precision** (exceeding random chance).
- EXP2 preserves high UP recall (**77.29%**) and UP F1 (**63.32%**).

---

## 6. Verification and Testing

### 6.1 Automated Pytest Suite
We executed the full automated pytest suite across all modules:
```bash
PYTHONPATH=. pytest tests/ -v
```
**Results:** **21 passed, 0 failed in 6.43s**

Detailed Test Breakdown:
- `test_calendar.py::test_nyse_skips_weekends`: **PASSED** (Weekend rollover to Monday verified).
- `test_calendar.py::test_nyse_skips_thanksgiving`: **PASSED** (Thanksgiving holiday skipping to Friday verified).
- `test_calendar.py::test_nyse_skips_good_friday`: **PASSED** (Good Friday holiday skipping to Monday verified).
- `test_calendar.py::test_nyse_skips_christmas`: **PASSED** (Christmas holiday rollover verified).
- `test_calendar.py::test_nse_skips_republic_day`: **PASSED** (Indian market holiday skipping verified).
- `test_calendar.py::test_calendar_safe_fallback`: **PASSED** (Safe business-day fallback on exception verified).
- `test_dashboard_forecast.py::test_dashboard_next_day_forecast_single_prediction`: **PASSED** (Single-day forecast, date, and price/pct delta verified).
- `test_dashboard_forecast.py::test_popular_stocks_dropdown_options`: **PASSED** (Presence of all required popular stock tickers verified).
- `test_data_leakage.py::test_no_temporal_overlap_in_development_holdout_split`: **PASSED** (Monotonic date separation verified).
- `test_data_leakage.py::test_walk_forward_expanding_folds_chronological_ordering`: **PASSED** (Expanding folds and boundary embargo verified).
- `test_evaluation.py::test_regression_metrics_exact_values`: **PASSED** (RMSE/MAE exact calculation verified).
- `test_evaluation.py::test_directional_metrics_confusion_matrix`: **PASSED** (Confusion matrix & balanced metrics verified).
- `test_evaluation.py::test_diebold_mariano_zero_loss_differential`: **PASSED** (HAC variance and p-value verified).
- `test_evaluation.py::test_conformal_calibrator_coverage`: **PASSED** (Non-conformity scoring & coverage verified).
- `test_exp2_integration.py::test_catboost_exp2_threshold_tuning`: **PASSED** (EXP2 threshold learning and price preservation verified).
- `test_exp2_integration.py::test_catboost_baseline_zero_threshold`: **PASSED** (Baseline $\tau=0.0$ default verified).
- `test_exp2_integration.py::test_walk_forward_evaluates_exp2_independently`: **PASSED** (Independent fold-wise threshold tuning verified).
- `test_exp2_integration.py::test_exp13_downside_specialist_non_default`: **PASSED** (EXP13 non-default specialist verified).
- `test_models.py::test_persistence_price_reconstruction`: **PASSED** (Persistence baseline verified).
- `test_models.py::test_tree_forecasters_produce_valid_predictions`: **PASSED** (Trees output valid positive prices).
- `test_models.py::test_deep_learning_models_handle_sequence_padding`: **PASSED** (Sequence models verified).

### 6.2 Streamlit Application End-to-End Verification
- **Execution Test:** Executed `run_next_day_forecast` directly on live market data (`AAPL` and `MSFT`).
- **Verified Primary Benchmark Output (AAPL):**
  - Training Partition: Fitted strictly on $N = 2,326$ development sessions (2015-04-08 to 2024-07-03). Zero diagnostic holdout sessions in training or tuning.
  - `last_date`: 2026-11-19
  - `last_close`: \$349.83
  - `next_date`: 2026-11-20 (Friday; authentic next trading session from `NYSEExchangeCalendar`)
  - `predicted_price`: \$350.25 (positive, finite dollar price)
  - `price_change`: +\$0.42 USD
  - `pct_change`: +0.12%
  - `predicted_direction`: UP (consistent with predicted return $\hat{y}_{t+1} = +0.1200\% > \tau^* = +0.1000\%$)
  - `learned_threshold`: $+0.001000$ (exact calibrated cutoff from development calibration split)
- **Verified Dynamic Transfer Output (MSFT):**
  - `last_date`: 2026-11-19
  - `last_close`: \$509.15
  - `next_date`: 2026-11-20 (Friday)
  - `predicted_price`: \$510.36
  - `price_change`: +\$1.21 USD (+0.24%)
  - `predicted_direction`: UP
- **Server Status:** Streamlit service running continuously on port 8501 returning HTTP 200 OK.

---

## 7. Confirmation of Evaluation Integrity & Holdout Disclosure

In strict adherence to the project's scientific constraints:
1. **Holdout Evaluation Disclosure:** The 596-session diagnostic holdout (spanning July 8, 2024 to November 18, 2026) was previously evaluated during Stage 2 and the boundary audit, and is therefore documented honestly as a **diagnostic holdout**, not as an "untouched" or unobserved dataset.
2. **Zero Diagnostic Holdout Leakage into Training/Tuning:** In this EXP2 integration, zero diagnostic holdout observations were used for model fitting, feature scaling, or threshold calibration. For AAPL, the model is trained strictly on the 2,326 clean development sessions (2015-04-08 to 2024-07-03) with boundary embargo applied.
3. **Honest Reporting & Metric Separation:** All EXP2 directional improvement claims (52.88% accuracy, 51.53% balanced accuracy, 34.13% DOWN F1) represent **development walk-forward cross-validation results** across 1,164 out-of-sample validation pairs, clearly distinguished from diagnostic holdout evaluations.
4. **Preservation of Baseline Comparisons:** The unweighted CatBoost regression baseline ($\tau=0.0$) and holdout Diebold-Mariano test results ($p=0.8699$) remain fully documented.
5. **Experimental Models Bounded:** EXP13 is preserved strictly as an experimental downside-specialist model and is not the default.

---

## 8. Remaining Limitations

1. **Stochastic Nature of Daily Prices:** Next-day stock prices possess low signal-to-noise ratios. No statistical model can eliminate false alarms or guarantee directional profitability across all market environments.
2. **Bull Market Drift vs. Downside Detection:** While EXP2 successfully increases down-day detection from 14% to 26%, equity market dynamics and positive secular drift inherently constrain down-day precision.
3. **Transaction Costs:** Real-world trading involves bid-ask spreads, borrow fees, and execution slippage, which are not modeled in this educational forecasting pipeline.

---

## 9. Final Audit Checklist & Verification Findings

| Audit Check Item | Status | Verification Detail |
| :--- | :---: | :--- |
| **1. Deployed Threshold (+0.001000)** | **VERIFIED** | Calculated exclusively from development data ($N=2,326$, 2015-04-08 to 2024-07-03) using an inner chronological 80/20 train/calibration split ($N_{\text{tr}}=1,860, N_{\text{val}}=466$). Grid search identifies $\tau^* = +0.001000$ (maximizing balanced accuracy subject to UP recall $\ge 50\%$). |
| **2. Zero Holdout in Regressor Training** | **VERIFIED** | The production forecaster isolates development rows (`usable_feats.loc[usable_feats.index < '2024-07-08'].iloc[:-1]`), training strictly on $N=2,326$ historical development observations. Zero diagnostic holdout sessions enter training or threshold tuning. |
| **3. Independent Fold-by-Fold Tuning** | **VERIFIED** | In walk-forward cross-validation (`src/evaluation/walk_forward.py`), every expanding fold instantiates a fresh `CatBoostEXP2Forecaster` that splits its training window into an inner 80/20 slice to learn $\tau^*$ independently before predicting on that fold's validation slice. |
| **4. Exchange Calendar Implementation** | **VERIFIED** | `src/data/calendar.py` implements `NYSEExchangeCalendar` and `NSEExchangeCalendar` skipping weekends and exchange holidays (Thanksgiving, Christmas, Good Friday, Republic Day, etc.) with a safe business-day fallback. Verified with 6 dedicated tests. |
| **5. Next-Trading-Day-Only UI** | **VERIFIED** | Dashboard in `app.py` has no horizon dropdown (no 5, 7, 15, or 30-day options) and zero recursive forecasting logic. It produces strictly one forecast for session $t+1$. |
| **6. Complete UI Components** | **VERIFIED** | Verified all visual components: Ticker dropdown with popular stocks & custom input, prominent "🚀 Predict Next Trading Day" button, 4 metric cards (Latest Close, Predicted Close, Expected Change, Predicted Direction), interactive Plotly chart with 60-day historical context and connected predicted point, and single-row forecast table. |
| **7. Expected Price & Pct Change Formulas** | **VERIFIED** | Price change ($\Delta P = \hat{P}_{t+1} - P_t$) and percentage change ($\frac{\Delta P}{P_t} \times 100\%$) are computed exactly, with explicit sign formatting (`+` for gains, `-` for losses) applied to both dollar amounts and percentages. |
| **8. Metric Separation & Holdout Disclosure** | **VERIFIED** | Documentation explicitly distinguishes development walk-forward CV from diagnostic holdout results. The holdout is disclosed as having undergone prior diagnostic evaluation and is never described as "untouched". |
| **9. Complete Pytest Suite** | **VERIFIED** | Full automated test suite passes: **24 passed, 0 failed in 6.63 seconds** across data leakage, evaluation, models, calendar, EXP2 integration, and UI classification test cases. |
| **10. Documentation Reconciliation** | **VERIFIED** | `implementation_report.md`, `README.md`, and research reports updated with complete audit findings, test results, and limitations. |

---

## 10. UI Directional Label Disambiguation & Verification

To eliminate any visual ambiguity between continuous regression price changes ($\Delta P = \hat{P}_{t+1} - P_t$) and the statistical EXP2 hurdle rule ($\hat{y}_{t+1} > \tau^*$), the Streamlit dashboard rendering in `app.py` was refined:

### 10.1 UI Display Logic
1. **Predicted Price Movement (Card 4 & Table Column 1):**
   - **PRICE INCREASE** (green `#15803D` badge `↗`) when $\Delta P > 0$.
   - **PRICE DECREASE** (red `#B91C1C` badge `↘`) when $\Delta P < 0$.
   - **NO PRICE CHANGE** (neutral `#475569` badge `—`) when $\Delta P = 0$.
2. **EXP2 Classification (Card 5 & Table Column 2):**
   - **UP — Above threshold** (green `#15803D` badge) when predicted return $\hat{y}_{t+1} > \tau^*$.
   - **Below UP threshold** (neutral `#334155` border badge) when predicted return $\hat{y}_{t+1} \le \tau^*$.
   - **Guarantee:** A positive price change is **NEVER** labeled with a red "DOWN" badge.
   - **Context Display:** Predicted log return and the calibrated threshold are displayed together (e.g. `Pred Return: +0.05% | Threshold: +0.13%`).
3. **Interactive Chart:**
   - Next-day forecast connector line is styled green (`#16A34A`) when expected price increases and red (`#DC2626`) only when expected price decreases.

### 10.2 Verification of Required Test Cases

| Scenario | Predicted Return | Learned Cutoff ($\tau^*$) | Price Movement Output | EXP2 Classification Output |
| :---: | :---: | :---: | :---: | :---: |
| **Case 1** | **+0.05%** | **+0.13%** | **`PRICE INCREASE ↗`** (Green) | **`Below UP threshold`** (Neutral Slate) |
| **Case 2** | **+0.20%** | **+0.13%** | **`PRICE INCREASE ↗`** (Green) | **`UP — Above threshold`** (Green) |
| **Case 3** | **-0.10%** | **+0.13%** | **`PRICE DECREASE ↘`** (Red) | **`Below UP threshold`** (Neutral Slate) |
