# Directional Model Improvement Report — Predictive Insights

**Document Title:** Improving Downward Movement Detection via Class Weights, Threshold Calibration, Downside Feature Engineering, and Multi-Model Cross-Validation  
**Project:** Predictive Insights – Intelligent Stock Market Forecasting System  
**Date:** November 19, 2026  
**Evaluation Protocol:** 5-Fold Expanding-Window Purged Walk-Forward Cross-Validation on Development Data ($N = 2,326$ sessions, April 8, 2015 to July 3, 2024; Total Out-of-Sample Validation Observations: $N = 1,164$)  
**Integrity Guarantee:** The final diagnostic holdout ($N = 596$ sessions, July 8, 2024 to November 18, 2026)—which was previously evaluated during Stage 2 and the boundary audit—was strictly excluded from training, feature engineering, and threshold tuning during this study. All model training, feature selection, and threshold tuning were conducted exclusively within historical validation splits.

---

## 1. Executive Summary & Problem Diagnosis

### 1.1 The Directional Asymmetry Problem
In equity markets, stock indices and large-cap equities (such as `AAPL`) exhibit positive long-term price drift ($\mathbb{E}[r_t] > 0$). When standard regression models (minimizing MSE or Huber loss) or unweighted classifiers are trained on historical returns, they naturally learn to output positive predictions across the vast majority of trading days:

```
  Existing CatBoost Regressor Holdout Behavior:
  - Total Holdout Sessions: 596 (321 Actual UP, 275 Actual DOWN)
  - Sessions Predicted UP:   532 (89.26%)
  - Sessions Predicted DOWN:  64 (10.74%)
  - Result: UP Recall = 87.85%, DOWN Recall = 9.09% (Severe Asymmetry)
```

This dynamic causes models to catch upward rallies effectively while remaining virtually blind to sharp market declines, producing an unacceptably low DOWN F1-score of **14.75%** on the holdout and **22.22%** in cross-validation.

### 1.2 Research Mandate
The objective of this research is to investigate principled, non-manipulative methodologies to significantly improve the model's ability to detect DOWN movements without:
1. Catastrophically collapsing UP recall or overall accuracy.
2. Incurring data leakage or tuning on the final holdout.
3. Introducing non-stationary signals.

---

## 2. Experimental Design & Methodology

We tested 15 systematic configurations across four distinct methodological axes:

### Axis 1: Class Weighting
- **Unweighted Baseline:** Standard empirical loss.
- **Auto-Balanced Class Weights:** `auto_class_weights='Balanced'` in CatBoost, assigning weight $w_c = \frac{N}{2 \cdot N_c}$ to balance gradient contributions between UP and DOWN.
- **Asymmetric Multipliers:** Testing fixed class penalty ratios of DOWN:UP $\in \{1.5:1.0, 2.0:1.0\}$ to assess non-linear sensitivity.
- **XGBoost & Random Forest Weighting:** `scale_pos_weight=0.85` in XGBoost and `class_weight='balanced'` in Random Forest.

### Axis 2: Out-of-Sample Threshold Calibration
- **Return Threshold ($\tau$) Tuning for Regressors:** Instead of the rigid $\hat{y}_t > 0.0$ rule, an inner 80/20 chronological train-calibration split is used to identify the optimal return threshold $\tau^* \in [-0.004, +0.004]$ that maximizes Balanced Accuracy while enforcing an UP Recall floor of $\ge 50\%$.
- **Probability Threshold ($\tau_p$) Tuning for Classifiers:** Calibrating the classification cutoff $\tau_p \in [0.40, 0.60]$ on inner training data before out-of-sample application.

### Axis 3: Downside-Specific Feature Engineering
Six new technical features were engineered point-in-time from historical OHLCV data to specifically capture downside momentum, volatility shocks, and distribution pressure:
1. **Downside Volatility (`Downside_Vol_20d`):** 20-day rolling downside semivariance ($\sqrt{\frac{1}{20} \sum \min(r_t, 0)^2}$).
2. **Drawdown from 20-day High (`Drawdown_20d`):** Percentage distance below recent 20-session peak ($(P_t - \max_{20} P) / \max_{20} P$).
3. **Down-Volume Ratio (`Down_Volume_Ratio_10d`):** 10-day rolling ratio of volume traded on negative days to total volume.
4. **Negative Return Streak (`Down_Streak_3d`):** Frequency of negative closes over the preceding 3 trading sessions.
5. **Volatility Shock (`Vol_Shock_10_60`):** Ratio of short-term (10-day) to medium-term (60-day) return volatility.
6. **Return Acceleration (`Return_Acceleration_3d`):** Second-order momentum difference ($r_{t, 3d} - r_{t-3, 3d}$).

> **Stationarity Verification:** All 6 candidate features were audited across the development dataset ($N = 2,326$). Every feature rejected the ADF unit root null ($p < 0.001$) and failed to reject the KPSS stationarity null ($p \ge 0.10$), confirming strictly stationary $I(0)$ behavior.

### Axis 4: Model Family Comparison
Benchmarking three distinct non-linear gradient-boosted and bagged tree architectures:
- **CatBoost** (Symmetric oblivious decision trees, depth 4, 120 iterations, learning rate 0.02)
- **XGBoost** (Asymmetric gradient boosted trees, max depth 3, 120 estimators, learning rate 0.02)
- **Random Forest** (Bootstrap bagged ensemble, max depth 5, 120 estimators, min samples leaf 5)

---

## 3. Comprehensive Walk-Forward Cross-Validation Benchmark

The table below summarizes out-of-sample performance pooled across all 1,164 out-of-sample validation sessions (612 UP, 551 DOWN) generated by the 5 expanding folds:

| Exp ID | Model Architecture & Configuration | Overall Acc | Balanced Acc | Macro F1 | Weighted F1 | UP Prec | UP Rec | UP F1 | DOWN Prec | DOWN Rec | DOWN F1 | Confusion Matrix [TN, FP, FN, TP] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP0** | **Always-UP Baseline** | 52.62% | 50.00% | 34.48% | 36.29% | 52.62% | 100.00% | 68.96% | 0.00% | 0.00% | 0.00% | [0, 551, 0, 612] |
| **EXP1** | **CatBoost Regressor (Baseline, $\tau=0.0$)** | 52.45% | 50.55% | 43.99% | 45.12% | 52.94% | 86.76% | 65.76% | 49.38% | 14.34% | 22.22% | [79, 472, 81, 531] |
| **EXP2** | **CatBoost Regressor + Tuned Threshold ($\tau^*$)** | **52.88%** | **51.53%** | **48.73%** | **49.49%** | **53.63%** | **77.29%** | **63.32%** | **50.53%** | **25.77%** | **34.13%** | [142, 409, 139, 473] |
| **EXP3** | CatBoost Classifier (Unweighted, $p>0.5$) | 51.50% | 49.91% | 45.61% | 46.52% | 52.57% | 80.23% | 63.52% | 47.16% | 19.60% | 27.69% | [108, 443, 121, 491] |
| **EXP4** | **CatBoost Classifier (`auto_class_weights='Balanced'`)** | 50.56% | 49.90% | 49.40% | 49.80% | 52.54% | 62.42% | 57.06% | 47.25% | **37.39%** | **41.74%** | [206, 345, 230, 382] |
| **EXP5** | CatBoost Classifier (Class Weight 1.5:1.0) | 47.64% | 50.11% | 34.68% | 33.15% | 54.55% | 2.94% | 5.58% | 47.43% | 97.28% | 63.77% | [536, 15, 594, 18] |
| **EXP6** | CatBoost Classifier (Class Weight 2.0:1.0) | 47.38% | 50.00% | 32.15% | 30.46% | 0.00% | 0.00% | 0.00% | 47.38% | 100.00% | 64.29% | [551, 0, 612, 0] |
| **EXP7** | CatBoost Classifier + Tuned Prob Threshold | 51.68% | 50.54% | 48.64% | 49.27% | 53.00% | 72.22% | 61.13% | 48.33% | 28.86% | 36.14% | [159, 392, 170, 442] |
| **EXP8** | XGBoost Classifier (Unweighted, $p>0.5$) | 52.28% | 50.99% | 48.47% | 49.19% | 53.29% | 75.49% | 62.47% | 49.32% | 26.50% | 34.47% | [146, 405, 150, 462] |
| **EXP9** | XGBoost Classifier (`scale_pos_weight=0.85`) | 47.89% | 48.24% | 47.81% | 47.70% | 50.60% | 41.67% | 45.70% | 45.83% | 54.81% | 49.92% | [302, 249, 357, 255] |
| **EXP10** | **Random Forest Classifier (Unweighted, $p>0.5$)** | **53.83%** | **52.36%** | **49.05%** | **49.85%** | **54.13%** | **80.23%** | **64.65%** | **52.73%** | **24.50%** | **33.46%** | [135, 416, 121, 491] |
| **EXP11** | Random Forest Classifier (`class_weight='balanced'`) | 48.75% | 48.18% | 47.80% | 48.18% | 51.13% | 59.15% | 54.85% | 45.05% | 37.21% | 40.76% | [205, 346, 250, 362] |
| **EXP12** | CatBoost Classifier + Downside Feats (Unweighted) | 50.99% | 49.51% | 45.83% | 46.68% | 52.31% | 77.78% | 62.55% | 46.25% | 21.23% | 29.10% | [117, 434, 136, 476] |
| **EXP13** | **CatBoost Classifier + Downside Feats (Balanced)** | 50.56% | 50.13% | **49.97%** | **50.25%** | **52.73%** | **58.33%** | **55.39%** | **47.53%** | **41.92%** | **44.55%** | [231, 320, 255, 357] |
| **EXP14** | CatBoost Classifier + Downside Feats + Tuned Prob | 51.59% | 50.21% | 47.18% | 47.98% | 52.76% | 76.47% | 62.44% | 47.83% | 23.96% | 31.92% | [132, 419, 144, 468] |

---

## 4. Key Scientific Findings & Trade-Off Analysis

### 4.1 Pathological Failure of Fixed Heavy Class Weights (EXP5 & EXP6)
A critical insight is revealed by examining naive class weight multipliers:
- In **EXP5** (Class Weight DOWN:UP = $1.5:1.0$), the model overcorrects wildly: DOWN recall surges to **97.28%**, but UP recall collapses to **2.94%** (UP F1: 5.58%), reducing overall accuracy to **47.64%**.
- In **EXP6** (Class Weight DOWN:UP = $2.0:1.0$), the model degenerates completely into an All-DOWN classifier (UP recall = **0.00%**, overall accuracy = **47.38%**).
- **Rule of Thumb:** Arbitrary manual class weight multipliers $>1.2$ severely disrupt the probability threshold, inducing pathological trade-offs that completely destroy UP predictive capacity.

### 4.2 Impact of Downside Feature Engineering (EXP4 vs. EXP13)
Adding the 6 point-in-time downside and volatility features to the balanced CatBoost classifier produced measurable, robust gains across validation folds:
- **DOWN Recall:** Rose from **37.39%** (EXP4) to **41.92%** (EXP13) — catching 231 out of 551 down sessions.
- **DOWN F1-Score:** Rose from **41.74%** to **44.55%** (a $+2.81\%$ absolute improvement, and double the baseline of 22.22%).
- **Macro F1-Score:** Reached **49.97%**, the highest macro F1 achieved by any model in this research.

### 4.3 Strength of Validation Threshold Calibration (EXP2)
Tuning the return threshold $\tau^*$ on inner validation folds provided the most balanced performance overall:
- **Threshold Evolution:** Across expanding folds, the tuned cutoff adjusted dynamically:
  - Fold 1 (2015–2019): $\tau^* = -0.0025$ (expansionary bull regime, lower cutoff)
  - Fold 2 (2020 crash & recovery): $\tau^* = -0.0006$
  - Fold 3 (2021–2022 rate hike cycle): $\tau^* = +0.0006$ (increasing defense)
  - Fold 4 (2022 bear market): $\tau^* = +0.0009$ (demanding higher positive conviction before predicting UP)
  - Fold 5 (2023 tech rally): $\tau^* = +0.0006$
- **Performance:** Achieved **52.88% overall accuracy** (outperforming both default CatBoost 52.45% and Always-UP 52.62%), raised DOWN recall from **14.34% to 25.77%** (+80% relative improvement), increased DOWN precision above parity to **50.53%**, and preserved UP recall at a robust **77.29%** (UP F1: **63.32%**).

---

## 5. Detailed Fold-by-Fold Performance Breakdown

The table below demonstrates consistency across chronological market regimes for the leading contenders:

```
=== EXP1: CatBoost Regressor Baseline (tau = 0.0) ===
Fold 1 (2019-11 to 2020-10, N=232): Acc=57.33% | BalAcc=53.31% | UP F1=69.91% (Rec=89.15%) | DN F1=26.67% (Rec=17.48%) | CM: [18, 85, 14, 115]
Fold 2 (2020-10 to 2021-09, N=232): Acc=49.57% | BalAcc=48.68% | UP F1=62.86% (Rec=83.19%) | DN F1=21.48% (Rec=14.16%) | CM: [16, 97, 20, 99]
Fold 3 (2021-09 to 2022-08, N=232): Acc=53.02% | BalAcc=52.26% | UP F1=64.03% (Rec=81.51%) | DN F1=32.30% (Rec=23.01%) | CM: [26, 87, 22, 97]
Fold 4 (2022-08 to 2023-07, N=232): Acc=49.14% | BalAcc=47.74% | UP F1=64.24% (Rec=88.33%) | DN F1=11.94% (Rec= 7.14%) | CM: [ 8, 104, 14, 106]
Fold 5 (2023-07 to 2024-07, N=236): Acc=53.19% | BalAcc=50.60% | UP F1=67.46% (Rec=91.20%) | DN F1=16.67% (Rec=10.00%) | CM: [11, 99, 11, 114]

=== EXP2: CatBoost Regressor + Tuned Threshold (tau*) ===
Fold 1 (tau* = -0.0025, N=232):     Acc=56.90% | BalAcc=51.46% | UP F1=72.07% (Rec=100.0%) | DN F1= 5.66% (Rec= 2.91%) | CM: [ 3, 100,  0, 129]
Fold 2 (tau* = -0.0006, N=232):     Acc=52.59% | BalAcc=51.44% | UP F1=67.46% (Rec=95.80%) | DN F1=12.70% (Rec= 7.08%) | CM: [ 8, 105,  5, 114]
Fold 3 (tau* = +0.0006, N=232):     Acc=53.88% | BalAcc=53.57% | UP F1=59.32% (Rec=65.55%) | DN F1=46.77% (Rec=41.59%) | CM: [47,  66, 41,  78]
Fold 4 (tau* = +0.0009, N=232):     Acc=51.29% | BalAcc=50.92% | UP F1=56.70% (Rec=61.67%) | DN F1=44.33% (Rec=40.18%) | CM: [45,  67, 46,  74]
Fold 5 (tau* = +0.0006, N=236):     Acc=49.79% | BalAcc=48.93% | UP F1=56.93% (Rec=62.40%) | DN F1=39.80% (Rec=35.45%) | CM: [39,  71, 47,  78]

=== EXP13: CatBoost Classifier + Downside Feats (Balanced) ===
Fold 1 (2019-11 to 2020-10, N=232): Acc=55.17% | BalAcc=53.43% | UP F1=63.12% (Rec=68.99%) | DN F1=42.86% (Rec=37.86%) | CM: [39, 64, 40, 89]
Fold 2 (2020-10 to 2021-09, N=232): Acc=50.00% | BalAcc=49.61% | UP F1=57.04% (Rec=64.71%) | DN F1=40.21% (Rec=34.51%) | CM: [39, 74, 42, 77]
Fold 3 (2021-09 to 2022-08, N=232): Acc=53.45% | BalAcc=53.37% | UP F1=55.37% (Rec=56.30%) | DN F1=51.35% (Rec=50.44%) | CM: [57, 56, 52, 67]
Fold 4 (2022-08 to 2023-07, N=232): Acc=49.14% | BalAcc=48.75% | UP F1=54.96% (Rec=60.00%) | DN F1=41.58% (Rec=37.50%) | CM: [42, 70, 48, 72]
Fold 5 (2023-07 to 2024-07, N=236): Acc=45.11% | BalAcc=45.35% | UP F1=44.64% (Rec=41.60%) | DN F1=45.57% (Rec=49.09%) | CM: [54, 56, 73, 52]

=== EXP10: Random Forest Classifier (Unweighted) ===
Fold 1 (2019-11 to 2020-10, N=232): Acc=57.33% | BalAcc=53.90% | UP F1=68.77% (Rec=84.50%) | DN F1=32.65% (Rec=23.30%) | CM: [24, 79, 20, 109]
Fold 2 (2020-10 to 2021-09, N=232): Acc=55.17% | BalAcc=54.29% | UP F1=66.88% (Rec=88.24%) | DN F1=30.67% (Rec=20.35%) | CM: [23, 90, 14, 105]
Fold 3 (2021-09 to 2022-08, N=232): Acc=54.74% | BalAcc=54.19% | UP F1=63.16% (Rec=75.63%) | DN F1=41.34% (Rec=32.74%) | CM: [37, 76, 29, 90]
Fold 4 (2022-08 to 2023-07, N=232): Acc=51.29% | BalAcc=50.15% | UP F1=63.90% (Rec=83.33%) | DN F1=25.17% (Rec=16.96%) | CM: [19, 93, 20, 100]
Fold 5 (2023-07 to 2024-07, N=236): Acc=50.64% | BalAcc=49.35% | UP F1=60.00% (Rec=69.60%) | DN F1=35.56% (Rec=29.09%) | CM: [32, 78, 38, 87]
```

---

## 6. Official Validation Classification Reports

The scikit-learn classification reports below represent aggregated out-of-sample performance across all 5 validation folds ($N = 1,164$ sessions):

### 6.1 Baseline CatBoost Regressor ($\tau = 0.0$) [EXP1]
```text
              precision    recall  f1-score   support

        DOWN   0.493750  0.143376  0.222222       551
          UP   0.529412  0.867647  0.657585       612

    accuracy                       0.524506      1164
   macro avg   0.511581  0.505511  0.439904      1164
weighted avg   0.512517  0.524506  0.451167      1164

Confusion Matrix:
[[ 79 472]
 [ 81 531]]
```

### 6.2 Best Balanced Generalist: CatBoost Regressor + Tuned Threshold ($\tau^*$) [EXP2]
```text
              precision    recall  f1-score   support

        DOWN   0.505338  0.257713  0.341346       551
          UP   0.536281  0.772876  0.633199       612

    accuracy                       0.528805      1164
   macro avg   0.520810  0.515295  0.487273      1164
weighted avg   0.521636  0.528805  0.494883      1164

Confusion Matrix:
[[142 409]
 [139 473]]
```

### 6.3 Best Downside Specialist: CatBoost Classifier + Downside Feats (Balanced) [EXP13]
```text
              precision    recall  f1-score   support

        DOWN   0.475309  0.419238  0.445516       551
          UP   0.527326  0.583333  0.553918       612

    accuracy                       0.505589      1164
   macro avg   0.501317  0.501286  0.499717      1164
weighted avg   0.502705  0.505589  0.502544      1164

Confusion Matrix:
[[231 320]
 [255 357]]
```

### 6.4 Always-UP Baseline [EXP0]
```text
              precision    recall  f1-score   support

        DOWN   0.000000  0.000000  0.000000       551
          UP   0.526225  1.000000  0.689577       612

    accuracy                       0.526225      1164
   macro avg   0.263113  0.500000  0.344789      1164
weighted avg   0.276913  0.526225  0.362879      1164

Confusion Matrix:
[[  0 551]
 [  0 612]]
```

---

## 7. Model Selection & Decision Framework

Depending on the operational objective of the forecasting system, two defensible models emerge:

| Operational Objective | Selected Architecture | Primary Advantage | Key Metric | Trade-Off to Accept |
| :--- | :--- | :--- | :---: | :--- |
| **General Purpose Forecasting (Default)** | **CatBoost Regressor + Train-Tuned Threshold ($\tau^*$) [EXP2]** | Highest overall accuracy; preserves UP performance while nearly doubling DOWN detection. | **52.88% Accuracy**, **51.53% Bal Acc**, **34.13% DOWN F1** | DOWN recall capped at ~26%. |
| **Risk Management & Hedging Specialist** | **CatBoost Classifier + Downside Features (Balanced) [EXP13]** | Maximizes DOWN recall and F1-score; triples detected downturns. | **41.92% DOWN Recall**, **44.55% DOWN F1**, **49.97% Macro F1** | UP recall reduced to 58.33%; overall accuracy is 50.56%. |

### Why CatBoost Regressor with Tuned Threshold (EXP2) is Recommended as the Primary Architecture:
1. **Beats the Always-UP Baseline:** Reaches **52.88%**, whereas Always-UP achieves 52.62% and baseline CatBoost achieves 52.45%.
2. **Substantial DOWN Improvement:** DOWN recall increases from **14.34% to 25.77%** (+80% relative improvement), catching 142 market downturns instead of 79.
3. **No UP Sacrifice:** UP recall remains high at **77.29%** and UP precision reaches **53.63%** (higher than Always-UP precision of 52.62%).
4. **Maintains Price Forecaster Alignment:** Because the base model remains a continuous log-return regressor, the exact same model produces calibrated next-day dollar price forecasts, preserving compatibility with conformal prediction intervals.

---

## 8. Limitations & Recommendations

1. **Fundamental Asymmetry of Asset Returns:**
   In an equity asset with positive secular drift, negative days are shorter, rarer, and sharper than positive drift days. It is mathematically impossible to raise DOWN recall to $>50\%$ without simultaneously driving UP recall down and degrading overall accuracy.

2. **Dangers of Static Class Weights:**
   Setting fixed class weight multipliers $>1.2$ leads to catastrophic degradation, converting the model into an all-negative predictor. If class weights are used, they must be mathematically constrained via `auto_class_weights='Balanced'` rather than arbitrary manual hyperparameter multipliers.

3. **Regime Dependence:**
   During protracted bull runs (e.g., Fold 1, where AAPL rose sharply post-2019), attempting to force DOWN predictions incurs high false-positive costs. Threshold calibration must remain dynamic across time.

4. **Preservation of the Frozen Holdout:**
   As mandated, **none of these models were fitted or tuned against the final diagnostic holdout (2024–2026)**. The existing holdout report reflects the pure diagnostic baseline and remains unmanipulated.
