# Final Directional Classification Metrics — Predictive Insights

**Document:** Complete Scikit-Learn Classification Report & Confusion Matrix  
**Dataset:** Final Purged Out-of-Sample Holdout ($N = 596$ trading sessions, July 8, 2024 to November 18, 2026)  
**Models Evaluated:**
1. **CatBoost Forecaster** (`CatBoostForecaster`, 120 iterations, depth 4, learning rate 0.02)
2. **Always-UP Baseline** (`MajorityClassForecaster`, constant positive classification)

---

## 1. Ground Truth Holdout Distribution

- **Evaluation Window:** 2024-07-08 to 2026-11-18 ($N = 596$ sessions)
- **Positive Class:** **UP** ($y_t > 0$, session close higher than previous close)
- **Negative Class:** **DOWN** ($y_t \le 0$, session close lower than or equal to previous close)

| Class | Label | Count (Support) | Percentage of Holdout |
| :--- | :---: | :---: | :---: |
| **DOWN** | 0 | 275 | 46.1409% |
| **UP** | 1 | 321 | 53.8591% |
| **Total** | — | **596** | **100.0000%** |

---

## 2. CatBoost Forecaster

### 2.1 Standard `sklearn.metrics.classification_report` Output

```text
              precision    recall  f1-score   support

        DOWN   0.390625  0.090909  0.147493       275
          UP   0.530075  0.878505  0.661196       321

    accuracy                       0.515101       596
   macro avg   0.460350  0.484707  0.404344       596
weighted avg   0.465732  0.515101  0.424168       596
```

### 2.2 Confusion Matrix

$$\begin{pmatrix} \text{Actual DOWN} \\ \text{Actual UP} \end{pmatrix} \times \begin{pmatrix} \text{Predicted DOWN} & \text{Predicted UP} \end{pmatrix} = \begin{bmatrix} 25 & 250 \\ 39 & 282 \end{bmatrix}$$

| Actual \ Predicted | Predicted DOWN (0) | Predicted UP (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual DOWN (0)** | **TN = 25** (4.19%) | **FP = 250** (41.95%) | 275 (46.14%) |
| **Actual UP (1)** | **FN = 39** (6.54%) | **TP = 282** (47.32%) | 321 (53.86%) |
| **Total Predicted** | **64** (10.74%) | **532** (89.26%) | **596** (100.00%) |

- **True Negatives (TN):** 25 (correctly predicted DOWN)
- **False Positives (FP):** 250 (falsely predicted UP when market fell)
- **False Negatives (FN):** 39 (falsely predicted DOWN when market rose)
- **True Positives (TP):** 282 (correctly predicted UP)

### 2.3 Detailed Metric Breakdown (Exact Values)

| Metric | Exact Fraction / Formula | Exact Float | Percentage |
| :--- | :---: | :---: | :---: |
| **UP Precision** | $\frac{282}{282 + 250} = \frac{282}{532}$ | $0.5300751879699248$ | **53.0075%** |
| **UP Recall** | $\frac{282}{282 + 39} = \frac{282}{321}$ | $0.8785046728971962$ | **87.8505%** |
| **UP F1-Score** | $2 \times \frac{0.530075 \times 0.878505}{0.530075 + 0.878505}$ | $0.6611957796014069$ | **66.1196%** |
| **UP Support** | — | $321$ | 321 sessions |
| **DOWN Precision** | $\frac{25}{25 + 39} = \frac{25}{64}$ | $0.3906250000000000$ | **39.0625%** |
| **DOWN Recall** | $\frac{25}{25 + 250} = \frac{25}{275} = \frac{1}{11}$ | $0.0909090909090909$ | **9.0909%** |
| **DOWN F1-Score** | $2 \times \frac{0.390625 \times (1/11)}{0.390625 + (1/11)}$ | $0.1474926253687316$ | **14.7493%** |
| **DOWN Support** | — | $275$ | 275 sessions |
| **Overall Accuracy** | $\frac{25 + 282}{596} = \frac{307}{596}$ | $0.5151006711409396$ | **51.5101%** |
| **Macro Avg Precision** | $\frac{0.390625 + 0.530075}{2}$ | $0.4603500939849624$ | **46.0350%** |
| **Macro Avg Recall** | $\frac{0.090909 + 0.878505}{2}$ | $0.4847068819031436$ | **48.4707%** (Balanced Accuracy) |
| **Macro Avg F1-Score** | $\frac{0.147493 + 0.661196}{2}$ | $0.4043442024850692$ | **40.4344%** |
| **Weighted Avg Precision** | $\frac{275 \times 0.390625 + 321 \times 0.530075}{596}$ | $0.4657315609703790$ | **46.5732%** |
| **Weighted Avg Recall** | $\frac{275 \times (1/11) + 321 \times (282/321)}{596} = \frac{307}{596}$ | $0.5151006711409396$ | **51.5101%** |
| **Weighted Avg F1-Score** | $\frac{275 \times 0.147493 + 321 \times 0.661196}{596}$ | $0.4241683174974040$ | **42.4168%** |

---

## 3. Always-UP Baseline (`MajorityClassForecaster`)

### 3.1 Standard `sklearn.metrics.classification_report` Output

```text
              precision    recall  f1-score   support

        DOWN   0.000000  0.000000  0.000000       275
          UP   0.538591  1.000000  0.700109       321

    accuracy                       0.538591       596
   macro avg   0.269295  0.500000  0.350055       596
weighted avg   0.290080  0.538591  0.377072       596
```

### 3.2 Confusion Matrix

$$\begin{pmatrix} \text{Actual DOWN} \\ \text{Actual UP} \end{pmatrix} \times \begin{pmatrix} \text{Predicted DOWN} & \text{Predicted UP} \end{pmatrix} = \begin{bmatrix} 0 & 275 \\ 0 & 321 \end{bmatrix}$$

| Actual \ Predicted | Predicted DOWN (0) | Predicted UP (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual DOWN (0)** | **TN = 0** (0.00%) | **FP = 275** (46.14%) | 275 (46.14%) |
| **Actual UP (1)** | **FN = 0** (0.00%) | **TP = 321** (53.86%) | 321 (53.86%) |
| **Total Predicted** | **0** (0.00%) | **596** (100.00%) | **596** (100.00%) |

- **True Negatives (TN):** 0
- **False Positives (FP):** 275
- **False Negatives (FN):** 0
- **True Positives (TP):** 321

### 3.3 Detailed Metric Breakdown (Exact Values)

| Metric | Exact Fraction / Formula | Exact Float | Percentage |
| :--- | :---: | :---: | :---: |
| **UP Precision** | $\frac{321}{596}$ | $0.5385906040268457$ | **53.8591%** |
| **UP Recall** | $\frac{321}{321}$ | $1.0000000000000000$ | **100.0000%** |
| **UP F1-Score** | $\frac{2 \times 321}{321 + 596} = \frac{642}{917}$ | $0.7001090512540894$ | **70.0109%** |
| **UP Support** | — | $321$ | 321 sessions |
| **DOWN Precision** | $0.0$ (no DOWN predictions, zero_division=0) | $0.0000000000000000$ | **0.0000%** |
| **DOWN Recall** | $\frac{0}{275}$ | $0.0000000000000000$ | **0.0000%** |
| **DOWN F1-Score** | $0.0$ | $0.0000000000000000$ | **0.0000%** |
| **DOWN Support** | — | $275$ | 275 sessions |
| **Overall Accuracy** | $\frac{321}{596}$ | $0.5385906040268457$ | **53.8591%** |
| **Macro Avg Precision** | $\frac{0.0 + 0.538591}{2}$ | $0.2692953020134228$ | **26.9295%** |
| **Macro Avg Recall** | $\frac{0.0 + 1.0}{2}$ | $0.5000000000000000$ | **50.0000%** |
| **Macro Avg F1-Score** | $\frac{0.0 + 0.700109}{2}$ | $0.3500545256270447$ | **35.0055%** |
| **Weighted Avg Precision** | $\frac{321 \times (321/596)}{596} = \frac{321^2}{596^2}$ | $0.2900798387460025$ | **29.0080%** |
| **Weighted Avg Recall** | $\frac{321 \times 1.0}{596} = \frac{321}{596}$ | $0.5385906040268457$ | **53.8591%** |
| **Weighted Avg F1-Score** | $\frac{321 \times (642/917)}{596}$ | $0.3770721567996018$ | **37.7072%** |

---

## 4. Head-to-Head Comparison & Critical Analysis

| Metric | CatBoost Forecaster | Always-UP Baseline | Delta (CatBoost - Baseline) | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **51.5101%** | **53.8591%** | **-2.3490%** | **Always-UP** |
| **Balanced Accuracy** | **48.4707%** | **50.0000%** | **-1.5293%** | **Always-UP** |
| **UP Precision** | **53.0075%** | **53.8591%** | **-0.8516%** | **Always-UP** |
| **UP Recall** | **87.8505%** | **100.0000%** | **-12.1495%** | **Always-UP** |
| **UP F1-Score** | **66.1196%** | **70.0109%** | **-3.8913%** | **Always-UP** |
| **DOWN Precision** | **39.0625%** | **0.0000%** | **+39.0625%** | **CatBoost** |
| **DOWN Recall** | **9.0909%** | **0.0000%** | **+9.0909%** | **CatBoost** |
| **DOWN F1-Score** | **14.7493%** | **0.0000%** | **+14.7493%** | **CatBoost** |
| **Macro F1-Score** | **40.4344%** | **35.0055%** | **+5.4289%** | **CatBoost** |
| **Weighted F1-Score** | **42.4168%** | **37.7072%** | **+4.7096%** | **CatBoost** |

### Key Scientific Insights:

1. **Why Always-UP Achieves Higher Accuracy:**
   The 596-session holdout period (July 2024 to November 2026) was characterized by an equity bull market in AAPL, where **53.86% of all days closed positive**. Simply predicting "UP" every day captures 100% of the positive sessions, yielding 53.86% raw accuracy.

2. **CatBoost's Positive Skew & DOWN Under-Prediction:**
   CatBoost predicts UP on **532 out of 596 sessions (89.26%)** and predicts DOWN on only **64 sessions (10.74%)**.
   - As a result, CatBoost catches **87.85% of UP sessions** (282 / 321), but correctly identifies only **9.09% of DOWN sessions** (25 / 275).
   - This heavy asymmetry explains why balanced accuracy is **48.47%** (below random chance of 50.00%).

3. **Defensible Conclusion:**
   CatBoost does **NOT** possess genuine market-timing alpha over a simple long-only posture on this holdout period. While it attempts bidirectional forecasting (yielding non-zero DOWN precision and recall, and slightly higher macro F1), its raw accuracy (51.51%) is strictly lower than naive trend-following (53.86%).
