# Stage 1 Research Summary: Data Preprocessing, Stationarity Audit & Target Selection

---

## 1. Data Split Arithmetic & Observation Tracking

* **Total Clean Dataset:** $N = 2,988$ trading sessions (2015-01-02 to 2026-11-18).
* **Nominal Split:** 80% Development / 20% Final Untouched Holdout.
* **Exact Integer Partition:**
  * **Development Set:** $\lceil 2,988 \times 0.80 \rceil = \mathbf{2,391\text{ sessions}}$ (Indices 0 to 2,390: 2015-01-02 to 2024-07-03).
  * **Final Holdout Set:** $2,988 - 2,391 = \mathbf{597\text{ sessions}}$ (Indices 2,391 to 2,987: 2024-07-05 to 2026-11-18) [**100% FROZEN**].
  * Check sum: $2,391 + 597 = \mathbf{2,988\text{ sessions}}$.

### Explanation of the 66 Dropped Observations within Development Set:
The development set contains 2,391 total sessions, but sub-train (1,743) + sub-val (582) = 2,325 usable sessions. The 66 excluded observations are accounted for as follows:
1. **65 Initial Warm-Up Sessions (Indices 0 to 64: 2015-01-02 to 2015-04-07):**
   * Rolling statistical indicators with windows up to 60 days (`Vol_60d`, `Rolling_Beta_60`, `Ratio_Close_SMA_50`) require 60 to 65 preceding rows before computing their first non-NaN value. These rows are discarded to prevent imputation distortion.
2. **1 Terminal Row (Index 2,390: 2024-07-03):**
   * Next-day forecasting targets (`Target_Next_Close` and `Target_Next_Log_Return`) are created via `.shift(-1)`. The final row of the development set has no forward target and is dropped.
3. **Usable Development Sessions:** $2,391 - 65 - 1 = \mathbf{2,325\text{ sessions}}$ (2015-04-08 to 2024-07-02).
   * **Sub-Train Split:** $1,743\text{ sessions}$ (2015-04-08 to 2022-03-08).
   * **Sub-Val Split:** $582\text{ sessions}$ (2022-03-09 to 2024-07-02).
   * Verification: $65 + 1,743 + 582 + 1 = 2,391$. No observations were accidentally dropped or double-counted.

---

## 2. Statistical Stationarity Audit (Evaluated on Development Set Only)

### Hypotheses & Methodology
* **Augmented Dickey-Fuller (ADF) Test:**
  * Null Hypothesis $H_0$: Unit root present (series is non-stationary).
  * Alternative Hypothesis $H_1$: Series is stationary or trend-stationary.
  * Rejection ($p < 0.05$) indicates evidence against a unit root.
* **KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin):**
  * Null Hypothesis $H_0$: Series is stationary around a deterministic constant.
  * Alternative Hypothesis $H_1$: Unit root present (series is non-stationary).
  * Rejection ($p < 0.05$) indicates non-stationarity.
* **KPSS $p$-Value Ceiling Explanation:**
  * In `statsmodels`, the asymptotic critical value lookup table is bounded at $p = 0.1000$ (critical value = 0.347). When the test statistic is smaller than 0.347, `statsmodels` reports $p = 0.1000$ with an interpolation warning, indicating $p \ge 0.10$ (strong failure to reject the stationarity null).

### Complete Test Results for All 30 Evaluated Features:

| Feature Name | Formulation / Description | ADF Statistic | ADF $p$-value | KPSS Statistic | KPSS $p$-value | Statistical Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `Close_Open_Ratio` | $(P_t - O_t) / O_t$ | -52.8320 | $0.0000$ | 0.1598 | 0.1000 | Evidence consistent with $I(0)$ |
| `Excess_Return_Over_Benchmark` | $r_{\text{AAPL},t} - r_{\text{SPY},t}$ | -34.2397 | $0.0000$ | 0.1339 | 0.1000 | Evidence consistent with $I(0)$ |
| `Log_Volume_Change` | $\Delta \ln(\text{Volume}_t)$ | -23.8905 | $0.0000$ | 0.0275 | 0.1000 | Evidence consistent with $I(0)$ |
| `Return_Lag_1` | $(P_{t-1} - P_{t-2}) / P_{t-2}$ | -15.0657 | $8.87 \times 10^{-28}$ | 0.1239 | 0.1000 | Evidence consistent with $I(0)$ |
| `Benchmark_Return_Lag_1` | $r_{\text{SPY},t-1}$ | -15.0090 | $1.06 \times 10^{-27}$ | 0.0527 | 0.1000 | Evidence consistent with $I(0)$ |
| `Log_Return_Lag_1` | $\ln(P_{t-1} / P_{t-2})$ | -14.9701 | $1.20 \times 10^{-27}$ | 0.1043 | 0.1000 | Evidence consistent with $I(0)$ |
| `Volume_Ratio_SMA_20` | $\text{Vol} / \text{SMA}_{20}(\text{Vol}) - 1.0$ | -14.3688 | $9.57 \times 10^{-27}$ | 0.0131 | 0.1000 | Evidence consistent with $I(0)$ |
| `Ratio_Close_SMA_5` | $P_t / \text{SMA}_5(P) - 1.0$ | -12.6151 | $1.62 \times 10^{-23}$ | 0.1153 | 0.1000 | Evidence consistent with $I(0)$ |
| `Return_Lag_5` | $(P_t - P_{t-5}) / P_{t-5}$ | -10.9737 | $7.76 \times 10^{-20}$ | 0.1205 | 0.1000 | Evidence consistent with $I(0)$ |
| `Log_Return_Lag_5` | $\ln(P_t / P_{t-5})$ | -10.9224 | $1.03 \times 10^{-19}$ | 0.1079 | 0.1000 | Evidence consistent with $I(0)$ |
| `Stoch_K_14` | Centered 14-day Stochastic $\%K$ | -10.8042 | $1.98 \times 10^{-19}$ | 0.1427 | 0.1000 | Evidence consistent with $I(0)$ |
| `RSI_14_Centered` | $(\text{RSI}_{14}/100) - 0.5$ | -10.6254 | $5.38 \times 10^{-19}$ | 0.1406 | 0.1000 | Evidence consistent with $I(0)$ |
| `Bollinger_Pct_B` | Bollinger Bands $\%B$ | -10.0977 | $1.08 \times 10^{-17}$ | 0.1279 | 0.1000 | Evidence consistent with $I(0)$ |
| `Ratio_Close_EMA_12` | $P_t / \text{EMA}_{12}(P) - 1.0$ | -10.0743 | $1.24 \times 10^{-17}$ | 0.1216 | 0.1000 | Evidence consistent with $I(0)$ |
| `Ratio_Close_SMA_20` | $P_t / \text{SMA}_{20}(P) - 1.0$ | -9.9934 | $1.97 \times 10^{-17}$ | 0.1212 | 0.1000 | Evidence consistent with $I(0)$ |
| `Ratio_SMA_5_20` | $\text{SMA}_5 / \text{SMA}_{20} - 1.0$ | -8.1688 | $8.74 \times 10^{-13}$ | 0.1171 | 0.1000 | Evidence consistent with $I(0)$ |
| `Ratio_Close_EMA_26` | $P_t / \text{EMA}_{26}(P) - 1.0$ | -7.9202 | $3.74 \times 10^{-12}$ | 0.1591 | 0.1000 | Evidence consistent with $I(0)$ |
| `High_Low_Ratio` | $(H_t - L_t) / P_t$ | -7.6856 | $1.46 \times 10^{-11}$ | 0.8903 | 0.0100 | Bounded $I(0)$ with changing variance |
| `NATR_14` | $\text{ATR}_{14} / P_t$ | -7.4961 | $4.37 \times 10^{-11}$ | 0.6946 | 0.0140 | Bounded $I(0)$ with changing variance |
| `MACD_Normalized` | $(\text{EMA}_{12} - \text{EMA}_{26}) / P_t$ | -7.0350 | $6.05 \times 10^{-10}$ | 0.1474 | 0.1000 | Evidence consistent with $I(0)$ |
| `Vol_20d` | Rolling 20-day return volatility | -6.4566 | $1.48 \times 10^{-8}$ | 0.5150 | 0.0383 | Bounded $I(0)$ with volatility clustering |
| `Ratio_Close_SMA_50` | $P_t / \text{SMA}_{50}(P) - 1.0$ | -6.3873 | $2.15 \times 10^{-8}$ | 0.1802 | 0.1000 | Evidence consistent with $I(0)$ |
| `Raw_Volume` | Raw trading volume | -6.2546 | $4.37 \times 10^{-8}$ | 4.0130 | 0.0100 | Bounded $I(0)$ with regime shifts |
| `Vol_10d` | Rolling 10-day return volatility | -5.7367 | $6.42 \times 10^{-7}$ | 0.5310 | 0.0347 | Bounded $I(0)$ with volatility clustering |
| `Vol_60d` | Rolling 60-day return volatility | -4.2361 | $5.72 \times 10^{-4}$ | 0.6453 | 0.0185 | Bounded $I(0)$ with volatility clustering |
| `Rolling_Beta_60` | Rolling 60-day market beta | -3.9995 | $1.41 \times 10^{-3}$ | 0.5786 | 0.0246 | Bounded $I(0)$ with regime shifts |
| `Raw_ATR_14` | Raw Average True Range (USD) | -3.3128 | $1.43 \times 10^{-2}$ | 6.1308 | 0.0100 | Scale-dependent level (EXCLUDED) |
| `Raw_SMA_20` | 20-day moving average (USD) | +0.3185 | $0.9782$ | 7.4142 | 0.0100 | **Unit root present $I(1)$ (EXCLUDED)** |
| `Raw_EMA_12` | 12-day exponential average (USD) | +0.4638 | $0.9837$ | 7.4216 | 0.0100 | **Unit root present $I(1)$ (EXCLUDED)** |
| `Close` | Raw closing price (USD) | +0.4749 | $0.9841$ | 7.4242 | 0.0100 | **Unit root present $I(1)$ (EXCLUDED)** |

*Note:* Passing these statistical tests does not constitute proof of strict stationarity. In financial markets, even series with rejected unit roots exhibit volatility clustering, regime shifts, and leptokurtic distributions.

---

## 3. Fair Target Representation Comparison

Evaluated on the **exact same 582 validation sessions (2022-03-09 to 2024-07-02)** using Ridge regression ($\alpha = 100.0$) on identical scaled stationary features:

| Target Formulation | Mathematical Form | Reconstructed Price Formula | Validation RMSE | Validation MAE | Directional Accuracy | Structural Assessment |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Persistence Baseline** | $\hat{P}_{t+1} = P_t$ | $\hat{P}_{t+1} = P_t$ | **2.8421 USD** | **2.1248 USD** | 47.59% | **Lowest error baseline** (All-DOWN tie rule) |
| **Continuous Log Return** | $y_{t+1} = \ln(P_{t+1}/P_t)$ | $\hat{P}_{t+1} = P_t e^{\hat{y}_{t+1}}$ | **2.8588 USD** | **2.1632 USD** | **52.06%** | **Best learned target representation** |
| **Relative Daily Return** | $r_{t+1} = (P_{t+1}-P_t)/P_t$ | $\hat{P}_{t+1} = P_t(1+\hat{r}_{t+1})$ | 2.8608 USD | 2.1646 USD | 51.72% | Comparable learned target |
| **Direct Price Level** | $P_{t+1}$ directly | $\hat{P}_{t+1} = \hat{y}$ | 39.7222 USD | 34.0327 USD | 50.86% | Catastrophic failure due to level drift |

### Clarification on Performance:
* The **Persistence baseline achieves the lowest validation RMSE (2.8421 USD) and MAE (2.1248 USD)**.
* Continuous log return is the **best-performing learned target representation** among the candidates tested, but it does **not** outperform the persistence baseline on this validation split.
* Continuous log returns are retained for Stage 2 because they ensure symmetric bounds, additivity, and eliminate drift collapse.

---

## 4. Fair Lookback Window Evaluation (Identical $N = 522$ Validation Sessions)

To ensure fair comparison, all candidate windows $L \in [5, 10, 15, 20, 30, 45, 60]$ were evaluated on the **exact same 522 validation sessions** (starting after the maximum window $L_{\max} = 60$):

| Lookback Window ($L$) | Validation RMSE | Validation MAE | Directional Accuracy | Comparison to Persistence |
| :---: | :---: | :---: | :---: | :--- |
| **Persistence Baseline** | **2.6997 USD** | **2.0096 USD** | **47.13%** | *Invariant to lookback window size* ($\hat{P}_{t+1} = P_t$) |
| **$L = 5$ sessions** | 2.7243 USD | 2.0217 USD | 49.81% | High-frequency noise; higher RMSE |
| **$L = 10$ sessions** | 2.7287 USD | 2.0362 USD | 50.57% | Sub-optimal error |
| **$L = 15$ sessions** | 2.7142 USD | 2.0198 USD | 51.15% | Lowest MAE among shorter windows |
| **$L = 20$ sessions** | 2.7557 USD | 2.0424 USD | 53.07% | Higher directional accuracy, slightly higher variance |
| **$L = 30$ sessions** | 2.7344 USD | 2.0435 USD | 50.38% | Moderate performance |
| **$L = 45$ sessions** | 2.7356 USD | 2.0429 USD | 53.45% | Strong directional signal |
| **$L = 60$ sessions** | **2.7328 USD** | **2.0497 USD** | **53.64%** | **Highest directional accuracy and lowest long-window RMSE** |

### Lookback Observations:
1. **Persistence is Invariant to Lookback:** The persistence benchmark predicts $\hat{P}_{t+1} = P_t$, which relies strictly on yesterday's closing price and is unaffected by the window length $L$.
2. **Window Selection:** $L = 60$ sessions is retained for multi-model benchmarking because it achieved the highest directional accuracy (53.64%) and lowest RMSE among the longer horizons on this validation split. No claim of capturing quarterly earnings cycles is asserted.
