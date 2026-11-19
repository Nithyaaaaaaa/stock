// Predictive Insights – Vanilla JavaScript Frontend

document.addEventListener("DOMContentLoaded", () => {
    const stockSelect = document.getElementById("stock-select");
    const customGroup = document.getElementById("custom-ticker-group");
    const customInput = document.getElementById("custom-ticker");
    const predictBtn = document.getElementById("predict-btn");
    const spinner = document.getElementById("loading-spinner");
    const alertBox = document.getElementById("alert-box");
    const resultsSection = document.getElementById("results-section");

    // Toggle custom ticker field
    stockSelect.addEventListener("change", () => {
        if (stockSelect.value === "CUSTOM") {
            customGroup.style.display = "block";
            customInput.focus();
        } else {
            customGroup.style.display = "none";
        }
    });

    // Run prediction on button click
    predictBtn.addEventListener("click", async () => {
        let ticker = stockSelect.value;
        if (ticker === "CUSTOM") {
            ticker = customInput.value.trim().toUpperCase();
            if (!ticker) {
                showAlert("Please enter a valid Yahoo Finance stock ticker symbol (e.g. AAPL, NVDA, RELIANCE.NS).");
                return;
            }
        }

        // UI state: loading
        clearAlert();
        spinner.style.display = "block";
        predictBtn.disabled = true;
        predictBtn.textContent = "Analyzing & Forecasting...";
        resultsSection.style.display = "none";

        try {
            const response = await fetch(`/api/predict?ticker=${encodeURIComponent(ticker)}`);
            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Failed to generate stock prediction.");
            }

            renderResults(data);
        } catch (error) {
            showAlert(error.message);
        } finally {
            spinner.style.display = "none";
            predictBtn.disabled = false;
            predictBtn.textContent = "🚀 Predict Next Trading Day";
        }
    });

    function showAlert(msg) {
        alertBox.textContent = `❌ ${msg}`;
        alertBox.style.display = "block";
    }

    function clearAlert() {
        alertBox.textContent = "";
        alertBox.style.display = "none";
    }

    function renderResults(data) {
        const curr = data.currency_symbol || "$";
        const priceDiff = data.price_change;
        const pctDiff = data.pct_change;
        const sign = priceDiff >= 0 ? "+" : "-";
        const deltaClass = priceDiff >= 0 ? "metric-delta-pos" : "metric-delta-neg";

        // Asset header badge
        const badgeHeader = document.getElementById("badge-header");
        if (data.ticker === "AAPL") {
            badgeHeader.innerHTML = "✅ <b>Primary Benchmark Asset</b> (Validated in 5-Fold Walk-Forward Cross-Validation & Purged Diagnostic Holdout)";
        } else {
            badgeHeader.innerHTML = `🧪 <b>Dynamic Asset Transfer</b> (Trained on ${data.ticker} history using the EXP2 CatBoost architecture)`;
        }

        // Metric Cards
        document.getElementById("latest-close-val").textContent = `${curr}${data.last_close.toFixed(2)}`;
        document.getElementById("latest-close-date").textContent = data.last_date;

        document.getElementById("pred-close-val").textContent = `${curr}${data.predicted_price.toFixed(2)}`;
        document.getElementById("pred-close-date").textContent = data.next_date;

        document.getElementById("delta-pct-val").className = deltaClass;
        document.getElementById("delta-pct-val").textContent = `${sign}${Math.abs(pctDiff).toFixed(2)}%`;
        document.getElementById("delta-price-val").className = deltaClass;
        document.getElementById("delta-price-val").textContent = `${sign}${curr}${Math.abs(priceDiff).toFixed(2)}`;

        // Price Movement Badge
        const movBadge = document.getElementById("price-mov-badge");
        const movSub = document.getElementById("price-mov-sub");
        let arrow = "—";
        let movClass = "price-badge-neutral";

        if (priceDiff > 1e-6) {
            arrow = "↗";
            movClass = "price-badge-up";
        } else if (priceDiff < -1e-6) {
            arrow = "↘";
            movClass = "price-badge-down";
        }
        movBadge.className = movClass;
        movBadge.textContent = `${data.price_movement} ${arrow}`;
        movSub.textContent = `${sign}${curr}${Math.abs(priceDiff).toFixed(2)} (${sign}${Math.abs(pctDiff).toFixed(2)}%)`;

        // EXP2 Classification Badge
        const exp2Badge = document.getElementById("exp2-badge");
        const exp2Sub = document.getElementById("exp2-sub");
        const isUpAbove = data.pred_log_return > data.learned_threshold;
        exp2Badge.className = isUpAbove ? "exp2-badge-up" : "exp2-badge-below";
        exp2Badge.textContent = data.exp2_classification;

        const predPctStr = (data.pred_log_return >= 0 ? "+" : "") + (data.pred_log_return * 100).toFixed(2) + "%";
        const tauPctStr = (data.learned_threshold >= 0 ? "+" : "") + (data.learned_threshold * 100).toFixed(2) + "%";
        exp2Sub.innerHTML = `Pred Return: <b>${predPctStr}</b> | Threshold: <b>${tauPctStr}</b>`;

        // Render Plotly Chart
        renderChart(data, curr, priceDiff);

        // Populate Forecast Table
        const tableBody = document.getElementById("table-body");
        tableBody.innerHTML = `
            <tr>
                <td><b>${data.ticker}</b> (${data.asset_name || data.ticker})</td>
                <td>${data.next_date}</td>
                <td>${curr}${data.last_close.toFixed(2)}</td>
                <td><b>${curr}${data.predicted_price.toFixed(2)}</b></td>
                <td class="${deltaClass}">${sign}${curr}${Math.abs(priceDiff).toFixed(2)} (${sign}${Math.abs(pctDiff).toFixed(2)}%)</td>
                <td><span class="${movClass}">${data.price_movement} ${arrow}</span></td>
                <td><span class="${isUpAbove ? 'exp2-badge-up' : 'exp2-badge-below'}">${data.exp2_classification}</span> <small style="color:#64748B;">(Pred: ${predPctStr}, Threshold: ${tauPctStr})</small></td>
            </tr>
        `;

        resultsSection.style.display = "block";
        resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function renderChart(data, curr, priceDiff) {
        if (!window.Plotly || !data.historical_dates || data.historical_dates.length === 0) {
            return;
        }

        const histTrace = {
            x: data.historical_dates,
            y: data.historical_prices,
            mode: "lines",
            name: "Historical Close",
            line: { color: "#334155", width: 2 }
        };

        const forecastColor = priceDiff >= 0 ? "#16A34A" : "#DC2626";
        const predTrace = {
            x: [data.historical_dates[data.historical_dates.length - 1], data.next_date.split(" (")[0]],
            y: [data.last_close, data.predicted_price],
            mode: "lines+markers",
            name: "Predicted Next Day",
            line: { color: forecastColor, width: 2.5, dash: "dash" },
            marker: { size: 8, color: forecastColor }
        };

        const layout = {
            margin: { l: 45, r: 25, t: 30, b: 40 },
            height: 420,
            template: "plotly_white",
            hovermode: "x unified",
            xaxis: { title: "Trading Session Date" },
            yaxis: { title: `Price (${curr})` },
            legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 }
        };

        Plotly.newPlot("chart-container", [histTrace, predTrace], layout, { responsive: true });
    }
});
