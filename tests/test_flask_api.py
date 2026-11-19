"""
Unit Tests: Flask Application & REST API Endpoints.
Verifies Vercel entry points, HTML frontend serving, and API predictions.
"""

import pytest
import json
from app import app
from api.index import app as vercel_app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_vercel_entry_point_exports_flask_instance():
    """Verify that api/index.py exports top-level app instance for Vercel."""
    assert vercel_app is not None
    assert hasattr(vercel_app, "route")
    assert hasattr(vercel_app, "wsgi_app")


def test_index_route_serves_html(client):
    """Verify GET / returns 200 OK and renders HTML frontend."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.content_type
    html = res.data.decode("utf-8")
    assert "Predictive Insights" in html
    assert "stock-select" in html
    assert "predict-btn" in html


def test_api_stocks_route(client):
    """Verify GET /api/stocks returns JSON list of supported tickers."""
    res = client.get("/api/stocks")
    assert res.status_code == 200
    assert "application/json" in res.content_type
    data = res.get_json()
    assert "Apple (AAPL)" in data
    assert data["Apple (AAPL)"] == "AAPL"


def test_api_health_route(client):
    """Verify GET /api/health returns healthy status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "CatBoost EXP2" in data["model_architecture"]


def test_api_predict_missing_ticker_returns_400(client):
    """Verify GET /api/predict without ticker returns 400 error."""
    res = client.get("/api/predict")
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "valid stock ticker" in data["error"]


def test_api_predict_get_aapl(client):
    """Verify GET /api/predict?ticker=AAPL executes EXP2 forecast and returns JSON."""
    res = client.get("/api/predict?ticker=AAPL")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["ticker"] == "AAPL"
    assert data["currency_symbol"] == "$"
    assert "last_close" in data and data["last_close"] > 0
    assert "predicted_price" in data and data["predicted_price"] > 0
    assert "price_movement" in data
    assert "exp2_classification" in data
    assert "learned_threshold" in data
    assert "historical_dates" in data and len(data["historical_dates"]) > 0
    assert "historical_prices" in data and len(data["historical_prices"]) > 0


def test_api_predict_post_msft(client):
    """Verify POST /api/predict with JSON payload executes forecast."""
    res = client.post(
        "/api/predict",
        data=json.dumps({"ticker": "MSFT"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["ticker"] == "MSFT"
    assert data["price_movement"] in ["PRICE INCREASE", "PRICE DECREASE", "NO PRICE CHANGE"]
    assert data["exp2_classification"] in ["UP — Above threshold", "Below UP threshold"]
