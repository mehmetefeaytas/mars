# SPDX-License-Identifier: AGPL-3.0-only
"""HTTP API testleri (ag cikisi gerektirmeyen uclar)."""

from fastapi.testclient import TestClient

from mars.api import AnalyzeParams, DEFAULT_BASKET, app, _settings_from


client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_meta_shape():
    r = client.get("/api/meta")
    assert r.status_code == 200
    d = r.json()
    assert d["verishield"] in {True, False}
    assert "basket" in d and len(d["basket"]) > 0
    assert "defaults" in d and "params" in d["defaults"]
    assert set(DEFAULT_BASKET) >= {"THYAO.IS", "BIMAS.IS"}


def test_settings_from_params():
    p = AnalyzeParams(lookback=300, news_penalty=0.2, tahta_modulation=0.5,
                      weights={"volume_z": 0.3})
    s = _settings_from(p)
    assert s.lookback_days == 300
    assert abs(s.news_penalty_strength - 0.2) < 1e-9
    assert abs(s.tahta_modulation - 0.5) < 1e-9
    assert abs(s.weights.volume_z - 0.3) < 1e-9


def test_analyze_unknown_ticker_returns_502():
    r = client.get("/api/analyze", params={"ticker": "NOSUCH.IS"})
    assert r.status_code == 502
    assert "Analiz basarisiz" in r.json()["detail"]


def test_screener_empty_tickers():
    r = client.post("/api/screener", json={"tickers": [], "params": {}})
    assert r.status_code == 200
    assert r.json()["rows"] == []
