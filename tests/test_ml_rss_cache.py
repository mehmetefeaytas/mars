# SPDX-License-Identifier: AGPL-3.0-only
"""ML katmani, RSS ve onbellek kapsami testleri."""

import numpy as np
import pandas as pd
import pytest

from mars.config import load_settings
from mars.ml import build_dataset, load_model, predict_scores, save_model, train
from mars.news.rss import auto_match, news_for_ticker, parse_rss
from mars.news.tickers import match_ticker, match_text_for_ticker
from mars.ingestion import _read_cache, _write_cache
from mars.explain import detailed_reasons

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item>
  <title>Türk Hava Yolları yolcu sayısını rekor seviyeye çıkardı</title>
  <description>THY açıkladı: yolcu sayısı arttı.</description>
  <link>https://ornek.com/1</link>
  <pubDate>Sun, 18 Oct 2026 08:00:00 GMT</pubDate>
</item>
<item>
  <title>Altın fiyatları yükseldi</title>
  <description>Gram altın rekor kırdı.</description>
  <link>https://ornek.com/2</link>
  <pubDate>Sun, 18 Oct 2026 09:00:00 GMT</pubDate>
</item>
</channel></rss>"""


# --- RSS ---

def test_parse_rss_items():
    items = parse_rss(SAMPLE_RSS, source="test")
    assert len(items) == 2
    assert "Türk Hava Yolları" in items[0].headline
    assert items[0].timestamp == pd.Timestamp("2026-10-18 08:00:00")
    assert items[0].url == "https://ornek.com/1"


def test_auto_match_finds_thyao():
    items = parse_rss(SAMPLE_RSS, source="test")
    matched = auto_match(items)
    assert "THYAO.IS" in matched
    assert len(matched["THYAO.IS"]) == 1


def test_news_for_ticker():
    items = parse_rss(SAMPLE_RSS, source="test")
    assert len(news_for_ticker("THYAO.IS", items)) == 1
    assert len(news_for_ticker("GARAN.IS", items)) == 0


def test_match_ticker_and_text():
    assert match_ticker("Aselsan yeni sözleşme imzaladı") == "ASELS.IS"
    assert match_text_for_ticker("Tüpraş refinery news", "TUPRS.IS") is True
    assert match_text_for_ticker("unrelated text", "TUPRS.IS") is False


# --- Onbellek kapsami (bug fix regresyonu) ---

def test_cache_rejects_noncovering_range(tmp_path):
    idx = pd.bdate_range("2023-01-02", periods=100)  # 2023-01-02 .. 2023-05-19
    df = pd.DataFrame({"close": np.arange(100.0)}, index=idx)
    p = tmp_path / "x.parquet"
    _write_cache(p, df)
    # Ayni aralik -> kabul
    assert _read_cache(p, ttl_hours=1, start="2023-01-02", end="2023-05-19") is not None
    # Cok sonraki aralik isteniyor (2026) -> RED (2023 hatasinin regresyonu)
    assert _read_cache(p, ttl_hours=1, start="2025-09-01", end="2026-10-01") is None


def test_cache_merges_union(tmp_path):
    idx1 = pd.bdate_range("2023-01-02", periods=50)
    idx2 = pd.bdate_range("2023-04-03", periods=50)
    df1 = pd.DataFrame({"close": np.arange(50.0)}, index=idx1)
    df2 = pd.DataFrame({"close": np.arange(50.0, 100.0)}, index=idx2)
    p = tmp_path / "y.parquet"
    _write_cache(p, df1)
    _write_cache(p, df2, existing=df1)
    merged = pd.read_parquet(p)
    assert len(merged) == 100
    assert merged.index.min() == idx1[0]
    assert merged.index.max() == idx2[-1]


# --- ML ---

def test_train_and_predict_roundtrip(tmp_path, monkeypatch):
    settings = load_settings()
    # Hizli egitim: az seed
    trained = train(list(range(1, 4)), settings=settings)
    assert trained.auc > 0.8  # sentetik ayrım öğrenilmeli
    assert set(trained.columns) >= {"sig_volume", "sig_index_divergence"}

    path = tmp_path / "m.joblib"
    save_model(trained, path=path, settings=settings)
    assert path.exists()

    # Yukle + tahmin (tum egitim sutunlari verilmeli — imputer 6 sutunla egitildi)
    row = {c: 0.85 for c in trained.columns}
    row_low = {c: 0.10 for c in trained.columns}
    scores = predict_scores(
        pd.DataFrame([row, row_low]),
        trained,
    )
    assert 0.0 <= scores.iloc[0] <= 1.0
    assert scores.iloc[0] > scores.iloc[1]  # yuksek sinyal -> yuksek olasilik


def test_ml_model_is_loaded_in_signals():
    """Kayitli model varsa compute_signals sig_ml uretir."""
    from mars.signals import compute_signals
    from mars.ingestion import align_stock_market
    from mars.synthetic import SyntheticConfig, generate_market, generate_stock

    settings = load_settings()
    if load_model(settings) is None:
        pytest.skip("egitilmis model yok (scripts/train_ml.py calistirilmali)")
    cfg = SyntheticConfig(n_days=200, seed=3)
    sf = compute_signals(align_stock_market(generate_stock(cfg), generate_market(cfg)), settings)
    assert "sig_ml" in sf.frame.columns
    assert sf.frame["sig_ml"].notna().any()


# --- Detayli aciklama ---

def test_detailed_reasons_structure():
    srow = pd.Series({
        "vol_z": 6.2, "volatility_z": 1.1, "illiquidity_z": 0.4,
        "abnormal_ret": 0.05, "log_ret": 0.06, "market_log_ret": 0.01,
        "sig_mahalanobis": 0.9, "sig_isolation_forest": 0.85, "sig_ml": 0.7,
    })
    frow = pd.Series({
        "contrib_sig_volume": 20.0, "contrib_sig_volatility": 3.0,
        "contrib_sig_liquidity": 1.0, "contrib_sig_index_divergence": 15.0,
        "contrib_sig_mahalanobis": 10.0, "contrib_sig_isolation_forest": 6.0,
        "contrib_sig_ml": 8.0, "suspicion": 63.0, "data_confidence": 90.0,
        "news_factor": 1.0,
    })
    reasons = detailed_reasons(srow, frow)
    assert len(reasons) >= 6
    assert all("detail" in r and "points" in r for r in reasons)
    # En yuksek katki en ustte
    assert reasons[0]["points"] == max(r["points"] for r in reasons)
    # Deger bazli metin icerigi
    assert "robust z = +6.2" in reasons[0]["detail"] or any("6.2" in r["detail"] for r in reasons)
