# SPDX-License-Identifier: AGPL-3.0-only
"""Acilanabilirlik katalogu: sinyaller okunabilir ve aciklayici olmali.

Amac: istatistik bilmeyen bir kullanicinin skoru artiran sinyalleri
(mahalanobis, iForest, index divergence...) anlayabilmesi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mars.explain import (
    _SIGNAL_INFO,
    contributions_row,
    detailed_reasons,
    signal_catalog,
    waterfall,
)


def _fusion_row() -> pd.Series:
    return pd.Series({
        "suspicion": 66.4,
        "data_confidence": 84.8,
        "contrib_sig_volume": 9.1,
        "contrib_sig_index_divergence": 17.3,
        "contrib_sig_volatility": 1.4,
        "contrib_sig_mahalanobis": 14.4,
        "contrib_sig_isolation_forest": 8.3,
        "contrib_sig_liquidity": 3.1,
        "contrib_sig_ml": 13.3,
        "contrib_sig_close_location": 0.0,
        "contrib_sig_pump_pattern": 6.9,
    })


def _signal_row() -> pd.Series:
    return pd.Series({
        "vol_z": 5.8, "volatility_z": 1.2, "illiquidity_z": 2.2,
        "abnormal_ret": 0.092, "log_ret": 0.095, "market_log_ret": 0.005,
        "sig_mahalanobis": 1.0, "sig_isolation_forest": 0.9, "sig_ml": 0.97,
        "clv": 0.8, "car5": 0.11, "up_streak": 4,
        "liquidity_tier": "sığ", "tahta_prior": 0.9,
    })


# --- Katalog ---
def test_catalog_covers_all_signals():
    cat = signal_catalog()
    assert len(cat) == len(_SIGNAL_INFO) == 9
    for item in cat:
        assert set(item) >= {"key", "column", "short", "title", "what"}
        assert item["column"].startswith("sig_")
        assert item["key"] == item["column"].replace("sig_", "")
        # Her aciklama gercek bir cumle olmali
        assert len(item["what"]) > 80, item["key"]
        assert item["what"].endswith("."), item["key"]


def test_catalog_has_no_technical_only_titles():
    # Baslik/mekanizma adlari sade dille aciklanmis olmali (parantez icinde terim).
    for item in signal_catalog():
        assert "—" in item["title"]
        for banned in ("index_divergence", "isolation_forest", "mahalanobis"):
            assert banned not in item["short"]


# --- Waterfall ---
def test_waterfall_is_sorted_and_descriptive():
    wf = waterfall(_fusion_row(), top_k=9)
    pts = [w["points"] for w in wf]
    assert pts == sorted(pts, reverse=True)
    assert wf[0]["key"] == "index_divergence"
    assert wf[0]["signal"] == "Endeksten kopma"
    assert "genel piyasadan" in wf[0]["title"].lower()
    for w in wf:
        assert w["title"] and "—" in w["title"]


def test_contributions_row_labels_turkish():
    df = contributions_row(_fusion_row())
    labels = list(df["signal"])
    assert "İşlem hacmi" in labels
    assert "Likidite (tahta) kolaylığı" in labels
    for raw in ("volume", "ml", "liquidity", "pump_pattern"):
        assert raw not in labels


# --- Gerekceler ---
def test_reasons_have_beginner_fields():
    reasons = detailed_reasons(_signal_row(), _fusion_row())
    by_key = {r["key"]: r for r in reasons}
    assert by_key["index_divergence"]["points"] == 17.3
    assert by_key["mahalanobis"]["what"].startswith("Sistem")
    assert "0.73" not in by_key["mahalanobis"]["detail"]  # gercek deger yazilir
    assert "1.00" in by_key["mahalanobis"]["detail"]
    for r in reasons:
        assert r["title"] and "—" in r["title"]
        assert r["what"], r["key"]
        assert r["detail"]


def test_context_reason_is_not_scored():
    reasons = detailed_reasons(_signal_row(), _fusion_row())
    ctx = [r for r in reasons if r["level"] == "bağlam"]
    assert len(ctx) == 1
    assert ctx[0]["points"] == 0.0
    assert ctx[0]["key"] == "context_tahta"
    assert "ayar" in ctx[0]["what"].lower()


def test_reasons_sorted_by_points_desc():
    reasons = detailed_reasons(_signal_row(), _fusion_row())
    pts = [r["points"] for r in reasons]
    assert pts == sorted(pts, reverse=True)


def test_reasons_survive_missing_values():
    # Bos/NaN satir cokmemeli
    sig = pd.Series({"liquidity_tier": "", "tahta_prior": np.nan})
    reasons = detailed_reasons(sig, _fusion_row())
    assert isinstance(reasons, list)
