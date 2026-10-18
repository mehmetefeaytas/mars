# SPDX-License-Identifier: AGPL-3.0-only
"""MARS Streamlit arayuzu — interaktif anomali ve risk karti.

Calistirma:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mars.config import load_settings  # noqa: E402
from mars.explain import waterfall  # noqa: E402
from mars.news.analyzer import analyze_news  # noqa: E402
from mars.news.sources import NewsItem  # noqa: E402
from mars.pipeline import analyze  # noqa: E402
from mars.synthetic import SyntheticConfig, generate_market, generate_stock  # noqa: E402
from mars.fusion.weighted import fuse  # noqa: E402
from mars.ingestion import align_stock_market  # noqa: E402
from mars.signals import compute_signals  # noqa: E402
from mars.models.market_model import rolling_abnormal_returns  # noqa: E402

st.set_page_config(page_title="MARS — Piyasa Anomali Skoru", layout="wide")

settings = load_settings()

SAMPLE_TICKERS = [
    "THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS", "SISE.IS",
    "KCHOL.IS", "BIMAS.IS", "TUPRS.IS", "PETKM.IS", "SASA.IS",
]


@st.cache_data(show_spinner=False)
def _run_synthetic(seed: int, kind: str) -> dict:
    cfg = SyntheticConfig(n_days=400, seed=seed)
    stock = generate_stock(cfg, n_injections=6, injection_kind=kind)
    market = generate_market(cfg)
    aligned = align_stock_market(stock, market)
    sf = compute_signals(aligned, settings)
    ar = rolling_abnormal_returns(sf.frame["log_ret"], sf.frame["market_log_ret"],
                                  window=settings.estimation_window)
    f = fuse(sf.frame, abnormal_returns=ar, settings=settings)
    return {"frame": sf.frame.join(f.frame), "inj": list(map(int, stock.attrs.get("injection_idx", [])))}


def _run_real(ticker: str, news_items: list[NewsItem]) -> dict:
    news_by_date = {}
    for item in news_items:
        if item.timestamp is not None:
            news_by_date.setdefault(pd.Timestamp(item.timestamp).normalize(), []).append(
                analyze_news(item, settings)
            )
    res = analyze(ticker, settings=settings, news_by_date=news_by_date or None)
    return {"frame": res.frame, "summary": res.summary(), "waterfall": res.waterfall(),
            "source": res.data_source}


st.title("MARS — BIST Piyasa Anomali ve Açıklanabilir Risk Skoru")
st.caption("Fark et → Skorla → Açıkla · Anomali skoru ≠ manipülasyon kanıtı")

with st.sidebar:
    st.header("Analiz")
    mode = st.radio("Mod", ["Gerçek veri (yfinance)", "Sentetik demo"], index=0)
    if mode == "Gerçek veri (yfinance)":
        ticker = st.selectbox("Hisse", SAMPLE_TICKERS, index=0)
        custom = st.text_input("veya ticker yaz (or. KONTR.IS)", "")
        if custom.strip():
            ticker = custom.strip().upper()
        st.subheader("Haber (opsiyonel)")
        headline = st.text_area("Başlık/özet", "")
        source = st.text_input("Kaynak", "kap")
        news_items = []
        if headline.strip():
            news_items = [NewsItem(headline=headline, source=source, official=source.lower() == "kap",
                                   timestamp=pd.Timestamp.utcnow().tz_localize(None).normalize())]
    else:
        kind = st.selectbox("Senaryo", ["pump", "subtle", "news_legit"], index=0)
        seed = st.slider("Seed", 1, 30, 7)
    run = st.button("Analiz Et", type="primary")

frame = None
summary = ""
wf: list[dict] = []
inj: list[int] = []

if run:
    try:
        if mode == "Gerçek veri (yfinance)":
            out = _run_real(ticker, news_items)
            frame = out["frame"]
            summary = out["summary"]
            wf = out["waterfall"]
            st.success(f"Veri kaynağı: {out['source']}")
        else:
            out = _run_synthetic(seed, kind)
            frame = out["frame"]
            inj = out["inj"]
            last = frame.iloc[-1]
            summary = f"Son gün şüphe skoru {last['suspicion']:.0f}/100."
            wf = waterfall(last)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analiz başarısız: {exc}")

if frame is not None:
    last = frame.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Şüphe Skoru", f"{last['suspicion']:.0f}/100")
    c2.metric("Veri Güveni", f"{last['data_confidence']:.0f}/100")
    c3.metric("Haber Faktörü", f"{last.get('news_factor', 1.0):.2f}")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Zaman Serisi — Şüphe Skoru")
        chart = pd.DataFrame({"Şüphe": frame["suspicion"]})
        if inj:
            chart["Enjeksiyon"] = 0.0
            chart.iloc[[i for i in inj if i < len(chart)], chart.columns.get_loc("Enjeksiyon")] = 100.0
        st.line_chart(chart, height=320)
        st.caption("Sentetik modda kırmızı çizgiler bilinen anomali günleridir.")

    with right:
        st.subheader("Sinyal Katkıları (şelale)")
        if wf:
            wdf = pd.DataFrame(wf)
            st.bar_chart(wdf.set_index("signal"), horizontal=True, height=320)
        st.subheader("Özet")
        st.write(summary)

    st.subheader("Açıklanabilir Risk Kartı")
    with st.expander("Öne çıkan sinyal katkıları", expanded=True):
        st.dataframe(pd.DataFrame(wf), use_container_width=True, hide_index=True)
    with st.expander("Haber notları"):
        notes = last.get("news_notes", [])
        st.write(notes if notes else "Bu gün için doğrulanmış haber sinyali yok.")
