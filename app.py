# SPDX-License-Identifier: AGPL-3.0-only
"""MARS Streamlit arayuzu — profesyonel trading-terminal tarzi dashboard.

Calistirma:
    PYTHONPATH=src .venv/bin/streamlit run app.py

Bolumler:
  * Şüpheli Hisseler — sepet taramasi, skor siralamasi
  * Hisse Analizi — mum grafigi + suphe serisi + sinyal katkilari + detayli gerekceler
  * Haber Akışı — RSS haberleri, polarite/guvenilirlik analizi, hisse eslestirme
  * LLM Açıklama — yerel model ile gunun yorumu (kural-tabanli fallback)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mars.config import FusionWeights, Settings, load_settings  # noqa: E402
from mars.explain import detailed_reasons, summarize_day, waterfall  # noqa: E402
from mars.fusion.weighted import fuse  # noqa: E402
from mars.news.analyzer import analyze_news  # noqa: E402
from mars.news.llm import explain_with_llm, llm_available  # noqa: E402
from mars.news.rss import auto_match, fetch_all_feeds, news_for_ticker  # noqa: E402
from mars.news.sources import CredibilityResult, NewsItem, NewsSignal, SentimentResult  # noqa: E402
from mars.news.verishield_adapter import verishield_available  # noqa: E402
from mars import ml as ml_module  # noqa: E402
from mars.pipeline import analyze  # noqa: E402

st.set_page_config(
    page_title="MARS · Piyasa Anomali Terminali",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Gorunum — trading terminal temasi
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .main, .block-container {
    font-family: 'Inter', -apple-system, sans-serif;
}
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px; }

/* ---- Ust bar ---- */
.mars-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 1.4rem; margin-bottom: 1.1rem;
    background: linear-gradient(120deg, #101625 0%, #0d1320 55%, #101a2b 100%);
    border: 1px solid #1e2739; border-radius: 16px;
}
.mars-logo { display: flex; align-items: center; gap: 14px; }
.mars-logo-mark {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #2dd4bf 0%, #0ea5e9 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 800; color: #04121a;
    box-shadow: 0 0 24px rgba(45,212,191,.35);
}
.mars-logo h1 { font-size: 1.35rem; font-weight: 800; margin: 0; letter-spacing: .4px; }
.mars-logo p { font-size: .78rem; color: #8b94a7; margin: 0; }
.mars-pills { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.pill {
    font-size: .7rem; font-weight: 600; padding: 5px 11px; border-radius: 999px;
    border: 1px solid #243046; color: #9aa4b8; background: #0d1320;
    white-space: nowrap;
}
.pill.on  { color: #34d399; border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }
.pill.off { color: #f87171; border-color: rgba(248,113,113,.35); background: rgba(248,113,113,.07); }
.pill.warn{ color: #fbbf24; border-color: rgba(251,191,36,.35); background: rgba(251,191,36,.08); }

/* ---- Kartlar ---- */
.card {
    background: linear-gradient(180deg, #121826 0%, #0f1420 100%);
    border: 1px solid #1e2739; border-radius: 14px;
    padding: 1.05rem 1.15rem; margin-bottom: .9rem;
}
.card h3 { font-size: .95rem; font-weight: 700; margin: 0 0 .6rem; color: #cbd5e1; }

/* ---- KPI ---- */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #121826 0%, #0f1420 100%);
    border: 1px solid #1e2739; border-radius: 14px; padding: .9rem 1rem;
}
[data-testid="stMetricLabel"] p { font-size: .72rem !important; color: #8b94a7 !important; }
[data-testid="stMetricValue"] { font-weight: 800 !important; }

/* ---- Sekmeler ---- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #1e2739; }
.stTabs [data-baseweb="tab"] {
    background: #0d1320; border: 1px solid #1e2739; border-radius: 10px 10px 0 0;
    padding: 8px 18px; font-size: .82rem; font-weight: 600; color: #8b94a7;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #143028, #0f1c1a) !important;
    color: #2dd4bf !important; border-color: rgba(45,212,191,.4) !important;
}

/* ---- Haber kartlari ---- */
.news-card {
    border: 1px solid #1e2739; border-radius: 12px; padding: .8rem 1rem;
    margin-bottom: .55rem; background: #0f1420;
}
.news-card .t { font-weight: 600; font-size: .92rem; color: #e2e8f0; }
.news-card .m { font-size: .74rem; color: #8b94a7; margin-top: 3px; }
.badge {
    display: inline-block; font-size: .68rem; font-weight: 700;
    padding: 2px 9px; border-radius: 999px; margin-right: 6px;
}
.b-pos { color: #34d399; background: rgba(52,211,153,.12); }
.b-neg { color: #f87171; background: rgba(248,113,113,.12); }
.b-neu { color: #9aa4b8; background: rgba(154,164,184,.12); }

/* ---- LLM kutusu ---- */
.llm-box {
    border: 1px solid rgba(45,212,191,.3); border-radius: 14px;
    padding: 1.1rem 1.2rem; background: linear-gradient(180deg, #0f1c1a, #0d1420);
    line-height: 1.65; font-size: .93rem;
}

.footer-note { color: #667085; font-size: .72rem; margin-top: 1.2rem; }
div[data-testid="stExpander"] { border: 1px solid #1e2739; border-radius: 12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Yardimcilar
# --------------------------------------------------------------------------
def _settings_with(**overrides) -> Settings:
    s = load_settings()
    for k, v in overrides.items():
        if hasattr(s, k):
            setattr(s, k, v)
    return s


@st.cache_data(ttl=900, show_spinner=False)
def _signals_for(
    ticker: str, lookback: int, rolling: int, est_window: int,
    iforest_trees: int, contamination: float,
):
    """Agir kisim (indirme + sinyal hesabi) onbellekten beslenir.

    Füzyon (agirliklar/haber) hafif oldugu icin onbellek DISINDA yapilir;
    boylece parametre degisikligi aninda yansir.
    """
    s = _settings_with(
        lookback_days=lookback, rolling_window=rolling, estimation_window=est_window,
        iforest_trees=iforest_trees, iforest_contamination=contamination,
    )
    try:
        res = analyze(ticker, settings=s, news_by_date=None)
        sf = res.signal_frame
        ml_ok = bool(sf["sig_ml"].notna().any()) if "sig_ml" in sf.columns else False
        return sf, ml_ok, res.data_source, ""
    except Exception as exc:  # noqa: BLE001
        return None, False, "", f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_news(max_items: int):
    s = _settings_with(rss_max_items=max_items)
    items, errors = fetch_all_feeds(s)
    payload = [
        {
            "headline": i.headline, "body": i.body, "source": i.source,
            "url": i.url, "timestamp": None if i.timestamp is None else str(i.timestamp),
        }
        for i in items
    ]
    return payload, errors


def _news_items_from_payload(payload: list[dict]) -> list[NewsItem]:
    out = []
    for p in payload:
        ts = pd.Timestamp(p["timestamp"]) if p["timestamp"] else None
        out.append(NewsItem(
            headline=p["headline"], body=p["body"], source=p["source"],
            url=p["url"], timestamp=ts,
        ))
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _analyze_news_cached(headline: str, body: str, source: str, ts_str: str | None):
    """Tek haber analizi (VeriShield + duygu) — onbellekli, seri hale getirilebilir."""
    s = load_settings()
    item = NewsItem(headline=headline, body=body, source=source,
                    timestamp=pd.Timestamp(ts_str) if ts_str else None)
    sig = analyze_news(item, s)
    return (
        sig.sentiment.polarity, sig.sentiment.label, sig.sentiment.confidence,
        sig.sentiment.backend, sig.credibility.score, sig.credibility.backend,
    )


def _rebuild_signal(item: NewsItem, analyzed: tuple) -> NewsSignal:
    """Onbellekten gelen demeti gercek NewsSignal nesnesine cevirir."""
    pol, lab, conf, _sback, cred, _cback = analyzed
    return NewsSignal(
        item=item,
        sentiment=SentimentResult(polarity=pol, label=lab, confidence=conf, backend="cache"),
        credibility=CredibilityResult(score=cred, backend="cache"),
    )


def _group_news_by_date(signals: list[NewsSignal]) -> dict[pd.Timestamp, list[NewsSignal]]:
    out: dict[pd.Timestamp, list[NewsSignal]] = {}
    for sig in signals:
        if sig.item.timestamp is not None:
            out.setdefault(pd.Timestamp(sig.item.timestamp).normalize(), []).append(sig)
    return out


def _risk_color(score: float) -> str:
    if score >= 65:
        return "#f87171"
    if score >= 35:
        return "#fbbf24"
    return "#34d399"


def _fmt_date(ts) -> str:
    return pd.Timestamp(ts).strftime("%d %b %Y")


# --------------------------------------------------------------------------
# Ust bar
# --------------------------------------------------------------------------
def _render_header(settings: Settings):
    ml_model = ml_module.load_model(settings)
    pills = []
    pills.append(
        f'<span class="pill on">VeriShield ✓ gomulu</span>'
        if verishield_available() else
        '<span class="pill warn">VeriShield — sezgisel fallback</span>'
    )
    if settings.ml_enabled and ml_model is not None:
        pills.append(
            f'<span class="pill on">ML modeli ✓ AUC {ml_model.auc:.3f}</span>'
        )
    else:
        pills.append('<span class="pill warn">ML modeli yok</span>')
    if settings.llm_enabled:
        pills.append(
            '<span class="pill on">LLM ✓</span>' if llm_available(settings)
            else '<span class="pill warn">LLM kapalı/erişilemez</span>'
        )
    else:
        pills.append('<span class="pill off">LLM kapalı</span>')
    if settings.rss_enabled:
        pills.append('<span class="pill on">RSS ✓</span>')
    else:
        pills.append('<span class="pill off">RSS kapalı</span>')
    now = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    pills.append(f'<span class="pill">{now}</span>')

    st.markdown(
        f"""
        <div class="mars-header">
          <div class="mars-logo">
            <div class="mars-logo-mark">M</div>
            <div>
              <h1>MARS <span style="color:#2dd4bf">Terminal</span></h1>
              <p>BIST Piyasa Anomali ve Açıklanabilir Risk Skorlama · Fark et → Skorla → Açıkla</p>
            </div>
          </div>
          <div class="mars-pills">{''.join(pills)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Kenar cubugu — parametreler
# --------------------------------------------------------------------------
def _render_sidebar(settings: Settings) -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Parametreler")
        mode = st.radio("Veri modu", ["Gerçek veri (yfinance)", "Sentetik demo"],
                        index=0, help="Sentetik mod ağ gerektirmez, bilinen enjeksiyon günleriyle gösterim yapar.")

        params: dict = {"mode": mode}

        if mode == "Gerçek veri (yfinance)":
            params["ticker"] = st.text_input(
                "Hisse kodu", value=st.session_state.get("ticker", "THYAO.IS"),
                help="ör. THYAO.IS, GARAN.IS"
            ).strip().upper()
        else:
            params["scenario"] = st.selectbox("Senaryo", ["pump", "subtle", "news_legit"], index=0)
            params["seed"] = st.slider("Seed", 1, 30, 7)

        with st.expander("📈 Piyasa & Motor", expanded=False):
            params["lookback"] = st.slider("Geçmiş uzunluğu (gün)", 180, 750, 400, 50)
            params["rolling"] = st.slider("Robust pencere (gün)", 20, 120, 60, 5,
                                          help="MAD z-skorları için yuvarlanan pencere")
            params["est_window"] = st.slider("Market model penceresi (gün)", 60, 250, 120, 10,
                                             help="Rolling beta tahmin penceresi (MacKinlay 1997)")
            params["iforest_trees"] = st.slider("iForest ağaç sayısı", 50, 500, 200, 50)
            params["contamination"] = st.slider("iForest contamination", 0.005, 0.10, 0.02, 0.005)

        with st.expander("⚖️ Füzyon ağırlıkları", expanded=False):
            params["w"] = FusionWeights(
                volume_z=st.slider("Hacim sapması", 0.0, 0.5, 0.22, 0.01),
                index_divergence=st.slider("Endeksten kopuş", 0.0, 0.5, 0.18, 0.01),
                volatility=st.slider("Oynaklık", 0.0, 0.5, 0.13, 0.01),
                mahalanobis=st.slider("MCD uzaklığı", 0.0, 0.5, 0.15, 0.01),
                isolation_forest=st.slider("Isolation Forest", 0.0, 0.5, 0.10, 0.01),
                liquidity=st.slider("Likidite (Amihud)", 0.0, 0.5, 0.07, 0.01),
                ml=st.slider("ML olasılığı", 0.0, 0.5, 0.15, 0.01),
            )
            params["news_penalty"] = st.slider(
                "Haber ceza gücü", 0.0, 0.6, 0.35, 0.05,
                help="Doğrulanmış haber skoru ne kadar düşürsün (Bayes güncellemesi)"
            )

        with st.expander("📰 Haberler & LLM", expanded=False):
            params["rss"] = st.checkbox("RSS akışları", value=settings.rss_enabled)
            params["rss_max"] = st.slider("Maks haber", 20, 120, 60, 10)
            params["llm"] = st.checkbox(
                "LLM açıklaması (yerel Ollama)", value=settings.llm_enabled,
                help="Ollama çalıştırın: ollama serve && ollama pull qwen2.5:7b-instruct"
            )
        return params


# --------------------------------------------------------------------------
# Grafikler
# --------------------------------------------------------------------------
def _price_chart(frame: pd.DataFrame) -> go.Figure:
    """Mum grafigi + hacim + suphe skoru."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.14, 0.34], vertical_spacing=0.04,
    )
    up = frame["close"] >= frame["open"]
    fig.add_trace(go.Candlestick(
        x=frame.index, open=frame["open"], high=frame["high"],
        low=frame["low"], close=frame["close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        name="Fiyat",
    ), row=1, col=1)
    colors = ["#22c55e" if u else "#ef4444" for u in up]
    fig.add_trace(go.Bar(
        x=frame.index, y=frame["volume"], marker_color=colors,
        name="Hacim", opacity=0.75,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["suspicion"], mode="lines",
        line=dict(color="#2dd4bf", width=2), name="Şüphe skoru",
        fill="tozeroy", fillcolor="rgba(45,212,191,0.10)",
    ), row=3, col=1)
    fig.add_hline(y=65, row=3, col=1, line=dict(color="#f87171", width=1, dash="dot"))
    fig.add_hline(y=35, row=3, col=1, line=dict(color="#fbbf24", width=1, dash="dot"))
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9aa4b8", size=11),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#1a2233", zerolinecolor="#1a2233")
    return fig


def _waterfall_chart(wf: list[dict]) -> go.Figure:
    labels = [w["signal"] for w in wf][::-1]
    values = [w["points"] for w in wf][::-1]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=["#2dd4bf" if v >= 8 else "#475569" for v in values]),
        text=[f"+{v:.1f}" for v in values], textposition="outside",
        textfont=dict(color="#cbd5e1"),
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9aa4b8", size=11), xaxis=dict(gridcolor="#1a2233"),
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------
# Sekme: Supheli Hisseler (screener)
# --------------------------------------------------------------------------
BASKET = [
    "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "EREGL.IS", "SISE.IS",
    "KCHOL.IS", "BIMAS.IS", "TUPRS.IS", "PETKM.IS", "SASA.IS", "FROTO.IS",
]


def _tab_screener(params: dict, settings: Settings):
    st.subheader("🔍 Şüpheli Hisseler — Sepet Taraması")
    st.caption("BIST sepeti son gün itibarıyla MARS şüphe skoruna göre sıralanır. "
               "İlk çalıştırmada veriler indirilir (ağ gerektirir), sonra önbellekten gelir.")

    col1, col2 = st.columns([3, 1])
    with col2:
        basket_str = st.text_area("Sepet (virgülle)", value=", ".join(BASKET), height=110)
        basket = [t.strip().upper() for t in basket_str.split(",") if t.strip()]
        run_scan = st.button("Sepeti Tara", type="primary", width='stretch')

    if not run_scan and "screener" not in st.session_state:
        st.info("Taramayı başlatmak için **Sepeti Tara** düğmesine basın.")
        return

    if run_scan or "screener" not in st.session_state or st.session_state.get("screener_params") != params:
        news_payload, _err = _fetch_news(params["rss_max"]) if params["rss"] else ([], [])
        news_items = _news_items_from_payload(news_payload)
        matched = auto_match(news_items)

        rows = []
        progress = st.progress(0.0, text="Sepet taranıyor…")
        for i, t in enumerate(basket):
            sf, ml_ok, src, err = _signals_for(
                t, params["lookback"], params["rolling"], params["est_window"],
                params["iforest_trees"], params["contamination"],
            )
            if sf is None:
                rows.append({"Hisse": t, "Şüphe": np.nan, "Veri Güveni": np.nan,
                             "Getiri (%)": np.nan, "AR (%)": np.nan,
                             "En Etkili Sinyal": f"hata: {err[:40]}", "Haber": 0})
            else:
                s = _settings_with(
                    lookback_days=params["lookback"], rolling_window=params["rolling"],
                    estimation_window=params["est_window"],
                    iforest_trees=params["iforest_trees"],
                    iforest_contamination=params["contamination"],
                    news_penalty_strength=params["news_penalty"],
                )
                news_sigs = [
                    _rebuild_signal(
                        item,
                        _analyze_news_cached(
                            item.headline, item.body, item.source,
                            str(item.timestamp) if item.timestamp is not None else None,
                        ),
                    )
                    for item in matched.get(t, [])[:5]
                ]
                fusion = fuse(
                    sf, news_by_date=_group_news_by_date(news_sigs) or None,
                    abnormal_returns=sf["abnormal_ret"], settings=s,
                    weights=params["w"],
                )
                last = fusion.frame.iloc[-1]
                wf = waterfall(fusion.frame.iloc[-1], top_k=1)
                ar_last = sf["abnormal_ret"].iloc[-1]
                rows.append({
                    "Hisse": t,
                    "Şüphe": round(float(last["suspicion"]), 1),
                    "Veri Güveni": round(float(last["data_confidence"]), 0),
                    "Getiri (%)": round(float(sf["log_ret"].iloc[-1]) * 100, 2),
                    "AR (%)": round(float(ar_last) * 100, 2)
                              if np.isfinite(ar_last) else np.nan,
                    "En Etkili Sinyal": wf[0]["signal"] if wf else "—",
                    "Haber": len(matched.get(t, [])),
                })
            progress.progress((i + 1) / len(basket), text=f"Sepet taranıyor… {t}")
        progress.empty()
        result = pd.DataFrame(rows).sort_values("Şüphe", ascending=False, na_position="last")
        st.session_state["screener"] = result
        st.session_state["screener_params"] = params

    result = st.session_state["screener"]
    st.dataframe(
        result,
        width='stretch', hide_index=True,
        column_config={
            "Şüphe": st.column_config.ProgressColumn(
                "Şüphe Skoru", min_value=0, max_value=100, format="%.1f",
            ),
            "Veri Güveni": st.column_config.ProgressColumn(
                "Veri Güveni", min_value=0, max_value=100, format="%.0f",
            ),
            "Getiri (%)": st.column_config.NumberColumn("Getiri (%)", format="%.2f"),
            "AR (%)": st.column_config.NumberColumn("AR (%)", format="%.2f"),
        },
    )
    st.download_button(
        "CSV indir", result.to_csv(index=False).encode("utf-8"),
        file_name=f"mars_screener_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv",
    )
    st.caption("⚠️ Şüphe skoru manipülasyon kanıtı değildir; karar-destek ölçütüdür.")


# --------------------------------------------------------------------------
# Sekme: Hisse Analizi
# --------------------------------------------------------------------------
def _tab_analysis(params: dict, settings: Settings):
    if params["mode"] == "Sentetik demo":
        _tab_analysis_synthetic(params, settings)
        return

    ticker = params.get("ticker", "THYAO.IS")
    st.subheader(f"📊 {ticker} — Detaylı Analiz")

    sf, ml_ok, src, err = _signals_for(
        ticker, params["lookback"], params["rolling"], params["est_window"],
        params["iforest_trees"], params["contamination"],
    )
    if sf is None:
        st.error(f"Analiz başarısız: {err}")
        st.info("Öneri: hisse kodunu .IS uzantısıyla yazın (ör. THYAO.IS) ve internet "
                "bağlantısını kontrol edin.")
        return

    # Haberler
    news_sigs: list[NewsSignal] = []
    if params["rss"]:
        news_payload, _err = _fetch_news(params["rss_max"])
        matched = news_for_ticker(ticker, _news_items_from_payload(news_payload))
        news_sigs = [
            _rebuild_signal(
                item,
                _analyze_news_cached(
                    item.headline, item.body, item.source,
                    str(item.timestamp) if item.timestamp is not None else None,
                ),
            )
            for item in matched[:8]
        ]
        if matched:
            st.caption(f"📰 Bu hisseyle eşleşen {len(matched)} haber bulundu (RSS).")

    s = _settings_with(
        lookback_days=params["lookback"], rolling_window=params["rolling"],
        estimation_window=params["est_window"], iforest_trees=params["iforest_trees"],
        iforest_contamination=params["contamination"],
        news_penalty_strength=params["news_penalty"],
    )
    fusion = fuse(
        sf, news_by_date=_group_news_by_date(news_sigs) or None,
        abnormal_returns=sf["abnormal_ret"], settings=s, weights=params["w"],
    )
    frame = sf.join(fusion.frame, how="left")
    st.session_state["last_frame"] = frame
    st.session_state["last_ticker"] = ticker

    # KPI kartlari
    last = frame.iloc[-1]
    prev = frame.iloc[-2] if len(frame) > 1 else last
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Şüphe Skoru", f"{last['suspicion']:.0f}/100",
              f"{last['suspicion'] - prev['suspicion']:+.1f} dün'e göre")
    k2.metric("Veri Güveni", f"{last['data_confidence']:.0f}/100")
    k3.metric("Günlük Getiri", f"{last['log_ret']*100:+.2f}%",
              f"{last['log_ret']*100 - prev['log_ret']*100:+.2f} pp")
    ar_now = last.get("abnormal_ret", float("nan"))
    k4.metric("Anormal Getiri (AR)", f"{ar_now*100:+.2f}%" if np.isfinite(ar_now) else "—")
    k5.metric("Haber Faktörü", f"{last.get('news_factor', 1.0):.2f}",
              "açıklayıcı" if last.get("news_factor", 1.0) < 0.999 else "etkisiz")

    st.plotly_chart(_price_chart(frame), width='stretch')

    # Gun secici + gerekceler
    st.markdown('<div class="card"><h3>🧩 Sinyal Katkıları ve Gerekçeler</h3>', unsafe_allow_html=True)
    day_options = frame.index[-90:]
    day_idx = st.select_slider(
        "İnceleme günü", options=list(range(len(day_options))),
        value=len(day_options) - 1,
        format_func=lambda i: day_options[i].strftime("%d %b %Y"),
    )
    day = day_options[day_idx]
    srow, frow = frame.loc[day], fusion.frame.loc[day]

    c_left, c_right = st.columns([1, 1.4])
    with c_left:
        st.plotly_chart(_waterfall_chart(waterfall(frow, top_k=7)), width='stretch')
    with c_right:
        st.markdown(f"**{_fmt_date(day)}** — Şüphe **{frow['suspicion']:.0f}/100** · "
                    f"veri güveni {frow['data_confidence']:.0f}/100")
        reasons = detailed_reasons(srow, frow)
        for r in reasons:
            color = _risk_color(min(100, r["points"] * 4))
            st.markdown(
                f"<div class='news-card'>"
                f"<span class='t' style='color:{color}'>● {r['signal']} (+{r['points']})</span>"
                f"<div class='m'>{r['detail']}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "⚙️ Analizi CSV indir",
        frame.drop(columns=[c for c in frame.columns if c == "news_notes"], errors="ignore")
        .round(4).to_csv().encode("utf-8"),
        file_name=f"mars_{ticker.replace('.', '_')}_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
    )
    st.caption(f"Veri kaynağı: {src} · ML sinyali: {'aktif' if ml_ok else 'yok (model eğitilmemiş)'}")


def _tab_analysis_synthetic(params: dict, settings: Settings):
    from mars.ingestion import align_stock_market  # noqa: PLC0415
    from mars.signals import compute_signals  # noqa: PLC0415
    from mars.synthetic import SyntheticConfig, generate_market, generate_stock  # noqa: PLC0415

    st.subheader(f"🧪 Sentetik Demo — senaryo: {params['scenario']}")
    cfg = SyntheticConfig(n_days=400, seed=params["seed"])
    stock = generate_stock(cfg, n_injections=6, injection_kind=params["scenario"])
    market = generate_market(cfg)
    sf = compute_signals(align_stock_market(stock, market), settings)
    fusion = fuse(sf.frame, abnormal_returns=sf.frame["abnormal_ret"],
                  settings=settings, weights=params["w"])
    frame = sf.frame.join(fusion.frame, how="left")
    st.session_state["last_frame"] = frame
    st.session_state["last_ticker"] = f"SENTETIK-{params['scenario']}"

    inj = list(map(int, stock.attrs.get("injection_idx", [])))
    last = frame.iloc[-1]
    k1, k2, k3 = st.columns(3)
    k1.metric("Şüphe Skoru (son gün)", f"{last['suspicion']:.0f}/100")
    k2.metric("Bilinen Anomali Günü", len(inj))
    k3.metric("Enjeksiyon Ort. Şüphe",
              f"{frame['suspicion'].iloc[inj].mean():.0f}/100" if inj else "—")

    fig = _price_chart(frame)
    if inj:
        fig.add_trace(go.Scatter(
            x=[frame.index[i] for i in inj if i < len(frame)],
            y=[frame["suspicion"].iloc[i] for i in inj if i < len(frame)],
            mode="markers", marker=dict(symbol="diamond", size=10, color="#f59e0b"),
            name="Bilinen enjeksiyon",
        ), row=3, col=1)
    st.plotly_chart(fig, width='stretch')
    st.caption("Turuncu elmaslar: senaryonun enjekte ettiği bilinen anomali günleri. "
               "Değerler sentetiktir — gerçek piyasa çıktısı değildir.")


# --------------------------------------------------------------------------
# Sekme: Haber Akisi
# --------------------------------------------------------------------------
def _tab_news(params: dict, settings: Settings):
    st.subheader("📰 Haber Akışı — RSS")
    if not params["rss"]:
        st.info("RSS kapalı. Kenar çubuğundan açabilirsiniz.")
        return
    with st.spinner("RSS akışları çekiliyor…"):
        news_payload, errors = _fetch_news(params["rss_max"])
    if errors:
        st.warning("Bazı akışlar alınamadı: " + "; ".join(errors))
    if not news_payload:
        st.error("Haber alınamadı (ağ/lisans).")
        return

    items = _news_items_from_payload(news_payload)
    matched = auto_match(items)
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam haber", len(items))
    c2.metric("Hisseyle eşleşen", sum(len(v) for v in matched.values()))
    c3.metric("Eşleşen hisse", len(matched))

    st.markdown('<div class="card"><h3>Son Haberler</h3>', unsafe_allow_html=True)
    for item in items[:25]:
        pol, lab, conf, sback, cred, cback = _analyze_news_cached(
            item.headline, item.body, item.source,
            str(item.timestamp) if item.timestamp is not None else None,
        )
        badge = "b-pos" if pol > 0.05 else "b-neg" if pol < -0.05 else "b-neu"
        lab_tr = {"pozitif": "POZİTİF", "negatif": "NEGATİF"}.get(lab, "NÖTR")
        ts = item.timestamp.strftime("%d %b %H:%M") if item.timestamp is not None else ""
        link = f"<a href='{item.url}' target='_blank' style='color:#2dd4bf;text-decoration:none'>{item.headline}</a>" if item.url else item.headline
        st.markdown(
            f"<div class='news-card'>"
            f"<div class='t'>{link}</div>"
            f"<div class='m'>{item.source} · {ts} · "
            f"<span class='badge {badge}'>{lab_tr} {pol:+.2f}</span>"
            f"<span class='badge b-neu'>Güvenilirlik {cred:.2f}</span>"
            f"<span class='badge b-neu'>polarite: {sback} · güvenilirlik: {cback}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Polarite: lexicon/BERT hibrit · Güvenilirlik: VeriShield 6-kanal motoru.")


# --------------------------------------------------------------------------
# Sekme: LLM Aciklama
# --------------------------------------------------------------------------
def _tab_llm(params: dict, settings: Settings):
    st.subheader("🤖 LLM ile Açıkla")
    frame = st.session_state.get("last_frame")
    ticker = st.session_state.get("last_ticker", "—")
    if frame is None or frame.empty:
        st.info("Önce **Hisse Analizi** sekmesinde bir analiz çalıştırın.")
        return

    s = _settings_with(news_penalty_strength=params["news_penalty"])
    day_options = frame.index[-90:]
    day_idx = st.select_slider(
        "Gün", options=list(range(len(day_options))), value=len(day_options) - 1,
        format_func=lambda i: day_options[i].strftime("%d %b %Y"),
        key="llm_day",
    )
    day = day_options[day_idx]
    srow = frame.loc[day]
    frow = frame.loc[day]

    reasons = detailed_reasons(srow, frow)
    lr = srow.get("log_ret", float("nan"))
    ar = srow.get("abnormal_ret", float("nan"))
    payload = {
        "ticker": ticker,
        "date": str(pd.Timestamp(day).date()),
        "suspicion": round(float(frow["suspicion"]), 1),
        "data_confidence": round(float(frow["data_confidence"]), 1),
        "log_ret": f"{lr*100:+.2f}%" if np.isfinite(lr) else "—",
        "abnormal_ret": f"{ar*100:+.2f}%" if np.isfinite(ar) else "—",
        "reasons": reasons,
        "news": [],
    }

    st.markdown('<div class="card"><h3>Kural-Tabanlı Açıklama (her zaman görünür)</h3>', unsafe_allow_html=True)
    st.markdown(summarize_day(frow, date=pd.Timestamp(day), signal_row=srow))
    for r in reasons[:5]:
        st.markdown(f"- **{r['signal']}** (+{r['points']}): {r['detail']}")
    st.markdown("</div>", unsafe_allow_html=True)

    llm_on = params["llm"]
    st.markdown('<div class="card"><h3>LLM Yorumu</h3>', unsafe_allow_html=True)
    if st.button("✨ LLM ile Açıkla", type="primary", disabled=not llm_on):
        if not llm_available(s):
            st.error("LLM'e ulaşılamadı. Ollama'yı başlatın: `ollama serve` ve "
                     "`ollama pull qwen2.5:7b-instruct`")
        else:
            with st.spinner(f"{s.llm_model} yorumluyor…"):
                text = explain_with_llm(payload, s)
            if text:
                st.markdown(f'<div class="llm-box">{text}</div>', unsafe_allow_html=True)
            else:
                st.warning("LLM yanıtı alınamadı.")
    if not llm_on:
        st.caption("LLM kapalı — kenar çubuğundan açın (yerel Ollama, veri dışarı çıkmaz).")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("⚠️ LLM yorumu bilgilendirme amaçlıdır; yatırım tavsiyesi değildir.")


# --------------------------------------------------------------------------
# Ana akis
# --------------------------------------------------------------------------
def main() -> None:
    settings = load_settings()
    _render_header(settings)
    params = _render_sidebar(settings)
    if "ticker" in params:
        st.session_state["ticker"] = params["ticker"]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Şüpheli Hisseler", "📊 Hisse Analizi", "📰 Haber Akışı", "🤖 LLM Açıklama"]
    )
    with tab1:
        if params["mode"] == "Gerçek veri (yfinance)":
            _tab_screener(params, settings)
        else:
            st.info("Sentetik modda sepet taraması yapılmaz. Gerçek veri moduna geçin.")
    with tab2:
        _tab_analysis(params, settings)
    with tab3:
        _tab_news(params, settings)
    with tab4:
        _tab_llm(params, settings)

    st.markdown(
        "<div class='footer-note'>MARS · BLMS405 dönem projesi · Anomali skoru ≠ manipülasyon "
        "kanıtı · yatırım tavsiyesi değildir · VeriShield (AGPL-3.0) gömülü</div>",
        unsafe_allow_html=True,
    )


main()
