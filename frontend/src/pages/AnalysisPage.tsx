import { useEffect, useRef, useState } from "react";
import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  AreaData,
  UTCTimestamp,
} from "lightweight-charts";
import { Analysis, fetchAnalysis } from "../api";
import { useApp } from "../state";
import { Card, ErrorBox, Spinner, fmtPct, scoreColor } from "../components/ui";
import { ParamsPanel } from "./ParamsPanel";

function toTs(date: string): UTCTimestamp {
  return Math.floor(new Date(date + "T00:00:00Z").getTime() / 1000) as UTCTimestamp;
}

function PriceChart({
  analysis,
  onSelectDay,
}: {
  analysis: Analysis;
  onSelectDay: (date: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const suspicionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || !suspicionRef.current) return;
    const chart: IChartApi = createChart(ref.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8b94a7",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#1a2233" },
        horzLines: { color: "#1a2233" },
      },
      width: ref.current.clientWidth,
      height: 340,
      rightPriceScale: { borderColor: "#1e2739" },
      timeScale: { borderColor: "#1e2739", timeVisible: false },
      crosshair: { mode: 0 },
    });
    const volumeChart: IChartApi = createChart(suspicionRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8b94a7",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#1a2233" },
        horzLines: { color: "#1a2233" },
      },
      width: suspicionRef.current.clientWidth,
      height: 130,
      rightPriceScale: { borderColor: "#1e2739", scaleMargins: { top: 0.1, bottom: 0 } },
      timeScale: { borderColor: "#1e2739", timeVisible: false },
    });

    const candles: CandlestickData[] = [];
    const volumes: HistogramData[] = [];
    const suspicion: AreaData[] = [];
    analysis.dates.forEach((d, i) => {
      const t = toTs(d);
      const o = analysis.open[i];
      const h = analysis.high[i];
      const l = analysis.low[i];
      const c = analysis.close[i];
      if (o != null && h != null && l != null && c != null) {
        candles.push({ time: t, open: o, high: h, low: l, close: c });
      }
      const v = analysis.volume[i];
      if (v != null) {
        volumes.push({
          time: t,
          value: v,
          color: (c ?? 0) >= (o ?? 0) ? "rgba(34,197,94,.55)" : "rgba(239,68,68,.55)",
        });
      }
      const s = analysis.suspicion[i];
      if (s != null) suspicion.push({ time: t, value: s });
    });

    const candleSeries: ISeriesApi<"Candlestick"> = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });
    candleSeries.setData(candles);

    const volumeSeries: ISeriesApi<"Histogram"> = chart.addHistogramSeries({
      priceScaleId: "vol",
      priceFormat: { type: "volume" },
    });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    volumeSeries.setData(volumes);

    const suspicionSeries: ISeriesApi<"Area"> = volumeChart.addAreaSeries({
      lineColor: "#2dd4bf",
      topColor: "rgba(45,212,191,0.25)",
      bottomColor: "rgba(45,212,191,0.02)",
      lineWidth: 2,
      priceScaleId: "right",
    });
    suspicionSeries.setData(suspicion);
    suspicionSeries.createPriceLine({ price: 65, color: "#f87171", lineWidth: 1, lineStyle: 2, title: "yüksek" });
    suspicionSeries.createPriceLine({ price: 35, color: "#fbbf24", lineWidth: 1, lineStyle: 2, title: "orta" });

    // Zaman ekseni senkronu
    const sync = (from: IChartApi, to: IChartApi) => {
      from.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) to.timeScale().setVisibleLogicalRange(range);
      });
    };
    sync(chart, volumeChart);
    sync(volumeChart, chart);

    chart.subscribeClick((param) => {
      if (param.time) {
        const d = new Date((param.time as number) * 1000).toISOString().slice(0, 10);
        onSelectDay(d);
      }
    });

    chart.timeScale().fitContent();
    volumeChart.timeScale().fitContent();

    const onResize = () => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth });
      if (suspicionRef.current) volumeChart.applyOptions({ width: suspicionRef.current.clientWidth });
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
      volumeChart.remove();
    };
  }, [analysis, onSelectDay]);

  return (
    <div className="chart-wrap">
      <div ref={ref} />
      <div className="legend" style={{ marginTop: 6 }}>
        <span><span className="dot" style={{ background: "#22c55e" }} />Yükseliş</span>
        <span><span className="dot" style={{ background: "#ef4444" }} />Düşüş</span>
        <span><span className="dot" style={{ background: "#2dd4bf" }} />Şüphe skoru (alt panel)</span>
        <span style={{ marginLeft: "auto" }}>Grafikte bir güne tıklayarak incele</span>
      </div>
      <div ref={suspicionRef} style={{ marginTop: 4 }} />
    </div>
  );
}

export default function AnalysisPage(): JSX.Element {
  const { ticker, params } = useApp();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dayIdx, setDayIdx] = useState<number | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchAnalysis(ticker, params)
      .then((a) => {
        setAnalysis(a);
        setDayIdx(a.dates.length - 1);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [ticker, params]);

  if (loading) return <Spinner label={`${ticker} analiz ediliyor…`} />;
  if (error) return <ErrorBox message={`${ticker}: ${error}`} />;
  if (!analysis) return <div className="empty">Analiz için hisse seçin.</div>;

  const i = dayIdx ?? analysis.dates.length - 1;
  const last = analysis.dates.length - 1;
  const suspicion = analysis.suspicion[i] ?? 0;
  const prevSuspicion = i > 0 ? analysis.suspicion[i - 1] ?? suspicion : suspicion;
  const logRet = analysis.log_ret[i];
  const ar = analysis.abnormal_ret[i];
  const newsFactor = analysis.news_factor[i] ?? 1;

  // Gunun katkilari (musteri tarafindan hesaplanir)
  const contribs = Object.entries(analysis.contributions)
    .map(([k, arr]) => ({ key: k.replace("contrib_sig_", "").replace(/_/g, " "), pts: arr[i] ?? 0 }))
    .filter((c) => c.pts > 0.05)
    .sort((a, b) => b.pts - a.pts);

  const kpis: { label: string; value: string; sub?: string; color?: string }[] = [
    {
      label: "Şüphe Skoru",
      value: `${suspicion.toFixed(0)}/100`,
      sub: `${(suspicion - prevSuspicion >= 0 ? "+" : "") + (suspicion - prevSuspicion).toFixed(1)} önceki güne göre`,
      color: scoreColor(suspicion),
    },
    { label: "Veri Güveni", value: `${(analysis.data_confidence[i] ?? 0).toFixed(0)}/100` },
    {
      label: "Günlük Getiri",
      value: fmtPct(logRet),
      color: (logRet ?? 0) >= 0 ? "var(--up)" : "var(--down)",
    },
    {
      label: "Anormal Getiri (AR)",
      value: fmtPct(ar),
      sub: "market modeli artığı",
      color: (ar ?? 0) >= 0 ? "var(--up)" : "var(--down)",
    },
    { label: "Haber Faktörü", value: newsFactor.toFixed(2), sub: newsFactor < 0.999 ? "açıklayıcı haber" : "etkisiz" },
    {
      label: "Tahta Bağlamı",
      value: analysis.liquidity_tier,
      sub: `önsel ${analysis.tahta_prior.toFixed(2)} · medyan hacim ${(analysis.adv_dollar_median / 1e9).toFixed(1)}B TL`,
      color: "var(--info)",
    },
  ];

  return (
    <>
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, flexWrap: "wrap", gap: 8 }}>
          <div>
            <span className="mono" style={{ fontSize: "1.3rem", fontWeight: 800 }}>{analysis.ticker}</span>
            <span style={{ color: "var(--text-faint)", fontSize: ".76rem", marginLeft: 10 }}>
              veri: {analysis.data_source} · {analysis.dates.length} gün · ML {analysis.ml_available ? "aktif" : "yok"}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button className="btn ghost" onClick={() => setDayIdx(Math.max(0, i - 1))}>←</button>
            <input
              type="date"
              value={analysis.dates[i]}
              min={analysis.dates[0]}
              max={analysis.dates[last]}
              onChange={(e) => {
                const idx = analysis.dates.indexOf(e.target.value);
                if (idx >= 0) setDayIdx(idx);
              }}
            />
            <button className="btn ghost" onClick={() => setDayIdx(Math.min(last, i + 1))}>→</button>
          </div>
        </div>

        <div className="kpi-grid">
          {kpis.map((k) => (
            <div className="kpi" key={k.label}>
              <div className="label">{k.label}</div>
              <div className="value" style={{ color: k.color }}>{k.value}</div>
              {k.sub ? <div className="sub">{k.sub}</div> : null}
            </div>
          ))}
        </div>

        <PriceChart analysis={analysis} onSelectDay={(d) => {
          const idx = analysis.dates.indexOf(d);
          if (idx >= 0) setDayIdx(idx);
        }} />
      </Card>

      <div className="grid-2" style={{ marginTop: 14 }}>
        <Card title={`Sinyal Katkıları — ${analysis.dates[i]}`}>
          {contribs.length ? (
            contribs.map((c) => (
              <div className="sig-row" key={c.key}>
                <span className="name">{c.key}</span>
                <div className="bar">
                  <div className="fill" style={{ width: `${Math.min(100, (c.pts / 25) * 100)}%` }} />
                </div>
                <span className="pts">+{c.pts.toFixed(1)}</span>
              </div>
            ))
          ) : (
            <div className="empty">Bu gün belirgin katkı yok.</div>
          )}
        </Card>

        <Card title="Gerekçeler (değer bazlı)">
          {i === last && analysis.reasons.length ? (
            analysis.reasons.map((r, idx) => (
              <div className="reason" key={idx}>
                <div className="head">
                  <b>{r.signal}</b>
                  <span className="pts">+{r.points}</span>
                </div>
                <p>{r.detail}</p>
              </div>
            ))
          ) : (
            <p style={{ fontSize: ".8rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
              {analysis.summary}
              <br /><br />
              <span style={{ color: "var(--text-faint)" }}>
                Gün-özel gerekçeler için <b>LLM Açıklama</b> sayfasını kullanın.
              </span>
            </p>
          )}
        </Card>
      </div>

      {analysis.news.length ? (
        <Card title={`Eşleşen Haberler (${analysis.news.length})`} className="" >
          {analysis.news.map((n, idx) => (
            <div className="news-card" key={idx}>
              {n.url ? <a href={n.url} target="_blank" rel="noreferrer" className="t">{n.headline}</a> : <span className="t">{n.headline}</span>}
              <div className="m">
                <span>{n.source}</span>
                <span className={`badge ${n.polarity > 0.05 ? "b-pos" : n.polarity < -0.05 ? "b-neg" : "b-neu"}`}>
                  {n.label} {n.polarity >= 0 ? "+" : ""}{n.polarity}
                </span>
                <span className="badge b-neu">güvenilirlik {n.credibility.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </Card>
      ) : null}

      <div style={{ marginTop: 14 }}>
        <ParamsPanel />
      </div>
    </>
  );
}
