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
import { Analysis, Reason, fetchAnalysis, fetchExplain } from "../api";
import { useApp } from "../state";
import { useTheme } from "../useTheme";
import { Card, ErrorBox, Spinner, fmtPct, scoreColor } from "../components/ui";
import { ParamsPanel } from "./ParamsPanel";

function toTs(date: string): UTCTimestamp {
  return Math.floor(new Date(date + "T00:00:00Z").getTime() / 1000) as UTCTimestamp;
}

interface ChartPalette {
  text: string;
  grid: string;
  border: string;
}

function PriceChart({
  analysis,
  onSelectDay,
  palette,
}: {
  analysis: Analysis;
  onSelectDay: (date: string) => void;
  palette: ChartPalette;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const suspicionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || !suspicionRef.current) return;
    const base = {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: palette.text,
        fontSize: 11,
      },
      grid: {
        vertLines: { color: palette.grid },
        horzLines: { color: palette.grid },
      },
    };
    const chart: IChartApi = createChart(ref.current, {
      ...base,
      width: ref.current.clientWidth,
      height: 340,
      rightPriceScale: { borderColor: palette.border },
      timeScale: { borderColor: palette.border, timeVisible: false },
      crosshair: { mode: 0 },
    });
    const volumeChart: IChartApi = createChart(suspicionRef.current, {
      ...base,
      width: suspicionRef.current.clientWidth,
      height: 130,
      rightPriceScale: { borderColor: palette.border, scaleMargins: { top: 0.1, bottom: 0 } },
      timeScale: { borderColor: palette.border, timeVisible: false },
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
          color: (c ?? 0) >= (o ?? 0) ? "rgba(63,182,139,.55)" : "rgba(239,83,80,.55)",
        });
      }
      const s = analysis.suspicion[i];
      if (s != null) suspicion.push({ time: t, value: s });
    });

    const candleSeries: ISeriesApi<"Candlestick"> = chart.addCandlestickSeries({
      upColor: "#3fb68b",
      downColor: "#ef5350",
      borderUpColor: "#3fb68b",
      borderDownColor: "#ef5350",
      wickUpColor: "#3fb68b",
      wickDownColor: "#ef5350",
    });
    candleSeries.setData(candles);

    const volumeSeries: ISeriesApi<"Histogram"> = chart.addHistogramSeries({
      priceScaleId: "vol",
      priceFormat: { type: "volume" },
    });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    volumeSeries.setData(volumes);

    const suspicionSeries: ISeriesApi<"Area"> = volumeChart.addAreaSeries({
      lineColor: "#f7941d",
      topColor: "rgba(247,148,29,0.22)",
      bottomColor: "rgba(247,148,29,0.02)",
      lineWidth: 2,
      priceScaleId: "right",
    });
    suspicionSeries.setData(suspicion);
    suspicionSeries.createPriceLine({ price: 65, color: "#ef5350", lineWidth: 1, lineStyle: 2, title: "yüksek" });
    suspicionSeries.createPriceLine({ price: 35, color: "#ffc53d", lineWidth: 1, lineStyle: 2, title: "orta" });

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
  }, [analysis, onSelectDay, palette]);

  return (
    <div className="chart-wrap">
      <div ref={ref} />
      <div className="legend" style={{ marginTop: 6 }}>
        <span><span className="dot" style={{ background: "#3fb68b" }} />Yükseliş</span>
        <span><span className="dot" style={{ background: "#ef5350" }} />Düşüş</span>
        <span><span className="dot" style={{ background: "#f7941d" }} />Şüphe skoru (alt panel)</span>
        <span style={{ marginLeft: "auto" }}>Grafikte bir güne tıklayarak inceleyin</span>
      </div>
      <div ref={suspicionRef} style={{ marginTop: 4 }} />
    </div>
  );
}

const PALETTES = {
  dark: { text: "#9a9a9a", grid: "#1d1d1d", border: "#262626" },
  light: { text: "#5c5c56", grid: "#ecece5", border: "#dfdfd8" },
};

export default function AnalysisPage(): JSX.Element {
  const { ticker, params } = useApp();
  const { theme } = useTheme();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dayIdx, setDayIdx] = useState<number | null>(null);
  const [dayReasons, setDayReasons] = useState<Reason[]>([]);
  const [expandedSig, setExpandedSig] = useState<string | null>(null);

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

  const selIdx = dayIdx ?? (analysis ? analysis.dates.length - 1 : 0);
  const isLastDay = analysis ? selIdx === analysis.dates.length - 1 : true;

  useEffect(() => {
    if (!analysis) return;
    if (isLastDay) {
      setDayReasons(analysis.reasons);
      return;
    }
    let alive = true;
    const date = analysis.dates[selIdx];
    fetchExplain(analysis.ticker, date, params, false)
      .then((r) => {
        if (alive) setDayReasons(r.reasons);
      })
      .catch(() => {
        if (alive) setDayReasons([]);
      });
    return () => {
      alive = false;
    };
  }, [analysis, selIdx, isLastDay, params]);

  if (loading) return <Spinner label={`${ticker} inceleniyor…`} />;
  if (error) return <ErrorBox message={`${ticker}: ${error}`} />;
  if (!analysis) return <div className="empty">Analiz için bir hisse seçin.</div>;

  const i = selIdx;
  const last = analysis.dates.length - 1;
  const suspicion = analysis.suspicion[i] ?? 0;
  const prevSuspicion = i > 0 ? analysis.suspicion[i - 1] ?? suspicion : suspicion;
  const logRet = analysis.log_ret[i];
  const ar = analysis.abnormal_ret[i];
  const newsFactor = analysis.news_factor[i] ?? 1;

  const catalog: Record<string, { short: string; title: string; what: string }> = {};
  analysis.signals_catalog.forEach((s) => {
    catalog[s.key] = s;
  });

  const contribs = Object.entries(analysis.contributions)
    .map(([k, arr]) => {
      const key = k.replace("contrib_sig_", "");
      return { key, label: catalog[key]?.short ?? key.replace(/_/g, " "), title: catalog[key]?.title ?? "", pts: arr[i] ?? 0 };
    })
    .filter((c) => c.pts > 0.05)
    .sort((a, b) => b.pts - a.pts);

  const contextReasons = dayReasons.filter((r) => r.level === "bağlam");
  const scoredReasons = dayReasons
    .filter((r) => r.level !== "bağlam")
    .sort((a, b) => b.points - a.points)

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
      label: "Endekse Göre Fazlası",
      value: fmtPct(ar),
      sub: "beklenenden sapma",
      color: (ar ?? 0) >= 0 ? "var(--up)" : "var(--down)",
    },
    { label: "Haber Etkisi", value: newsFactor.toFixed(2), sub: newsFactor < 0.999 ? "haber skoru düşürdü" : "etkisiz" },
    {
      label: "Tahta Bağlamı",
      value: analysis.liquidity_tier,
      sub: `likidite önseli ${analysis.tahta_prior.toFixed(2)}`,
      color: "var(--info)",
    },
  ];

  const matchedNews = analysis.news.filter((n) => !n.general);

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

        <PriceChart
          analysis={analysis}
          palette={PALETTES[theme]}
          onSelectDay={(d) => {
            const idx = analysis.dates.indexOf(d);
            if (idx >= 0) setDayIdx(idx);
          }}
        />
      </Card>

      <div className="grid-2" style={{ marginTop: 14 }}>
        <Card title={`Skoru En Çok Artıranlar — ${analysis.dates[i]}`}>
          <p className="hint" style={{ fontSize: ".74rem" }}>
            Şüphe skoru 100 üzerinden hesaplanan bir puandır. Aşağıdaki her çubuk, o sinyalin
            skora <b>kaç puan eklediğini</b> gösterir; çubuk ne kadar uzunsa o gün skoru o kadar
            çok yükseltmiştir. Örneğin <b>+17.3</b>, o sinyalin skora 17.3 puan kattığı anlamına gelir.
            Bir maddeye tıklayarak <b>ne anlama geldiğini</b> görebilirsiniz.
          </p>
          {contribs.length ? (
            contribs.map((c) => {
              const cat = catalog[c.key];
              const reason = dayReasons.find((r) => r.key === c.key && r.level !== "bağlam");
              const detail = reason?.detail;
              const what = reason?.what ?? cat?.what;
              const open = expandedSig === c.key;
              const toggle = () => setExpandedSig(open ? null : c.key);
              return (
                <div key={c.key}>
                  <div
                    className={`sig-row${open ? " open" : ""}`}
                    role="button"
                    tabIndex={0}
                    aria-expanded={open}
                    title={c.title}
                    onClick={toggle}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggle();
                      }
                    }}
                  >
                    <span className="name">{c.label}</span>
                    <div className="bar">
                      <div className="fill" style={{ width: `${Math.min(100, (c.pts / 25) * 100)}%` }} />
                    </div>
                    <span className="pts">+{c.pts.toFixed(1)}</span>
                    <span className="chev" aria-hidden="true">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                        strokeLinecap="round" strokeLinejoin="round">
                        <path d="m6 9 6 6 6-6" />
                      </svg>
                    </span>
                  </div>
                  {open ? (
                    <div className="sig-detail">
                      {detail ? <p>{detail}</p> : null}
                      {what ? (
                        <p className="what">
                          <b>Bu ne demek?</b> {what}
                        </p>
                      ) : null}
                      {!detail && !what ? <p>{c.title}</p> : null}
                    </div>
                  ) : null}
                </div>
              );
            })
          ) : (
            <div className="empty">Bu gün belirgin bir katkı yok.</div>
          )}
        </Card>

        <Card title="Neden Bu Skor? (kısa özet)">
          {i === last && analysis.reasons.length ? (
            analysis.reasons.map((r, idx) => (
              <div className="reason" key={idx}>
                <div className="head">
                  <b>{r.signal}</b>
                  <span className="pts">{r.points > 0 ? `+${r.points}` : "—"}</span>
                </div>
                <p>{r.detail}</p>
              </div>
            ))
          ) : (
            <p style={{ fontSize: ".8rem", color: "var(--text-muted)", lineHeight: 1.7 }}>
              {analysis.summary}
              <br /><br />
              <span style={{ color: "var(--text-faint)" }}>
                Seçtiğiniz güne özel açıklama için aşağıdaki <b>“Bu sinyaller ne anlama
                geliyor?”</b> listesini ya da <b>LLM Açıklama</b> sayfasını kullanın.
              </span>
            </p>
          )}
        </Card>
      </div>

      {scoredReasons.length || contextReasons.length ? (
        <Card
          title={`Bu Sinyaller Ne Anlama Geliyor? — ${analysis.dates[i]}`}
        >
          <p className="hint" style={{ fontSize: ".74rem" }}>
            Her sinyal için üç şey yazıyor: <b>neyi ölçtüğü</b>, o gün <b>ne olduğu</b>
            (gerçek sayılarla) ve bunun skora <b>kaç puan</b> eklediği. İstatistik bilgisi
            gerektirmez; terimler parantez içinde açıklanır.
          </p>
          {scoredReasons.map((r) => (
            <div className="reason" key={r.key}>
              <div className="head">
                <b>{r.title}</b>
                <span className="pts">{r.points > 0 ? `+${r.points} puan` : "puan katmadı"}</span>
              </div>
              <p>{r.detail}</p>
              {r.what ? (
                <p className="what">
                  <b>Bu ne demek?</b> {r.what}
                </p>
              ) : null}
            </div>
          ))}
          {contextReasons.map((r) => (
            <div className="context-note" key={r.key}>
              <div className="head" style={{ display: "flex", justifyContent: "space-between", gap: 10, marginBottom: 3 }}>
                <b>{r.title}</b>
                <span className="tag">puan üretmez, ayar</span>
              </div>
              <p>{r.detail}</p>
              {r.what ? (
                <p className="what" style={{ marginTop: 6 }}>
                  <b>Bu ne demek?</b> {r.what}
                </p>
              ) : null}
            </div>
          ))}
        </Card>
      ) : (
        <Card title="Bu Sinyaller Ne Anlama Geliyor?">
          <p className="hint" style={{ fontSize: ".72rem" }}>
            Bu gün için ayrıntılı gerekçe üretilemedi; özet yukarıdaki kartta yer alıyor.
          </p>
        </Card>
      )}

      {analysis.news.length ? (
        <Card
          title={
            matchedNews.length
              ? `Bu Hisseyle İlgili Haberler (${matchedNews.length})`
              : "Bu hisseyle eşleşen haber bulunamadı — güncel piyasa haberleri"
          }
        >
          {matchedNews.length ? null : (
            <p className="hint" style={{ fontSize: ".72rem" }}>
              Hisseye özel bir haber yakalanamadı; bilgi amaçlı genel piyasa akışı gösteriliyor.
            </p>
          )}
          {analysis.news.map((n, idx) => (
            <div className="news-card" key={idx}>
              {n.url ? <a href={n.url} target="_blank" rel="noreferrer" className="t">{n.headline}</a> : <span className="t">{n.headline}</span>}
              <div className="m">
                <span>{n.source}</span>
                {n.timestamp ? <span>{new Date(n.timestamp).toLocaleDateString("tr-TR")}</span> : null}
                {n.general ? <span className="badge b-cat">genel piyasa</span> : null}
                <span className={`badge ${n.polarity > 0.05 ? "b-pos" : n.polarity < -0.05 ? "b-neg" : "b-neu"}`}>
                  {n.label} {n.polarity >= 0 ? "+" : ""}{n.polarity}
                </span>
                <span className="badge b-neu">güvenilirlik {n.credibility.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </Card>
      ) : (
        <Card title="Haberler">
          <p className="hint" style={{ fontSize: ".78rem", marginBottom: 0 }}>
            Şu an gösterilecek haber yok. Ayarlar sayfasından RSS akışlarını kontrol edin.
          </p>
        </Card>
      )}

      <div style={{ marginTop: 14 }}>
        <ParamsPanel />
      </div>
    </>
  );
}
