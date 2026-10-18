import { useEffect, useMemo, useState } from "react";
import { ScreenerRow, fetchMeta, fetchScreener } from "../api";
import { useApp } from "../state";
import { Card, ErrorBox, ScoreBar, Spinner, fmtPct } from "../components/ui";
import { useNavigate } from "react-router-dom";

type SortKey = "suspicion" | "log_ret" | "abnormal_ret" | "ticker";

export default function ScreenerPage(): JSX.Element {
  const { params, setTicker } = useApp();
  const navigate = useNavigate();
  const [rows, setRows] = useState<ScreenerRow[]>([]);
  const [errors, setErrors] = useState<{ ticker: string; error: string }[]>([]);
  const [basket, setBasket] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("suspicion");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    fetchMeta()
      .then((m) => setBasket(m.basket))
      .catch(() => setBasket(["THYAO.IS"]));
  }, []);

  const run = (tickers: string[]) => {
    setLoading(true);
    setError(null);
    fetchScreener(tickers, params)
      .then((r) => {
        setRows(r.rows);
        setErrors(r.errors);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (basket.length) run(basket);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [basket.length, params]);

  const sorted = useMemo(() => {
    const arr = [...rows];
    arr.sort((a, b) => {
      const av = (a[sortKey] as number | string | null) ?? (sortKey === "ticker" ? "" : -999);
      const bv = (b[sortKey] as number | string | null) ?? (sortKey === "ticker" ? "" : -999);
      const cmp = typeof av === "string" ? av.localeCompare(String(bv)) : (av as number) - (bv as number);
      return sortAsc ? cmp : -cmp;
    });
    return arr;
  }, [rows, sortKey, sortAsc]);

  const exportCsv = () => {
    const header = "ticker,suspicion,data_confidence,log_ret,abnormal_ret,top_signal,liquidity_tier,tahta_prior,news_count,date";
    const lines = sorted.map((r) =>
      [r.ticker, r.suspicion, r.data_confidence, r.log_ret, r.abnormal_ret, r.top_signal, r.liquidity_tier, r.tahta_prior, r.news_count, r.date].join(",")
    );
    const blob = new Blob([header + "\n" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mars_screener_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const th = (key: SortKey, label: string) => (
    <th
      onClick={() => {
        if (sortKey === key) setSortAsc(!sortAsc);
        else {
          setSortKey(key);
          setSortAsc(false);
        }
      }}
    >
      {label} {sortKey === key ? (sortAsc ? "↑" : "↓") : ""}
    </th>
  );

  return (
    <>
      <Card title="🔍 Şüpheli Hisseler — Sepet Taraması">
        <p className="hint" style={{ fontSize: ".78rem" }}>
          İzleme listesindeki tüm hisseler, son işlem günü için tek tek puanlanır ve
          en dikkat çekiciler üstte olacak şekilde sıralanır. <b>İlk tarama</b> verileri
          indirdiği için birkaç dakika sürebilir; sonraki taramalar önbellekten hızlı gelir.
          Bir hisseye tıklayarak detaylı incelemeye geçebilirsiniz.
        </p>
        <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
          <button className="btn" onClick={() => run(basket)} disabled={loading}>
            {loading ? "Taranıyor…" : "Yeniden Tara"}
          </button>
          <button className="btn ghost" onClick={exportCsv} disabled={!sorted.length}>
            CSV indir
          </button>
          {errors.length ? (
            <span style={{ color: "var(--warn)", fontSize: ".76rem", alignSelf: "center" }}>
              {errors.length} hisse alınamadı
            </span>
          ) : null}
        </div>

        {error ? <ErrorBox message={error} /> : null}
        {loading ? <Spinner label="Sepet taranıyor — veriler indiriliyor / analiz ediliyor…" /> : null}

        {!loading && sorted.length ? (
          <table className="table">
            <thead>
              <tr>
                {th("ticker", "Hisse")}
                {th("suspicion", "Şüphe Skoru")}
                {th("log_ret", "Günlük Getiri")}
                {th("abnormal_ret", "Anormal Getiri (AR)")}
                <th>Tahta Bağlamı</th>
                <th>En Etkili Sinyal</th>
                <th>Haber</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr
                  key={r.ticker}
                  style={{ cursor: "pointer" }}
                  onClick={() => {
                    setTicker(r.ticker);
                    navigate("/analysis");
                  }}
                >
                  <td className="mono" style={{ fontWeight: 700 }}>{r.ticker}</td>
                  <td><ScoreBar value={r.suspicion} /></td>
                  <td className="mono" style={{ color: (r.log_ret ?? 0) >= 0 ? "var(--up)" : "var(--down)" }}>
                    {fmtPct(r.log_ret)}
                  </td>
                  <td className="mono" style={{ color: (r.abnormal_ret ?? 0) >= 0 ? "var(--up)" : "var(--down)" }}>
                    {fmtPct(r.abnormal_ret)}
                  </td>
                  <td>
                    <span className="badge b-tahta">{r.liquidity_tier}</span>{" "}
                    <span className="mono" style={{ color: "var(--text-faint)", fontSize: ".72rem" }}>
                      önsel {r.tahta_prior.toFixed(2)}
                    </span>
                  </td>
                  <td>{r.top_signal} <span className="mono" style={{ color: "var(--accent)" }}>+{r.top_points}</span></td>
                  <td className="mono">{r.news_count || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
        {!loading && !sorted.length && !error ? (
          <div className="empty">Tarama başlatmak için “Yeniden Tara” düğmesine basın.</div>
        ) : null}
      </Card>
    </>
  );
}
