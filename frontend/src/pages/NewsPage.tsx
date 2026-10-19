import { useEffect, useState } from "react";
import { NewsItem, fetchNews } from "../api";
import { Card, ErrorBox, Spinner, Pill } from "../components/ui";

const CATEGORY_LABEL: Record<string, string> = {
  hisse: "hisse",
  piyasa: "piyasa",
  sirket: "şirket",
  diger: "diğer",
};

export default function NewsPage(): JSX.Element {
  const [items, setItems] = useState<NewsItem[]>([]);
  const [matched, setMatched] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchNews(60)
      .then((r) => {
        setItems(r.items);
        setMatched(r.matched);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <>
      <Card title="Borsa ve Piyasa Haberleri">
        <p className="hint">
          Yalnızca <b>borsa/ekonomi ile ilgili</b> haberler listelenir; magazin, spor
          ve genel siyaset haberleri elenir. Haberler BloombergHT ve CNN Türk ekonomi
          akışlarından alınır. “Duygu” haberin olumlu/olumsuz tonunu, “güvenilirlik”
          ise kaynağın ne kadar sağlam olduğunu gösterir.
        </p>
        {Object.keys(matched).length ? (
          <div className="status-line">
            <span style={{ color: "var(--text-muted)", fontSize: ".76rem" }}>Hisselerle eşleşenler:</span>
            {Object.entries(matched).map(([t, n]) => (
              <Pill key={t}>{t.replace(".IS", "")} · {n}</Pill>
            ))}
          </div>
        ) : null}
        {error ? <ErrorBox message={error} /> : null}
        {loading ? <Spinner label="Haberler çekiliyor ve analiz ediliyor…" /> : null}
        {!loading && items.length === 0 ? (
          <div className="empty">Şu an borsa ile ilgili haber bulunamadı.</div>
        ) : null}
        {!loading &&
          items.map((n, idx) => (
            <div className="news-card" key={idx}>
              {n.url ? (
                <a href={n.url} target="_blank" rel="noreferrer" className="t">
                  {n.headline}
                </a>
              ) : (
                <span className="t">{n.headline}</span>
              )}
              <div className="m">
                <span>{n.source}</span>
                <span>
                  {n.timestamp
                    ? new Date(n.timestamp).toLocaleString("tr-TR", { dateStyle: "short", timeStyle: "short" })
                    : ""}
                </span>
                {n.category ? (
                  <span className="badge b-cat">{CATEGORY_LABEL[n.category] ?? n.category}</span>
                ) : null}
                {n.ticker ? <span className="badge b-tahta">{n.ticker.replace(".IS", "")}</span> : null}
                <span className={`badge ${n.polarity > 0.05 ? "b-pos" : n.polarity < -0.05 ? "b-neg" : "b-neu"}`}>
                  {n.label} {n.polarity >= 0 ? "+" : ""}
                  {n.polarity}
                </span>
                <span className="badge b-neu">güvenilirlik {n.credibility.toFixed(2)}</span>
              </div>
            </div>
          ))}
      </Card>
    </>
  );
}
