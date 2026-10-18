import { useEffect, useState } from "react";
import { NewsItem, fetchNews } from "../api";
import { Card, ErrorBox, Spinner } from "../components/ui";

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
      <Card title="RSS Haber Akışı — polarite ve güvenilirlik analizi">
        <p style={{ fontSize: ".78rem", color: "var(--text-muted)", marginBottom: 12 }}>
          BloombergHT + CNN Türk Ekonomi akışları. Polarite: lexicon/BERT hibrit ·
          Güvenilirlik: VeriShield 6-kanal doğrulama motoru. Hisseyle eşleşen haberler
          analizde haber düzeltmesini besler.
        </p>
        {matched && Object.keys(matched).length ? (
          <p style={{ fontSize: ".76rem", marginBottom: 10 }}>
            <b style={{ color: "var(--accent)" }}>Eşleşen hisseler:</b>{" "}
            {Object.entries(matched).map(([t, n]) => `${t} (${n})`).join(", ")}
          </p>
        ) : null}
        {error ? <ErrorBox message={error} /> : null}
        {loading ? <Spinner label="Haberler çekiliyor ve analiz ediliyor…" /> : null}
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
                <span>{n.timestamp ? new Date(n.timestamp).toLocaleString("tr-TR", { dateStyle: "short", timeStyle: "short" }) : ""}</span>
                <span className={`badge ${n.polarity > 0.05 ? "b-pos" : n.polarity < -0.05 ? "b-neg" : "b-neu"}`}>
                  {n.label.toUpperCase()} {n.polarity >= 0 ? "+" : ""}
                  {n.polarity}
                </span>
                <span className="badge b-neu">güvenilirlik {n.credibility.toFixed(2)}</span>
                <span className="badge b-neu">{n.sentiment_backend} / {n.credibility_backend}</span>
              </div>
            </div>
          ))}
      </Card>
    </>
  );
}
