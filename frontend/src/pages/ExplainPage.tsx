import { useState } from "react";
import { ExplainResult, fetchExplain } from "../api";
import { useApp } from "../state";
import { Card, ErrorBox, Spinner, scoreColor, Pill } from "../components/ui";
import { useNavigate } from "react-router-dom";

export default function ExplainPage(): JSX.Element {
  const { ticker, params } = useApp();
  const navigate = useNavigate();
  const [date, setDate] = useState<string>("");
  const [result, setResult] = useState<ExplainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = (d: string | null) => {
    setLoading(true);
    setError(null);
    fetchExplain(ticker, d, params)
      .then((r) => {
        setResult(r);
        setDate(r.date);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  return (
    <>
      <Card title="🤖 Bu sayfa ne yapar?">
        <p className="hint">
          Seçtiğiniz bir günde hissede <b>olağan dışı bir şey olup olmadığını</b> yapay
          zekâ sayesinde sade bir Türkçeyle okursunuz. Yapay zekâ; grafik, hacim, endeks
          karşılaştırması ve haberleri birlikte değerlendirip “bugün neden dikkat çekici?”
          sorusunu <b>sayılarla</b> yanıtlar. Yorum bilgilendirme amaçlıdır, yatırım
          tavsiyesi değildir.
        </p>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 6 }}>
          <span className="mono" style={{ fontWeight: 800, fontSize: "1.1rem" }}>{ticker}</span>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            title="Boş bırakırsanız son işlem günü kullanılır"
          />
          <button className="btn" onClick={() => run(date || null)} disabled={loading}>
            {loading ? "Hazırlanıyor…" : "Açıkla"}
          </button>
          <span className="tip" style={{ color: "var(--text-faint)", fontSize: ".72rem" }}>
            Tarih boş bırakılırsa son işlem günü seçilir.
          </span>
        </div>
        {error ? <ErrorBox message={error} /> : null}
        {loading ? <Spinner label="Veriler hazırlanıyor ve yapay zekâ yorumluyor…" /> : null}
      </Card>

      {result ? (
        <>
          <Card>
            <div className="kpi-grid">
              <div className="kpi">
                <div className="label">Şüphe Skoru</div>
                <div className="value" style={{ color: scoreColor(result.suspicion) }}>
                  {result.suspicion.toFixed(0)}/100
                </div>
                <div className="sub">yüksek = daha dikkat çekici</div>
              </div>
              <div className="kpi">
                <div className="label">Veri Güveni</div>
                <div className="value">{result.data_confidence.toFixed(0)}/100</div>
                <div className="sub">verinin ne kadarı mevcut</div>
              </div>
              <div className="kpi">
                <div className="label">Açıklamayı Kim Yazdı?</div>
                <div className="value" style={{ fontSize: "1rem" }}>
                  {result.llm ? "🤖 Yapay zekâ" : "📖 Kural motoru"}
                </div>
                <div className="sub">{result.date}</div>
              </div>
            </div>
            <p style={{ fontSize: ".9rem", lineHeight: 1.75 }}>{result.summary}</p>
          </Card>

          <Card title="🤖 Yapay Zekâ Yorumu">
            {result.llm ? (
              <div className="llm-box">{result.llm}</div>
            ) : (
              <>
                <p style={{ fontSize: ".86rem", color: "var(--warn)", lineHeight: 1.7 }}>
                  Yapay zekâ yorumu alınamadı{result.llm_error ? `: ${result.llm_error}` : "."}{" "}
                  Aşağıdaki “Gerekçeler” bölümü her durumda çalışır.
                </p>
                <div className="status-line">
                  <button className="btn ghost sm" onClick={() => navigate("/settings")}>
                    ⚙️ Ayarları aç
                  </button>
                  <Pill status="warn">Model seçin / anahtar girin</Pill>
                </div>
              </>
            )}
          </Card>

          <Card title="📊 Gerekçeler (sayılarla, sade anlatım)">
            <p className="hint" style={{ fontSize: ".72rem" }}>
              Yapay zekâ yorumundan bağımsız olarak, skoru neyin yükselttiğini madde
              madde gösterir. Yorum değişse bile bu gerekçeler hep aynı kurallarla üretilir.
            </p>
            {result.reasons.map((r, idx) => (
              <div className="reason" key={idx}>
                <div className="head">
                  <b>{r.signal}</b>
                  <span className="pts">+{r.points}</span>
                </div>
                <p>{r.detail}</p>
              </div>
            ))}
          </Card>
        </>
      ) : null}
    </>
  );
}
