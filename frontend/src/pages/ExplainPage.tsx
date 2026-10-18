import { useState } from "react";
import { ExplainResult, fetchExplain } from "../api";
import { useApp } from "../state";
import { Card, ErrorBox, Spinner, scoreColor } from "../components/ui";

export default function ExplainPage(): JSX.Element {
  const { ticker, params } = useApp();
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
      <Card title="🤖 LLM ile Açıkla — günün yorumu">
        <p style={{ fontSize: ".8rem", color: "var(--text-muted)", marginBottom: 14, lineHeight: 1.6 }}>
          Seçilen günün sinyal verileri ve haber bilgileri yerel LLM'e (Ollama) gönderilir;
          model, skoru <b>sayılarla gerekçelendirerek</b> Türkçe açıklar. LLM kapalıysa
          kural-tabanlı açıklama gösterilir. Yorum bilgilendirme amaçlıdır — yatırım
          tavsiyesi değildir.
        </p>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
          <span className="mono" style={{ fontWeight: 800, fontSize: "1.1rem" }}>{ticker}</span>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
          <button className="btn" onClick={() => run(date || null)} disabled={loading}>
            {loading ? "Analiz ediliyor…" : "Açıkla"}
          </button>
        </div>
        {error ? <ErrorBox message={error} /> : null}
        {loading ? <Spinner label="Sinyaller hesaplanıyor ve LLM yorumluyor…" /> : null}
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
              </div>
              <div className="kpi">
                <div className="label">Veri Güveni</div>
                <div className="value">{result.data_confidence.toFixed(0)}/100</div>
              </div>
              <div className="kpi">
                <div className="label">Açıklama Kaynağı</div>
                <div className="value" style={{ fontSize: "1rem" }}>
                  {result.llm_backend === "ollama" ? "🤖 Yerel LLM" : "📖 Kural-tabanlı"}
                </div>
              </div>
            </div>
            <p style={{ fontSize: ".88rem", lineHeight: 1.7 }}>{result.summary}</p>
          </Card>

          {result.llm ? (
            <Card title="LLM Yorumu">
              <div className="llm-box">{result.llm}</div>
            </Card>
          ) : (
            <Card title="LLM Yorumu">
              <p style={{ fontSize: ".82rem", color: "var(--warn)" }}>
                LLM kapalı veya erişilemez. Ollama'yı başlatın:
                <code style={{ display: "block", marginTop: 8, fontFamily: "var(--mono)" }}>
                  ollama serve && ollama pull qwen2.5:7b-instruct
                </code>
                Aşağıdaki kural-tabanlı gerekçeler her zaman görünürdür.
              </p>
            </Card>
          )}

          <Card title="Değer Bazlı Gerekçeler">
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
