import { AnalyzeParams } from "../api";
import { useApp } from "../state";
import { Card } from "../components/ui";

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  fmt,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  return (
    <div className="param">
      <label>
        {label}
        <span className="val">{fmt ? fmt(value) : value}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

const WEIGHT_LABELS: Record<string, string> = {
  volume_z: "Hacim",
  index_divergence: "Endeks kopuşu",
  volatility: "Oynaklık",
  mahalanobis: "MCD uzaklığı",
  isolation_forest: "iForest",
  liquidity: "Likidite",
  ml: "ML",
  close_location: "Kapanış konumu",
  pump_pattern: "Çok günlük desen",
};

export function ParamsPanel(): JSX.Element {
  const { params, setParams } = useApp();

  const upd = (patch: Partial<AnalyzeParams>) => setParams({ ...params, ...patch });
  const updWeight = (key: string, v: number) => {
    const weights = { ...(params.weights ?? {}) };
    weights[key] = v;
    upd({ weights });
  };

  const weights = params.weights;

  return (
    <Card title="⚙️ Parametreler — analiz motoru">
      <div className="params-grid">
        <Slider label="Geçmiş uzunluğu (gün)" value={params.lookback} min={180} max={750} step={50} onChange={(v) => upd({ lookback: v })} />
        <Slider label="Robust pencere (gün)" value={params.rolling} min={20} max={120} step={5} onChange={(v) => upd({ rolling: v })} />
        <Slider label="Market model penceresi (gün)" value={params.est_window} min={60} max={250} step={10} onChange={(v) => upd({ est_window: v })} />
        <Slider label="iForest ağaç sayısı" value={params.iforest_trees} min={50} max={500} step={50} onChange={(v) => upd({ iforest_trees: v })} />
        <Slider label="iForest contamination" value={params.contamination} min={0.005} max={0.1} step={0.005} onChange={(v) => upd({ contamination: v })} fmt={(v) => v.toFixed(3)} />
        <Slider label="Haber ceza gücü" value={params.news_penalty} min={0} max={0.6} step={0.05} onChange={(v) => upd({ news_penalty: v })} fmt={(v) => v.toFixed(2)} />
        <Slider label="Tahta (likidite) modülasyonu" value={params.tahta_modulation} min={0} max={1} step={0.05} onChange={(v) => upd({ tahta_modulation: v })} fmt={(v) => v.toFixed(2)} />
        <div className="param">
          <label>
            RSS haber akışları
            <span className="val">{params.rss ? "açık" : "kapalı"}</span>
          </label>
          <button className="btn ghost" style={{ marginTop: 4 }} onClick={() => upd({ rss: !params.rss })}>
            {params.rss ? "Kapat" : "Aç"}
          </button>
        </div>
      </div>

      <details style={{ marginTop: 14 }}>
        <summary style={{ cursor: "pointer", color: "var(--text-muted)", fontSize: ".8rem", fontWeight: 600 }}>
          Füzyon ağırlıkları (0 = varsayılan)
        </summary>
        <div className="params-grid" style={{ marginTop: 10 }}>
          {Object.entries(WEIGHT_LABELS).map(([key, label]) => (
            <Slider
              key={key}
              label={label}
              value={weights ? weights[key] ?? 0 : 0}
              min={0}
              max={0.5}
              step={0.01}
              onChange={(v) => updWeight(key, v)}
              fmt={(v) => (weights ? v.toFixed(2) : "oto")}
            />
          ))}
        </div>
        {weights ? (
          <button className="btn ghost" style={{ marginTop: 10 }} onClick={() => upd({ weights: null })}>
            Varsayılan ağırlıklara dön
          </button>
        ) : null}
      </details>
    </Card>
  );
}

export default ParamsPanel;
