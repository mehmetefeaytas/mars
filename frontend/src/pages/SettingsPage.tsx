import { useEffect, useState } from "react";
import {
  EffectiveSettings,
  ProviderPreset,
  acceptTerms,
  fetchModels,
  fetchSettings,
  saveSettings,
  testLlm,
} from "../api";
import { Card, ErrorBox, Spinner, Pill } from "../components/ui";

type Draft = Omit<EffectiveSettings, "llm_api_key" | "llm_api_key_set"> & { llm_api_key: string };

function toDraft(s: EffectiveSettings): Draft {
  const { llm_api_key: _k, llm_api_key_set: _s, ...rest } = s;
  return { ...rest, llm_api_key: "" };
}

export default function SettingsPage(): JSX.Element {
  const [draft, setDraft] = useState<Draft | null>(null);
  const [providers, setProviders] = useState<ProviderPreset[]>([]);
  const [keySet, setKeySet] = useState(false);
  const [keyMask, setKeyMask] = useState("");
  const [storePath, setStorePath] = useState("");
  const [llmAvailable, setLlmAvailable] = useState(false);
  const [terms, setTerms] = useState<{ supported: boolean; accepted: boolean | null; current_version?: number }>({ supported: false, accepted: true });
  const [models, setModels] = useState<{ id: string }[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    fetchSettings()
      .then((s) => {
        setDraft(toDraft(s.current));
        setProviders(s.providers);
        setKeySet(s.current.llm_api_key_set);
        setKeyMask(s.current.llm_api_key);
        setStorePath(s.store_path);
        setLlmAvailable(s.llm_available);
        setTerms({
          supported: Boolean(s.evren_terms?.supported),
          accepted: s.evren_terms?.accepted ?? null,
          current_version: s.evren_terms?.current_version,
        });
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const upd = <K extends keyof Draft>(k: K, v: Draft[K]) => setDraft((d) => (d ? { ...d, [k]: v } : d));

  const pickProvider = (id: string) => {
    const preset = providers.find((p) => p.id === id);
    setDraft((d) => (d ? { ...d, llm_provider: id, llm_base_url: preset?.default_url || d.llm_base_url } : d));
  };

  const save = () => {
    if (!draft) return;
    setSaving(true);
    setError(null);
    setNotice(null);
    const patch: Record<string, unknown> = { ...draft };
    if (!draft.llm_api_key) delete patch.llm_api_key;
    saveSettings(patch)
      .then((r) => {
        setKeySet(r.current.llm_api_key_set);
        setKeyMask(r.current.llm_api_key);
        setLlmAvailable(r.llm_available);
        setDraft((d) => (d ? { ...d, llm_api_key: "" } : d));
        setNotice("Ayarlar kaydedildi.");
        return fetchSettings();
      })
      .then((s) => setTerms({
        supported: Boolean(s.evren_terms?.supported),
        accepted: s.evren_terms?.accepted ?? null,
        current_version: s.evren_terms?.current_version,
      }))
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setSaving(false));
  };

  const runTest = () => {
    setTesting(true);
    setError(null);
    setNotice(null);
    testLlm()
      .then((r) => {
        if (r.ok) setNotice(`✓ Bağlantı başarılı: ${r.message}`);
        else if (r.terms_required) setNotice(`⚠ Kullanım şartları onaylanmalı (sürüm ${r.version ?? "?"}).`);
        else setError(`LLM testi başarısız: ${r.message}`);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setTesting(false));
  };

  const listModels = () => {
    setError(null);
    fetchModels()
      .then((r) => {
        setModels(r.models);
        setNotice(`${r.models.length} model listelendi.`);
      })
      .catch((e) => setError(String(e.message ?? e)));
  };

  const doAcceptTerms = () => {
    setError(null);
    acceptTerms(terms.current_version ?? 1)
      .then((r) => {
        setTerms((t) => ({ ...t, accepted: true }));
        setNotice(`Kullanım şartları onaylandı (sürüm ${r.accepted_version}).`);
      })
      .catch((e) => setError(String(e.message ?? e)));
  };

  if (loading || !draft) return <Spinner label="Ayarlar yükleniyor…" />;

  const remote = ["evren", "openai", "custom"].includes(draft.llm_provider);

  return (
    <>
      {error ? <ErrorBox message={error} /> : null}
      {notice ? (
        <div className="card" style={{ borderColor: "rgba(45,212,191,.4)" }}>
          <p style={{ color: "var(--accent)", fontSize: ".86rem" }}>{notice}</p>
        </div>
      ) : null}

      <Card title="🤖 Yapay zekâ açıklaması (LLM)">
        <p className="hint">
          LLM Açıklama sayfasında kullanılan modeli buradan seçersiniz. Yerel model
          (Ollama) bilgisayarınızda çalışır ve internete veri göndermez. EVREN gibi
          uzak sağlayıcılar için adres ve API anahtarı gerekir.
        </p>

        <div className="settings-grid">
          <div className="field">
            <label>Sağlayıcı</label>
            <select value={draft.llm_provider} onChange={(e) => pickProvider(e.target.value)}>
              {providers.map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
            <span className="tip">{providers.find((p) => p.id === draft.llm_provider)?.hint}</span>
          </div>

          <div className="field">
            <label>Model</label>
            <div className="row">
              <input
                type="text"
                list="model-list"
                value={draft.llm_model}
                onChange={(e) => upd("llm_model", e.target.value)}
                placeholder="or. qwen2.5:3b-instruct / qwen3.8-flash-next"
              />
              <button className="btn ghost sm" onClick={listModels} type="button">Listele</button>
            </div>
            <datalist id="model-list">
              {models.map((m) => <option key={m.id} value={m.id} />)}
            </datalist>
          </div>

          <div className="field">
            <label>Sunucu adresi (base URL)</label>
            <input
              type="text"
              value={draft.llm_base_url}
              onChange={(e) => upd("llm_base_url", e.target.value)}
              placeholder="http://localhost:11434/v1"
            />
          </div>

          <div className="field">
            <label>API anahtarı {keySet ? <span className="tip">(kayıtlı: {keyMask})</span> : null}</label>
            <input
              type="password"
              value={draft.llm_api_key}
              onChange={(e) => upd("llm_api_key", e.target.value)}
              placeholder={keySet ? "Değiştirmek için yazın (boş = mevcut kalır)" : remote ? "evren_llm_…" : "gerekmez"}
              autoComplete="off"
            />
          </div>

          <div className="field">
            <label>Zaman aşımı (saniye)</label>
            <input type="number" min={10} max={600} value={draft.llm_timeout}
              onChange={(e) => upd("llm_timeout", Number(e.target.value))} />
          </div>

          <div className="field">
            <label>Yanıt uzunluğu üst sınırı (token)</label>
            <input type="number" min={128} max={8192} value={draft.llm_max_tokens}
              onChange={(e) => upd("llm_max_tokens", Number(e.target.value))} />
            <span className="tip">Akıl yürüten modellerde en az 2048 önerilir.</span>
          </div>

          <div className="field">
            <label>Yaratıcılık (temperature)</label>
            <input type="number" min={0} max={1.5} step={0.1} value={draft.llm_temperature}
              onChange={(e) => upd("llm_temperature", Number(e.target.value))} />
          </div>

          <div className="field">
            <label>Seçenekler</label>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 2 }}>
              <label className="switch">
                <input type="checkbox" checked={draft.llm_enabled}
                  onChange={(e) => upd("llm_enabled", e.target.checked)} />
                <span className="track" />
                LLM açıklaması etkin
              </label>
              {remote ? (
                <label className="switch">
                  <input type="checkbox" checked={draft.llm_thinking}
                    onChange={(e) => upd("llm_thinking", e.target.checked)} />
                  <span className="track" />
                  Düşünme (reasoning) modu açık
                </label>
              ) : null}
            </div>
          </div>
        </div>

        <div className="status-line">
          <Pill status={llmAvailable ? "on" : "warn"}>
            {llmAvailable ? "LLM erişilebilir" : "LLM erişilemez"}
          </Pill>
          <Pill status={draft.llm_enabled ? "on" : "off"}>
            {draft.llm_enabled ? "Açık" : "Kapalı"}
          </Pill>
          <button className="btn ghost sm" onClick={runTest} disabled={testing} type="button">
            {testing ? "Test ediliyor…" : "Bağlantıyı test et"}
          </button>
          {terms.supported && terms.accepted === false ? (
            <button className="btn sm" onClick={doAcceptTerms} type="button">
              Kullanım şartlarını onayla (sürüm {terms.current_version ?? "?"})
            </button>
          ) : terms.supported && terms.accepted ? (
            <Pill status="on">Kullanım şartları onaylı</Pill>
          ) : null}
        </div>
      </Card>

      <Card title="📰 Haber kaynakları (RSS)">
        <p className="hint">
          Yalnızca borsa/ekonomi ile ilgili haberler gösterilir. Akış adreslerini
          virgülle ayırarak yazın.
        </p>
        <div className="settings-grid">
          <div className="field">
            <label>RSS akışları</label>
            <textarea
              rows={3}
              value={draft.rss_feeds}
              onChange={(e) => upd("rss_feeds", e.target.value)}
            />
          </div>
          <div className="field">
            <label>En fazla haber sayısı</label>
            <input type="number" min={10} max={300} value={draft.rss_max_items}
              onChange={(e) => upd("rss_max_items", Number(e.target.value))} />
            <div style={{ marginTop: 8 }}>
              <label className="switch">
                <input type="checkbox" checked={draft.rss_enabled}
                  onChange={(e) => upd("rss_enabled", e.target.checked)} />
                <span className="track" />
                Haber akışı etkin
              </label>
            </div>
          </div>
        </div>
      </Card>

      <Card title="⚙️ Analiz motoru varsayılanları">
        <p className="hint">
          Bu değerler analiz motorunun temel ayarlarıdır. Hisse sayfasındaki
          “Analiz Ayarları” panelinden anlık olarak da değiştirilebilir.
        </p>
        <div className="settings-grid">
          <div className="field">
            <label>Geçmiş uzunluğu (gün)</label>
            <input type="number" min={120} max={1000} value={draft.lookback_days}
              onChange={(e) => upd("lookback_days", Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Hareketli pencere (gün)</label>
            <input type="number" min={20} max={120} value={draft.rolling_window}
              onChange={(e) => upd("rolling_window", Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Model tahmin penceresi (gün)</label>
            <input type="number" min={60} max={250} value={draft.estimation_window}
              onChange={(e) => upd("estimation_window", Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Anormallik ağacı sayısı</label>
            <input type="number" min={50} max={500} step={50} value={draft.iforest_trees}
              onChange={(e) => upd("iforest_trees", Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Anormallik oranı</label>
            <input type="number" min={0.005} max={0.1} step={0.005} value={draft.iforest_contamination}
              onChange={(e) => upd("iforest_contamination", Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Haber düzeltme gücü</label>
            <input type="number" min={0} max={0.6} step={0.05} value={draft.news_penalty_strength}
              onChange={(e) => upd("news_penalty_strength", Number(e.target.value))} />
          </div>
          <div className="field">
            <label>Likidite (tahta) modülasyonu</label>
            <input type="number" min={0} max={1} step={0.05} value={draft.tahta_modulation}
              onChange={(e) => upd("tahta_modulation", Number(e.target.value))} />
          </div>
          <div className="field">
            <label>VeriShield</label>
            <label className="switch" style={{ marginTop: 4 }}>
              <input type="checkbox" checked={draft.verishield_enabled}
                onChange={(e) => upd("verishield_enabled", e.target.checked)} />
              <span className="track" />
              Haber güvenilirlik motoru etkin
            </label>
          </div>
        </div>
      </Card>

      <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 16, flexWrap: "wrap" }}>
        <button className="btn" onClick={save} disabled={saving}>
          {saving ? "Kaydediliyor…" : "Ayarları Kaydet"}
        </button>
        <button className="btn ghost" onClick={load} type="button">Geri al / Yenile</button>
        <span className="tip" style={{ color: "var(--text-faint)", fontSize: ".7rem" }}>
          Kayıt yeri: <span className="mono">{storePath}</span> (bu dosya .env değerlerini geçersiz kılar)
        </span>
      </div>
    </>
  );
}
