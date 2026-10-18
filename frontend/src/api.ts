export interface MetaInfo {
  version: string;
  verishield: boolean;
  ml: { available: boolean; auc: number | null; coefficients: Record<string, number> };
  llm: { enabled: boolean; available: boolean; model: string };
  rss: { enabled: boolean; feeds: string[] };
  basket: string[];
  defaults: { params: AnalyzeParams; weights: Record<string, number> };
}

export interface AnalyzeParams {
  lookback: number;
  rolling: number;
  est_window: number;
  iforest_trees: number;
  contamination: number;
  news_penalty: number;
  tahta_modulation: number;
  weights: Record<string, number> | null;
  rss: boolean;
  llm: boolean | null;
}

export interface Reason {
  signal: string;
  detail: string;
  points: number;
  level: string;
}

export interface WaterfallItem {
  signal: string;
  points: number;
}

export interface NewsBrief {
  headline: string;
  source: string;
  url: string | null;
  timestamp: string | null;
  polarity: number;
  label: string;
  credibility: number;
  general?: boolean;
}

export interface Analysis {
  ticker: string;
  data_source: string;
  ml_available: boolean;
  dates: string[];
  open: (number | null)[];
  high: (number | null)[];
  low: (number | null)[];
  close: (number | null)[];
  volume: (number | null)[];
  suspicion: (number | null)[];
  suspicion_base: (number | null)[];
  data_confidence: (number | null)[];
  news_factor: (number | null)[];
  log_ret: (number | null)[];
  abnormal_ret: (number | null)[];
  car5: (number | null)[];
  clv: (number | null)[];
  signals: Record<string, (number | null)[]>;
  contributions: Record<string, (number | null)[]>;
  tahta_prior: number;
  liquidity_tier: string;
  adv_dollar_median: number;
  news: NewsBrief[];
  news_is_general: boolean;
  summary: string;
  waterfall: WaterfallItem[];
  reasons: Reason[];
}

export interface ScreenerRow {
  ticker: string;
  suspicion: number | null;
  data_confidence: number | null;
  log_ret: number | null;
  abnormal_ret: number | null;
  top_signal: string;
  top_points: number;
  liquidity_tier: string;
  tahta_prior: number;
  news_count: number;
  date: string;
}

export interface NewsItem {
  headline: string;
  body: string;
  source: string;
  url: string | null;
  timestamp: string | null;
  polarity: number;
  label: string;
  confidence: number;
  sentiment_backend: string;
  credibility: number;
  credibility_backend: string;
  category?: string;
  ticker?: string | null;
}

export interface ExplainResult {
  ticker: string;
  date: string;
  suspicion: number;
  data_confidence: number;
  summary: string;
  reasons: Reason[];
  llm: string | null;
  llm_backend: string;
  llm_error: string | null;
}

export interface EffectiveSettings {
  llm_enabled: boolean;
  llm_provider: string;
  llm_base_url: string;
  llm_model: string;
  llm_api_key: string;
  llm_api_key_set: boolean;
  llm_timeout: number;
  llm_thinking: boolean;
  llm_max_tokens: number;
  llm_temperature: number;
  rss_enabled: boolean;
  rss_feeds: string;
  rss_max_items: number;
  sentiment_backend: string;
  verishield_enabled: boolean;
  lookback_days: number;
  rolling_window: number;
  estimation_window: number;
  iforest_trees: number;
  iforest_contamination: number;
  news_penalty_strength: number;
  tahta_modulation: number;
}

export interface ProviderPreset {
  id: string;
  label: string;
  default_url: string;
  needs_key: boolean;
  hint: string;
}

export interface SettingsResponse {
  current: EffectiveSettings;
  overrides: Record<string, unknown>;
  providers: ProviderPreset[];
  store_path: string;
  evren_terms: { supported: boolean; accepted: boolean | null; current_version?: number; error?: string };
  llm_available: boolean;
}

export function defaultParams(): AnalyzeParams {
  return {
    lookback: 400,
    rolling: 60,
    est_window: 120,
    iforest_trees: 200,
    contamination: 0.02,
    news_penalty: 0.35,
    tahta_modulation: 1.0,
    weights: null,
    rss: true,
    llm: null,
  };
}

export function paramsToQuery(p: AnalyzeParams): string {
  const q = new URLSearchParams({
    lookback: String(p.lookback),
    rolling: String(p.rolling),
    est_window: String(p.est_window),
    iforest_trees: String(p.iforest_trees),
    contamination: String(p.contamination),
    news_penalty: String(p.news_penalty),
    tahta_modulation: String(p.tahta_modulation),
    rss: String(p.rss),
  });
  return q.toString();
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* yoksay */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function fetchMeta(): Promise<MetaInfo> {
  return jsonOrThrow<MetaInfo>(await fetch("/api/meta"));
}

export async function fetchAnalysis(ticker: string, p: AnalyzeParams): Promise<Analysis> {
  return jsonOrThrow<Analysis>(await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}&${paramsToQuery(p)}`));
}

export async function fetchScreener(tickers: string[], p: AnalyzeParams): Promise<{ rows: ScreenerRow[]; errors: { ticker: string; error: string }[] }> {
  const res = await fetch("/api/screener", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickers, params: p }),
  });
  return jsonOrThrow(res);
}

export async function fetchNews(limit = 60): Promise<{ items: NewsItem[]; matched: Record<string, number> }> {
  return jsonOrThrow(await fetch(`/api/news?limit=${limit}`));
}

export async function fetchExplain(ticker: string, date: string | null, p: AnalyzeParams): Promise<ExplainResult> {
  const res = await fetch("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, date, params: p }),
  });
  return jsonOrThrow(res);
}

export async function fetchSettings(): Promise<SettingsResponse> {
  return jsonOrThrow(await fetch("/api/settings"));
}

export async function saveSettings(patch: Record<string, unknown>): Promise<{ current: EffectiveSettings; llm_available: boolean }> {
  const res = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  return jsonOrThrow(res);
}

export async function fetchModels(): Promise<{ models: { id: string; owned_by?: string; task?: string }[]; provider: string }> {
  return jsonOrThrow(await fetch("/api/settings/models"));
}

export async function testLlm(): Promise<{ ok: boolean; message: string; terms_required?: boolean; version?: number; provider?: string; model?: string }> {
  return jsonOrThrow(await fetch("/api/settings/llm/test", { method: "POST" }));
}

export async function fetchTerms(): Promise<{ supported?: boolean; accepted?: boolean; current_version?: number }> {
  return jsonOrThrow(await fetch("/api/settings/terms"));
}

export async function acceptTerms(version: number): Promise<{ accepted_version: number }> {
  const res = await fetch("/api/settings/terms/accept", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version }),
  });
  return jsonOrThrow(res);
}
