"""Merkezi yapilandirma.

Tum ayarlar ortam degiskenleri (ve varsa .env) uzerinden okunur. Kod icinde
sir tutulmaz. Varsayilanlar makul ve tek makinede calisacak sekilde secildi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Basit .env okuyucu (ek bagimlilik olmadan)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class FusionWeights:
    """Birlesik suphe skoru agirliklari (sum=1 olacak sekilde normalize edilir).

    ml: denetimli siniflandirici sinyali — model kayitli degilse sinyal NaN olur
    ve agirlik otomatik olarak mevcut sinyallere yeniden dagitilir.
    """

    volume_z: float = 0.18        # Hacim sapmasi (robust MAD z) — likidite baglaminda module edilir
    index_divergence: float = 0.16  # Endeks ayrisimasi (market model CAR)
    volatility: float = 0.10      # Oynaklik patlamasi (robust)
    mahalanobis: float = 0.13     # Robust cok degiskenli uzaklik (MCD)
    isolation_forest: float = 0.08  # iForest cikti
    liquidity: float = 0.06       # Likidite soku (Amihud) — likidite baglaminda module edilir
    ml: float = 0.12              # Denetimli ML olasiligi (model varsa)
    close_location: float = 0.08  # CLV / kapanis konumu (marking the close izi)
    pump_pattern: float = 0.09    # Cok gunlu pump deseni (CAR kalıciligi + seri)

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def normalized(self) -> dict[str, float]:
        total = sum(self.as_dict().values())
        if total <= 0:
            n = len(self.as_dict())
            return {k: 1.0 / n for k in self.as_dict()}
        return {k: v / total for k, v in self.as_dict().items()}


@dataclass
class Settings:
    # --- Proje ---
    project_root: Path = field(default_factory=lambda: Path.cwd())

    # --- Veri ---
    data_dir: Path = field(default_factory=lambda: Path("data"))
    cache_ttl_hours: int = 12
    lookback_days: int = 400
    benchmark_ticker: str = "XU100.IS"

    # --- Anomali ---
    mad_scale: float = 1.4826
    iforest_trees: int = 200
    iforest_contamination: float = 0.02
    rolling_window: int = 60
    estimation_window: int = 120
    random_state: int = 42

    # --- Haber / VeriShield ---
    verishield_enabled: bool = True
    verishield_mode: str = "embedded"  # embedded | http
    verishield_base_url: str = "http://localhost:8000"

    # --- Haber / RSS ---
    rss_enabled: bool = True
    rss_feeds: str = (
        "https://www.bloomberght.com/rss/tum-haberler.xml,"
        "https://www.cnnturk.com/feed/rss/ekonomi/news"
    )
    rss_max_items: int = 60

    # --- ML ---
    ml_enabled: bool = True

    # --- Duygu / LLM ---
    sentiment_backend: str = "auto"  # auto | bert | lexicon
    sentiment_model: str = "savasy/bert-base-turkish-sentiment-cased"
    llm_enabled: bool = False
    llm_provider: str = "ollama"     # ollama | evren | openai | custom
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b-instruct"
    llm_api_key: str = ""
    llm_timeout: int = 180
    llm_thinking: bool = False       # akil yuruten modellerde dusunmeyi kapat
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.3

    # --- Füzyon ---
    weights: FusionWeights = field(default_factory=FusionWeights)
    news_penalty_strength: float = 0.35  # haber destegi supheyi bu orana kadar dusurur
    tahta_modulation: float = 1.0        # 0 = likidite-baglami modulasyonu kapali

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"


def load_settings(dotenv: Path | None = None) -> Settings:
    """Ortam degiskenlerinden ayarlari yukler."""
    if dotenv is None:
        dotenv = Path.cwd() / ".env"
    _load_dotenv(dotenv)

    s = Settings()
    s.data_dir = Path(_env_str("MARS_DATA_DIR", "data"))
    s.cache_ttl_hours = _env_int("MARS_CACHE_TTL_HOURS", s.cache_ttl_hours)
    s.lookback_days = _env_int("MARS_LOOKBACK_DAYS", s.lookback_days)
    s.benchmark_ticker = _env_str("MARS_BENCHMARK_TICKER", s.benchmark_ticker)

    s.mad_scale = _env_float("MARS_MAD_SCALE", s.mad_scale)
    s.iforest_trees = _env_int("MARS_IFOREST_TREES", s.iforest_trees)
    s.iforest_contamination = _env_float(
        "MARS_IFOREST_CONTAMINATION", s.iforest_contamination
    )
    s.rolling_window = _env_int("MARS_ROLLING_WINDOW", s.rolling_window)
    s.estimation_window = _env_int("MARS_ESTIMATION_WINDOW", s.estimation_window)

    s.verishield_enabled = _env_bool("MARS_VERISHIELD_ENABLED", s.verishield_enabled)
    s.verishield_mode = _env_str("MARS_VERISHIELD_MODE", s.verishield_mode)
    s.verishield_base_url = _env_str("MARS_VERISHIELD_BASE_URL", s.verishield_base_url)

    s.rss_enabled = _env_bool("MARS_RSS_ENABLED", s.rss_enabled)
    s.rss_feeds = _env_str("MARS_RSS_FEEDS", s.rss_feeds)
    s.rss_max_items = _env_int("MARS_RSS_MAX_ITEMS", s.rss_max_items)

    s.ml_enabled = _env_bool("MARS_ML_ENABLED", s.ml_enabled)

    s.llm_timeout = _env_int("MARS_LLM_TIMEOUT", s.llm_timeout)

    s.sentiment_backend = _env_str("MARS_SENTIMENT_BACKEND", s.sentiment_backend)
    s.sentiment_model = _env_str("MARS_SENTIMENT_MODEL", s.sentiment_model)
    s.llm_enabled = _env_bool("MARS_LLM_ENABLED", s.llm_enabled)
    s.llm_provider = _env_str("MARS_LLM_PROVIDER", s.llm_provider)
    s.llm_base_url = _env_str("MARS_LLM_BASE_URL", s.llm_base_url)
    s.llm_model = _env_str("MARS_LLM_MODEL", s.llm_model)
    s.llm_api_key = _env_str("MARS_LLM_API_KEY", s.llm_api_key)
    s.llm_thinking = _env_bool("MARS_LLM_THINKING", s.llm_thinking)
    s.llm_max_tokens = _env_int("MARS_LLM_MAX_TOKENS", s.llm_max_tokens)
    s.llm_temperature = _env_float("MARS_LLM_TEMPERATURE", s.llm_temperature)

    s.news_penalty_strength = _env_float(
        "MARS_NEWS_PENALTY_STRENGTH", s.news_penalty_strength
    )
    s.tahta_modulation = _env_float("MARS_TAHTA_MODULATION", s.tahta_modulation)

    # Arayuzden kaydedilen calisma-zamani ayarlari env'in uzerine uygula.
    _apply_overrides(s)
    return s


def _apply_overrides(s: Settings) -> None:
    """`data/settings.local.json` icindeki kullanici ayarlarini uygular."""
    try:
        from .settings_store import load_overrides, store_path  # noqa: PLC0415

        overrides = load_overrides(store_path())
    except Exception:  # noqa: BLE001
        return
    for key, value in overrides.items():
        if hasattr(s, key):
            setattr(s, key, value)
