import { ReactNode, useEffect, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { MetaInfo, fetchMeta } from "../api";
import { useApp } from "../state";
import { useTheme } from "../useTheme";
import { Pill } from "./ui";

function IconScreener(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.2-3.2" />
      <path d="m8 11.5 2-2.5 1.7 3 1.8-2" />
    </svg>
  );
}

function IconCandles(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 4v3M7 15v5M17 3v3M17 12v6" />
      <rect x="5" y="7" width="4" height="8" rx="1" />
      <rect x="15" y="6" width="4" height="6" rx="1" />
    </svg>
  );
}

function IconNews(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Z" />
      <path d="M4 22a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2" />
      <path d="M18 14h-8" />
      <path d="M15 18h-5" />
      <path d="M10 6h8v4h-8V6Z" />
    </svg>
  );
}

function IconSparkles(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
      <path d="M5 3v4" />
      <path d="M3 5h4" />
      <path d="M19 17v4" />
      <path d="M17 19h4" />
    </svg>
  );
}

function IconSliders(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3" />
      <path d="M14 2v4M8 10v4M16 18v4" />
    </svg>
  );
}

function IconSun(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
  );
}

function IconMoon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
    </svg>
  );
}

const NAV: { to: string; icon: () => JSX.Element; label: string }[] = [
  { to: "/screener", icon: IconScreener, label: "Şüpheli Hisseler" },
  { to: "/analysis", icon: IconCandles, label: "Hisse Analizi" },
  { to: "/news", icon: IconNews, label: "Haber Akışı" },
  { to: "/explain", icon: IconSparkles, label: "LLM Açıklama" },
  { to: "/settings", icon: IconSliders, label: "Ayarlar" },
];

const TITLES: Record<string, string> = {
  "/screener": "Şüpheli Hisseler — Sepet Taraması",
  "/analysis": "Hisse Analizi",
  "/news": "Haber Akışı",
  "/explain": "LLM ile Açıklama",
  "/settings": "Ayarlar",
};

export default function Layout(): JSX.Element {
  const location = useLocation();
  const navigate = useNavigate();
  const { ticker, setTicker } = useApp();
  const { theme, toggle } = useTheme();
  const [meta, setMeta] = useState<MetaInfo | null>(null);
  const [search, setSearch] = useState(ticker);

  useEffect(() => {
    fetchMeta().then(setMeta).catch(() => setMeta(null));
  }, [location.pathname]);

  useEffect(() => setSearch(ticker), [ticker]);

  const submitSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const t = search.trim().toUpperCase();
    if (t) {
      setTicker(t.includes(".") ? t : `${t}.IS`);
      navigate("/analysis");
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-mark">M</div>
          <div className="logo-text">
            <h1>
              MARS <span>Terminal</span>
            </h1>
            <p>Piyasa Anomali & Risk</p>
          </div>
        </div>

        {NAV.map((n) => (
          <div
            key={n.to}
            className={`nav-item${location.pathname === n.to ? " active" : ""}`}
            onClick={() => navigate(n.to)}
          >
            <span className="ico">
              <n.icon />
            </span>
            {n.label}
          </div>
        ))}

        <div className="section">İzleme Listesi</div>
        {(meta?.basket ?? ["THYAO.IS"]).slice(0, 14).map((t) => (
          <div
            key={t}
            className="watch-item"
            onClick={() => {
              setTicker(t);
              navigate("/analysis");
            }}
          >
            <span>{t.replace(".IS", "")}</span>
            <span className="s" style={{ color: ticker === t ? "var(--accent)" : undefined }}>
              {ticker === t ? "●" : ""}
            </span>
          </div>
        ))}

        <div className="footer">
          <Pill status={meta?.verishield ? "on" : "warn"}>
            VeriShield {meta?.verishield ? "✓" : "—"}
          </Pill>
          <Pill status={meta?.ml.available ? "on" : "warn"}>
            ML {meta?.ml.available ? `✓ ${meta.ml.auc?.toFixed(3)}` : "yok"}
          </Pill>
          <Pill status={meta?.llm.available ? "on" : "off"}>
            LLM {meta?.llm.available ? "✓" : "kapalı"}
          </Pill>
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <h2>{TITLES[location.pathname] ?? "MARS"}</h2>
          <form className="search" onSubmit={submitSearch}>
            <input
              type="text"
              value={search}
              placeholder="Hisse kodu (THYAO)"
              onChange={(e) => setSearch(e.target.value)}
            />
            <button className="btn" type="submit">
              Analiz Et
            </button>
          </form>
          <button
            className="theme-toggle"
            onClick={toggle}
            title={theme === "dark" ? "Açık temaya geç" : "Koyu temaya geç"}
            type="button"
          >
            {theme === "dark" ? <IconSun /> : <IconMoon />}
          </button>
        </div>
        <div className="content">
          <Outlet />
          <div className="footer-note">
            MARS · BLMS405 dönem projesi · Anomali skoru ≠ manipülasyon kanıtı · yatırım
            tavsiyesi değildir · v{meta?.version ?? "?"}
          </div>
        </div>
      </div>
    </div>
  );
}
