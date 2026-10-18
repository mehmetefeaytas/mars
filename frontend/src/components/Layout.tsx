import { ReactNode, useEffect, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { MetaInfo, fetchMeta } from "../api";
import { useApp } from "../state";
import { useTheme } from "../useTheme";
import { Pill } from "./ui";

const NAV = [
  { to: "/screener", icon: "🔍", label: "Şüpheli Hisseler" },
  { to: "/analysis", icon: "📊", label: "Hisse Analizi" },
  { to: "/news", icon: "📰", label: "Haber Akışı" },
  { to: "/explain", icon: "🤖", label: "LLM Açıklama" },
  { to: "/settings", icon: "⚙️", label: "Ayarlar" },
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
            <span className="ico">{n.icon}</span>
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
            {theme === "dark" ? "☀️" : "🌙"}
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
