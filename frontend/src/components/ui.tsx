import { ReactNode } from "react";

export function Card({ title, children, className }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <div className={`card${className ? " " + className : ""}`}>
      {title ? <h3>{title}</h3> : null}
      {children}
    </div>
  );
}

export function Pill({ status, children }: { status?: "on" | "off" | "warn"; children: ReactNode }) {
  return <span className={`pill ${status ?? ""}`}>{children}</span>;
}

export function ScoreBar({ value }: { value: number | null }) {
  const v = value ?? 0;
  const color = v >= 65 ? "var(--down)" : v >= 35 ? "var(--warn)" : "var(--up)";
  return (
    <div className="score-cell">
      <div className="score-bar">
        <div className="fill" style={{ width: `${Math.min(100, Math.max(0, v))}%`, background: color }} />
      </div>
      <span className="score-num" style={{ color }}>{v.toFixed(0)}</span>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div className="spinner" />
      {label ? <div className="empty">{label}</div> : null}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="card" style={{ borderColor: "rgba(239,83,80,.4)", marginBottom: 14 }}>
      <p style={{ color: "var(--down)", fontSize: ".86rem", display: "flex", alignItems: "center", gap: 8 }}>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
          strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
          style={{ flexShrink: 0 }}>
          <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
          <path d="M12 9v4" />
          <path d="M12 17h.01" />
        </svg>
        {message}
      </p>
    </div>
  );
}

export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;
}

export function scoreColor(v: number | null | undefined): string {
  const x = v ?? 0;
  return x >= 65 ? "var(--down)" : x >= 35 ? "var(--warn)" : "var(--up)";
}
