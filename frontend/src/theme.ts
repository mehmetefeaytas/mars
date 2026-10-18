export type ThemeName = "dark" | "light";

const KEY = "mars.theme";

export function getTheme(): ThemeName {
  try {
    return localStorage.getItem(KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function applyTheme(t: ThemeName): void {
  document.documentElement.dataset.theme = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* yoksay */
  }
  window.dispatchEvent(new Event("mars-theme"));
}

export function initTheme(): void {
  applyTheme(getTheme());
}
