import { useEffect, useState } from "react";
import { ThemeName, applyTheme, getTheme } from "./theme";

export function useTheme(): { theme: ThemeName; toggle: () => void; setTheme: (t: ThemeName) => void } {
  const [theme, setState] = useState<ThemeName>(getTheme);
  useEffect(() => {
    const handler = () => setState(getTheme());
    window.addEventListener("mars-theme", handler);
    return () => window.removeEventListener("mars-theme", handler);
  }, []);
  return {
    theme,
    setTheme: (t: ThemeName) => applyTheme(t),
    toggle: () => applyTheme(theme === "dark" ? "light" : "dark"),
  };
}
