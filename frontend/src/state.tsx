import { createContext, useContext, useState, ReactNode } from "react";
import { AnalyzeParams, defaultParams } from "./api";

interface AppState {
  params: AnalyzeParams;
  setParams: (p: AnalyzeParams) => void;
  ticker: string;
  setTicker: (t: string) => void;
}

export const AppContext = createContext<AppState>({
  params: defaultParams(),
  setParams: () => undefined,
  ticker: "THYAO.IS",
  setTicker: () => undefined,
});

export function AppProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useState<AnalyzeParams>(defaultParams());
  const [ticker, setTicker] = useState("THYAO.IS");
  return (
    <AppContext.Provider value={{ params, setParams, ticker, setTicker }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp(): AppState {
  return useContext(AppContext);
}
