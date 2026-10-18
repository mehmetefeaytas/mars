import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import "./theme.css";
import Layout from "./components/Layout";
import ScreenerPage from "./pages/ScreenerPage";
import AnalysisPage from "./pages/AnalysisPage";
import NewsPage from "./pages/NewsPage";
import ExplainPage from "./pages/ExplainPage";
import { AppProvider } from "./state";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/screener" replace />} />
            <Route path="/screener" element={<ScreenerPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/news" element={<NewsPage />} />
            <Route path="/explain" element={<ExplainPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppProvider>
  </React.StrictMode>
);
