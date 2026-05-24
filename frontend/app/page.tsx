"use client";

import { useEffect, useState } from "react";
import TopBar from "@/components/shell/TopBar";
import Nav, { type NavRoute } from "@/components/shell/Nav";
import AitoPanel from "@/components/shell/AitoPanel";
import ErrorState from "@/components/shell/ErrorState";
import { apiFetch, ApiError } from "@/lib/api";
import type { AitoPanelConfig } from "@/lib/types";

type ExampleResponse = {
  message: string;
  aito_url: string;
  tables: string[];
};

// One AitoPanelConfig per page (or per route segment). For a multi-page
// demo, you'd typically build one per page.tsx.
const PANEL_CONFIG: AitoPanelConfig = {
  operation: "GET /api/v1/schema",
  stats: [
    { value: "1", label: "Aito DB" },
    { value: "~ms", label: "round-trip" },
    { value: "0", label: "ML pipelines" },
  ],
  description:
    "This demo's <strong>hello world</strong>: the backend calls Aito's <code>/schema</code> to enumerate tables, " +
    "and we render the list. Replace with your real Aito query, and the panel automatically shows the live request/response.",
  query: JSON.stringify({ method: "GET", path: "/api/v1/schema" }, null, 2),
  links: [
    { label: "Aito docs", url: "https://aito.ai/docs/" },
    { label: "Source on GitHub", url: "https://github.com/AitoDotAI" },
  ],
};

// For a single-page template, the sidebar is empty (Nav hides itself).
// Add routes here once you have multiple pages.
const ROUTES: NavRoute[] = [];

export default function Home() {
  const [data, setData] = useState<ExampleResponse | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [lastResponseMs, setLastResponseMs] = useState<number | null>(null);
  const [lastQuery, setLastQuery] = useState<object | null>(null);

  useEffect(() => {
    const t0 = performance.now();
    apiFetch<ExampleResponse>("/api/example")
      .then((d) => {
        setData(d);
        setLastResponseMs(Math.round(performance.now() - t0));
        setLastQuery({ method: "GET", path: "/api/example" });
      })
      .catch((e) => setError(e));
  }, []);

  return (
    <div className="app">
      <Nav routes={ROUTES} />
      <main className="main">
        <TopBar
          brand="Hello demo"
          breadcrumb="Template"
          title="Hello from Aito"
          githubUrl="https://github.com/AitoDotAI"
        />
        <div className="content">
          {error && <ErrorState error={error} />}
          {data && (
            <div className="card">
              <div className="card-header">
                <div className="card-title">{data.message}</div>
              </div>
              <div style={{ padding: "16px 20px" }}>
                <p style={{ color: "var(--text3)", marginBottom: 12 }}>
                  Aito DB: <code>{data.aito_url}</code>
                </p>
                <p style={{ color: "var(--text2)", marginBottom: 8 }}>
                  <strong>{data.tables.length}</strong> table{data.tables.length === 1 ? "" : "s"}:
                </p>
                <ul style={{ paddingLeft: 18 }}>
                  {data.tables.map((t) => (
                    <li key={t}>
                      <code>{t}</code>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          {!data && !error && (
            <div className="card" style={{ padding: 24, color: "var(--text3)" }}>
              Loading…
            </div>
          )}
        </div>
      </main>
      <AitoPanel config={PANEL_CONFIG} lastQuery={lastQuery} lastResponseMs={lastResponseMs} />
    </div>
  );
}
