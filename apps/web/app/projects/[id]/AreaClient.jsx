"use client";

import React, { useMemo, useState } from "react";

/**
 * LuxIA — AreaClient (AutOPILOT + PRO toggle)
 *
 * - One-click AUTOPILOT: generates 3 concepts via Engine
 * - Optional PRO Radiance: if toggle ON, starts a PRO job right after concepts generation
 *
 * Requirements (Vercel env):
 * - NEXT_PUBLIC_ENGINE_URL (e.g. https://luxia.onrender.com)
 * - NEXT_PUBLIC_LUXIA_TOKEN (same as Engine LUXIA_TOKEN)
 *
 * Notes:
 * - This component is intentionally self-contained to avoid “missing imports”.
 * - If your project already has a richer AreaClient, merge only the PRO toggle + autopilot handler parts.
 */

function getEngineUrl() {
  const raw = process.env.NEXT_PUBLIC_ENGINE_URL || "";
  return raw.replace(/\/+$/, "");
}

function getAuthHeaders() {
  const token = process.env.NEXT_PUBLIC_LUXIA_TOKEN || "";
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function engineFetch(path, opts = {}) {
  const base = getEngineUrl();
  if (!base) throw new Error("NEXT_PUBLIC_ENGINE_URL mancante");
  const url = `${base}${path.startsWith("/") ? path : `/${path}`}`;

  const headers = {
    "Content-Type": "application/json",
    ...(opts.headers || {}),
    ...getAuthHeaders(),
  };

  const res = await fetch(url, { ...opts, headers });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Engine ${res.status}: ${text || res.statusText}`);
  }
  // Some endpoints may return empty body (204). Keep safe:
  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("application/json")) return { ok: true };
  return res.json();
}

export default function AreaClient({ projectId, initialAreas = [], onResults }) {
  const [areas, setAreas] = useState(initialAreas);
  const [loading, setLoading] = useState(false);
  const [proEnabled, setProEnabled] = useState(false);
  const [status, setStatus] = useState("");

  const canRun = useMemo(() => !!projectId, [projectId]);

  // AUTOPILOT:
  // 1) generate 3 concepts (NumPy fast)
  // 2) if PRO toggle enabled, start PRO Radiance async job
  const runAutopilot = async () => {
    if (!canRun) return;
    setLoading(true);
    setStatus("Avvio Autopilot…");

    try {
      // 1) Generate concepts (fast engine)
      setStatus("Generazione 3 concept (Fast Engine)…");
      const concepts = await engineFetch(`/projects/${projectId}/concepts`, {
        method: "POST",
        body: JSON.stringify({
          mode: "fast",
          areas: areas,
        }),
      });

      if (typeof onResults === "function") onResults(concepts);

      // 2) Optionally start PRO Radiance job
      if (proEnabled) {
        setStatus("PRO attivo: avvio job Radiance…");
        const job = await engineFetch(`/projects/${projectId}/pro/radiance/jobs`, {
          method: "POST",
          body: JSON.stringify({
            daylight: true,
          }),
        });
        setStatus(`Job PRO avviato: ${job?.job_id || "ok"} (risultati in Jobs/PRO)`);
      } else {
        setStatus("Autopilot completato (Fast).");
      }
    } catch (e) {
      setStatus(`Errore: ${e?.message || String(e)}`);
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <button
          onClick={runAutopilot}
          disabled={!canRun || loading}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid #1e3a8a",
            background: loading ? "#93c5fd" : "#2563eb",
            color: "white",
            fontWeight: 700,
            cursor: loading ? "not-allowed" : "pointer",
          }}
          title="One-click: genera concept + calcoli (e PRO se attivo)"
        >
          🚀 AUTOPILOT
        </button>

        <label style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600 }}>
          <input
            type="checkbox"
            checked={proEnabled}
            onChange={(e) => setProEnabled(e.target.checked)}
            disabled={loading}
          />
          PRO Radiance (on/off)
        </label>

        <span style={{ color: "#64748b", fontSize: 13 }}>
          {proEnabled ? "Autopilot avvierà anche Radiance PRO." : "Autopilot userà solo Fast Engine (NumPy)."}
        </span>
      </div>

      {status ? (
        <div style={{ padding: 10, borderRadius: 10, background: "#f1f5f9", color: "#0f172a" }}>
          <b>Stato:</b> {status}
        </div>
      ) : null}

      {/* Debug block (can be removed) */}
      <details style={{ padding: 10, borderRadius: 10, border: "1px solid #e2e8f0" }}>
        <summary style={{ cursor: "pointer", fontWeight: 700 }}>Aree (debug)</summary>
        <pre style={{ fontSize: 12, overflowX: "auto" }}>{JSON.stringify(areas, null, 2)}</pre>
      </details>
    </div>
  );
}
