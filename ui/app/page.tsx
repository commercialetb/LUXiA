"use client";

import { useMemo, useState } from "react";

type AnalyzeResp = any;

function b64ToBlob(b64: string, mime: string) {
  const byteChars = atob(b64);
  const byteNumbers = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) byteNumbers[i] = byteChars.charCodeAt(i);
  return new Blob([new Uint8Array(byteNumbers)], { type: mime });
}

export default function Page() {
  const [file, setFile] = useState<File | null>(null);

  // LuxIA settings (simple jobs style)
  const [autonomyLevel, setAutonomyLevel] = useState<number>(1); // 1 = autonoma (default)
  const [designerMode, setDesignerMode] = useState<boolean>(true);
  const [goal, setGoal] = useState<string>("balanced");
  const [daylightMode, setDaylightMode] = useState<boolean>(false); // calcola solo se richiesto

  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);

  const [resp, setResp] = useState<AnalyzeResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [userInputs, setUserInputs] = useState<Record<string, any>>({});
  const [chatHistory, setChatHistory] = useState<Array<{ role: "user" | "assistant"; content: string }>>([]);
  const [chatMsg, setChatMsg] = useState<string>("");

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  const analyze = async () => {
    if (!file) return;

    setBusy(true);
    setLoading(true);
    setErr(null);
    setResp(null);

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append(
        "options",
        JSON.stringify({
          autonomy_level: autonomyLevel,
          designer_mode: designerMode,
          goal,
          daylight_mode: daylightMode,
          ...userInputs,
        })
      );

      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      const data = await r.json().catch(() => null);

      if (!r.ok || !data?.ok) {
        setErr(data?.error || "Errore analisi");
      } else {
        setResp(data);

        // if engine asks for missing inputs, keep them available to fill
        if (Array.isArray(data?.needs_user_input) && data.needs_user_input.length) {
          const next: Record<string, any> = { ...userInputs };
          for (const q of data.needs_user_input) {
            const k = q?.key;
            if (k && next[k] === undefined) next[k] = q?.default ?? "";
          }
          setUserInputs(next);
        }
      }
    } catch (e: any) {
      setErr(e?.message || "Errore analisi");
    } finally {
      setBusy(false);
      setLoading(false);
    }
  };

  const sendChat = async () => {
    const message = chatMsg.trim();
    if (!message) return;
    setChatMsg("");
    const optimistic = [...chatHistory, { role: "user" as const, content: message }];
    setChatHistory(optimistic);

    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: optimistic, context: resp?.extracted || {} }),
    });
    const data = await r.json().catch(() => null);
    if (!r.ok || !data?.ok) {
      setChatHistory([...optimistic, { role: "assistant" as const, content: data?.error || "Errore chat" }]);
      return;
    }
    setChatHistory(data.history || [...optimistic, { role: "assistant" as const, content: data.assistant || "" }]);
  };

  const downloadPdf = () => {
    const b64 = resp?.report_pdf_base64;
    if (!b64) return;
    const blob = b64ToBlob(b64, "application/pdf");
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "LuxIA_Report.pdf";
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadDxf = () => {
    const b64 = resp?.dxf_out_base64 || resp?.extracted?.dxf_out_base64;
    if (!b64) return;
    const blob = b64ToBlob(b64, "application/dxf");
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "LuxIA_Illuminazione.dxf";
    a.click();
    URL.revokeObjectURL(url);
  };

  const issues = resp?.vision?.issues || [];
  const pass = resp?.vision?.pass === true;

  return (
    <div className="main">
      <div>
        <div className="h1">LuxIA</div>
        <div className="p">Carica la planimetria. LuxIA analizza, verifica e genera il report.</div>
      </div>

      <div className="card">
        {Array.isArray(resp?.needs_user_input) && resp?.needs_user_input.length > 0 && (
          <div className="card" style={{ marginBottom: 12, border: "1px dashed #999" }}>
            <div style={{ fontWeight: 800, fontSize: 16 }}>Servono alcuni dati per essere precisi</div>
            <div className="small" style={{ marginBottom: 8 }}>
              Inseriscili qui e rilancia “Analizza”. (Se lasci i default, LuxIA procede con assunzioni.)
            </div>
            <div className="grid">
              {resp.needs_user_input.map((q: any, i: number) => (
                <div key={q.key || i} className="field">
                  <label className="label">{q.question || q.key}</label>
                  <input
                    className="inp"
                    value={userInputs?.[q.key] ?? ""}
                    onChange={(e) => setUserInputs((s) => ({ ...s, [q.key]: e.target.value }))}
                    placeholder={String(q.default ?? "")}
                  />
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10 }}>
              <button className="btnP" onClick={analyze} disabled={!file || loading}>
                Analizza (con questi dati)
              </button>
            </div>
          </div>
        )}

        <div className="row">
          <input
            type="file"
            accept="application/pdf,image/*,.dxf,.dwg"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />

          <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
            <label className="small">
              Autonoma
              <select
                className="input"
                value={autonomyLevel}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setAutonomyLevel(v);
                  if (v >= 1) setDesignerMode(true);
                  if (v === 0) setDesignerMode(false);
                }}
                style={{ marginLeft: 8 }}
              >
                <option value={0}>0 (solo analisi)</option>
                <option value={1}>1 (default)</option>
                <option value={2}>2 (aggressiva)</option>
              </select>
            </label>

            <label className="small">
              Obiettivo
              <select className="input" value={goal} onChange={(e) => setGoal(e.target.value)} style={{ marginLeft: 8 }}>
                <option value="balanced">bilanciato</option>
                <option value="min_qty">min quantità</option>
                <option value="min_power">min potenza</option>
              </select>
            </label>

            <label className="small" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input type="checkbox" checked={designerMode} onChange={(e) => setDesignerMode(e.target.checked)} />
              Designer layer
            </label>

            <label className="small" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input type="checkbox" checked={daylightMode} onChange={(e) => setDaylightMode(e.target.checked)} />
              Daylight (solo se richiesto)
            </label>
          </div>

          <button className="btnP" onClick={analyze} disabled={!file || busy}>
            {busy ? "Analisi…" : "Analizza"}
          </button>
        </div>

        {previewUrl && (
          <div style={{ marginTop: 12 }}>
            <div className="small">{file?.name}</div>
            {file?.type !== "application/pdf" ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="preview" style={{ maxWidth: "100%", borderRadius: 12, marginTop: 10 }} />
            ) : (
              <a className="small" href={previewUrl} target="_blank" rel="noreferrer">
                Apri PDF
              </a>
            )}
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 800, fontSize: 16 }}>Chat LuxIA</div>
        <div className="small" style={{ marginBottom: 8 }}>
          Chiedimi qualsiasi cosa sul progetto (UNI, scelte apparecchi, alternative, ecc.).
        </div>
        <div style={{ maxHeight: 220, overflow: "auto", border: "1px solid #ddd", borderRadius: 10, padding: 10 }}>
          {chatHistory.length === 0 ? (
            <div className="small">Scrivi un messaggio per iniziare.</div>
          ) : (
            chatHistory.map((m, idx) => (
              <div key={idx} style={{ marginBottom: 8 }}>
                <div className="small" style={{ opacity: 0.7 }}>{m.role === "user" ? "Tu" : "LuxIA"}</div>
                <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
              </div>
            ))
          )}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <input className="inp" value={chatMsg} onChange={(e) => setChatMsg(e.target.value)} placeholder="Scrivi qui…" />
          <button className="btnG" onClick={sendChat}>Invia</button>
        </div>
      </div>

      {err && (
        <div className="card">
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Errore</div>
          <div className="p" style={{ margin: 0 }}>{err}</div>
        </div>
      )}

      {resp && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontWeight: 800, fontSize: 16 }}>Esito verifica</div>
              <div className="small">
                Provider: {resp.vision?.provider || "n/d"} • Confidenza: {(resp.vision?.confidence ?? 0).toFixed(2)}
              </div>
            </div>
            <div className="row">
              {pass && (
                <>
                  <button className="btnP" onClick={downloadPdf} disabled={!resp.report_pdf_base64}>Scarica Report PDF</button>
                  <button className="btnG" onClick={downloadDxf} disabled={!(resp.dxf_out_base64 || resp.extracted?.dxf_out_base64)}>Scarica DXF</button>
                </>
              )}
              {!pass && (
                <button className="btnP" onClick={() => { setResp(null); setErr(null); }}>
                  Carica versione corretta
                </button>
              )}
            </div>
          </div>

          {!pass && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontWeight: 700, marginBottom: 8 }}>Cosa non torna</div>
              <ul className="list">
                {issues.length ? issues.map((it: any, i: number) => (
                  <li key={i}>{it.detail || JSON.stringify(it)}</li>
                )) : <li>Verifica non superata.</li>}
              </ul>
            </div>
          )}

          {pass && (
            <div style={{ marginTop: 14 }} className="small">
              Conforme: {pass ? "✅" : "❌"} • Livelli: {resp.extracted?.levels?.length || 1} • Ambienti: {resp.extracted?.rooms?.length || 0} • Apparecchi: {resp.extracted?.lights?.length || 0}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
