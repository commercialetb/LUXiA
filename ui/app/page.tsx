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
  const [busy, setBusy] = useState(false);
  const [resp, setResp] = useState<AnalyzeResp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    setErr(null);
    setResp(null);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("options", JSON.stringify({
      write_lights_dxf: true,
      verify: true,
      overlay: true,
      multilevel: true
    }));

    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await r.json().catch(() => null);
    if (!r.ok || !data?.ok) {
      setErr(data?.error || "Errore analisi");
    } else {
      setResp(data);
    }
    setBusy(false);
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
        <div className="row">
          <input
            type="file"
            accept="application/pdf,image/*,.dxf,.dwg"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
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
              <a className="small" href={previewUrl} target="_blank" rel="noreferrer">Apri PDF</a>
            )}
          </div>
        )}
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
                <button className="btnP" onClick={() => { setResp(null); setErr(null); }} >Carica versione corretta</button>
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
