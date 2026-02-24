"use client";
import { useEffect, useState } from "react";

const themes = [
  { v: "dark_elegant", label: "Dark Elegant" },
  { v: "clean_office", label: "Clean Office" },
  { v: "retail_high_contrast", label: "Retail High Contrast" },
  { v: "hospitality_soft", label: "Hospitality Soft" },
];

export default function ProjectStylePackClient({ projectId }) {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [pack, setPack] = useState({
    mood: "",
    cct_default: 3000,
    contrast_level: "medium",
    accent_ratio: 0.20,
    uniformity_target: 0.60,
    density_bias: 1.00,
    presentation_theme: "clean_office",
  });

  async function load() {
    setLoading(true);
    setMsg("");
    try {
      const res = await fetch(`/api/projects/${projectId}/style-pack`, { cache: "no-store" });
      const j = await res.json();
      if (j.ok && j.pack) setPack((p) => ({ ...p, ...j.pack }));
    } catch (e) {} finally {
      setLoading(false);
    }
  }

  async function save() {
    setSaving(true);
    setMsg("");
    try {
      const res = await fetch(`/api/projects/${projectId}/style-pack`, {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ pack })
      });
      const j = await res.json();
      if (!j.ok) throw new Error(j.error || "save failed");
      setMsg("✅ Salvato");
    } catch (e) {
      setMsg("❌ " + String(e?.message || e));
    } finally {
      setSaving(false);
    }
  }

  useEffect(() => { load(); }, [projectId]);

  return (
    <div style={{ background:"#fff", border:"1px solid #e5e7eb", borderRadius: 14, padding: 14, marginTop: 14 }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap: 12, flexWrap:"wrap" }}>
        <div>
          <h3 style={{ margin:0 }}>🎛️ Style Pack (Progetto)</h3>
          <p style={{ margin:"6px 0 0 0", fontSize: 13, color:"#6b7280" }}>
            Parametri progettuali per questo progetto: LuxIA li usa per concept, layout e rendering.
          </p>
        </div>
        <div style={{ display:"flex", gap: 10, alignItems:"center" }}>
          <button disabled={saving || loading} onClick={save}
            style={{ padding:"10px 12px", borderRadius: 12, border:"1px solid #0ea5e9", background:"#0ea5e9", color:"#fff", fontWeight:700 }}>
            {saving ? "Salvo…" : "💾 Salva"}
          </button>
          <button disabled={saving || loading} onClick={load}
            style={{ padding:"10px 12px", borderRadius: 12, border:"1px solid #e5e7eb", background:"#fff" }}>
            ↻ Ricarica
          </button>
        </div>
      </div>

      <div style={{ marginTop: 12, display:"grid", gridTemplateColumns:"repeat(3, 1fr)", gap: 12 }}>
        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>CCT default</b>
          <div style={{ fontSize: 12, color:"#6b7280", marginTop: 6 }}>{pack.cct_default} K</div>
          <input type="range" min="2700" max="5000" step="100" value={pack.cct_default}
            onChange={(e)=>setPack({ ...pack, cct_default: Number(e.target.value) })}
            style={{ width:"100%", marginTop: 10 }}
          />
        </div>

        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>Density bias</b>
          <div style={{ fontSize: 12, color:"#6b7280", marginTop: 6 }}>{Number(pack.density_bias).toFixed(2)}×</div>
          <input type="range" min="0.80" max="1.30" step="0.01" value={pack.density_bias}
            onChange={(e)=>setPack({ ...pack, density_bias: Number(e.target.value) })}
            style={{ width:"100%", marginTop: 10 }}
          />
        </div>

        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>Uniformità target</b>
          <div style={{ fontSize: 12, color:"#6b7280", marginTop: 6 }}>{Number(pack.uniformity_target).toFixed(2)}</div>
          <input type="range" min="0.30" max="0.80" step="0.01" value={pack.uniformity_target}
            onChange={(e)=>setPack({ ...pack, uniformity_target: Number(e.target.value) })}
            style={{ width:"100%", marginTop: 10 }}
          />
        </div>

        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>Accent ratio</b>
          <div style={{ fontSize: 12, color:"#6b7280", marginTop: 6 }}>{Math.round(Number(pack.accent_ratio)*100)}%</div>
          <input type="range" min="0.00" max="0.50" step="0.01" value={pack.accent_ratio}
            onChange={(e)=>setPack({ ...pack, accent_ratio: Number(e.target.value) })}
            style={{ width:"100%", marginTop: 10 }}
          />
        </div>

        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>Contrasto</b>
          <select value={pack.contrast_level} onChange={(e)=>setPack({ ...pack, contrast_level: e.target.value })}
            style={{ width:"100%", marginTop: 10, padding:"10px 12px", borderRadius: 12, border:"1px solid #e5e7eb" }}>
            <option value="low">Low (soft)</option>
            <option value="medium">Medium</option>
            <option value="high">High (dramatic)</option>
          </select>
        </div>

        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>Tema presentazione</b>
          <select value={pack.presentation_theme} onChange={(e)=>setPack({ ...pack, presentation_theme: e.target.value })}
            style={{ width:"100%", marginTop: 10, padding:"10px 12px", borderRadius: 12, border:"1px solid #e5e7eb" }}>
            {themes.map(t => <option key={t.v} value={t.v}>{t.label}</option>)}
          </select>
        </div>
      </div>

      <div style={{ marginTop: 10, fontSize: 13, color: msg.startsWith("✅") ? "#16a34a" : (msg.startsWith("❌") ? "#dc2626" : "#6b7280") }}>
        {msg || " "}
      </div>
    </div>
  );
}
