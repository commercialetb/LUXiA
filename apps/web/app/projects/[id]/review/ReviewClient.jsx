"use client";
import { useEffect, useMemo, useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/browser";
import VoiceAssistant from "@/components/VoiceAssistant";

function badge(text) {
  return <span style={{ padding:"2px 8px", border:"1px solid #24304a", borderRadius:999, fontSize:12, opacity:0.9 }}>{text}</span>;
}

async function downloadExport(kind, payload, token) {
  const baseUrl = process.env.NEXT_PUBLIC_ENGINE_URL;
  const url = baseUrl + (kind === "dxf" ? "/exports/layout.dxf" : "/exports/layout.json");
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-LuxIA-Token": token },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const t = await res.text();
    alert("Export failed: " + t);
    return;
  }
  const blob = await res.blob();
  const a = document.createElement("a");
  const ext = kind === "dxf" ? "dxf" : "json";
  a.href = URL.createObjectURL(blob);
  a.download = `luxia_layout.${ext}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}


async function downloadPPT(projectId) {
  const res = await fetch(`/api/exports/ppt?project_id=${projectId}`);
  if (!res.ok) {
    const t = await res.text();
    alert("PPT export failed: " + t);
    return;
  }
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `LuxIA_${projectId}.pptx`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}

export default function ReviewClient({ project }) {
  const supabase = supabaseBrowser();

  const [areas, setAreas] = useState([]);
  const [concepts, setConcepts] = useState([]);
  const [msg, setMsg] = useState("");
  const [speakText, setSpeakText] = useState("");
  const [saving, setSaving] = useState(false);
  const [styles, setStyles] = useState([]);
  const [activeStyleId, setActiveStyleId] = useState(project.active_style_id || "");

  const [feedback, setFeedback] = useState({}); // area_id -> text
  const [choice, setChoice] = useState({}); // area_id -> concept_id

  const grouped = useMemo(() => {
    const byArea = {};
    for (const a of areas) byArea[a.id] = { area: a, concepts: [] };
    for (const c of concepts) {
      if (!byArea[c.area_id]) continue;
      byArea[c.area_id].concepts.push(c);
    }
    return Object.values(byArea);
  }, [areas, concepts]);

  async function loadAll() {
    setMsg("");
    // load styles
    try {
      const res = await fetch(`/api/designer/styles?tenantId=${project.tenant_id}`);
      const j = await res.json();
      if (j.ok) setStyles(j.styles || []);
    } catch (e) {
      // ignore
    }

    const { data: a, error: e1 } = await supabase
      .from("areas")
      .select("*")
      .eq("project_id", project.id)
      .order("created_at", { ascending: true });

    if (e1) return setMsg("Errore aree: " + e1.message);
    setAreas(a || []);

    const areaIds = (a || []).map(x => x.id);
    if (!areaIds.length) { setConcepts([]); return; }

    const { data: c, error: e2 } = await supabase
      .from("concepts")
      .select("id, area_id, concept_type, solution, metrics, renders, created_at")
      .in("area_id", areaIds)
      .order("created_at", { ascending: true });

    if (e2) return setMsg("Errore concept: " + e2.message);
    setConcepts(c || []);

    // preload decisions (optional)
    const { data: d } = await supabase
      .from("decisions")
      .select("area_id, chosen_concept_id, feedback_text")
      .eq("project_id", project.id);

    const ch = {};
    const fb = {};
    for (const r of (d || [])) {
      if (r.chosen_concept_id) ch[r.area_id] = r.chosen_concept_id;
      if (r.feedback_text) fb[r.area_id] = r.feedback_text;
    }
    setChoice(ch);
    setFeedback(fb);
  }

  useEffect(() => { loadAll(); }, [project.id]);

  async function saveDecisions() {
    setMsg("");
    setSaving(true);
    try {
      const { data: auth } = await supabase.auth.getUser();
      const user = auth?.user;
      if (!user) throw new Error("Devi essere loggato.");

      const rows = [];
      for (const a of areas) {
        const cid = choice[a.id];
        const fb  = (feedback[a.id] || "").trim();
        if (!cid && !fb) continue;
        rows.push({
          tenant_id: project.tenant_id,
          project_id: project.id,
          area_id: a.id,
          chosen_concept_id: cid || null,
          feedback_text: fb || null,
          edits: {},
          created_by: user.id,
        });
      }

      if (!rows.length) {
        setSaving(false);
        return setMsg("Niente da salvare: scegli almeno un concept o scrivi feedback.");
      }

      const { error: e1 } = await supabase.from("decisions").insert(rows);
      if (e1) throw new Error(e1.message);

// v18: record learning events into Designer Brain (Team)
for (const a of areas) {
  const cid = choice[a.id];
  if (!cid) continue;
  try {
    await fetch("/api/designer/select-concept", {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ projectId: project.id, areaId: a.id, conceptId: cid })
    });
  } catch (e) {
    // ignore (non-blocking)
  }
}

      // lightweight learning into studio_profile.style_tokens
      const counts = { comfort: 0, efficiency: 0, architectural: 0 };
      for (const a of areas) {
        const cid = choice[a.id];
        if (!cid) continue;
        const c = concepts.find(x => x.id === cid);
        if (!c) continue;
        counts[c.concept_type] = (counts[c.concept_type] || 0) + 1;
      }

      const { data: sp } = await supabase
        .from("studio_profile")
        .select("tenant_id, style_tokens")
        .eq("tenant_id", project.tenant_id)
        .maybeSingle();

      const current = sp?.style_tokens || {};
      const next = {
        ...current,
        concept_votes: {
          comfort: (current?.concept_votes?.comfort || 0) + counts.comfort,
          efficiency: (current?.concept_votes?.efficiency || 0) + counts.efficiency,
          architectural: (current?.concept_votes?.architectural || 0) + counts.architectural,
        },
        last_project_id: project.id,
        last_saved_at: new Date().toISOString(),
      };

      const { error: e2 } = await supabase
        .from("studio_profile")
        .upsert({
          tenant_id: project.tenant_id,
          style_tokens: next,
          updated_at: new Date().toISOString(),
        }, { onConflict: "tenant_id" });

      if (e2) throw new Error("Decisioni ok, ma update studio_profile fallito: " + e2.message);

      setMsg("✅ Decisioni salvate. LuxIA ha aggiornato il profilo stile dello studio.");
    } catch (e) {
      setMsg("Errore: " + e.message);
    } finally {
      setSaving(false);
    }
  }

  const hasConcepts = concepts.length > 0;

  return (
    <section style={card()}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", gap: 12, flexWrap:"wrap" }}>
        <div>
          <h2 style={{ margin: 0 }}>Review Concept & Learning</h2>
          <div style={{ opacity: 0.85, fontSize: 13 }}>
            Seleziona l’opzione per ogni area e aggiungi feedback: LuxIA impara.
          </div>
        </div>
        <div style={{ display:"flex", gap:10, alignItems:"center", flexWrap:"wrap" }}>
  <span style={{ fontSize: 12, opacity: 0.9 }}>Stile attivo:</span>
  <select
    value={activeStyleId}
    onChange={async (e) => {
      const sid = e.target.value;
      setActiveStyleId(sid);
      await fetch("/api/designer/styles", {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ tenantId: project.tenant_id, projectId: project.id, action: "set_active", styleId: sid })
      });
    }}
    style={{ padding:"8px 10px", borderRadius: 10, border:"1px solid #24304a", background:"#020617", color:"#e5e7eb", minWidth: 240 }}
  >
    <option value="">(Team Default)</option>
    {(styles || []).map(s => (
      <option key={s.id} value={s.id}>{s.is_default ? "⭐ " : ""}{s.name}</option>
    ))}
  </select>
  <button
    onClick={async ()=>{
      const clientName = prompt("Nome cliente/stile (es: TELEDIFESA, Hotel X)");
      if (!clientName) return;
      const res = await fetch("/api/designer/styles", {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ tenantId: project.tenant_id, projectId: project.id, action: "create_client", clientName })
      });
      const j = await res.json();
      if (j.ok && j.style?.id) {
        setActiveStyleId(j.style.id);
        await loadAll();
      }
    }}
    style={btn("#0ea5e9")}
  >
    ➕ Nuovo stile cliente
  </button>
</div>

<button disabled={saving} onClick={saveDecisions} style={btn(saving ? "#334155" : "#16a34a")}>
          {saving ? "Salvataggio..." : "Salva scelte (Learning)"}
        </button>
      </div>

      {!areas.length && (
        <div style={{ marginTop: 12, opacity: 0.9 }}>
          Nessuna area. Torna alla pagina progetto e aggiungi le aree.
        </div>
      )}

      {areas.length > 0 && !hasConcepts && (
        <div style={{ marginTop: 12, opacity: 0.9 }}>
          Nessun concept ancora. Torna alla pagina progetto e clicca “Genera 3 concept”.
        </div>
      )}

      <div style={{ marginTop: 12, display:"grid", gap: 12 }}>
        {grouped.map(({ area, concepts: cs }) => (
          <div key={area.id} style={{ padding: 12, border:"1px solid #24304a", borderRadius: 14, background:"#0b1530" }}>
            <div style={{ display:"flex", justifyContent:"space-between", gap:10, flexWrap:"wrap" }}>
              <div>
                <div style={{ fontWeight: 900, fontSize: 15 }}>{area.name}</div>
                <div style={{ opacity: 0.85, fontSize: 13 }}>
                  {badge(area.tipo_locale)} <span style={{ marginLeft: 8, opacity:0.85 }}>{Number(area.superficie_m2).toFixed(1)} m² • h {Number(area.altezza_m).toFixed(2)} m</span>
                </div>
              </div>
              <div style={{ alignSelf:"center", opacity: 0.8, fontSize: 12 }}>
                Selezionato: <b>{choice[area.id] ? "✅" : "—"}</b>
              </div>
            </div>

            <div style={{ marginTop: 10, display:"grid", gridTemplateColumns:"repeat(3, 1fr)", gap:10 }}>
              {cs.slice(0,3).map((c) => {
                const selected = choice[area.id] === c.id;
                const title =
                  c.concept_type === "comfort" ? "Comfort" :
                  c.concept_type === "efficiency" ? "Efficienza" : "Architetturale";
                const notes = c.solution?.notes || "";
                const lum = c.solution?.luminaire?.name || c.solution?.luminaire?.id || "";
                const Em = c.metrics?.Em ?? null;
                const wm2 = c.metrics?.wm2 ?? null;
                return (
                  <button
                    key={c.id}
                    onClick={() => setChoice(prev => ({ ...prev, [area.id]: c.id }))}
                    style={{
                      textAlign:"left",
                      padding:12,
                      borderRadius: 12,
                      border: selected ? "2px solid #60a5fa" : "1px solid #24304a",
                      background: selected ? "#0f1a33" : "#0b1220",
                      color:"#e5e7eb",
                      cursor:"pointer",
                    }}
                  >
                    <div style={{ fontWeight: 900 }}>{title}</div>
                    <div style={{ marginTop: 6, opacity: 0.9, fontSize: 13 }}>{notes}</div>
                    {lum && <div style={{ marginTop: 8, fontSize: 12, opacity: 0.85 }}>🔦 {lum}</div>}
                    {(Em !== null || wm2 !== null) && (
                      <div style={{ marginTop: 4, fontSize: 12, opacity: 0.85 }}>
                        📏 Em≈{Em ?? "—"} lux • ⚡ {wm2 ?? "—"} W/m²
                      </div>
                    )}
                    <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>
                      (Step successivo: calcoli, render, norme complete)
                    </div>
                  </button>
                );
              })}
            </div>

            <div style={{ marginTop: 10 }}>
              <label style={lab()}>Feedback</label>
              <textarea
                value={feedback[area.id] || ""}
                onChange={(e)=>setFeedback(prev=>({ ...prev, [area.id]: e.target.value }))}
                placeholder="Es: Opzione 2 ma 3000K e più uniforme vicino alle postazioni VDT..."
                style={{ ...inp(), minHeight: 80 }}
              />
            </div>
          </div>
        ))}
      </div>

      {msg && <div style={{ marginTop: 12, color: "#fde68a" }}>{msg}</div>}

      <div style={{ marginTop: 14, display:"flex", gap:10, flexWrap:"wrap", alignItems:"center" }}>
  <button
    onClick={() => downloadPPT(project.id)}
    disabled={!hasConcepts}
    style={btn(!hasConcepts ? "#334155" : "#2563eb")}
    title={!hasConcepts ? "Genera prima i concept" : "Esporta presentazione"}
  >
    ⬇️ Scarica PPTX (Concept)
  </button>
  <span style={{ fontSize:12, opacity:0.8 }}>
    (v13) Include riepilogo aree + 3 concept per area. Rendering photoreal arriverà in v15.
  </span>
</div>

<div style={{ marginTop: 14, display:"flex", gap:10, flexWrap:"wrap" }}>
        <a href={`/projects/${project.id}`} style={{ color:"#93c5fd" }}>← Torna al progetto</a>
        <a href="/" style={{ color:"#93c5fd" }}>Home</a>
      </div>
    </section>
  );
}

const card = ()=>({ background:"#0f172a", border:"1px solid #24304a", borderRadius:14, padding:14 });
const inp  = ()=>({ width:"100%", padding:10, borderRadius:10, border:"1px solid #24304a", background:"#020617", color:"#e5e7eb" });
const btn  = (c)=>({ background:c, border:"none", padding:"10px 14px", borderRadius:10, color:"white", fontWeight:900, cursor:"pointer" });
const lab  = ()=>({ fontSize:12, opacity:0.8, display:"block", marginBottom:6 });
