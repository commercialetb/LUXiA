"use client";
import { useEffect, useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/browser";
import VoiceAssistant from "@/components/VoiceAssistant";
import { engineApi } from "@/lib/engine";

export default function AreaClient({ project }) {
  const supabase = supabaseBrowser();
  const [areas, setAreas] = useState([]);
  const [msg, setMsg] = useState("");
  const [projectBrands, setProjectBrands] = useState([]);
  const [allBrands, setAllBrands] = useState([]);
  const [catalog, setCatalog] = useState([]);
  const [constraints, setConstraints] = useState("");


  // new area fields
  const [name, setName] = useState("");
  const [tipo, setTipo] = useState("Ufficio VDT");
  const [sup, setSup] = useState(30);
  const [h, setH] = useState(2.7);

async function loadBrandsCatalog() {
  setMsg("");
  const { data: pb, error: e1 } = await supabase
    .from("project_brands")
    .select("brand")
    .eq("project_id", project.id);

  if (e1) return setMsg("Errore brand progetto: " + e1.message);
  const selected = (pb || []).map(x => x.brand);
  setProjectBrands(selected);

  const { data: lum, error: e2 } = await supabase
    .from("luminaires")
    .select("id, brand, model, family, type, mounting, distribution, flux_lm, watt, cct, cri, ugr, ip, dimmable, notes")
    .eq("tenant_id", project.tenant_id);

  if (e2) return setMsg("Errore catalogo: " + e2.message);
  setCatalog(lum || []);
  const brands = Array.from(new Set((lum || []).map(x => x.brand))).sort();
  setAllBrands(brands);
}

async function toggleBrand(brand) {
  setMsg("");
  const has = projectBrands.includes(brand);
  if (has) {
    const { error } = await supabase
      .from("project_brands")
      .delete()
      .eq("project_id", project.id)
      .eq("brand", brand);
    if (error) return setMsg("Errore rimozione brand: " + error.message);
    setProjectBrands(prev => prev.filter(b => b !== brand));
  } else {
    const { error } = await supabase
      .from("project_brands")
      .insert({ project_id: project.id, tenant_id: project.tenant_id, brand });
    if (error) return setMsg("Errore aggiunta brand: " + error.message);
    setProjectBrands(prev => [...prev, brand]);
  }
}

async function loadAreas() {
    const { data, error } = await supabase.from("areas").select("*").eq("project_id", project.id).order("created_at", { ascending: true });
    if (error) return setMsg("Errore aree: " + error.message);
    setAreas(data || []);
  }

  useEffect(() => { loadAreas(); loadBrandsCatalog(); }, [project.id]);

  async function addArea(e) {
    e.preventDefault();
    setMsg("");
    if (!name.trim()) return setMsg("Nome area mancante.");
    const { error } = await supabase.from("areas").insert({
      project_id: project.id,
      name,
      tipo_locale: tipo,
      superficie_m2: Number(sup),
      altezza_m: Number(h),
      vdt: (tipo === "Ufficio VDT"),
    });
    if (error) return setMsg("Errore insert area: " + error.message);
    setName("");
    await loadAreas();
  }
const designer_area_bias = (
  project?.designer_area_bias ??
  project?.stylepack?.designer_area_bias ??
  {}
);
  async function generateConcepts(priority = "mix") {
    setMsg("");
    if (!areas.length) return setMsg("Aggiungi prima almeno un'area.");
    try {
      // call Engine (placeholder) to get 3 concept per area
      // Fetch studio_profile to make concepts style-aware (learning)
      const { data: sp } = await supabase.from("studio_profile").select("style_tokens").eq("tenant_id", project.tenant_id).maybeSingle();
      const style_tokens = sp?.style_tokens || {};
// Fetch Project Style Pack (editable sliders)
let project_style_pack = null;
try {
  const res = await fetch(`/api/projects/${project.id}/style-pack`, { cache: "no-store" });
  const j = await res.json();
  project_style_pack = j?.pack || null;
} catch (e) {
  project_style_pack = null;
}

// Fetch Designer Brain profile (team DNA) for active style
let designer_stats = null;
if (project.active_style_id) {
  const { data: dp } = await supabase
    .from("designer_team_profile")
    .select("stats")
    .eq("tenant_id", project.tenant_id)
    .eq("style_id", project.active_style_id)
    .maybeSingle();
  designer_stats = dp?.stats || null;
}

      const payload = {
        areas: areas.map(a => ({ name: a.name, tipo_locale: a.tipo_locale, superficie_m2: Number(a.superficie_m2), altezza_m: Number(a.altezza_m) })),
        style_tokens,
        allowed_brands: projectBrands,
        catalog,
        priority,
        constraints,
        designer_stats,
        designer_area_bias,
      };
      const r = await engineApi(`/projects/${project.id}/concepts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      // Persist basic concepts (placeholder rows) in DB
      const inserts = [];
      for (const block of (r.result || [])) {
        const areaRow = areas.find(x => x.name === block.area) || areas.find(x => x.name.startsWith(block.area)) || null;
        if (!areaRow) continue;
        for (const c of (block.concepts || [])) {
          const mapType = c.concept_type || (c.name.toLowerCase().includes("comfort") ? "comfort" :
                          c.name.toLowerCase().includes("eff") ? "efficiency" : "architectural");
          inserts.push({
            area_id: areaRow.id,
            concept_type: mapType,
            solution: { 
              notes: c.notes, 
              engine_id: c.id, 
              concept_type: mapType,
              luminaire: c.luminaire || null,
              style_bias: c.style_bias,
              style_snapshot: payload.style_tokens || {},
            },
            metrics: c.calc || {},
            renders: {}
          });
        }
      }
      if (inserts.length) {
        const { error } = await supabase.from("concepts").insert(inserts);
        if (error) return setMsg("Errore salvataggio concept: " + error.message);
      }
      setMsg("Concept generati e salvati. (Step successivo: calcoli+render+scelta+learning)");
    } catch (e) {
      setMsg("Errore Engine: " + e.message);
    }
  }

  return (
    <section style={card()}>
      <h2 style={{ marginTop: 0 }}>Aree</h2>

      <form onSubmit={addArea} style={{ display:"grid", gap:8, gridTemplateColumns:"2fr 1.2fr 0.8fr 0.8fr auto", alignItems:"end" }}>
        <div>
          <label style={lab()}>Nome</label>
          <input value={name} onChange={(e)=>setName(e.target.value)} style={inp()} placeholder="Es. V1 Ufficio operativo" />
        </div>
        <div>
          <label style={lab()}>Tipo locale</label>
          <select value={tipo} onChange={(e)=>setTipo(e.target.value)} style={inp()}>
            <option>Ufficio VDT</option>
            <option>Sala riunioni</option>
            <option>Corridoio</option>
            <option>Reception</option>
            <option>Bagno/WC</option>
            <option>Archivio</option>
            <option>Ingresso</option>
            <option>Mensa/Ristoro</option>
            <option>Locale tecnico</option>
          </select>
        </div>
        <div>
          <label style={lab()}>m²</label>
          <input type="number" step="0.1" value={sup} onChange={(e)=>setSup(e.target.value)} style={inp()} />
        </div>
        <div>
          <label style={lab()}>h (m)</label>
          <input type="number" step="0.05" value={h} onChange={(e)=>setH(e.target.value)} style={inp()} />
        </div>
        <button style={btn("#16a34a")} type="submit">+ Area</button>
      </form>

<div style={{ marginTop: 10 }}>
  <div style={{ padding: 12, border:"1px solid #24304a", borderRadius: 14, background:"#0b1530", marginBottom: 12 }}>
    <div style={{ fontWeight: 900, marginBottom: 6 }}>Brand attivi nel progetto (multi-brand)</div>
    <div style={{ opacity: 0.85, fontSize: 13, marginBottom: 10 }}>
      Seleziona i brand da cui LuxIA può scegliere i prodotti. Se non selezioni nulla, LuxIA genera concept “brand-neutral”.
    </div>
    <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
      {allBrands.length ? allBrands.map(b => (
        <button key={b} onClick={() => toggleBrand(b)} style={{
          padding:"6px 10px", borderRadius:999,
          border: projectBrands.includes(b) ? "2px solid #60a5fa" : "1px solid #24304a",
          background: projectBrands.includes(b) ? "#0f1a33" : "#0b1220",
          color:"#e5e7eb", cursor:"pointer"
        }}>{b}</button>
      )) : <span style={{ opacity:0.85 }}>Nessun catalogo ancora. (Aggiungi luminaires nel DB)</span>}
    </div>
  </div>

        {!areas.length ? (
          <div style={{ opacity: 0.85 }}>Nessuna area inserita.</div>
        ) : (
          <table style={{ width:"100%", marginTop: 10, borderCollapse:"collapse" }}>
            <thead>
              <tr>
                <th style={th()}>Nome</th>
                <th style={th()}>Tipo</th>
                <th style={th()}>m²</th>
                <th style={th()}>h</th>
              </tr>
            </thead>
            <tbody>
              {areas.map(a => (
                <tr key={a.id}>
                  <td style={td()}>{a.name}</td>
                  <td style={td()}>{a.tipo_locale}</td>
                  <td style={td()}>{Number(a.superficie_m2).toFixed(1)}</td>
                  <td style={td()}>{Number(a.altezza_m).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div style={{ marginTop: 12, display:"flex", gap:10 }}>
        <button style={btn("#2563eb")} onClick={generateConcepts}>Genera 3 concept (Engine)</button>
        <a href={`/projects/${project.id}/review`} style={{ alignSelf:"center", color:"#93c5fd" }}>🧠 Review & Learning</a>
        <a href="/" style={{ alignSelf:"center", color:"#93c5fd" }}>← Home</a>
      </div>

      {msg && <div style={{ marginTop: 10, color: "#fde68a" }}>{msg}</div>}
    </section>
  );
}

const card = ()=>({ background:"#0f172a", border:"1px solid #24304a", borderRadius:14, padding:14 });
const inp  = ()=>({ width:"100%", padding:10, borderRadius:10, border:"1px solid #24304a", background:"#020617", color:"#e5e7eb" });
const btn  = (c)=>({ background:c, border:"none", padding:"10px 14px", borderRadius:10, color:"white", fontWeight:800, cursor:"pointer" });
const lab  = ()=>({ fontSize:12, opacity:0.8, display:"block", marginBottom:4 });
const th   = ()=>({ textAlign:"left", padding:"8px 6px", borderBottom:"1px solid #24304a", fontSize:12, opacity:0.8 });
const td   = ()=>({ padding:"8px 6px", borderBottom:"1px solid #1f2a44", fontSize:13 });
