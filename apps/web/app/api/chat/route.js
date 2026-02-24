export const runtime = "nodejs";
import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";
import { engineApi } from "@/lib/engine";
import PptxGenJS from "pptxgenjs";
import { createClient } from "@supabase/supabase-js";

function supabaseRoute() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anon) throw new Error("Missing Supabase env vars");
  const cookieStore = cookies();
  return createServerClient(url, anon, {
    cookies: {
      get(name) { return cookieStore.get(name)?.value; },
      set(name, value, options) { cookieStore.set({ name, value, ...options }); },
      remove(name, options) { cookieStore.set({ name, value: "", ...options, maxAge: 0 }); },
    },
  });
}


function supabaseAdmin() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceKey) throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY");
  return createClient(url, serviceKey);
}

function wantsRender(message) {
  const t = String(message || "").toLowerCase();
  return t.includes("render") || t.includes("fotoreal") || t.includes("fotorealist") || t.includes("immagine realistica");
}

function wantsPptx(text) {
  const t = (text || "").toLowerCase();
  return t.includes("ppt") || t.includes("pptx") || t.includes("presentazione") || t.includes("powerpoint");
}

async function buildPptxBuffer({ project, areas, conceptsByArea }) {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "LuxIA";
  pptx.company = "LuxIA";
  pptx.title = `LuxIA – ${project.name}`;

  const TITLE = "1A365D";
  const ACCENT = "2B6CB0";
  const GRAY = "4B5563";

function drawMiniPlan(slide, box, area, concept) {
  try {
    const { x, y, w, h } = box;
    const sup = Number(area.surface_m2 || area.surface || area.sup || 0) || 0;
    const side = sup > 0 ? Math.sqrt(sup) : 5;
    // frame
    slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, fill:{ color:"F7FAFC" }, line:{ color:"CBD5E0", width:1 }});
    // inner room rect with padding
    const pad = 0.18;
    const rx = x + pad, ry = y + pad, rw = Math.max(0.1, w - 2*pad), rh = Math.max(0.1, h - 2*pad);
    slide.addShape(pptx.ShapeType.rect, { x:rx, y:ry, w:rw, h:rh, fill:{ color:"FFFFFF" }, line:{ color:"A0AEC0", width:1 }});
    const coords = (concept?.calc?.coords || concept?.coords || []);
    // scale coords from meters to box
    const sx = rw / side;
    const sy = rh / side;
    for (const c of coords) {
      const lx = Number(c[0] ?? c.x ?? 0);
      const ly = Number(c[1] ?? c.y ?? 0);
      const px = rx + lx * sx;
      const py = ry + ly * sy;
      slide.addShape(pptx.ShapeType.ellipse, {
        x: px - 0.06, y: py - 0.06, w: 0.12, h: 0.12,
        fill:{ color:"FBBF24" },
        line:{ color:"111827", width:0.5 }
      });
    }
    // caption
    const em = concept?.calc?.E_m ?? concept?.E_m;
    const et = concept?.calc?.E_t ?? concept?.E_t;
    slide.addText(`${area.name || "Area"} • Em ${fmt(em)} lux / Et ${fmt(et)}`, {
      x: x+0.12, y: y+h-0.32, w: w-0.24, h: 0.26,
      fontFace:"Aptos", fontSize:10, color:"374151"
    });
  } catch (e) {
    // ignore
  }
}

  // Cover
  {
    const s = pptx.addSlide();
    s.addShape(pptx.ShapeType.rect, { x:0, y:0, w:13.33, h:7.5, fill:{ color: TITLE }});
    s.addText("LuxIA — Concept Lighting", { x:0.7, y:2.2, w:12, h:1, fontFace:"Aptos", fontSize:38, color:"FFFFFF", bold:true });
    s.addText(project.name || "Progetto", { x:0.7, y:3.25, w:12, h:0.6, fontFace:"Aptos", fontSize:20, color:"90CDF4" });
    s.addText(`Data: ${new Date().toLocaleDateString("it-IT")}`, { x:0.7, y:4.0, w:12, h:0.4, fontFace:"Aptos", fontSize:12, color:"BEE3F8" });
  }

  // Summary slide
  {
    const s = pptx.addSlide();
    s.addText("Riepilogo aree", { x:0.6, y:0.3, w:12.2, h:0.6, fontFace:"Aptos", fontSize:26, color: TITLE, bold:true });
    const rows = [["Area","Tipo","m²","h","Concept"]];
    for (const a of areas) {
      const c0 = (conceptsByArea[a.id] || [])[0];
      rows.push([a.name, a.tipo_locale, String(a.superficie_m2), String(a.altezza_m), c0?.concept_type || "—"]);
    }
    s.addTable(rows, {
      x:0.6, y:1.2, w:12.2, h:5.8,
      fontFace:"Aptos", fontSize:12,
      border:{ type:"solid", color:"CBD5E0", pt:1 },
      fill:"F7FAFC",
      colW:[3.4,3.0,1.0,1.0,3.6],
      rowH:0.35,
      color:"111827",
      valign:"mid",
      autoFit:true,
      header:true
    });
  }

  // Per-area slides
  for (const a of areas) {
    const s = pptx.addSlide();
    s.addText(a.name, { x:0.6, y:0.3, w:12.2, h:0.5, fontFace:"Aptos", fontSize:24, color: TITLE, bold:true });
    s.addText(`${a.tipo_locale} • ${a.superficie_m2} m² • h ${a.altezza_m} m`, { x:0.6, y:0.85, w:12.2, h:0.3, fontFace:"Aptos", fontSize:12, color: GRAY });

    const concepts = conceptsByArea[a.id] || [];
    const boxY = 1.35;
    const boxW = 12.2/3 - 0.1;
    for (let i=0; i<3; i++) {
      const c = concepts[i];
      const x = 0.6 + i*(boxW+0.15);
      s.addShape(pptx.ShapeType.roundRect, { x, y:boxY, w:boxW, h:5.8, fill:{ color:"FFFFFF" }, line:{ color:"E5E7EB", width:1 } });
      s.addShape(pptx.ShapeType.rect, { x, y:boxY, w:boxW, h:0.5, fill:{ color: ACCENT }});
      s.addText(c?.concept_type || ["Comfort","Efficienza","Architetturale"][i], { x:x+0.15, y:boxY+0.1, w:boxW-0.3, h:0.3, fontFace:"Aptos", fontSize:12, color:"FFFFFF", bold:true });

      // Mini plan + layout points
      drawMiniPlan(s, { x:x+0.15, y:boxY+0.72, w:boxW-0.3, h:2.55 }, a, c);


      const m = c?.calc || c?.metrics || {};
      const lines = [
        `Lampade: ${m.n ?? "—"}`,
        `Em: ${m.E_m ?? "—"} lux (target ${m.E_t ?? "—"})`,
        `W: ${m.W_t ?? "—"}  |  W/m²: ${m.wm2 ?? "—"}`,
        `UGR ok: ${m.ok_ugr ?? "—"}  |  Ra ok: ${m.ok_ra ?? "—"}`,
      ];
      const sceneNames = (m.scenes || []).map(s=>`${s.name}:${Math.round((s.dimming||1)*100)}%`).slice(0,6);
      if (sceneNames.length) lines.push("Scene: " + sceneNames.join(", "));

      s.addText(lines.join("\n"), { x:x+0.2, y:boxY+0.7, w:boxW-0.4, h:4.8, fontFace:"Aptos", fontSize:11, color:"111827" });

      const prod = c?.fixture?.model || c?.lampada || c?.solution?.lampada || "";
      if (prod) s.addText(String(prod).slice(0,60), { x:x+0.2, y:boxY+5.35, w:boxW-0.4, h:0.35, fontFace:"Aptos", fontSize:9, color: GRAY });
    }
  }

  const buf = await pptx.write("arraybuffer");
  return Buffer.from(buf);
}


function pickBrandsFromText(text, brands) {
  const t = (text || "").toLowerCase();
  const hits = brands.filter(b => t.includes(String(b).toLowerCase()));
  return hits;
}

export async function POST(req) {
  try {
    const body = await req.json();
    const projectId = body.project_id;
    const message = body.message || "";
    if (!projectId || !message.trim()) {
      return NextResponse.json({ ok:false, error:"Missing project_id or message" }, { status: 400 });
    }

    const supabase = supabaseRoute();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return NextResponse.json({ ok:false, error:"Unauthorized" }, { status: 401 });

    // Insert user message
    await supabase.from("project_chat_messages").insert({
      project_id: projectId,
      created_by: user.id,
      role: "user",
      content: message,
      meta: { client: "web", ts: Date.now() },
    });

    
// RAG: retrieve relevant knowledge chunks (full-text) to steer decisions
let ragContext = "";
try {
  const { data: hits } = await supabase.rpc("search_knowledge", {
    q: message,
    p_project_id: projectId,
    max_rows: 6,
  });
  if (hits && hits.length) {
    ragContext = hits.map(h => `- ${h.title || "Doc"}: ${String(h.content || "").slice(0, 700)}`).join("\n\n");
  }
} catch (e) {
  // ignore RAG failures in MVP
  ragContext = "";
}

// Create job
    const { data: job, error: jobErr } = await supabase
      .from("project_jobs")
      .insert({
        project_id: projectId,
        created_by: user.id,
        job_type: "generate_concepts",
        status: "running",
        input: { prompt: message },
        started_at: new Date().toISOString(),
      })
      .select("id")
      .single();

    if (jobErr) throw new Error(jobErr.message);

    // Load project + areas
    const { data: project, error: pErr } = await supabase
      .from("projects")
      .select("id, tenant_id, name, project_type")
      .eq("id", projectId)
      .single();
    if (pErr) throw new Error(pErr.message);

    const { data: areas, error: aErr } = await supabase
      .from("areas")
      .select("id, name, tipo_locale, superficie_m2, altezza_m")
      .eq("project_id", projectId)
      .order("created_at", { ascending: true });
    if (aErr) throw new Error(aErr.message);

    // Load catalog for tenant
    const { data: catalog, error: cErr } = await supabase
      .from("luminaires")
      .select("brand, model, type, mounting, distribution, flux_lm, watt, cri, ugr, cct, dimmable")
      .eq("tenant_id", project.tenant_id)
      .limit(5000);
    if (cErr) throw new Error(cErr.message);

    const distinctBrands = Array.from(new Set((catalog || []).map(r => r.brand).filter(Boolean)));
    let allowedBrands = pickBrandsFromText(message, distinctBrands);

    // Defaults: prefer BEGA if present, else all brands (MVP)
    if (!allowedBrands.length) {
      if (distinctBrands.includes("BEGA")) allowedBrands = ["BEGA"];
      else allowedBrands = distinctBrands.slice(0, 5);
    }

    const catalogPayload = (catalog || []).map(r => ({
      brand: r.brand,
      model: r.model,
      type: r.type,
      mounting: r.mounting,
      distribution: r.distribution,
      flux_lm: r.flux_lm,
      watt: r.watt,
      cri: r.cri,
      ugr: r.ugr,
      cct: r.cct,
      dimmable: r.dimmable,
    }));

    const areasPayload = (areas || []).map(a => ({
      id: a.id,
      name: a.name,
      tipo_locale: a.tipo_locale,
      superficie_m2: Number(a.superficie_m2),
      height_m: Number(a.altezza_m),
    }));

    const engineRes = await engineApi(`/projects/${projectId}/concepts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        areas: areasPayload,
        allowed_brands: allowedBrands,
        catalog: catalogPayload,
        style_tokens: {},
        priority: "mix",
        constraints: ragContext ? (message + "\n\n[KNOWLEDGE]\n" + ragContext) : message,
      }),
    });

    // Persist concepts per area (replace existing)
    for (const block of (engineRes.result || [])) {
      const areaName = block.area;
      const areaRow = (areas || []).find(a => a.name === areaName) || null;
      const areaId = areaRow?.id || null;
      if (!areaId) continue;

      // wipe previous concepts for the area (MVP)
      await supabase.from("concepts").delete().eq("area_id", areaId);

      for (const c of (block.concepts || [])) {
        await supabase.from("concepts").insert({
          area_id: areaId,
          concept_type: c.concept_type,
          solution: c,
          metrics: c.calc || {},
          renders: { scenes: (c.calc && c.calc.scenes) ? c.calc.scenes : [] },
        });
      }
    }

    
// Enqueue Blender render jobs (v16) on demand
if (wantsRender) {
  try {
    const mood = /hospital/i.test(message) ? "hospitality_warm"
                : /retail/i.test(message) ? "retail_bright"
                : /industr/i.test(message) ? "industrial_raw"
                : /minimal/i.test(message) ? "minimal_white"
                : "office_clean";
    const quality = /high|alta/i.test(message) ? "high" : (/draft|bozza/i.test(message) ? "draft" : "medium");
    for (const block of (engineRes.result || [])) {
      const areaName = block.area;
      const areaRow = (areas || []).find(a => a.name === areaName) || null;
      const areaId = areaRow?.id || null;
      if (!areaId) continue;

      const { data: cRows } = await supabase
  .from("concepts")
  .select("id, concept_type, solution, metrics")
  .eq("area_id", areaId);

const conceptsToRender = (cRows || []).filter(x => ["comfort","efficienza","architetturale"].some(k => String(x.concept_type||"").toLowerCase().includes(k)));
if (!conceptsToRender.length) continue;


      for (const chosen of conceptsToRender) {
          const calc = chosen.solution?.calc || chosen.metrics || {};
      const coords = calc.coords || [];
      const sup = Number(areaRow.superficie_m2 || areaRow.sup || 36);
      const lato = Math.sqrt(Math.max(sup, 1));
      const height = Number(areaRow.altezza_m || 2.7);

      await supabase.from("project_jobs").insert({
        project_id: projectId,
        created_by: user.id,
        job_type: "render_blender_v16",
        status: "queued",
        payload: {
          project_id: projectId,
          owner_id: user.id,
          concept_id: chosen.id,
              concept_type: chosen.concept_type || null,
          area_name: areaName,
          mood,
          quality,
          cameras: ["technical","client"],
          width: 1600,
          height: 1200,
          scene: {
            room: { width_m: lato, depth_m: lato, height_m: height },
            luminaires: (coords || []).map(p => ({ x: Number(p[0]), y: Number(p[1]), z: height - 0.05, power: 1.0, cct: 3500 }))
          }
        },
        input: { from: "chat_autopilot" },
        started_at: null,
        finished_at: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
          }

    }
    await supabase.from("project_chat_messages").insert({
      project_id: projectId,
      created_by: user.id,
      role: "assistant",
      content: "🎬 Ho messo in coda i render fotorealistici (Cycles) per tutti e 3 i concept (Comfort/Efficienza/Architetturale) con 2 camere. Avvia il worker Blender quando vuoi: `python workers/blender_worker.py`.",
      meta: { type: "render_jobs_enqueued" }
    });
  } catch (e) {
    // ignore
  }
}
const summary = `Ho generato ${engineRes.result?.length || 0} aree con 3 concept ciascuna (Comfort/Efficienza/Architetturale) usando: ${allowedBrands.join(", ") || "—"}.\nApri **Review & Learning** per vedere risultati ed export.`;

    await supabase.from("project_chat_messages").insert({
      project_id: projectId,
      created_by: user.id,
      role: "assistant",
      content: summary,
      meta: { job_id: job.id, brands: allowedBrands, kind: "concepts", rag_used: Boolean(ragContext) },
    });
// Autopilot: if user asked for a PPTX/presentation, generate and store it in Supabase Storage
if (wantsPptx(message)) {
  try {
    const supabaseA = supabaseAdmin();

    // Load concepts freshly from DB
    const { data: conceptsAll } = await supabase
      .from("concepts")
      .select("area_id, concept_type, solution, metrics")
      .in("area_id", (areas || []).map(a => a.id))
      .order("created_at", { ascending: true });

    const conceptsByArea = {};
    for (const c of (conceptsAll || [])) {
      if (!conceptsByArea[c.area_id]) conceptsByArea[c.area_id] = [];
      const sol = c.solution || {};
      // normalize
      conceptsByArea[c.area_id].push({
        concept_type: c.concept_type,
        ...sol,
        calc: sol.calc || c.metrics || {},
      });
    }

    const pptBuf = await buildPptxBuffer({
      project,
      areas,
      conceptsByArea,
    });

    const owner = user.id;
    const filePath = `${owner}/${projectId}/LuxIA_${(project.name || "progetto").replace(/\s+/g,"_")}_${Date.now()}.pptx`;

    const { error: upErr } = await supabaseA.storage
      .from("exports")
      .upload(filePath, pptBuf, {
        contentType: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        upsert: true,
      });
    if (upErr) throw upErr;

    await supabase.from("project_exports").insert({
      project_id: projectId,
      created_by: user.id,
      kind: "pptx",
      file_path: filePath,
      meta: { source: "autopilot", title: `LuxIA – ${project.name}` },
    });

    const { data: signed } = await supabaseA.storage.from("exports").createSignedUrl(filePath, 60 * 60);
    const link = signed?.signedUrl ? `\n\nPPTX pronta (link 1h): ${signed.signedUrl}` : "";

    await supabase.from("project_chat_messages").insert({
      project_id: projectId,
      created_by: user.id,
      role: "assistant",
      content: `✅ Ho generato automaticamente la presentazione PPTX del progetto.` + link,
      meta: { kind: "export", export_kind: "pptx", file_path: filePath },
    });
  } catch (e2) {
    await supabase.from("project_chat_messages").insert({
      project_id: projectId,
      created_by: user.id,
      role: "assistant",
      content: `⚠️ Concept creati, ma non sono riuscita a generare la PPTX in autopilot. Motivo: ${String(e2?.message || e2)}`,
      meta: { kind: "export_error" },
    });
  }
}



    await supabase.from("project_jobs").update({
      status: "done",
      output: engineRes,
      finished_at: new Date().toISOString(),
    }).eq("id", job.id);

    return NextResponse.json({ ok:true, job_id: job.id, result: engineRes });
  } catch (e) {
    return NextResponse.json({ ok:false, error: String(e?.message || e) }, { status: 500 });
  }
}


export async function GET(req) {
  try {
    const { searchParams } = new URL(req.url);
    const projectId = searchParams.get("project_id");
    if (!projectId) return NextResponse.json({ ok:false, error:"Missing project_id" }, { status: 400 });

    const supabase = supabaseRoute();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return NextResponse.json({ ok:false, error:"Unauthorized" }, { status: 401 });

    const { data: messages, error } = await supabase
      .from("project_chat_messages")
      .select("id, role, content, meta, created_at")
      .eq("project_id", projectId)
      .order("created_at", { ascending: true })
      .limit(300);

    if (error) throw new Error(error.message);
    return NextResponse.json({ ok:true, messages: messages || [] });
  } catch (e) {
    
// Autopilot: if user asked for a render/fotorealistico, queue a render_blender job
if (wantsRender(message)) {
  try {
    await supabase.from("project_jobs").insert({
      project_id: projectId,
      created_by: user.id,
      type: "render_blender",
      status: "queued",
      meta: { source: "autopilot", note: "Run local Blender worker to produce PNGs." }
    });
    await supabase.from("project_chat_messages").insert({
      project_id: projectId,
      created_by: user.id,
      role: "assistant",
      content: "🖼️ Render richiesto: ho messo in coda un job `render_blender`. Avvia il worker locale (workers/blender_worker.py) per generare i PNG e caricarli su Supabase.",
      meta: { kind: "render_queue" },
    });
  } catch (e) {
    // ignore render queue errors
  }
}
return NextResponse.json({ ok:false, error: String(e?.message || e) }, { status: 500 });
  }
}function detectMoodHint(text) {
  const t = String(text || "").toLowerCase();
  const rules = [
    { mood: "hospitality_warm", keys: ["hotel", "ospital", "lounge", "spa", "warm", "caldo", "accogliente", "soft"] },
    { mood: "retail_bright", keys: ["retail", "shop", "negozio", "vetrina", "bright", "brillante", "contrasto", "high contrast"] },
    { mood: "office_clean", keys: ["ufficio", "office", "vdt", "clean", "neutro", "bianco", "produttiv"] },
    { mood: "industrial_raw", keys: ["industr", "warehouse", "capannone", "cemento", "metallo", "raw"] },
    { mood: "minimal_white", keys: ["minimal", "minimalista", "essenziale", "tutto bianco", "white"] },
  ];
  for (const r of rules) if (r.keys.some(k => t.includes(k))) return r.mood;
  return null;
}


