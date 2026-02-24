export const runtime = "nodejs";
import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";
import PptxGenJS from "pptxgenjs";

// Server Supabase client (user session)
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

async function fetchImageAsDataUri(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Image fetch failed");
  const buf = Buffer.from(await res.arrayBuffer());
  const b64 = buf.toString("base64");
  return "data:image/png;base64," + b64;
}

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number" && Number.isFinite(v)) return String(v);
  return String(v);
}

export async function GET(req) {
  try {
    const { searchParams } = new URL(req.url);
    const projectId = searchParams.get("project_id");
    if (!projectId) return NextResponse.json({ ok:false, error:"Missing project_id" }, { status: 400 });

    const supabase = supabaseRoute();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return NextResponse.json({ ok:false, error:"Unauthorized" }, { status: 401 });

    const { data: project, error: pErr } = await supabase
      .from("projects")
      .select("id, name, project_type, tenant_id, created_at")
      .eq("id", projectId)
      .single();
    if (pErr) throw new Error(pErr.message);

const { data: spRow } = await supabase
  .from("project_style_packs")
  .select("pack")
  .eq("project_id", projectId)
  .limit(1)
  .maybeSingle();
const stylePack = spRow?.pack || {};


    const { data: areas, error: aErr } = await supabase
      .from("areas")
      .select("id, name, tipo_locale, superficie_m2, altezza_m, created_at")
      .eq("project_id", projectId)
      .order("created_at", { ascending: true });
    if (aErr) throw new Error(aErr.message);

    const { data: concepts, error: cErr } = await supabase
      .from("concepts")
      .select("id, area_id, concept_type, solution, metrics, renders, created_at")
      .in("area_id", (areas || []).map(a => a.id))
      .order("created_at", { ascending: true });
    if (cErr) throw new Error(cErr.message);

    // Build PPT
    const pptx = new PptxGenJS();
    pptx.layout = "LAYOUT_WIDE";
    pptx.author = "LuxIA";
    pptx.company = "LuxIA";
    pptx.subject = "Lighting concept";
    pptx.title = `LuxIA – ${project.name}`;


// v22: PPT theme driven by Style Pack
const presTheme = String(stylePack?.presentation_theme || "clean_office");
const theme = {
  clean_office: { TITLE:"1A365D", ACCENT:"2B6CB0", GRAY:"4B5563", BG:"FFFFFF" },
  dark_elegant: { TITLE:"F8FAFC", ACCENT:"38BDF8", GRAY:"CBD5E1", BG:"0B1220" },
  retail_high_contrast: { TITLE:"111827", ACCENT:"F97316", GRAY:"374151", BG:"FFFFFF" },
  hospitality_soft: { TITLE:"1F2937", ACCENT:"A78BFA", GRAY:"4B5563", BG:"FFF7ED" },
}[presTheme] || { TITLE:"1A365D", ACCENT:"2B6CB0", GRAY:"4B5563", BG:"FFFFFF" };

    const TITLE = theme.TITLE;
    const ACCENT = theme.ACCENT;
    const GRAY = theme.GRAY;

function addBG(slide) {
  slide.addShape(pptx.ShapeType.rect, { x:0, y:0, w:13.333, h:7.5, fill:{ color: theme.BG }, line:{ color: theme.BG } });
}

function drawMiniPlan(slide, box, area, concept) {
  try {
    const { x, y, w, h } = box;
    const sup = Number(area.superficie_m2 || area.surface_m2 || area.sup || 0) || 0;
    const side = sup > 0 ? Math.sqrt(sup) : 5;
    slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, fill:{ color:"F7FAFC" }, line:{ color:"CBD5E0", width:1 }});
    const pad = 0.12;
    const rx = x + pad, ry = y + pad, rw = Math.max(0.1, w - 2*pad), rh = Math.max(0.1, h - 2*pad);
    slide.addShape(pptx.ShapeType.rect, { x:rx, y:ry, w:rw, h:rh, fill:{ color:"FFFFFF" }, line:{ color:"A0AEC0", width:1 }});
    const coords = (concept?.solution?.calc?.coords || concept?.metrics?.coords || concept?.solution?.coords || []);
    const sx = rw / side, sy = rh / side;
    for (const c of coords) {
      const lx = Number(c[0] ?? c.x ?? 0);
      const ly = Number(c[1] ?? c.y ?? 0);
      const px = rx + lx * sx;
      const py = ry + ly * sy;
      slide.addShape(pptx.ShapeType.ellipse, { x:px-0.05, y:py-0.05, w:0.10, h:0.10, fill:{ color:"FBBF24" }, line:{ color:"111827", width:0.5 }});
    }
  } catch (_) {}
}


    // Slide 1: Cover
    {
      const s = pptx.addSlide();
      s.addShape(pptx.ShapeType.rect, { x:0, y:0, w:13.33, h:7.5, fill: { color: TITLE }});
      s.addText("LuxIA — Concept Illuminotecnico", {
        x:0.7, y:1.6, w:12, h:0.6, fontFace:"Aptos", fontSize: 34, color: "FFFFFF", bold:true
      });
      s.addText(project.name || "Progetto", {
        x:0.7, y:2.35, w:12, h:0.5, fontFace:"Aptos", fontSize: 20, color: "90CDF4", bold:true
      });
      s.addText(`Data: ${new Date().toLocaleDateString("it-IT")}`, {
        x:0.7, y:6.7, w:12, h:0.4, fontFace:"Aptos", fontSize: 12, color: "CBD5E0"
      });
    }

    // Slide 2: Summary table
    {
      const s = pptx.addSlide();
      s.addText("Riepilogo Aree e KPI", { x:0.6, y:0.4, w:12, h:0.5, fontFace:"Aptos", fontSize: 22, color: TITLE, bold:true });

      const rows = [["Area","Tipo","m²","h","Concept","Em (lux)","W","W/m²","UGR","Ra"]];
      for (const a of (areas || [])) {
        const cc = (concepts || []).filter(x => x.area_id === a.id);
        const best = cc.find(x => x.concept_type === "Comfort") || cc[0];
        const m = best?.metrics || best?.solution?.calc || {};
        const sol = best?.solution || {};
        rows.push([
          (a.name || "").slice(0,22),
          (a.tipo_locale || "").slice(0,18),
          fmt(a.superficie_m2),
          fmt(a.altezza_m),
          best ? best.concept_type : "—",
          fmt(m.E_m ?? m.em_lux ?? m.Em),
          fmt(m.W_t ?? m.w_total),
          fmt(m.wm2 ?? m.w_m2),
          fmt(sol.ugr ?? m.ugr ?? sol?.luminaire?.ugr),
          fmt(sol.ra ?? m.ra ?? sol?.luminaire?.cri),
        ]);
      }

      s.addTable(rows, {
        x:0.4, y:1.2, w:12.5,
        fontFace:"Aptos", fontSize: 10,
        border: { type:"solid", color:"CBD5E0", pt:1 },
        fill: "FFFFFF",
        color: "111827",
        valign: "middle",
        rowH: 0.32,
        colW: [2.2, 1.6, 0.7, 0.5, 1.2, 0.9, 0.6, 0.7, 0.6, 0.6],
        header: true
      });
    }

    // Slides per area (3 concepts)
    for (const a of (areas || [])) {
      const s = pptx.addSlide();
      s.addShape(pptx.ShapeType.rect, { x:0, y:0, w:13.33, h:0.9, fill: { color: ACCENT }});
      s.addText(`${a.name} — ${a.tipo_locale}`, { x:0.6, y:0.15, w:12.2, h:0.6, fontFace:"Aptos", fontSize: 18, color:"FFFFFF", bold:true });
      s.addText(`Superficie: ${fmt(a.superficie_m2)} m² • Altezza: ${fmt(a.altezza_m)} m`, { x:0.6, y:1.1, w:12, h:0.3, fontFace:"Aptos", fontSize: 12, color: GRAY });

      const cc = (concepts || []).filter(x => x.area_id === a.id);
      const order = ["Comfort","Efficienza","Architetturale"];
      const sorted = order.map(t => cc.find(x => x.concept_type === t)).filter(Boolean);

      // Optional: embed latest renders (v17) — client + technical
try {
  const { data: rr } = await supabase
    .from("project_renders")
    .select("storage_path, camera, created_at")
    .eq("project_id", projectId)
    .eq("area_name", a.name)
    .in("camera", ["client","technical"])
    .order("created_at", { ascending: false })
    .limit(10);

  const latest = (cam) => (rr || []).find(x => x.camera === cam);

  const boxX = 8.7, boxW = 4.4;
  const boxH = 1.55; // two stacked images
  const pad = 0.05;

  // Client (top)
  const rClient = latest("client");
  if (rClient?.storage_path) {
    const { data: signed } = await supabase.storage.from("renders").createSignedUrl(rClient.storage_path, 3600);
    if (signed?.signedUrl) {
      const imgData = await fetchImageAsDataUri(signed.signedUrl);
      s.addShape(ppt.ShapeType.rect, { x:boxX, y:1.6, w:boxW, h:boxH, fill:{ color:"F8FAFC" }, line:{ color:"E5E7EB" }});
      s.addImage({ data: imgData, x:boxX+pad, y:1.6+pad, w:boxW-2*pad, h:boxH-2*pad });
      s.addText("Render (camera cliente)", { x:boxX, y:1.6+boxH, w:boxW, h:0.25, fontFace:"Aptos", fontSize: 10, color: GRAY });
    }
  }

  // Technical (bottom)
  const rTech = latest("technical");
  if (rTech?.storage_path) {
    const { data: signed2 } = await supabase.storage.from("renders").createSignedUrl(rTech.storage_path, 3600);
    if (signed2?.signedUrl) {
      const imgData2 = await fetchImageAsDataUri(signed2.signedUrl);
      const y0 = 3.55;
      s.addShape(ppt.ShapeType.rect, { x:boxX, y:y0, w:boxW, h:boxH, fill:{ color:"F8FAFC" }, line:{ color:"E5E7EB" }});
      s.addImage({ data: imgData2, x:boxX+pad, y:y0+pad, w:boxW-2*pad, h:boxH-2*pad });
      s.addText("Render (camera tecnica)", { x:boxX, y:y0+boxH, w:boxW, h:0.25, fontFace:"Aptos", fontSize: 10, color: GRAY });
    }
  }
} catch (e) {
  // ignore missing renders
}


let y = 1.6;
      for (const c of sorted) {
        const sol = c.solution || {};
        const m = c.metrics || sol.calc || {};
        const lum = sol.luminaire || sol.lamp || sol.fixture || {};
        const title = `${c.concept_type} — ${lum.brand || sol.brand || ""} ${lum.model || sol.model || ""}`.trim();

        s.addText(title || c.concept_type, { x:0.6, y, w:12.1, h:0.35, fontFace:"Aptos", fontSize: 14, color: TITLE, bold:true });
        y += 0.35;

        const bullets = [
          `N apparecchi: ${fmt(m.n ?? sol.n)}`,
          `Em: ${fmt(m.E_m ?? m.em_lux)} lux (target ${fmt(m.E_t ?? m.et_lux)})`,
          `Potenza: ${fmt(m.W_t ?? m.w_total)} W  •  ${fmt(m.wm2 ?? m.w_m2)} W/m²`,
          `UGR: ${fmt(lum.ugr ?? sol.ugr ?? m.ugr)}  •  Ra/CRI: ${fmt(lum.cri ?? lum.ra ?? sol.ra ?? m.ra)}`,
          `Scene: ${(m.scenes || []).map(x => x.name || x.id).filter(Boolean).join(", ") || "—"}`,
        ];

        s.addText(bullets.map(t => `• ${t}`).join("\n"), {
          x:0.9, y, w:7.6, h:1.0,
          fontFace:"Aptos", fontSize: 11, color: GRAY,
          valign:"top"
        });

        // Mini-plan + layout points (from calc coords)
        drawMiniPlan(s, { x:8.6, y, w:4.0, h:1.0 }, a, c);

        y += 1.1;
      }

      // Layout coords note
      const any = sorted[0]?.solution || cc[0]?.solution || {};
      const coords = any?.calc?.coords || any?.coords || [];
      const coordText = coords.length ? coords.slice(0, 18).map(p => `(${p[0]},${p[1]})`).join("  ") + (coords.length>18?" …":"") : "—";
      s.addText(`Layout (prime coordinate): ${coordText}`, { x:0.6, y:6.9, w:12.2, h:0.4, fontFace:"Aptos", fontSize: 10, color: "6B7280" });
    }

    const buf = await pptx.write("nodebuffer");

    return new NextResponse(buf, {
      status: 200,
      headers: {
        "Content-Type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "Content-Disposition": `attachment; filename="LuxIA_${(project.name || "progetto").replace(/[^a-z0-9_-]+/gi,"_")}.pptx"`,
      },
    });
  } catch (e) {
    return NextResponse.json({ ok:false, error: String(e?.message || e) }, { status: 500 });
  }
}
