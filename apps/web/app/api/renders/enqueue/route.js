import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

function sbAdmin() {
  if (!serviceKey) throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY");
  return createClient(url, serviceKey);
}

// POST { projectId, conceptId, areaName, coords, room:{w,d,h}, mood, quality, cameras:["technical","client"], stylePack? }
export async function POST(req) {
  try {
    const body = await req.json();
    const {
      projectId, ownerId, conceptId, conceptType, areaName,
      coords, room, mood="office_clean", quality="medium",
      cameras=["technical","client"],
      width=1600, height=1200, cct=3500
    } = body || {};
// v22 overrides from style pack (if present)
const sp = stylePack || {};
const mood2 = sp.mood || null;
const cct2 = Number(sp.cct_default || 0) || null;
const contrast_level = sp.contrast_level || "medium";
const accent_ratio = Number(sp.accent_ratio || 0.2);
const quality2 = (sp.presentation_theme === "dark_elegant" || contrast_level === "high") ? "high" : quality;



    if (!projectId) return NextResponse.json({ ok:false, error:"Missing projectId" }, { status: 400 });

    const supabase = sbAdmin();

// v22: auto-apply project Style Pack to render payload
let stylePack = body?.stylePack || null;
if (!stylePack) {
  try {
    const { data: row } = await supabase
      .from("project_style_packs")
      .select("pack")
      .eq("project_id", projectId)
      .limit(1)
      .maybeSingle();
    stylePack = row?.pack || null;
  } catch (e) {
    stylePack = null;
  }
}

    // Build scene payload for blender script
    const rw = room?.width_m || room?.w || 6.0;
    const rd = room?.depth_m || room?.d || 6.0;
    const rh = room?.height_m || room?.h || 2.7;

    const luminaires = (coords || []).map((p) => ({
      x: Number(p[0] ?? p.x ?? rw*0.5),
      y: Number(p[1] ?? p.y ?? rd*0.5),
      z: rh - 0.05,
      power: 1.0,
      cct: Number(cct2 || cct || 3500),
    }));

    // boost a subset as accents
    try {
      const n = luminaires.length;
      const k = Math.max(0, Math.min(n, Math.round(n * accent_ratio)));
      for (let i = 0; i < k; i++) {
        luminaires[i].power = 1.35;
      }
    } catch (e) {}

    const payload = {
      project_id: projectId,
      owner_id: ownerId || null,
      concept_id: conceptId || null,
      concept_type: conceptType || null,
      area_name: areaName || "Area",
      mood: (mood2 || mood), quality: quality2, cameras, width, height,
      scene: {
        room: { width_m: rw, depth_m: rd, height_m: rh },
        luminaires
      }
    };

    const { data, error } = await supabase.from("project_jobs").insert({
      project_id: projectId,
      owner_id: ownerId || null,
      job_type: "render_blender_v16",
      status: "queued",
      payload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }).select().single();

    if (error) throw error;

    return NextResponse.json({ ok:true, jobId: data.id });
  } catch (e) {
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 500 });
  }
}
