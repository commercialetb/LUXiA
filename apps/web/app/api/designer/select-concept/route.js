import { NextResponse } from "next/server";
import { supabaseServer } from "@/lib/supabase/server";

// POST { projectId, areaId, conceptId }
export async function POST(req) {
  try {
    const supabase = supabaseServer();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return NextResponse.json({ ok:false, error:"Unauthorized" }, { status: 401 });

    const body = await req.json().catch(() => ({}));
    const { projectId, areaId, conceptId } = body || {};
    if (!projectId || !areaId || !conceptId) {
      return NextResponse.json({ ok:false, error:"Missing projectId/areaId/conceptId" }, { status: 400 });
    }

    // Pull project -> tenant + active style (RLS-protected)
    const { data: proj, error: pErr } = await supabase
      .from("projects")
      .select("id, tenant_id, active_style_id")
      .eq("id", projectId)
      .single();
    if (pErr) throw pErr;

    const tenantId = proj.tenant_id;
    let styleId = proj.active_style_id;

    // Ensure a default style exists (RLS allows insert/update for tenant users)
    if (!styleId) {
      const { data: st, error: stErr } = await supabase
        .from("designer_styles")
        .select("id")
        .eq("tenant_id", tenantId)
        .eq("is_default", true)
        .limit(1);
      if (stErr) throw stErr;

      if (st?.[0]?.id) {
        styleId = st[0].id;
      } else {
        const { data: created, error: cErr } = await supabase
          .from("designer_styles")
          .insert({ tenant_id: tenantId, name: "Team Default", scope: "tenant", is_default: true })
          .select()
          .single();
        if (cErr) throw cErr;
        styleId = created.id;
      }

      const { error: upErr } = await supabase.from("projects").update({ active_style_id: styleId }).eq("id", projectId);
      if (upErr) throw upErr;
    }

    // Load area + concept for metrics
    const { data: area, error: aErr } = await supabase
      .from("areas")
      .select("id, name, tipo_locale, superficie_m2, altezza_m")
      .eq("id", areaId)
      .single();
    if (aErr) throw aErr;

    const { data: con, error: cErr } = await supabase
      .from("concepts")
      .select("id, concept_type, solution, metrics")
      .eq("id", conceptId)
      .single();
    if (cErr) throw cErr;

    const calc = con.metrics || con.solution?.calc || {};
    const n = Number(calc.n ?? calc.N ?? 0);
    const sup = Number(area.superficie_m2 || 0);
    const density = sup ? (n / sup) : null;
    const wm2 = Number(calc.wm2 || 0);
    const ugr = Number(calc.ugr || 0);

    // Brand & CCT attempt
    const selectedBrand = (con.solution?.luminaire?.brand) || (con.solution?.brand) || null;
    const selectedCct = Number(con.solution?.cct || con.solution?.CCT || 0) || null;
    const selectedMood = con.solution?.mood || null;

    // Record learning event (RLS)
    const { error: insErr } = await supabase.from("designer_learning_events").insert({
      tenant_id: tenantId,
      style_id: styleId,
      project_id: projectId,
      area_name: area.name,
      area_type: area.tipo_locale,
      selected_concept_type: con.concept_type,
      selected_brand: selectedBrand,
      selected_cct: selectedCct,
      selected_mood: selectedMood,
      n_luminaires: n,
      area_m2: sup,
      density,
      wm2,
      ugr,
      created_by: user.id,
    });
    if (insErr) throw insErr;

    // Recompute aggregated profile (RPC is SECURITY DEFINER)
    const { data: stats, error: rErr } = await supabase.rpc("recompute_designer_profile", { p_tenant_id: tenantId, p_style_id: styleId });
    if (rErr) throw rErr;

    return NextResponse.json({ ok:true, styleId, stats });
  } catch (e) {
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 500 });
  }
}
