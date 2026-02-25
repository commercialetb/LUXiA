import { NextResponse } from "next/server";
import { supabaseServer } from "@/lib/supabase/server";

// POST { projectId }
export async function POST(req) {
  try {
    const supabase = supabaseServer();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return NextResponse.json({ ok:false, error:"Unauthorized" }, { status: 401 });

    const { projectId } = await req.json();
    if (!projectId) return NextResponse.json({ ok:false, error:"Missing projectId" }, { status: 400 });

    const { data: proj, error: pErr } = await supabase
      .from("projects")
      .select("id, tenant_id, name, client_name, active_style_id")
      .eq("id", projectId)
      .single();
    if (pErr) throw pErr;

    if (proj.active_style_id) return NextResponse.json({ ok:true, activeStyleId: proj.active_style_id });

    const tenantId = proj.tenant_id;
    const clientName = (proj.client_name || "").trim();

    // Ensure Team Default exists
    const { data: defRows, error: defErr } = await supabase
      .from("designer_styles")
      .select("id")
      .eq("tenant_id", tenantId)
      .eq("is_default", true)
      .limit(1);
    if (defErr) throw defErr;

    let defId = defRows?.[0]?.id;
    if (!defId) {
      const { data: created, error: cErr } = await supabase
        .from("designer_styles")
        .insert({ tenant_id: tenantId, name: "Team Default", scope: "tenant", is_default: true })
        .select()
        .single();
      if (cErr) throw cErr;
      defId = created.id;
    }

    // If client name present: find or create a client style and set active
    if (clientName) {
      const { data: ex, error: exErr } = await supabase
        .from("designer_styles")
        .select("id")
        .eq("tenant_id", tenantId)
        .eq("scope", "client")
        .ilike("client_name", clientName)
        .limit(1);
      if (exErr) throw exErr;

      let sid = ex?.[0]?.id;
      if (!sid) {
        const { data: created2, error: c2Err } = await supabase
          .from("designer_styles")
          .insert({ tenant_id: tenantId, name: `Cliente: ${clientName}`, scope: "client", client_name: clientName, is_default: false })
          .select()
          .single();
        if (c2Err) throw c2Err;
        sid = created2.id;
      }

      const { error: upErr } = await supabase.from("projects").update({ active_style_id: sid }).eq("id", projectId);
      if (upErr) throw upErr;

      return NextResponse.json({ ok:true, activeStyleId: sid, createdFromClient: true });
    }

    // fallback to team default
    const { error: upErr2 } = await supabase.from("projects").update({ active_style_id: defId }).eq("id", projectId);
    if (upErr2) throw upErr2;

    return NextResponse.json({ ok:true, activeStyleId: defId, createdFromClient: false });
  } catch (e) {
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 500 });
  }
}
