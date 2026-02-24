import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

function sbAdmin() {
  if (!serviceKey) throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY");
  return createClient(url, serviceKey);
}

// POST { projectId }
export async function POST(req) {
  try {
    const supabase = sbAdmin();
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
    const { data: defRows } = await supabase
      .from("designer_styles")
      .select("id")
      .eq("tenant_id", tenantId)
      .eq("is_default", true)
      .limit(1);

    let defId = defRows?.[0]?.id;
    if (!defId) {
      const { data: created } = await supabase
        .from("designer_styles")
        .insert({ tenant_id: tenantId, name: "Team Default", scope: "tenant", is_default: true })
        .select()
        .single();
      defId = created.id;
    }

    // If client name present: find or create a client style and set active
    if (clientName) {
      const { data: ex } = await supabase
        .from("designer_styles")
        .select("id")
        .eq("tenant_id", tenantId)
        .eq("scope", "client")
        .ilike("client_name", clientName)
        .limit(1);

      let sid = ex?.[0]?.id;
      if (!sid) {
        const { data: created2 } = await supabase
          .from("designer_styles")
          .insert({ tenant_id: tenantId, name: `Cliente: ${clientName}`, scope: "client", client_name: clientName, is_default: false })
          .select()
          .single();
        sid = created2.id;
      }
      await supabase.from("projects").update({ active_style_id: sid }).eq("id", projectId);
      return NextResponse.json({ ok:true, activeStyleId: sid, createdFromClient: true });
    }

    // fallback to team default
    await supabase.from("projects").update({ active_style_id: defId }).eq("id", projectId);
    return NextResponse.json({ ok:true, activeStyleId: defId, createdFromClient: false });
  } catch (e) {
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 500 });
  }
}
