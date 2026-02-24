import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

function sbAdmin() {
  if (!serviceKey) throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY");
  return createClient(url, serviceKey);
}

// GET ?tenantId=...
export async function GET(req) {
  try {
    const supabase = sbAdmin();
    const { searchParams } = new URL(req.url);
    const tenantId = searchParams.get("tenantId");
    if (!tenantId) return NextResponse.json({ ok:false, error:"Missing tenantId" }, { status: 400 });

    const { data, error } = await supabase
      .from("designer_styles")
      .select("id,name,scope,client_name,is_default,created_at")
      .eq("tenant_id", tenantId)
      .order("is_default", { ascending: false });
    if (error) throw error;
    return NextResponse.json({ ok:true, styles: data || [] });
  } catch (e) {
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 500 });
  }
}

// POST { tenantId, projectId, action: "set_active"|"create_client", styleId?, clientName?, name? }
export async function POST(req) {
  try {
    const supabase = sbAdmin();
    const body = await req.json();
    const { tenantId, projectId, action, styleId, clientName, name } = body || {};
    if (!tenantId || !projectId) return NextResponse.json({ ok:false, error:"Missing tenantId/projectId" }, { status: 400 });

    if (action === "set_active") {
      if (!styleId) return NextResponse.json({ ok:false, error:"Missing styleId" }, { status: 400 });
      const { error } = await supabase.from("projects").update({ active_style_id: styleId }).eq("id", projectId);
      if (error) throw error;
      return NextResponse.json({ ok:true });
    }

    if (action === "create_client") {
      const nm = name || `Cliente: ${clientName || "Nuovo"}`;
      const { data, error } = await supabase.from("designer_styles").insert({
        tenant_id: tenantId,
        name: nm,
        scope: "client",
        client_name: clientName || null,
        is_default: false
      }).select().single();
      if (error) throw error;
      await supabase.from("projects").update({ active_style_id: data.id }).eq("id", projectId);
      return NextResponse.json({ ok:true, style: data });
    }

    return NextResponse.json({ ok:false, error:"Unknown action" }, { status: 400 });
  } catch (e) {
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 500 });
  }
}
