import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

function sbAdmin() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) throw new Error("Missing SUPABASE env (NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)");
  return createClient(url, key, { auth: { persistSession: false } });
}

export async function GET(req, { params }) {
  try {
    const supabase = sbAdmin();
    const projectId = params.id;

    const { data: proj, error: e0 } = await supabase
      .from("projects")
      .select("id, tenant_id")
      .eq("id", projectId)
      .maybeSingle();
    if (e0) throw e0;
    if (!proj) return NextResponse.json({ ok: false, error: "Project not found" }, { status: 404 });

    const { data: row, error: e1 } = await supabase
      .from("project_style_packs")
      .select("pack, updated_at")
      .eq("tenant_id", proj.tenant_id)
      .eq("project_id", projectId)
      .maybeSingle();
    if (e1) throw e1;

    return NextResponse.json({ ok: true, pack: row?.pack || null, updated_at: row?.updated_at || null });
  } catch (e) {
    return NextResponse.json({ ok: false, error: String(e?.message || e) }, { status: 500 });
  }
}

export async function POST(req, { params }) {
  try {
    const supabase = sbAdmin();
    const projectId = params.id;
    const body = await req.json();
    const pack = body?.pack || {};

    const { data: proj, error: e0 } = await supabase
      .from("projects")
      .select("id, tenant_id")
      .eq("id", projectId)
      .maybeSingle();
    if (e0) throw e0;
    if (!proj) return NextResponse.json({ ok: false, error: "Project not found" }, { status: 404 });

    const { data, error } = await supabase.rpc("upsert_project_style_pack", {
      p_tenant_id: proj.tenant_id,
      p_project_id: projectId,
      p_pack: pack,
    });
    if (error) throw error;

    return NextResponse.json({ ok: true, pack: data });
  } catch (e) {
    return NextResponse.json({ ok: false, error: String(e?.message || e) }, { status: 500 });
  }
}
