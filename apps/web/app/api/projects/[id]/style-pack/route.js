import { NextResponse } from "next/server";
import { supabaseServer } from "@/lib/supabase/server";

// GET  /api/projects/:id/style-pack
export async function GET(req, { params }) {
  try {
    const supabase = supabaseServer();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return NextResponse.json({ ok: false, error: "Unauthorized" }, { status: 401 });

    const projectId = params.id;

    // Project must be readable by this user via RLS.
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

// POST /api/projects/:id/style-pack  body: { pack }
export async function POST(req, { params }) {
  try {
    const supabase = supabaseServer();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return NextResponse.json({ ok: false, error: "Unauthorized" }, { status: 401 });

    const projectId = params.id;
    const body = await req.json().catch(() => ({}));
    const pack = body?.pack || {};

    const { data: proj, error: e0 } = await supabase
      .from("projects")
      .select("id, tenant_id")
      .eq("id", projectId)
      .maybeSingle();
    if (e0) throw e0;
    if (!proj) return NextResponse.json({ ok: false, error: "Project not found" }, { status: 404 });

    // RLS should allow the tenant user to upsert their project's pack.
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
