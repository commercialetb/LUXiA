import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(req: Request) {
  const base = process.env.ENGINE_BASE_URL;
  if (!base) return NextResponse.json({ ok: false, error: "ENGINE_BASE_URL missing" }, { status: 500 });

  const form = await req.formData();
  const file = form.get("file") as File | null;
  const options = (form.get("options") as string) || "{}";
  if (!file) return NextResponse.json({ ok: false, error: "Missing file" }, { status: 400 });

  const fd = new FormData();
  fd.append("file", file);
  fd.append("options", options);

  const r = await fetch(base.replace(/\/$/, "") + "/planimetry/analyze", { method: "POST", body: fd });
  const txt = await r.text();
  let data: any = null;
  try { data = JSON.parse(txt); } catch { data = { ok: false, error: txt }; }
  return NextResponse.json(data, { status: r.status });
}
