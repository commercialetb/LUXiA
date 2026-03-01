import { NextResponse } from "next/server";

const ENGINE_URL = process.env.ENGINE_URL || process.env.NEXT_PUBLIC_ENGINE_URL || "";

export async function POST(req: Request) {
  try {
    if (!ENGINE_URL) {
      return NextResponse.json({ ok: false, error: "ENGINE_URL non configurato" }, { status: 500 });
    }
    const body = await req.json();
    const r = await fetch(`${ENGINE_URL}/assistant/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    return new NextResponse(text, { status: r.status, headers: { "Content-Type": "application/json" } });
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: String(e) }, { status: 500 });
  }
}
