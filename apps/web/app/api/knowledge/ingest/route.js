export const runtime = "nodejs";
import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import pdf from "pdf-parse";
import mammoth from "mammoth";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

function sbAdmin() {
  if (!serviceKey) throw new Error("Missing SUPABASE_SERVICE_ROLE_KEY");
  return createClient(url, serviceKey);
}

function chunkText(text, maxChars = 1400, overlap = 200) {
  const t = (text || "").replace(/\r/g, "");
  const chunks = [];
  let i = 0;
  while (i < t.length) {
    const end = Math.min(t.length, i + maxChars);
    chunks.push(t.slice(i, end));
    i = end - overlap;
    if (i < 0) i = 0;
    if (end === t.length) break;
  }
  return chunks.map((c) => c.trim()).filter(Boolean);
}

async function extractText(buf, mime, filename) {
  const ext = (filename.split(".").pop() || "").toLowerCase();
  if (mime === "application/pdf" || ext === "pdf") {
    const res = await pdf(buf);
    return res.text || "";
  }
  if (mime === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || ext === "docx") {
    const res = await mammoth.extractRawText({ buffer: buf });
    return res.value || "";
  }
  if (mime?.startsWith("text/") || ["txt","md"].includes(ext)) {
    return buf.toString("utf-8");
  }
  return "";
}

export async function POST(req) {
  try {
    const body = await req.json();
    const { documentId } = body || {};
    if (!documentId) return NextResponse.json({ ok:false, error:"Missing documentId" }, { status: 400 });

    const supabase = sbAdmin();

    const { data: doc, error: docErr } = await supabase
      .from("knowledge_documents")
      .select("id, project_id, owner_id, title, file_path, mime_type")
      .eq("id", documentId)
      .single();
    if (docErr) throw docErr;

    const { data: job, error: jobErr } = await supabase
      .from("knowledge_jobs")
      .insert({
        document_id: doc.id,
        project_id: doc.project_id,
        owner_id: doc.owner_id,
        status: "running",
      })
      .select()
      .single();
    if (jobErr) throw jobErr;

    const { data: file, error: dlErr } = await supabase.storage.from("knowledge").download(doc.file_path);
    if (dlErr) throw dlErr;

    const arrayBuf = await file.arrayBuffer();
    const buf = Buffer.from(arrayBuf);

    const text = await extractText(buf, doc.mime_type || "", doc.title || "");

    if (!text.trim()) {
      await supabase.from("knowledge_jobs").update({
        status: "done",
        stats: { chunks: 0, note: "No text extracted (binary/image/dwg/dxf). Stored as reference." },
        updated_at: new Date().toISOString(),
      }).eq("id", job.id);

      return NextResponse.json({ ok:true, status:"done", chunks: 0 });
    }

    const chunks = chunkText(text, 1400, 200);

    await supabase.from("knowledge_chunks").delete().eq("document_id", doc.id);

    const rows = chunks.map((c, idx) => ({
      document_id: doc.id,
      project_id: doc.project_id,
      chunk_index: idx,
      content: c,
      meta: { title: doc.title, source: "upload" },
    }));

    const batchSize = 100;
    for (let i = 0; i < rows.length; i += batchSize) {
      const batch = rows.slice(i, i + batchSize);
      const { error: insErr } = await supabase.from("knowledge_chunks").insert(batch);
      if (insErr) throw insErr;
    }

    await supabase.from("knowledge_jobs").update({
      status: "done",
      stats: { chunks: rows.length },
      updated_at: new Date().toISOString(),
    }).eq("id", job.id);

    return NextResponse.json({ ok:true, status:"done", chunks: rows.length });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 500 });
  }
}
