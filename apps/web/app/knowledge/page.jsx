"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

function sb() {
  return createClient(supabaseUrl, supabaseAnon);
}

export default function KnowledgePage() {
  const supabase = useMemo(() => sb(), []);
  const [session, setSession] = useState(null);
  const [docs, setDocs] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [projectId, setProjectId] = useState("");

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session || null));
    const { data: sub } = supabase.auth.onAuthStateChange((_e, s) => setSession(s));
    refresh();
    return () => sub.subscription.unsubscribe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refresh() {
    const { data: sess } = await supabase.auth.getSession();
    if (!sess.session) return;

    const { data: d1 } = await supabase
      .from("knowledge_documents")
      .select("id, title, file_path, mime_type, tags, project_id, created_at")
      .order("created_at", { ascending: false })
      .limit(50);
    setDocs(d1 || []);

    const { data: j1 } = await supabase
      .from("knowledge_jobs")
      .select("id, document_id, status, error, stats, created_at, updated_at")
      .order("created_at", { ascending: false })
      .limit(50);
    setJobs(j1 || []);
  }

  async function uploadDoc(file) {
    if (!file) return;
    setUploading(true);
    try {
      const { data: sess } = await supabase.auth.getSession();
      if (!sess.session) {
        alert("Login required");
        return;
      }

      const owner = sess.session.user.id;
      const path = `${owner}/${Date.now()}_${file.name.replace(/\s+/g, "_")}`;

      const { error: upErr } = await supabase.storage.from("knowledge").upload(path, file, {
        upsert: false,
        contentType: file.type || "application/octet-stream",
      });
      if (upErr) throw upErr;

      const { data: doc, error: insErr } = await supabase
        .from("knowledge_documents")
        .insert({
          title: file.name,
          file_path: path,
          mime_type: file.type || null,
          project_id: projectId || null,
          owner_id: owner,
          tags: [],
        })
        .select()
        .single();
      if (insErr) throw insErr;

      await fetch("/api/knowledge/ingest", {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ documentId: doc.id }),
      });

      await refresh();
      alert("Documento caricato. Ingestione avviata.");
    } catch (e) {
      console.error(e);
      alert("Upload/ingest error: " + (e?.message || String(e)));
    } finally {
      setUploading(false);
    }
  }

  async function search() {
    if (!q.trim()) return;
    const { data, error } = await supabase.rpc("search_knowledge", {
      q,
      p_project_id: projectId || null,
      max_rows: 8,
    });
    if (error) return alert("Search error: " + error.message);
    setHits(data || []);
  }

  if (!session) {
    return (
      <div style={{ padding: 22 }}>
        <h2>Knowledge</h2>
        <p>Fai login per usare la knowledge base.</p>
        <Link href="/login">Vai al login</Link>
      </div>
    );
  }

  return (
    <div style={{ padding: 22, maxWidth: 1100, margin: "0 auto" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", gap: 12 }}>
        <h2 style={{ margin:0 }}>📚 Knowledge Base</h2>
        <div style={{ display:"flex", gap: 12 }}>
          <Link href="/projects">Progetti</Link>
          <Link href="/knowledge">Knowledge</Link>
        </div>
      </div>

      <div style={{ marginTop: 12, background:"#fff", border:"1px solid #e5e7eb", borderRadius: 12, padding: 14 }}>
        <div style={{ display:"flex", gap: 10, flexWrap:"wrap", alignItems:"center" }}>
          <label style={{ fontSize: 13, color:"#374151" }}>
            Project filter (optional):
            <input
              value={projectId}
              onChange={(e) => setProjectId(e.target.value)}
              placeholder="UUID progetto (opzionale)"
              style={{ marginLeft: 10, padding:"8px 10px", borderRadius: 10, border:"1px solid #e5e7eb", width: 360 }}
            />
          </label>
          <label style={{ fontSize: 13, color:"#374151" }}>
            Upload:
            <input
              type="file"
              onChange={(e) => uploadDoc(e.target.files?.[0])}
              disabled={uploading}
              style={{ marginLeft: 10 }}
              accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.dxf,.dwg"
            />
          </label>
          <button className="btn" onClick={refresh} disabled={uploading}>Aggiorna</button>
        </div>
        <p style={{ marginTop: 10, fontSize: 13, color:"#6b7280" }}>
          v12: ingestione testo da PDF/DOCX/TXT per RAG. Immagini e DWG/DXF restano come reference (OCR/parse in v13+).
        </p>
      </div>

      <div style={{ marginTop: 14, display:"grid", gridTemplateColumns:"1.2fr 0.8fr", gap: 14 }}>
        <div style={{ background:"#fff", border:"1px solid #e5e7eb", borderRadius: 12, padding: 14 }}>
          <h3 style={{ marginTop: 0 }}>🔎 Ricerca (RAG)</h3>
          <div style={{ display:"flex", gap: 10 }}>
            <input
              value={q}
              onChange={(e)=>setQ(e.target.value)}
              placeholder="Es: UNI 11630, UGR ufficio, uniformità..."
              style={{ flex:1, padding:"10px 12px", borderRadius: 10, border:"1px solid #e5e7eb" }}
            />
            <button className="btn" onClick={search}>Cerca</button>
          </div>

          <div style={{ marginTop: 12 }}>
            {hits.length === 0 ? (
              <p style={{ fontSize: 13, color:"#6b7280" }}>Nessun risultato (o fai una ricerca).</p>
            ) : hits.map((h) => (
              <div key={h.chunk_id} style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12, marginBottom: 10 }}>
                <div style={{ display:"flex", justifyContent:"space-between", gap: 10 }}>
                  <b style={{ fontSize: 13 }}>{h.title || "Documento"}</b>
                  <span style={{ fontSize: 12, color:"#6b7280" }}>rank {Number(h.rank || 0).toFixed(3)}</span>
                </div>
                <pre style={{ whiteSpace:"pre-wrap", fontSize: 12, color:"#111827", marginTop: 8 }}>
{h.content?.slice(0, 900)}
                </pre>
              </div>
            ))}
          </div>
        </div>

        <div style={{ background:"#fff", border:"1px solid #e5e7eb", borderRadius: 12, padding: 14 }}>
          <h3 style={{ marginTop: 0 }}>📄 Documenti</h3>
          {docs.length === 0 ? (
            <p style={{ fontSize: 13, color:"#6b7280" }}>Nessun documento caricato.</p>
          ) : docs.map((d) => (
            <div key={d.id} style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 10, marginBottom: 10 }}>
              <b style={{ fontSize: 13 }}>{d.title || "Documento"}</b>
              <div style={{ fontSize: 12, color:"#6b7280", marginTop: 6 }}>
                {d.mime_type || "—"} • {new Date(d.created_at).toLocaleString()}
              </div>
              <div style={{ fontSize: 12, color:"#6b7280", marginTop: 6 }}>
                docId: <code>{d.id}</code>
              </div>
            </div>
          ))}

          <h3 style={{ marginTop: 18 }}>⚙️ Jobs</h3>
          {jobs.length === 0 ? (
            <p style={{ fontSize: 13, color:"#6b7280" }}>Nessun job.</p>
          ) : jobs.slice(0, 12).map((j) => (
            <div key={j.id} style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 10, marginBottom: 10 }}>
              <b style={{ fontSize: 13 }}>{j.status}</b>
              <div style={{ fontSize: 12, color:"#6b7280", marginTop: 6 }}>
                docId: <code>{j.document_id}</code>
              </div>
              {j.error ? <div style={{ fontSize: 12, color:"#b91c1c", marginTop: 6 }}>{j.error}</div> : null}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
