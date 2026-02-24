"use client";
import { useEffect, useMemo, useRef, useState } from "react";

function Bubble({ role, content, meta, created_at }) {
  const isUser = role === "user";
  return (
    <div style={{ display:"flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 10 }}>
      <div style={{
        maxWidth: "78%",
        background: isUser ? "#1d4ed8" : "#111827",
        color: "white",
        padding: "10px 12px",
        borderRadius: 14,
        borderTopRightRadius: isUser ? 6 : 14,
        borderTopLeftRadius: isUser ? 14 : 6,
        boxShadow: "0 6px 18px rgba(0,0,0,.18)"
      }}>
        <div style={{ whiteSpace:"pre-wrap", lineHeight: 1.35, fontSize: 14 }}>{content}</div>
        <div style={{ display:"flex", justifyContent:"space-between", gap:10, marginTop: 6, opacity: 0.75, fontSize: 11 }}>
          <span>{role}</span>
          <span>{created_at ? new Date(created_at).toLocaleString() : ""}</span>
        </div>
        {meta?.job_id ? (
          <div style={{ marginTop: 6, fontSize: 12, opacity: 0.9 }}>
            <span style={{ background:"#0f172a", padding:"2px 8px", borderRadius: 999 }}>job: {meta.job_id.slice(0,8)}…</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function ChatClient({ projectId, initialMessages }) {
  const [messages, setMessages] = useState(initialMessages || []);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:"smooth" }); }, [messages, busy]);

  async function send() {
    const msg = text.trim();
    if (!msg || busy) return;
    setText("");
    setBusy(true);

    // optimistic
    const optimistic = { id: "tmp_" + Date.now(), role: "user", content: msg, created_at: new Date().toISOString(), meta: {} };
    setMessages(m => [...m, optimistic]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ project_id: projectId, message: msg }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Request failed");

      // reload messages (simple)
      const r2 = await fetch(`/api/chat?project_id=${projectId}`, { method: "GET" });
      if (r2.ok) {
        const d2 = await r2.json();
        if (d2.ok) setMessages(d2.messages || []);
      } else {
        // fallback: append assistant summary from response if present
        const assistantText = data?.result ? "Operazione completata. Vai su Review & Learning." : "Operazione completata.";
        setMessages(m => [...m, { id:"tmp_a_"+Date.now(), role:"assistant", content: assistantText, created_at: new Date().toISOString(), meta:{} }]);
      }
    } catch (e) {
      setMessages(m => [...m, { id:"tmp_err_"+Date.now(), role:"assistant", content: "Errore: " + (e?.message || e), created_at: new Date().toISOString(), meta:{} }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <div style={{ display:"flex", gap: 10, flexWrap:"wrap", marginBottom: 12 }}>
        <div style={{ padding:"8px 10px", border:"1px solid rgba(255,255,255,.12)", borderRadius: 10, background:"rgba(17,24,39,.55)" }}>
          <div style={{ fontWeight: 800, marginBottom: 4 }}>Prompt rapidi</div>
          <div style={{ display:"flex", gap: 8, flexWrap:"wrap" }}>
            <button className="btn" onClick={() => setText("Crea 3 concept per tutte le aree con marca BEGA. Restituisci calcoli, layout e scene.")}>BEGA — 3 concept</button>
            <button className="btn" onClick={() => setText("Crea concept con priorità efficienza energetica. Preferisci 4000K e riduci W/m².")}>Efficienza</button>
            <button className="btn" onClick={() => setText("Crea concept più architetturale con accenti e scene. Preferisci 3000K.")}>Architetturale</button>
          </div>
        </div>
      </div>

      <div style={{
        border:"1px solid rgba(255,255,255,.10)",
        borderRadius: 14,
        padding: 14,
        background:"rgba(0,0,0,.25)",
        maxHeight: "62vh",
        overflow:"auto"
      }}>
        {messages.map(m => (
          <Bubble key={m.id} role={m.role} content={m.content} meta={m.meta} created_at={m.created_at} />
        ))}
        {busy ? (
          <div style={{ opacity: 0.8, fontSize: 13, padding: 8 }}>LuxIA sta lavorando…</div>
        ) : null}
        <div ref={bottomRef} />
      </div>

      <div style={{ display:"flex", gap: 10, marginTop: 12 }}>
        <input
          value={text}
          onChange={(e)=>setText(e.target.value)}
          onKeyDown={(e)=>{ if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); send(); }}}
          placeholder="Scrivi: 'Crea 3 concept BEGA…' oppure 'Priorità comfort UGR≤19…'"
          style={{
            flex: 1,
            padding: "12px 12px",
            borderRadius: 12,
            border:"1px solid rgba(255,255,255,.14)",
            background:"rgba(17,24,39,.45)",
            color:"white"
          }}
        />
        <button className="btn" onClick={send} disabled={busy} style={{ minWidth: 120 }}>
          {busy ? "..." : "Invia"}
        </button>
      </div>

      <p style={{ opacity: 0.8, marginTop: 10, fontSize: 13 }}>
        Suggerimento: nomina la marca nel testo (es. “BEGA”, “iGuzzini”). Se non la specifichi, LuxIA usa BEGA se presente nel catalogo.
      </p>
    </section>
  );
}
