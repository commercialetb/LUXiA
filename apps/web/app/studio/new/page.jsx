"use client";
import { useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/browser";

export default function NewStudio() {
  const supabase = supabaseBrowser();
  const [name, setName] = useState("");
  const [msg, setMsg] = useState("");

  async function createStudio(e) {
    e.preventDefault();
    setMsg("");

    const { data: auth } = await supabase.auth.getUser();
    const user = auth?.user;
    if (!user) return setMsg("Devi essere loggato.");

    // 1) create tenant
    const { data: t, error: e1 } = await supabase.from("tenants").insert({ name }).select().single();
    if (e1) return setMsg("Errore tenant: " + e1.message);

    // 2) add membership as owner
    const { error: e2 } = await supabase.from("tenant_users").insert({ tenant_id: t.id, user_id: user.id, role: "owner" });
    if (e2) return setMsg("Errore membership: " + e2.message);

    setMsg("Studio creato! Torna alla home.");
  }

  return (
    <main style={{ maxWidth: 520, margin: "60px auto", padding: 20 }}>
      <h1>Crea Studio</h1>
      <form onSubmit={createStudio} style={{ background:"#0f172a", border:"1px solid #24304a", borderRadius:14, padding:14, display:"grid", gap:10 }}>
        <label>Nome Studio</label>
        <input value={name} onChange={(e)=>setName(e.target.value)} style={{ padding:10, borderRadius:10, border:"1px solid #24304a", background:"#020617", color:"#e5e7eb" }} />
        <button type="submit" style={{ background:"#16a34a", border:"none", padding:"10px 14px", borderRadius:10, color:"white", fontWeight:800, cursor:"pointer" }}>
          Crea
        </button>
        {msg && <div style={{ marginTop: 6, color: "#fde68a" }}>{msg}</div>}
      </form>
    </main>
  );
}
