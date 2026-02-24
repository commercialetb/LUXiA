"use client";
import { useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/browser";

export default function Signup() {
  const supabase = supabaseBrowser();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState("");

  async function onSignup(e) {
    e.preventDefault();
    setMsg("");
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) return setMsg("Errore: " + error.message);
    setMsg("Account creato. Ora fai login.");
  }

  return (
    <main style={wrap()}>
      <h1>LuxIA — Signup</h1>
      <form onSubmit={onSignup} style={card()}>
        <label>Email</label>
        <input style={inp()} value={email} onChange={(e)=>setEmail(e.target.value)} />
        <label style={{ marginTop: 10 }}>Password</label>
        <input style={inp()} type="password" value={password} onChange={(e)=>setPassword(e.target.value)} />
        <button style={btn()} type="submit">Crea account</button>
        <a href="/auth/login" style={{ marginTop: 10, display: "block", color: "#93c5fd" }}>Vai al login</a>
        {msg && <div style={{ marginTop: 10, color: "#fde68a" }}>{msg}</div>}
      </form>
    </main>
  );
}

const wrap = ()=>({ maxWidth: 420, margin: "60px auto", padding: 20 });
const card = ()=>({ background:"#0f172a", border:"1px solid #24304a", borderRadius:14, padding:14, display:"grid", gap:8 });
const inp = ()=>({ padding:10, borderRadius:10, border:"1px solid #24304a", background:"#020617", color:"#e5e7eb" });
const btn = ()=>({ background:"#16a34a", border:"none", padding:"10px 14px", borderRadius:10, color:"white", fontWeight:800, cursor:"pointer", marginTop:10 });
