"use client";
import { useEffect, useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/browser";

export default function CatalogPage() {
  const supabase = supabaseBrowser();
  const [tenantId, setTenantId] = useState("");
  const [tenants, setTenants] = useState([]);
  const [rows, setRows] = useState([]);
  const [msg, setMsg] = useState("");

  const [brand, setBrand] = useState("BEGA");
  const [model, setModel] = useState("");
  const [flux, setFlux] = useState(3000);
  const [watt, setWatt] = useState(25);
  const [cri, setCri] = useState(90);
  const [ugr, setUgr] = useState(19);
  const [cct, setCct] = useState("3000K");

  useEffect(() => {
    (async () => {
      const { data: auth } = await supabase.auth.getUser();
      if (!auth?.user) { window.location.href = "/auth/login"; return; }

      const { data, error } = await supabase
        .from("tenant_users")
        .select("tenant_id, role, tenants(name)")
        .eq("user_id", auth.user.id);

      if (error) return setMsg(error.message);
      setTenants(data || []);
      if ((data || []).length === 1) setTenantId(data[0].tenant_id);
    })();
  }, []);

  async function loadCatalog(tid) {
    if (!tid) return;
    const { data, error } = await supabase
      .from("luminaires")
      .select("*")
      .eq("tenant_id", tid)
      .order("created_at", { ascending: false });
    if (error) return setMsg(error.message);
    setRows(data || []);
  }

  useEffect(() => { loadCatalog(tenantId); }, [tenantId]);

  async function addRow(e) {
    e.preventDefault();
    setMsg("");
    if (!tenantId) return setMsg("Seleziona uno Studio.");
    if (!model.trim()) return setMsg("Inserisci modello.");

    const { error } = await supabase.from("luminaires").insert({
      tenant_id: tenantId,
      brand, model,
      flux_lm: Number(flux),
      watt: Number(watt),
      cri: Number(cri),
      ugr: Number(ugr),
      cct
    });

    if (error) return setMsg(error.message);
    setModel("");
    await loadCatalog(tenantId);
  }

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
      <h1>Catalogo Apparecchi (multi-brand)</h1>

      <div style={{ margin: "10px 0 16px", display:"flex", gap:10, flexWrap:"wrap" }}>
        <a href="/" style={{ color:"#93c5fd" }}>← Home</a>
      </div>

      <section style={card()}>
        <div style={{ display:"flex", gap:10, flexWrap:"wrap", alignItems:"center" }}>
          <label style={{ fontSize:12, opacity:0.8 }}>Studio:</label>
          <select value={tenantId} onChange={(e)=>setTenantId(e.target.value)} style={inp()}>
            <option value="">— seleziona —</option>
            {tenants.map(t => <option key={t.tenant_id} value={t.tenant_id}>{t.tenants?.name || t.tenant_id}</option>)}
          </select>
        </div>

        <form onSubmit={addRow} style={{ display:"grid", gridTemplateColumns:"1fr 2fr 1fr 1fr 1fr 1fr 1fr auto", gap:8, alignItems:"end", marginTop: 10 }}>
          <div><label style={lab()}>Brand</label><input value={brand} onChange={(e)=>setBrand(e.target.value)} style={inp()} /></div>
          <div><label style={lab()}>Modello</label><input value={model} onChange={(e)=>setModel(e.target.value)} style={inp()} placeholder="Es. Downlight 3000lm 25W" /></div>
          <div><label style={lab()}>lm</label><input type="number" value={flux} onChange={(e)=>setFlux(e.target.value)} style={inp()} /></div>
          <div><label style={lab()}>W</label><input type="number" value={watt} onChange={(e)=>setWatt(e.target.value)} style={inp()} /></div>
          <div><label style={lab()}>Ra</label><input type="number" value={cri} onChange={(e)=>setCri(e.target.value)} style={inp()} /></div>
          <div><label style={lab()}>UGR</label><input type="number" value={ugr} onChange={(e)=>setUgr(e.target.value)} style={inp()} /></div>
          <div><label style={lab()}>CCT</label><input value={cct} onChange={(e)=>setCct(e.target.value)} style={inp()} /></div>
          <button type="submit" style={btn("#16a34a")}>+ Aggiungi</button>
        </form>

        {msg && <div style={{ marginTop: 10, color:"#fde68a" }}>{msg}</div>}
      </section>

      <section style={{ marginTop: 14 }}>
        <h2 style={{ marginBottom: 8 }}>Apparecchi</h2>
        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse" }}>
            <thead>
              <tr>
                {["Brand","Modello","lm","W","Ra","UGR","CCT","Creato"].map(h => (
                  <th key={h} style={th()}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.id}>
                  <td style={td()}>{r.brand}</td>
                  <td style={td()}>{r.model}</td>
                  <td style={td()}>{r.flux_lm}</td>
                  <td style={td()}>{r.watt}</td>
                  <td style={td()}>{r.cri}</td>
                  <td style={td()}>{r.ugr ?? "—"}</td>
                  <td style={td()}>{r.cct ?? "—"}</td>
                  <td style={td()}>{String(r.created_at || "").slice(0,10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

const card = ()=>({ background:"#0f172a", border:"1px solid #24304a", borderRadius:14, padding:14 });
const inp  = ()=>({ padding:10, borderRadius:10, border:"1px solid #24304a", background:"#020617", color:"#e5e7eb" });
const btn  = (c)=>({ background:c, border:"none", padding:"10px 14px", borderRadius:10, color:"white", fontWeight:900, cursor:"pointer" });
const lab  = ()=>({ fontSize:12, opacity:0.8, display:"block", marginBottom:4 });
const th   = ()=>({ textAlign:"left", padding:"8px 6px", borderBottom:"1px solid #24304a", fontSize:12, opacity:0.8 });
const td   = ()=>({ padding:"8px 6px", borderBottom:"1px solid #1f2a44", fontSize:13 });
