"use client";
import { useEffect, useState } from "react";
import { supabaseBrowser } from "@/lib/supabase/browser";

export default function NewProject() {
  const supabase = supabaseBrowser();

  const [tenants, setTenants] = useState([]);
  const [tenantId, setTenantId] = useState("");
  const [name, setName] = useState("");
  const [ptype, setPtype] = useState("office");
  const [file, setFile] = useState(null);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    (async () => {
      const { data: auth } = await supabase.auth.getUser();
      const user = auth?.user;
      if (!user) { window.location.href = "/auth/login"; return; }

      const { data, error } = await supabase
        .from("tenant_users")
        .select("tenant_id, role, tenants(name)")
        .eq("user_id", user.id);

      if (error) { setMsg("Errore: " + error.message); return; }
      setTenants(data || []);
      if ((data || []).length === 1) setTenantId(data[0].tenant_id);
    })();
  }, []);

  async function onCreate(e) {
    e.preventDefault();
    setMsg("");

    const { data: auth } = await supabase.auth.getUser();
    const user = auth?.user;
    if (!user) return setMsg("Devi essere loggato.");

    if (!tenantId) return setMsg("Seleziona uno Studio.");
    if (!name.trim()) return setMsg("Inserisci un nome progetto.");
    if (!file) return setMsg("Carica una planimetria (PDF/JPG/PNG).");

    // 1) create project row
    const { data: prj, error: e1 } = await supabase.from("projects").insert({
      tenant_id: tenantId,
      name,
      project_type: ptype,
      created_by: user.id,
    }).select().single();

    if (e1) return setMsg("Errore progetto: " + e1.message);

    // 2) upload planimetry
    const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
    const path = `${tenantId}/${prj.id}/${safeName}`;

    const { error: e2 } = await supabase.storage.from("LuxIA").upload(path, file, {
      upsert: true,
      contentType: file.type || "application/octet-stream",
    });
    if (e2) return setMsg("Errore upload: " + e2.message);

    // 3) save storage path into project
    const { error: e3 } = await supabase.from("projects").update({
      planimetry_path: path,
    }).eq("id", prj.id);

    if (e3) return setMsg("Errore update progetto: " + e3.message);

    window.location.href = `/projects/${prj.id}`;
  }

  return (
    <main style={{ maxWidth: 720, margin: "40px auto", padding: 20 }}>
      <h1>Nuovo Progetto</h1>

      <form onSubmit={onCreate} style={card()}>
        <label>Studio</label>
        <select value={tenantId} onChange={(e)=>setTenantId(e.target.value)} style={inp()}>
          <option value="">— seleziona —</option>
          {tenants.map((t)=>(
            <option key={t.tenant_id} value={t.tenant_id}>
              {(t.tenants?.name || t.tenant_id) + ` (${t.role})`}
            </option>
          ))}
        </select>

        <label>Nome progetto</label>
        <input value={name} onChange={(e)=>setName(e.target.value)} style={inp()} placeholder="Es. Uffici Teledifesa — 1° piano" />

        <label>Tipo progetto</label>
        <select value={ptype} onChange={(e)=>setPtype(e.target.value)} style={inp()}>
          <option value="office">Uffici</option>
          <option value="hospitality">Hotel / Ospitalità</option>
          <option value="retail">Retail</option>
          <option value="residential">Residenziale</option>
        </select>

        <label>Planimetria (PDF/JPG/PNG)</label>
        <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={(e)=>setFile(e.target.files?.[0] || null)} style={inp()} />

        <button type="submit" style={btn("#16a34a")}>Crea progetto</button>

        {msg && <div style={{ marginTop: 8, color: "#fde68a" }}>{msg}</div>}
        <div style={{ marginTop: 10, opacity: 0.8, fontSize: 12 }}>
          Bucket Storage richiesto: <b>LuxIA</b>
        </div>
      </form>
    </main>
  );
}

const card = ()=>({ background:"#0f172a", border:"1px solid #24304a", borderRadius:14, padding:14, display:"grid", gap:10 });
const inp  = ()=>({ padding:10, borderRadius:10, border:"1px solid #24304a", background:"#020617", color:"#e5e7eb" });
const btn  = (c)=>({ background:c, border:"none", padding:"10px 14px", borderRadius:10, color:"white", fontWeight:800, cursor:"pointer", marginTop: 6 });
