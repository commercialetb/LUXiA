"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

function sb() {
  return createClient(supabaseUrl, supabaseAnon);
}

function pct(n, total) {
  if (!total) return "0%";
  return Math.round((n / total) * 100) + "%";
}

export default function DesignerBrain() {
  const supabase = useMemo(() => sb(), []);
  const [session, setSession] = useState(null);
  const [tenantId, setTenantId] = useState("");
  const [styleId, setStyleId] = useState("");
  const [areaType, setAreaType] = useState("");
  const [compare, setCompare] = useState({ team: null, active: null });
  const [styles, setStyles] = useState([]);
  const [areaTypes, setAreaTypes] = useState([]);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session || null));
    const { data: sub } = supabase.auth.onAuthStateChange((_e, s) => setSession(s));
    return () => sub.subscription.unsubscribe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!session) return;
    (async () => {
      // Get first org membership
      const { data: om } = await supabase.from("tenant_users").select("tenant_id").limit(1);
      const oid = om?.[0]?.tenant_id || "";
      setTenantId(oid);

      const { data: st } = await supabase.from("designer_styles").select("id,name,is_default,scope,client_name").eq("tenant_id", oid).order("is_default", { ascending: false });
      setStyles(st || []);
      const def = (st || []).find(x => x.is_default) || (st || [])[0];
      if (def?.id) setStyleId(def.id);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.user?.id]);

  useEffect(() => {
    if (!tenantId || !styleId) return;
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tenantId, styleId]);

  async function refresh() {
    setLoading(true);
    try {
      const { data: activeStats, error: e1 } = await supabase.rpc("get_designer_profile", {
        p_tenant_id: tenantId,
        p_style_id: styleId,
        p_area_type: areaType || null
      });
      if (e1) throw e1;

      const { data: defId, error: e0 } = await supabase.rpc("get_default_designer_style", { p_tenant_id: tenantId });
      if (e0) throw e0;

      let teamStats = null;
      if (defId) {
        const { data: tStats, error: e2 } = await supabase.rpc("get_designer_profile", {
          p_tenant_id: tenantId,
          p_style_id: defId,
          p_area_type: areaType || null
        });
        if (e2) throw e2;
        teamStats = tStats;
      }

      setCompare({ team: teamStats, active: activeStats });
      setProfile({ stats: activeStats, updated_at: new Date().toISOString() });
    } finally {
      setLoading(false);
    }
  }

  async function recompute() {
    setLoading(true);
    try {
      const { data, error } = await supabase.rpc("recompute_designer_profile", { p_tenant_id: tenantId, p_style_id: styleId });
      if (error) return alert("RPC error: " + error.message);
      setProfile({ stats: data, updated_at: new Date().toISOString() });
    } finally {
      setLoading(false);
    }
  }

  if (!session) {
    return (
      <div style={{ padding: 22 }}>
        <h2>Designer Brain</h2>
        <p>Login richiesto.</p>
        <Link href="/login">Vai al login</Link>
      </div>
    );
  }

  const stats = profile?.stats || {};
  const n = Number(stats.n || 0);

  function renderDist(title, obj) {
    const entries = Object.entries(obj || {}).sort((a,b)=>Number(b[1])-Number(a[1])).slice(0,10);
    return (
      <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
        <b>{title}</b>
        {entries.length === 0 ? (
          <p style={{ fontSize: 13, color:"#6b7280", marginTop: 8 }}>Nessun dato.</p>
        ) : (
          <div style={{ marginTop: 8 }}>
            {entries.map(([k,v]) => (
              <div key={k} style={{ display:"flex", justifyContent:"space-between", fontSize: 13, padding:"6px 0", borderBottom:"1px dashed #e5e7eb" }}>
                <span>{k}</span>
                <span><b>{v}</b> <span style={{ color:"#6b7280" }}>({pct(Number(v), n)})</span></span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={{ padding: 22, maxWidth: 1100, margin:"0 auto" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", gap: 12 }}>
        <h2 style={{ margin: 0 }}>🧠 Designer Brain (Team DNA)</h2>
        <div style={{ display:"flex", gap: 12 }}>
          <Link href="/projects">Progetti</Link>
          <Link href="/knowledge">Knowledge</Link>
          <Link href="/designer-brain">Designer Brain</Link>
        </div>
      </div>

      <div style={{ marginTop: 12, background:"#fff", border:"1px solid #e5e7eb", borderRadius: 12, padding: 14 }}>
        <div style={{ display:"flex", gap: 12, alignItems:"center", flexWrap:"wrap" }}>
          <label style={{ fontSize: 13, color:"#374151" }}>
            Style:
            <select value={styleId} onChange={(e)=>setStyleId(e.target.value)} style={{ marginLeft: 10, padding:"8px 10px", borderRadius: 10, border:"1px solid #e5e7eb" }}>
              {styles.map(s => (
                <option key={s.id} value={s.id}>
                  {s.is_default ? "⭐ " : ""}{s.name}{s.scope === "client" ? ` (cliente: ${s.client_name || "—"})` : ""}
                </option>
              ))}
            </select>
          </label>
<label style={{ fontSize: 13, color:"#374151" }}>
  Filtro area type:
  <select value={areaType} onChange={(e)=>setAreaType(e.target.value)} style={{ marginLeft: 10, padding:"8px 10px", borderRadius: 10, border:"1px solid #e5e7eb" }}>
    <option value="">(tutte)</option>
    {(areaTypes || []).map(t => (
      <option key={t} value={t}>{t}</option>
    ))}
  </select>
</label>

          <button className="btn" onClick={refresh} disabled={loading}>Aggiorna</button>
          <button className="btn" onClick={recompute} disabled={loading}>Ricalcola profilo</button>
          <span style={{ fontSize: 12, color:"#6b7280" }}>
            eventi: <b>{n}</b> • updated: {profile?.updated_at ? new Date(profile.updated_at).toLocaleString() : "—"}
          </span>
        </div>
        <p style={{ marginTop: 10, fontSize: 13, color:"#6b7280" }}>
          LuxIA impara dalle scelte confermate del team: concept, brand, CCT, mood, densità e target. Usa questo profilo per ordinare e adattare i concept futuri.
        </p>
      </div>

      <div style={{ marginTop: 14, display:"grid", gridTemplateColumns:"repeat(2, 1fr)", gap: 12 }}>
        {renderDist("Concept più scelti", stats.concept)}
        {renderDist("Brand preferiti", stats.brand)}
        {renderDist("CCT preferite", stats.cct)}
        {renderDist("Mood preferiti", stats.mood)}
      </div>

      <div style={{ marginTop: 12, display:"grid", gridTemplateColumns:"repeat(3, 1fr)", gap: 12 }}>
        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>Densità media</b>
          <div style={{ fontSize: 22, marginTop: 8 }}>{Number(stats.avg_density || 0).toFixed(3)} <span style={{ fontSize: 12, color:"#6b7280" }}>n/m²</span></div>
        </div>
        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>W/m² medio</b>
          <div style={{ fontSize: 22, marginTop: 8 }}>{Number(stats.avg_wm2 || 0).toFixed(2)} <span style={{ fontSize: 12, color:"#6b7280" }}>W/m²</span></div>
        </div>
        <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
          <b>UGR medio</b>
          <div style={{ fontSize: 22, marginTop: 8 }}>{Number(stats.avg_ugr || 0).toFixed(1)}</div>
        </div>
      
<div style={{ marginTop: 14, background:"#fff", border:"1px solid #e5e7eb", borderRadius: 12, padding: 14 }}>
  <h3 style={{ margin:"0 0 10px 0" }}>🔁 Team vs Stile Attivo {areaType ? `(Filtro: ${areaType})` : ""}</h3>
  <div style={{ display:"grid", gridTemplateColumns:"repeat(2, 1fr)", gap: 12 }}>
    <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
      <b>Team Default</b>
      <p style={{ marginTop: 6, fontSize: 13, color:"#6b7280" }}>Eventi: <b>{Number(compare.team?.n || 0)}</b></p>
      {renderDist("Concept (Team)", compare.team?.concept || {})}
    </div>
    <div style={{ border:"1px solid #e5e7eb", borderRadius: 12, padding: 12 }}>
      <b>Stile attivo</b>
      <p style={{ marginTop: 6, fontSize: 13, color:"#6b7280" }}>Eventi: <b>{Number(compare.active?.n || 0)}</b></p>
      {renderDist("Concept (Stile)", compare.active?.concept || {})}
    </div>
  </div>
</div>

</div>
    </div>
  );
}
