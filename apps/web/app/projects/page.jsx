import { createServer } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function ProjectsIndex() {
  const supabase = createServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return (
      <div className="card">
        <h1 style={{ marginTop: 0 }}>Progetti</h1>
        <p>Devi fare login per vedere i progetti.</p>
        <a className="btn" href="/auth/login">Vai al login</a>
      </div>
    );
  }

  const { data: memberships, error: mErr } = await supabase
    .from("tenant_users")
    .select("tenant_id, role")
    .eq("user_id", user.id);

  if (mErr) {
    return (
      <div className="card">
        <h1 style={{ marginTop: 0 }}>Progetti</h1>
        <p style={{ color: "#fca5a5" }}>Errore lettura memberships: {mErr.message}</p>
      </div>
    );
  }

  const tenantIds = (memberships || []).map((m) => m.tenant_id);
  const { data: projects, error: pErr } = await supabase
    .from("projects")
    .select("id, tenant_id, name, project_type, created_at")
    .in("tenant_id", tenantIds.length ? tenantIds : ["00000000-0000-0000-0000-000000000000"])
    .order("created_at", { ascending: false });

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <h1 style={{ margin: 0 }}>Progetti</h1>
        <a className="btn" href="/projects/new" style={{ marginLeft: "auto" }}>+ Nuovo progetto</a>
      </div>

      {pErr && (
        <div className="card" style={{ borderColor: "rgba(239,68,68,0.35)" }}>
          <p style={{ color: "#fca5a5" }}>Errore lettura progetti: {pErr.message}</p>
        </div>
      )}

      {!projects?.length ? (
        <div className="card">
          <p>Nessun progetto trovato. Crea il primo da qui:</p>
          <a className="btn" href="/projects/new">+ Crea progetto</a>
        </div>
      ) : (
        <div className="card">
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "#94a3b8" }}>
                <th style={{ padding: "10px 8px" }}>Nome</th>
                <th style={{ padding: "10px 8px" }}>Tipo</th>
                <th style={{ padding: "10px 8px" }}>Creato</th>
                <th style={{ padding: "10px 8px" }} />
              </tr>
            </thead>
            <tbody>
              {projects.map((p) => (
                <tr key={p.id} style={{ borderTop: "1px solid rgba(148,163,184,0.15)" }}>
                  <td style={{ padding: "10px 8px", fontWeight: 700 }}>{p.name}</td>
                  <td style={{ padding: "10px 8px" }}>{p.project_type || "-"}</td>
                  <td style={{ padding: "10px 8px", color: "#cbd5e1" }}>{new Date(p.created_at).toLocaleString()}</td>
                  <td style={{ padding: "10px 8px", textAlign: "right" }}>
                    <a className="btn" href={`/projects/${p.id}`}>Apri</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
