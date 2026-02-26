import { redirect } from "next/navigation";
import { supabaseServer } from "@/lib/supabase/server";

export default async function StudioIndexPage() {
  const supabase = supabaseServer();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/auth/login");

  const { data: memberships, error } = await supabase
    .from("tenant_users")
    .select("tenant_id, role, tenants(id, name)")
    .eq("user_id", user.id);

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
      <h1 style={{ color: "#fff", marginTop: 0 }}>Studi</h1>

      {error && (
        <div style={warn()}>
          Errore nel caricare gli studi: {error.message}
        </div>
      )}

      {!memberships?.length ? (
        <div style={{ color: "#94a3b8" }}>
          Nessuno studio trovato. Crea il primo da <b>/studio/new</b>.
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 }}>
          {memberships.map((m) => (
            <div key={m.tenant_id} style={card()}>
              <div style={{ fontWeight: 900, color: "#fff", fontSize: 16 }}>
                {m.tenants?.name || m.tenant_id}
              </div>
              <div style={{ color: "#94a3b8", fontSize: 13, marginTop: 6 }}>Ruolo: {m.role}</div>
              <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
                <a href={`/projects?tenant=${m.tenant_id}`} style={pill()}>Vedi progetti</a>
                <a href={`/projects/new?tenant=${m.tenant_id}`} style={pill()}>+ Nuovo progetto</a>
              </div>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}

const card = () => ({ background: "#0f172a", border: "1px solid #24304a", borderRadius: 14, padding: 14 });
const pill = () => ({
  display: "inline-block",
  padding: "8px 10px",
  borderRadius: 999,
  border: "1px solid rgba(148,163,184,0.35)",
  color: "#e5e7eb",
  textDecoration: "none",
  fontSize: 13,
});
const warn = () => ({
  padding: 12,
  borderRadius: 10,
  border: "1px solid rgba(251,191,36,0.35)",
  background: "rgba(251,191,36,0.08)",
  color: "#fbbf24",
  marginBottom: 14,
});
