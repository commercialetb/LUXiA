import { redirect } from "next/navigation";
import { supabaseServer } from "@/lib/supabase/server";

export default async function Home() {
  const supabase = supabaseServer();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) redirect("/auth/login");

  const { data: memberships } = await supabase
    .from("tenant_users")
    .select("tenant_id, role, tenants(name)")
    .eq("user_id", user.id);

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
      <header style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom: 14 }}>
        <div>
          <div style={{ fontWeight: 900, fontSize: 20 }}>💡 LuxIA SaaS MVP</div>
          <div style={{ opacity: 0.85, fontSize: 13 }}>Loggato come: {user.email}</div>
        </div>
        <form action="/auth/logout" method="post">
          <button style={btn()}>Logout</button>
        </form>
      </header>

      <section style={card()}>
        <h2 style={{ marginTop: 0 }}>I tuoi Studi (Tenants)</h2>
        {!memberships?.length ? (
          <p style={{ opacity: 0.9 }}>
            Nessuno studio trovato. Vai su <b>/studio/new</b> per crearne uno (MVP). Poi crea un progetto su <b>/projects/new</b>. Gestisci il catalogo su <b>/catalog</b>.
          </p>
        ) : (
          <ul>
            {memberships.map((m) => (
              <li key={m.tenant_id}>
                <b>{m.tenants?.name || m.tenant_id}</b> — ruolo: {m.role}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section style={{ ...card(), marginTop: 12 }}>
        <h2 style={{ marginTop: 0 }}>Prossimo step UI</h2>
        <ol style={{ lineHeight: 1.7 }}>
          <li>Wizard crea studio (tenant) + auto-add owner</li>
          <li>Wizard crea progetto + upload planimetria su Storage</li>
          <li>Pagina progetto: aree → concept → scelta → learning</li>
        </ol>
      </section>
    </main>
  );
}

const card = ()=>({ background:"#0f172a", border:"1px solid #24304a", borderRadius:14, padding:14 });
const btn  = ()=>({ background:"#2563eb", border:"none", padding:"10px 14px", borderRadius:10, color:"white", fontWeight:800, cursor:"pointer" });
