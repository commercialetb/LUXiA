import { redirect } from "next/navigation";
import { supabaseServer } from "@/lib/supabase/server";
import ReviewClient from "./ReviewClient";

export default async function ReviewPage({ params }) {
  const supabase = supabaseServer();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/auth/login");

  const { data: project, error } = await supabase
    .from("projects")
    .select("id, tenant_id, active_style_id, name, project_type, planimetry_path, created_at")
    .eq("id", params.id)
    .single();

  if (error || !project) {
    return (
      <main style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
        <h1>Progetto non trovato</h1>
        <a href="/" style={{ color:"#93c5fd" }}>← Home</a>
      </main>
    );
  }

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
      <header style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom: 14, gap: 10, flexWrap:"wrap" }}>
        <div>
          <div style={{ fontWeight: 900, fontSize: 20 }}>🧠 Review — {project.name}</div>
          <div style={{ opacity: 0.85, fontSize: 13 }}>
            Bucket: LuxIA • Learning attivo (decisions + studio_profile)
          </div>
        </div>
        <a href={`/projects/${project.id}`} style={{ color:"#93c5fd" }}>← Progetto</a>
      </header>

      <ReviewClient project={project} />
    </main>
  );
}
