import { redirect } from "next/navigation";
import { supabaseServer } from "@/lib/supabase/server";
import AreaClient from "./AreaClient";
import ProjectStylePackClient from "./ProjectStylePackClient";

export default async function ProjectPage({ params }) {
  const supabase = supabaseServer();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/auth/login");

  const { data: project, error } = await supabase
    .from("projects")
    .select("id, tenant_id, active_style_id, name, client_name, project_type, planimetry_path, created_at")
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
      <header style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom: 14 }}>
        <div>
          <div style={{ fontWeight: 900, fontSize: 20 }}>📁 {project.name}</div>
          <div style={{ opacity: 0.85, fontSize: 13 }}>
            Tipo: {project.project_type} • Bucket: LuxIA • Planimetria: {project.planimetry_path || "—"}
          </div>
        </div>
        <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}><a href={`/projects/${project.id}/review`} style={{ color:"#93c5fd" }}>🧠 Review & Learning</a><a href="/designer-brain" style={{ color:"#93c5fd" }}>🧬 Designer Brain</a><a href={`/projects/${project.id}/chat`} style={{ color:"#93c5fd" }}>💬 Chat</a><a href="/projects/new" style={{ color:"#93c5fd" }}>+ Nuovo progetto</a></div>
      </header>

      <AreaClient project={project} />
      <ProjectStylePackClient projectId={project.id} />
    </main>
  );
}
