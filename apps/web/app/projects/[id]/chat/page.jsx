import { redirect } from "next/navigation";
import { supabaseServer } from "@/lib/supabase/server";
import ChatClient from "./ChatClient";

export default async function ProjectChatPage({ params }) {
  const supabase = supabaseServer();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/auth/login");

  const { data: project, error: pErr } = await supabase
    .from("projects")
    .select("id, name")
    .eq("id", params.id)
    .single();
  if (pErr || !project) {
    return (
      <main style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
        <h1>Progetto non trovato</h1>
        <a href="/" style={{ color:"#93c5fd" }}>← Home</a>
      </main>
    );
  }

  const { data: messages } = await supabase
    .from("project_chat_messages")
    .select("id, role, content, meta, created_at")
    .eq("project_id", params.id)
    .order("created_at", { ascending: true })
    .limit(200);

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
      <header style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom: 14 }}>
        <div style={{ fontWeight: 900, fontSize: 20 }}>💬 Chat — {project.name}</div>
        <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
          <a href={`/projects/${project.id}`} style={{ color:"#93c5fd" }}>← Progetto</a>
          <a href={`/projects/${project.id}/review`} style={{ color:"#93c5fd" }}>🧠 Review & Learning</a>
        </div>
      </header>

      <ChatClient projectId={project.id} initialMessages={messages || []} />
    </main>
  );
}
