-- LuxIA v14: Project exports (stored files like PPTX/PDF/DXF)
create table if not exists public.project_exports (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references public.projects(id) on delete cascade,
  created_by uuid references auth.users(id) on delete cascade,
  kind text not null, -- pptx|pdf|dxf|zip|gltf
  file_path text not null,
  meta jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

alter table public.project_exports enable row level security;

create policy "project_exports_select" on public.project_exports
for select using (
  exists (
    select 1 from public.project_members pm
    where pm.project_id = project_exports.project_id and pm.user_id = auth.uid()
  )
);

create policy "project_exports_insert" on public.project_exports
for insert with check (
  created_by = auth.uid()
  and exists (
    select 1 from public.project_members pm
    where pm.project_id = project_exports.project_id and pm.user_id = auth.uid()
  )
);
