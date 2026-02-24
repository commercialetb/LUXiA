-- LuxIA v16: Blender Cycles render jobs params (4:3, moods, cameras)
-- Extends project_jobs payload schema via convention (no table alteration needed)
-- Adds a helper table to track render assets per concept/area.

create table if not exists public.project_renders (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references public.projects(id) on delete cascade,
  owner_id uuid references auth.users(id) on delete cascade,
  concept_id uuid,
  area_name text,
  camera text not null, -- technical | client
  mood text,
  quality text,
  width int,
  height int,
  storage_path text not null,
  created_at timestamptz default now()
);

alter table public.project_renders enable row level security;

create policy "project_renders_select" on public.project_renders
for select using (
  owner_id = auth.uid()
  or exists (
    select 1 from public.project_members pm
    where pm.project_id = project_renders.project_id
      and pm.user_id = auth.uid()
  )
);

create policy "project_renders_insert" on public.project_renders
for insert with check (
  owner_id = auth.uid()
  and (
    project_id is null
    or exists (
      select 1 from public.project_members pm
      where pm.project_id = project_renders.project_id
        and pm.user_id = auth.uid()
    )
  )
);

create policy "project_renders_delete" on public.project_renders
for delete using ( owner_id = auth.uid() );
