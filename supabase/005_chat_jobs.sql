-- LuxIA SaaS v11: Project chat + jobs orchestration (MVP)
-- Run AFTER previous schema/RLS scripts.

-- Chat messages (per project)
create table if not exists public.project_chat_messages (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  created_by uuid not null references auth.users(id) on delete restrict,
  role text not null check (role in ('user','assistant','system')),
  content text not null,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists project_chat_messages_project_id_idx
  on public.project_chat_messages(project_id, created_at);

-- Jobs (long tasks)
create table if not exists public.project_jobs (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  created_by uuid not null references auth.users(id) on delete restrict,
  job_type text not null, -- generate_concepts|render|ppt|export|ingest
  status text not null default 'queued' check (status in ('queued','running','done','error')),
  input jsonb not null default '{}'::jsonb,
  output jsonb not null default '{}'::jsonb,
  error text null,
  started_at timestamptz null,
  finished_at timestamptz null,
  created_at timestamptz not null default now()
);

create index if not exists project_jobs_project_id_idx
  on public.project_jobs(project_id, created_at);

-- RLS
alter table public.project_chat_messages enable row level security;
alter table public.project_jobs enable row level security;

-- Helper: tenant membership
-- tenant_users(user_id, tenant_id)
-- projects(tenant_id)

create policy "chat read for tenant members" on public.project_chat_messages
for select to authenticated
using (
  exists (
    select 1
    from public.projects p
    join public.tenant_users tu on tu.tenant_id = p.tenant_id
    where p.id = project_chat_messages.project_id
      and tu.user_id = auth.uid()
  )
);

create policy "chat insert for tenant members" on public.project_chat_messages
for insert to authenticated
with check (
  created_by = auth.uid()
  and exists (
    select 1
    from public.projects p
    join public.tenant_users tu on tu.tenant_id = p.tenant_id
    where p.id = project_chat_messages.project_id
      and tu.user_id = auth.uid()
  )
);

create policy "jobs read for tenant members" on public.project_jobs
for select to authenticated
using (
  exists (
    select 1
    from public.projects p
    join public.tenant_users tu on tu.tenant_id = p.tenant_id
    where p.id = project_jobs.project_id
      and tu.user_id = auth.uid()
  )
);

create policy "jobs insert for tenant members" on public.project_jobs
for insert to authenticated
with check (
  created_by = auth.uid()
  and exists (
    select 1
    from public.projects p
    join public.tenant_users tu on tu.tenant_id = p.tenant_id
    where p.id = project_jobs.project_id
      and tu.user_id = auth.uid()
  )
);

create policy "jobs update own" on public.project_jobs
for update to authenticated
using (created_by = auth.uid())
with check (created_by = auth.uid());
