-- LuxIA v21: Project-based Style Pack Engine (editable sliders)

create table if not exists public.project_style_packs (
  id uuid primary key default uuid_generate_v4(),
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  pack jsonb not null default '{}'::jsonb,
  created_by uuid references auth.users(id) on delete set null,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique(tenant_id, project_id)
);

alter table public.project_style_packs enable row level security;

create policy "project_style_packs_select" on public.project_style_packs
for select using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = project_style_packs.tenant_id
      and tu.user_id = auth.uid()
  )
);

create policy "project_style_packs_insert" on public.project_style_packs
for insert with check (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = project_style_packs.tenant_id
      and tu.user_id = auth.uid()
  )
);

create policy "project_style_packs_update" on public.project_style_packs
for update using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = project_style_packs.tenant_id
      and tu.user_id = auth.uid()
  )
);

create or replace function public.get_project_style_pack(p_project_id uuid)
returns jsonb
language sql
security definer
as $$
  select coalesce(psp.pack, '{}'::jsonb)
  from public.project_style_packs psp
  where psp.project_id = p_project_id
  limit 1;
$$;

create or replace function public.upsert_project_style_pack(p_tenant_id uuid, p_project_id uuid, p_pack jsonb)
returns jsonb
language plpgsql
security definer
as $$
declare
  v jsonb;
begin
  insert into public.project_style_packs(tenant_id, project_id, pack, created_by, created_at, updated_at)
  values (p_tenant_id, p_project_id, p_pack, auth.uid(), now(), now())
  on conflict (tenant_id, project_id)
  do update set pack = excluded.pack, updated_at = now()
  returning pack into v;

  return v;
end;
$$;
