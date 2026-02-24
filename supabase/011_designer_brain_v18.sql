-- LuxIA v18: Designer Brain (Team) + Client Styles (tenant-based)
-- Stores learning events, aggregated profiles, and project style selection.

-- 1) Styles (team default + client/project variants)
create table if not exists public.designer_styles (
  id uuid primary key default uuid_generate_v4(),
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  name text not null,
  scope text not null default 'tenant', -- tenant | client | project
  client_name text,
  is_default boolean default false,
  created_by uuid references auth.users(id) on delete set null,
  created_at timestamptz default now()
);

alter table public.designer_styles enable row level security;

create policy "designer_styles_select" on public.designer_styles
for select using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = designer_styles.tenant_id
      and tu.user_id = auth.uid()
  )
);

create policy "designer_styles_insert" on public.designer_styles
for insert with check (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = designer_styles.tenant_id
      and tu.user_id = auth.uid()
  )
);

create policy "designer_styles_update" on public.designer_styles
for update using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = designer_styles.tenant_id
      and tu.user_id = auth.uid()
  )
);

-- 2) Link project -> active style
alter table public.projects
  add column if not exists active_style_id uuid references public.designer_styles(id);

-- 3) Learning events
create table if not exists public.designer_learning_events (
  id uuid primary key default uuid_generate_v4(),
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  style_id uuid not null references public.designer_styles(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  area_name text,
  area_type text,
  selected_concept_type text, -- comfort|efficiency|architectural
  selected_brand text,
  selected_cct int,
  selected_mood text,
  n_luminaires int,
  area_m2 numeric,
  density numeric, -- n/m2
  wm2 numeric,
  ugr numeric,
  created_by uuid references auth.users(id) on delete set null,
  created_at timestamptz default now()
);

alter table public.designer_learning_events enable row level security;

create policy "designer_learning_events_select" on public.designer_learning_events
for select using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = designer_learning_events.tenant_id
      and tu.user_id = auth.uid()
  )
);

create policy "designer_learning_events_insert" on public.designer_learning_events
for insert with check (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = designer_learning_events.tenant_id
      and tu.user_id = auth.uid()
  )
);

-- 4) Aggregated profile (updated by RPC)
create table if not exists public.designer_team_profile (
  id uuid primary key default uuid_generate_v4(),
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  style_id uuid not null references public.designer_styles(id) on delete cascade,
  stats jsonb not null default '{}'::jsonb,
  updated_at timestamptz default now(),
  unique(tenant_id, style_id)
);

alter table public.designer_team_profile enable row level security;

create policy "designer_team_profile_select" on public.designer_team_profile
for select using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = designer_team_profile.tenant_id
      and tu.user_id = auth.uid()
  )
);

create policy "designer_team_profile_update" on public.designer_team_profile
for update using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = designer_team_profile.tenant_id
      and tu.user_id = auth.uid()
  )
);

create policy "designer_team_profile_upsert" on public.designer_team_profile
for insert with check (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = designer_team_profile.tenant_id
      and tu.user_id = auth.uid()
  )
);

-- 5) RPC: recompute profile stats from events
create or replace function public.recompute_designer_profile(p_tenant_id uuid, p_style_id uuid)
returns jsonb
language plpgsql
security definer
as $$
declare
  v_stats jsonb;
begin
  with ev as (
    select * from public.designer_learning_events
    where tenant_id = p_tenant_id and style_id = p_style_id
  ),
  concept as (
    select selected_concept_type as k, count(*) as c from ev
    where selected_concept_type is not null
    group by selected_concept_type
  ),
  brand as (
    select selected_brand as k, count(*) as c from ev
    where selected_brand is not null
    group by selected_brand
  ),
  cct as (
    select selected_cct::text as k, count(*) as c from ev
    where selected_cct is not null
    group by selected_cct
  ),
  mood as (
    select selected_mood as k, count(*) as c from ev
    where selected_mood is not null
    group by selected_mood
  ),
  agg as (
    select
      coalesce(avg(density),0) as avg_density,
      coalesce(avg(wm2),0) as avg_wm2,
      coalesce(avg(ugr),0) as avg_ugr,
      count(*) as n
    from ev
  )
  select jsonb_build_object(
    'n', (select n from agg),
    'concept', (select coalesce(jsonb_object_agg(k,c), '{}'::jsonb) from concept),
    'brand', (select coalesce(jsonb_object_agg(k,c), '{}'::jsonb) from brand),
    'cct', (select coalesce(jsonb_object_agg(k,c), '{}'::jsonb) from cct),
    'mood', (select coalesce(jsonb_object_agg(k,c), '{}'::jsonb) from mood),
    'avg_density', (select avg_density from agg),
    'avg_wm2', (select avg_wm2 from agg),
    'avg_ugr', (select avg_ugr from agg)
  ) into v_stats;

  insert into public.designer_team_profile(tenant_id, style_id, stats, updated_at)
  values (p_tenant_id, p_style_id, v_stats, now())
  on conflict (tenant_id, style_id)
  do update set stats = excluded.stats, updated_at = now();

  return v_stats;
end;
$$;

