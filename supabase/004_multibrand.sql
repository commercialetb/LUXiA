-- LuxIA SaaS v6: Multi-brand catalog + per-project brand selection
-- Run AFTER previous schema/RLS scripts.

create table if not exists public.luminaires (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  brand text not null,
  model text not null,
  family text,
  type text,
  mounting text,
  distribution text,
  flux_lm numeric not null,
  watt numeric not null,
  cct text,
  cri int default 80,
  ugr int,
  ip text,
  dimmable boolean default true,
  notes text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_luminaires_tenant_brand on public.luminaires(tenant_id, brand);

create table if not exists public.project_brands (
  project_id uuid not null references public.projects(id) on delete cascade,
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  brand text not null,
  created_at timestamptz default now(),
  primary key (project_id, brand)
);

create index if not exists idx_project_brands_project on public.project_brands(project_id);

alter table public.luminaires enable row level security;
alter table public.project_brands enable row level security;

drop policy if exists "luminaires select" on public.luminaires;
create policy "luminaires select"
on public.luminaires for select
to authenticated
using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = luminaires.tenant_id
      and tu.user_id = auth.uid()
  )
);

drop policy if exists "luminaires insert" on public.luminaires;
create policy "luminaires insert"
on public.luminaires for insert
to authenticated
with check (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = luminaires.tenant_id
      and tu.user_id = auth.uid()
      and tu.role in ('owner','admin')
  )
);

drop policy if exists "luminaires update" on public.luminaires;
create policy "luminaires update"
on public.luminaires for update
to authenticated
using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = luminaires.tenant_id
      and tu.user_id = auth.uid()
      and tu.role in ('owner','admin')
  )
)
with check (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = luminaires.tenant_id
      and tu.user_id = auth.uid()
      and tu.role in ('owner','admin')
  )
);

drop policy if exists "luminaires delete" on public.luminaires;
create policy "luminaires delete"
on public.luminaires for delete
to authenticated
using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = luminaires.tenant_id
      and tu.user_id = auth.uid()
      and tu.role in ('owner','admin')
  )
);

drop policy if exists "project_brands select" on public.project_brands;
create policy "project_brands select"
on public.project_brands for select
to authenticated
using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = project_brands.tenant_id
      and tu.user_id = auth.uid()
  )
);

drop policy if exists "project_brands insert" on public.project_brands;
create policy "project_brands insert"
on public.project_brands for insert
to authenticated
with check (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = project_brands.tenant_id
      and tu.user_id = auth.uid()
      and tu.role in ('owner','admin')
  )
);

drop policy if exists "project_brands delete" on public.project_brands;
create policy "project_brands delete"
on public.project_brands for delete
to authenticated
using (
  exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = project_brands.tenant_id
      and tu.user_id = auth.uid()
      and tu.role in ('owner','admin')
  )
);

-- Seed sample luminaires for every tenant (placeholders)
insert into public.luminaires (tenant_id, brand, model, family, type, mounting, distribution, flux_lm, watt, cct, cri, ugr, ip, dimmable, notes)
select t.id, v.brand, v.model, v.family, v.type, v.mounting, v.distribution, v.flux_lm, v.watt, v.cct, v.cri, v.ugr, v.ip, v.dimmable, v.notes
from public.tenants t
cross join (
  values
    ('BEGA','Downlight 3000lm 25W','Downlight','downlight','recessed','wide',3000,25,'3000K',90,16,'IP20',true,'Placeholder sample'),
    ('iGuzzini','Lineare 4500lm 35W','Laser Blade','linear','recessed','wide',4500,35,'4000K',90,18,'IP20',true,'Placeholder sample'),
    ('Flos','Sospensione 2800lm 22W','Aim','pendant','pendant','wide',2800,22,'3000K',90,15,'IP20',true,'Placeholder sample'),
    ('Delta Light','Downlight 2500lm 20W','Tweeter','downlight','recessed','wide',2500,20,'3000K',90,14,'IP44',true,'Placeholder sample')
) as v(brand,model,family,type,mounting,distribution,flux_lm,watt,cct,cri,ugr,ip,dimmable,notes)
on conflict do nothing;
