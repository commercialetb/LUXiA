-- LuxIA SaaS: RLS policies
-- Enable Row Level Security and allow access only within user's tenant.

alter table public.tenants enable row level security;
alter table public.tenant_users enable row level security;
alter table public.studio_profile enable row level security;
alter table public.projects enable row level security;
alter table public.areas enable row level security;
alter table public.concepts enable row level security;
alter table public.decisions enable row level security;

-- Helper: check membership
create or replace function public.is_tenant_member(tid uuid)
returns boolean language sql stable as $$
  select exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = tid and tu.user_id = auth.uid()
  );
$$;

-- tenants: select only if member
drop policy if exists tenants_select on public.tenants;
create policy tenants_select on public.tenants
for select using (public.is_tenant_member(id));

-- tenants: insert allowed for authenticated users (they create their studio)
drop policy if exists tenants_insert on public.tenants;
create policy tenants_insert on public.tenants
for insert to authenticated
with check (true);

-- tenant_users: members can see their rows
drop policy if exists tenant_users_select on public.tenant_users;
create policy tenant_users_select on public.tenant_users
for select using (user_id = auth.uid() or public.is_tenant_member(tenant_id));

-- tenant_users: only owners can manage membership
create or replace function public.is_tenant_owner(tid uuid)
returns boolean language sql stable as $$
  select exists (
    select 1 from public.tenant_users tu
    where tu.tenant_id = tid and tu.user_id = auth.uid() and tu.role = 'owner'
  );
$$;

drop policy if exists tenant_users_insert on public.tenant_users;
create policy tenant_users_insert on public.tenant_users
for insert to authenticated
with check (public.is_tenant_owner(tenant_id) or user_id = auth.uid());

drop policy if exists tenant_users_update on public.tenant_users;
create policy tenant_users_update on public.tenant_users
for update to authenticated
using (public.is_tenant_owner(tenant_id))
with check (public.is_tenant_owner(tenant_id));

drop policy if exists tenant_users_delete on public.tenant_users;
create policy tenant_users_delete on public.tenant_users
for delete to authenticated
using (public.is_tenant_owner(tenant_id));

-- studio_profile: members can select/update
drop policy if exists studio_profile_select on public.studio_profile;
create policy studio_profile_select on public.studio_profile
for select using (public.is_tenant_member(tenant_id));

drop policy if exists studio_profile_upsert on public.studio_profile;
create policy studio_profile_upsert on public.studio_profile
for insert to authenticated with check (public.is_tenant_member(tenant_id));
create policy studio_profile_update on public.studio_profile
for update to authenticated
using (public.is_tenant_member(tenant_id))
with check (public.is_tenant_member(tenant_id));

-- projects: members can CRUD
drop policy if exists projects_select on public.projects;
create policy projects_select on public.projects
for select using (public.is_tenant_member(tenant_id));

drop policy if exists projects_insert on public.projects;
create policy projects_insert on public.projects
for insert to authenticated
with check (public.is_tenant_member(tenant_id) and created_by = auth.uid());

drop policy if exists projects_update on public.projects;
create policy projects_update on public.projects
for update to authenticated
using (public.is_tenant_member(tenant_id))
with check (public.is_tenant_member(tenant_id));

drop policy if exists projects_delete on public.projects;
create policy projects_delete on public.projects
for delete to authenticated
using (public.is_tenant_member(tenant_id));

-- areas inherit by project
create or replace function public.project_tenant(pid uuid)
returns uuid language sql stable as $$
  select tenant_id from public.projects where id = pid
$$;

drop policy if exists areas_select on public.areas;
create policy areas_select on public.areas
for select using (public.is_tenant_member(public.project_tenant(project_id)));

drop policy if exists areas_insert on public.areas;
create policy areas_insert on public.areas
for insert to authenticated
with check (public.is_tenant_member(public.project_tenant(project_id)));

drop policy if exists areas_update on public.areas;
create policy areas_update on public.areas
for update to authenticated
using (public.is_tenant_member(public.project_tenant(project_id)))
with check (public.is_tenant_member(public.project_tenant(project_id)));

drop policy if exists areas_delete on public.areas;
create policy areas_delete on public.areas
for delete to authenticated
using (public.is_tenant_member(public.project_tenant(project_id)));

-- concepts inherit by area->project
create or replace function public.area_project(aid uuid)
returns uuid language sql stable as $$
  select project_id from public.areas where id = aid
$$;

drop policy if exists concepts_select on public.concepts;
create policy concepts_select on public.concepts
for select using (public.is_tenant_member(public.project_tenant(public.area_project(area_id))));

drop policy if exists concepts_insert on public.concepts;
create policy concepts_insert on public.concepts
for insert to authenticated
with check (public.is_tenant_member(public.project_tenant(public.area_project(area_id))));

drop policy if exists concepts_update on public.concepts;
create policy concepts_update on public.concepts
for update to authenticated
using (public.is_tenant_member(public.project_tenant(public.area_project(area_id))))
with check (public.is_tenant_member(public.project_tenant(public.area_project(area_id))));

drop policy if exists concepts_delete on public.concepts;
create policy concepts_delete on public.concepts
for delete to authenticated
using (public.is_tenant_member(public.project_tenant(public.area_project(area_id))));

-- decisions: members within tenant
drop policy if exists decisions_select on public.decisions;
create policy decisions_select on public.decisions
for select using (public.is_tenant_member(tenant_id));

drop policy if exists decisions_insert on public.decisions;
create policy decisions_insert on public.decisions
for insert to authenticated
with check (public.is_tenant_member(tenant_id) and created_by = auth.uid());
