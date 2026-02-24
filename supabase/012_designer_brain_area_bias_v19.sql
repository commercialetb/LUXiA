-- LuxIA v19: Client name + auto-style + area-type bias

-- 1) project client name
alter table public.projects
  add column if not exists client_name text;

-- 2) RPC: get concept distribution for a given area type (within style)
create or replace function public.get_designer_bias_for_area(p_tenant_id uuid, p_style_id uuid, p_area_type text)
returns jsonb
language plpgsql
security definer
as $$
declare
  v jsonb;
begin
  with ev as (
    select * from public.designer_learning_events
    where tenant_id = p_tenant_id and style_id = p_style_id
  ),
  ev_area as (
    select * from ev where area_type = p_area_type
  ),
  concept_all as (
    select selected_concept_type as k, count(*) as c from ev
    where selected_concept_type is not null
    group by selected_concept_type
  ),
  concept_area as (
    select selected_concept_type as k, count(*) as c from ev_area
    where selected_concept_type is not null
    group by selected_concept_type
  )
  select jsonb_build_object(
    'concept_all', (select coalesce(jsonb_object_agg(k,c), '{}'::jsonb) from concept_all),
    'concept_area', (select coalesce(jsonb_object_agg(k,c), '{}'::jsonb) from concept_area),
    'n_all', (select count(*) from ev),
    'n_area', (select count(*) from ev_area)
  ) into v;

  return v;
end;
$$;

