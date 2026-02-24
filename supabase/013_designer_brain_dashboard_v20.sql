-- LuxIA v20: Designer Brain dashboard filters + Team vs Client compare + mood hints

alter table public.projects add column if not exists client_name text;
alter table public.projects add column if not exists mood_hint text;

create or replace function public.get_default_designer_style(p_tenant_id uuid)
returns uuid
language sql
security definer
as $$
  select id
  from public.designer_styles
  where tenant_id = p_tenant_id and is_default = true
  order by created_at asc
  limit 1;
$$;

create or replace function public.get_designer_profile(p_tenant_id uuid, p_style_id uuid, p_area_type text default null)
returns jsonb
language plpgsql
security definer
as $$
declare
  v_stats jsonb;
begin
  with ev as (
    select * from public.designer_learning_events
    where tenant_id = p_tenant_id
      and style_id = p_style_id
      and (p_area_type is null or area_type = p_area_type)
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

  return v_stats;
end;
$$;
