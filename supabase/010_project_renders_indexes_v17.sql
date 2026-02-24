-- LuxIA v17: indexes + concept_type (optional)
alter table public.project_renders
  add column if not exists concept_type text;

create index if not exists project_renders_proj_area_cam_idx
  on public.project_renders(project_id, area_name, camera, created_at desc);

create index if not exists project_renders_concept_idx
  on public.project_renders(concept_id);
