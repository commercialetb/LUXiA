-- LuxIA SaaS: Core schema (multi-tenant)
-- Run in Supabase SQL editor.

create extension if not exists "uuid-ossp";

-- Tenants (studios)
create table if not exists public.tenants (
  id uuid primary key default uuid_generate_v4(),
  name text not null,
  created_at timestamptz not null default now()
);

-- Tenant membership (maps auth.users -> tenants)
create table if not exists public.tenant_users (
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null default 'editor', -- owner|editor|viewer
  created_at timestamptz not null default now(),
  primary key (tenant_id, user_id)
);

-- Studio preferences (LuxIA "firma")
create table if not exists public.studio_profile (
  tenant_id uuid primary key references public.tenants(id) on delete cascade,
  preferred_brands jsonb not null default '["BEGA"]'::jsonb,
  cct_preference jsonb not null default '{"default":3000}'::jsonb,
  optics_preference jsonb not null default '{}'::jsonb,
  style_tokens jsonb not null default '{}'::jsonb,
  pdf_theme jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

-- Projects
create table if not exists public.projects (
  id uuid primary key default uuid_generate_v4(),
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  name text not null,
  project_type text not null default 'office', -- office|hospitality|retail|residential
  created_by uuid not null references auth.users(id) on delete restrict,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  planimetry_path text null  -- storage path (bucket/object)
);

-- Areas
create table if not exists public.areas (
  id uuid primary key default uuid_generate_v4(),
  project_id uuid not null references public.projects(id) on delete cascade,
  name text not null,
  tipo_locale text not null,
  superficie_m2 numeric not null,
  altezza_m numeric not null default 2.7,
  vdt boolean not null default false,
  notes text null,
  created_at timestamptz not null default now()
);

-- Concepts
create table if not exists public.concepts (
  id uuid primary key default uuid_generate_v4(),
  area_id uuid not null references public.areas(id) on delete cascade,
  concept_type text not null, -- comfort|efficiency|architectural
  solution jsonb not null default '{}'::jsonb,
  metrics jsonb not null default '{}'::jsonb,
  renders jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- Decisions (learning signal)
create table if not exists public.decisions (
  id uuid primary key default uuid_generate_v4(),
  tenant_id uuid not null references public.tenants(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  area_id uuid not null references public.areas(id) on delete cascade,
  chosen_concept_id uuid null references public.concepts(id) on delete set null,
  feedback_text text null,
  edits jsonb not null default '{}'::jsonb,
  created_by uuid not null references auth.users(id) on delete restrict,
  created_at timestamptz not null default now()
);

-- Helpful trigger to update updated_at on projects
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_projects_updated_at on public.projects;
create trigger trg_projects_updated_at
before update on public.projects
for each row execute function public.set_updated_at();
