-- LuxIA v12: Knowledge Base + RAG (free-first)
-- Creates knowledge docs/chunks tables and a full-text search RPC.
-- Optional: if pgvector enabled, you can later add embeddings.

create table if not exists public.knowledge_documents (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references public.projects(id) on delete cascade,
  owner_id uuid references auth.users(id) on delete cascade,
  title text,
  file_path text not null,
  mime_type text,
  source_type text default 'upload',
  tags text[] default '{}',
  created_at timestamptz default now()
);

alter table public.knowledge_documents enable row level security;

create policy "knowledge_documents_select" on public.knowledge_documents
for select using (
  owner_id = auth.uid()
  or exists (
    select 1 from public.project_members pm
    where pm.project_id = knowledge_documents.project_id
      and pm.user_id = auth.uid()
  )
);

create policy "knowledge_documents_insert" on public.knowledge_documents
for insert with check (
  owner_id = auth.uid()
  and (
    project_id is null
    or exists (
      select 1 from public.project_members pm
      where pm.project_id = knowledge_documents.project_id
        and pm.user_id = auth.uid()
    )
  )
);

create policy "knowledge_documents_delete" on public.knowledge_documents
for delete using ( owner_id = auth.uid() );

create table if not exists public.knowledge_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references public.knowledge_documents(id) on delete cascade,
  project_id uuid references public.projects(id) on delete cascade,
  chunk_index int not null,
  content text not null,
  meta jsonb default '{}'::jsonb,
  tsv tsvector generated always as (to_tsvector('simple', coalesce(content,''))) stored,
  created_at timestamptz default now()
);

alter table public.knowledge_chunks enable row level security;

create policy "knowledge_chunks_select" on public.knowledge_chunks
for select using (
  exists (
    select 1 from public.knowledge_documents d
    where d.id = knowledge_chunks.document_id
      and (
        d.owner_id = auth.uid()
        or exists (
          select 1 from public.project_members pm
          where pm.project_id = d.project_id and pm.user_id = auth.uid()
        )
      )
  )
);

create policy "knowledge_chunks_insert" on public.knowledge_chunks
for insert with check (
  exists (
    select 1 from public.knowledge_documents d
    where d.id = knowledge_chunks.document_id and d.owner_id = auth.uid()
  )
);

create index if not exists knowledge_chunks_tsv_idx on public.knowledge_chunks using gin (tsv);

create table if not exists public.knowledge_jobs (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references public.knowledge_documents(id) on delete cascade,
  project_id uuid references public.projects(id) on delete cascade,
  owner_id uuid references auth.users(id) on delete cascade,
  status text not null default 'queued',
  error text,
  stats jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

alter table public.knowledge_jobs enable row level security;

create policy "knowledge_jobs_select" on public.knowledge_jobs
for select using (owner_id = auth.uid());

create policy "knowledge_jobs_insert" on public.knowledge_jobs
for insert with check (owner_id = auth.uid());

create policy "knowledge_jobs_update" on public.knowledge_jobs
for update using (owner_id = auth.uid());

create or replace function public.search_knowledge(
  q text,
  p_project_id uuid default null,
  max_rows int default 8
)
returns table (
  chunk_id uuid,
  document_id uuid,
  title text,
  content text,
  rank real,
  meta jsonb
)
language sql
stable
as $$
  select
    c.id as chunk_id,
    c.document_id,
    d.title,
    c.content,
    ts_rank_cd(c.tsv, plainto_tsquery('simple', q)) as rank,
    c.meta
  from public.knowledge_chunks c
  join public.knowledge_documents d on d.id = c.document_id
  where (p_project_id is null or c.project_id = p_project_id)
    and c.tsv @@ plainto_tsquery('simple', q)
    and (
      d.owner_id = auth.uid()
      or exists (
        select 1 from public.project_members pm
        where pm.project_id = d.project_id and pm.user_id = auth.uid()
      )
    )
  order by rank desc
  limit max_rows;
$$;
