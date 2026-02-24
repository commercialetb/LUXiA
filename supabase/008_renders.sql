-- LuxIA v15: Rendering pipeline tables/policies (free-first)
-- Uses project_jobs + project_exports already present.
-- Create a storage bucket named `renders` (private recommended).
-- This migration adds optional helper enum values and storage policies (best-effort).

-- Storage policies for bucket 'renders'
alter table storage.objects enable row level security;

-- Allow authenticated users to read their tenant render files via signed URLs (usually served by service role).
-- We keep policies permissive for MVP. Tighten later per tenant_id.
create policy if not exists "renders_select_auth"
on storage.objects for select
to authenticated
using (bucket_id = 'renders');

create policy if not exists "renders_insert_auth"
on storage.objects for insert
to authenticated
with check (bucket_id = 'renders');

create policy if not exists "renders_update_auth"
on storage.objects for update
to authenticated
using (bucket_id = 'renders')
with check (bucket_id = 'renders');

create policy if not exists "renders_delete_auth"
on storage.objects for delete
to authenticated
using (bucket_id = 'renders');
