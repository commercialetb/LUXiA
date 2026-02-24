-- LuxIA SaaS: Storage policies for bucket 'LuxIA'
-- 1) Create bucket in Supabase dashboard: name = LuxIA (recommended: private)
-- 2) Run this in SQL editor

-- Ensure RLS is enabled for storage.objects (usually is)
alter table storage.objects enable row level security;

-- Allow authenticated users to upload/select objects in LuxIA bucket.
-- NOTE: For stricter tenant scoping, we can embed tenant_id into the object path and validate it via a function.
drop policy if exists "LuxIA upload" on storage.objects;
create policy "LuxIA upload"
on storage.objects
for insert to authenticated
with check (bucket_id = 'LuxIA');

drop policy if exists "LuxIA select" on storage.objects;
create policy "LuxIA select"
on storage.objects
for select to authenticated
using (bucket_id = 'LuxIA');

drop policy if exists "LuxIA delete" on storage.objects;
create policy "LuxIA delete"
on storage.objects
for delete to authenticated
using (bucket_id = 'LuxIA');
