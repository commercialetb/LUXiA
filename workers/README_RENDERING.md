# LuxIA v15 — Rendering (Free-first)

## What you get now
- Webapp can queue a `render_blender` job (via chat keyword "render"/"fotorealistico")
- A local worker (`workers/blender_worker.py`) polls Supabase and generates placeholder PNGs
- PNGs are uploaded to Supabase Storage bucket `renders`
- Exports are recorded in `project_exports` with kind `render_png`

## Setup
1) Supabase: create storage bucket `renders` (private recommended)
2) Run SQL migration: `supabase/008_renders.sql`
3) In webapp env: set `SUPABASE_SERVICE_ROLE_KEY`
4) On your machine:
   - `pip install supabase python-dotenv`
   - create `.env` with:
     - `SUPABASE_URL=...`
     - `SUPABASE_SERVICE_ROLE_KEY=...`
5) Run worker:
   - `python workers/blender_worker.py`

## Next upgrade (real photoreal)
Replace placeholder generator with Blender headless:
- Keep same job queue
- Use a template `.blend` and a python script that:
  - imports the plan/layout (DXF or JSON)
  - places fixtures
  - sets materials/IES
  - renders to PNG
  - uploads PNGs to `renders`
