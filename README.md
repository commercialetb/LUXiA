# LuxIA SaaS MVP — Next.js + Supabase (Auth+DB+Storage) + FastAPI Engine

This MVP gives you:
- Next.js App Router UI with Supabase Auth (SSR cookies) and protected routes.
- Supabase SQL schema + RLS for multi-tenant (studios) + projects.
- FastAPI Engine placeholder (still token-based) — next step is to validate Supabase JWT.

## 1) Create Supabase project
Create a project on Supabase (or https://database.new). Then copy:
- Project URL
- anon public key

## 2) Run the SQL schema
In Supabase SQL editor, run:
- `supabase/001_schema.sql`
- `supabase/002_rls.sql`

## 3) Configure env
Copy `apps/web/.env.local.example` -> `apps/web/.env.local` and fill keys.

## 4) Run Web
```bash
cd apps/web
npm install
npm run dev
```
Open http://localhost:3000

## 5) Run Engine (optional)
```bash
cd apps/engine
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn engine_api:app --host 127.0.0.1 --port 8787
```

## Notes
Supabase Auth for Next.js App Router + SSR client is based on Supabase docs. citeturn0search0turn0search2turn0search11


## Storage bucket
Create a bucket named **LuxIA** and run `supabase/003_storage_policies.sql`.


## Review & Learning
After generating concepts, open `/projects/[id]/review` to select options and save learning signals.


## Style-aware concepts (learning loop)
- Review saves decisions and increments `studio_profile.style_tokens.concept_votes`.
- Concept generation fetches `studio_profile.style_tokens` and sends it to Engine.
- Engine orders/annotates concepts based on learned preference.


## BEGA rule-based selection + numeric estimate
- Engine v0.5 returns per concept: `luminaire` (BEGA placeholder) + `calc` (lumen-method estimate: Em, W/m², ok flags).
- Web stores `concepts.metrics` and shows Em/Wm² + luminaire in Review.


## Multi-brand (per-project)
- Run `supabase/004_multibrand.sql` to create `luminaires` and `project_brands` + RLS.
- Add luminaires via `/catalog` (tenant owner/admin).
- In each project, select enabled brands; concept generation matches only those brands. Empty selection => brand-neutral.


## Voice (free)
- Added browser-native voice: TTS via `speechSynthesis`, dictation via Web Speech API (Chrome/Edge).
- Components: `apps/web/components/VoiceAssistant.jsx`


## Autopilot optimizer (v8)
- Engine `/projects/{id}/concepts`: supports `priority` = norms|efficiency|mix and free-text `constraints`.
- Web: constraints box (voice dictation appends here) + Autopilot buttons.


## Layout + Uniformity proxy (v9)
- Engine returns `coords`, `layout`, `u0` (uniformity proxy), and room `dims`.
- Review UI displays U0 and layout.


## Export layout + Scenes (v10)
- Engine adds `scenes` presets per area and export endpoints:
  - POST `/exports/layout.json`
  - POST `/exports/layout.dxf`
- Web review page: download Layout JSON / DXF.


## Knowledge Base (v12)
1) In Supabase create a storage bucket named `knowledge`.
2) Run SQL: `supabase/006_knowledge_rag.sql`
3) Set env var `SUPABASE_SERVICE_ROLE_KEY` in web app.


## PPTX Export (v13)
- From Review page, click **Scarica PPTX (Concept)**.
- Server route: `/api/exports/ppt?project_id=...` (requires login cookies).


## v14 Autopilot Exports
- Create a Supabase Storage bucket named `exports` (private recommended).
- `SUPABASE_SERVICE_ROLE_KEY` must be set in the web app to enable server-side upload.
- When a chat message contains 'ppt/pptx/presentazione/powerpoint', LuxIA will generate and store a PPTX automatically.


## Rendering (v15)
- Create Supabase storage bucket `renders`.
- Run SQL `supabase/008_renders.sql`.
- Start local worker: `python workers/blender_worker.py`.


## v16 Photoreal Rendering (Blender Cycles, 4:3)
- Run SQL: `supabase/009_render_jobs_v16.sql`
- Create storage bucket: `renders` (private)
- Start worker: `python workers/blender_worker.py`
- Env: `BLENDER_BIN` path to blender executable.


## v17
- Render enqueue now targets all 3 concepts (Comfort/Efficienza/Architetturale).
- PPT embeds both client + technical renders when available.
- Run SQL: `supabase/010_project_renders_indexes_v17.sql`


## v18 — Designer Brain (Team DNA)
- Run `supabase/011_designer_brain_v18.sql`.
- Set `SUPABASE_SERVICE_ROLE_KEY` in the web app env (server-side only) to enable Designer Brain APIs.
- New page: `/designer-brain`.
- Review page logs confirmed concept selections into `designer_learning_events` and recomputes `designer_team_profile`.
- Engine now accepts optional `designer_stats` to bias concept ordering.


## v19 — Auto client style + Area-type DNA
- Run `supabase/012_designer_brain_area_bias_v19.sql`.
- Projects now have `client_name`.
- App auto-assigns `active_style_id` using client_name (or Team Default).
- Engine uses `get_designer_bias_for_area` to bias concept ordering per room type.


## v20 — Designer Brain Dashboard + Area Filter + Auto-tag mood
- Run `supabase/013_designer_brain_dashboard_v20.sql`.
- `/designer-brain`: filtro per tipo locale + confronto Team vs Stile attivo.
- Chat: auto-tag `projects.mood_hint` + aggiorna `studio_profiles.style_tokens.mood_hint`.


## v21 — Project Style Pack (editable sliders)
- Run `supabase/014_project_style_pack_v21.sql`.
- Project page shows Style Pack panel with sliders.
- Engine consumes `project_style_pack` to bias concept ordering and apply CCT default.


## v22 — Style Pack applied to Layout + Optics + Rendering + PPT
- Engine: density_bias scales sizing, uniformity_target affects layout margins, contrast/accent influence distribution.
- Render enqueue: auto-applies project_style_packs to mood/cct/contrast + accent power.
- Blender: view contrast uses style_pack.contrast_level.
- PPT: theme is driven by stylePack.presentation_theme.
