#!/usr/bin/env python3
"""LuxIA v16 Blender Worker (Cycles, 4:3)

- Polls Supabase table `project_jobs` for jobs type `render_blender_v16` with status `queued`.
- For each job, downloads the scene payload from `job.payload` (JSON) and invokes Blender headless:
    blender -b --python workers/render_blender_scene.py -- <scene_json_path> <out_png_path> <camera> <mood> <quality>
- Uploads PNG to Supabase bucket `renders` and stores a row in `project_renders`.
- Marks job done/error.

Environment (.env):
  SUPABASE_URL=
  SUPABASE_SERVICE_ROLE_KEY=
  RENDERS_BUCKET=renders
  BLENDER_BIN=blender   (or full path)
"""
import os, time, json, tempfile, subprocess, traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET = os.getenv("RENDERS_BUCKET", "renders")
BLENDER_BIN = os.getenv("BLENDER_BIN", "blender")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("[LuxIA] Blender worker v16 started.")
    while True:
        try:
            # Fetch one queued render job
            jobs = sb.table("project_jobs")                 .select("*")                 .eq("status","queued")                 .eq("job_type","render_blender_v16")                 .order("created_at", desc=False)                 .limit(1)                 .execute().data

            if not jobs:
                time.sleep(POLL_SECONDS)
                continue

            job = jobs[0]
            job_id = job["id"]
            payload = job.get("payload") or {}
            project_id = payload.get("project_id") or job.get("project_id")
            owner_id = payload.get("owner_id") or job.get("owner_id")
            concept_id = payload.get("concept_id")
            concept_type = payload.get("concept_type")
            area_name = payload.get("area_name","Area")
            scene = payload.get("scene") or {}

            mood = payload.get("mood","office_clean")
            quality = payload.get("quality","medium")  # draft|medium|high
            cams = payload.get("cameras") or ["technical","client"]
            width = int(payload.get("width",1600))
            height = int(payload.get("height",1200))

            # mark running
            sb.table("project_jobs").update({
                "status":"running",
                "updated_at": utcnow()
            }).eq("id", job_id).execute()

            # render each camera
            for cam in cams:
                render_one(sb, project_id, owner_id, concept_id, area_name, scene,
                           camera=cam, mood=mood, quality=quality, width=width, height=height)

            # mark done
            sb.table("project_jobs").update({
                "status":"done",
                "updated_at": utcnow(),
                "result": {"renders": len(cams), "mood": mood, "quality": quality, "size":[width,height]}
            }).eq("id", job_id).execute()

            print(f"[LuxIA] Job done: {job_id}")

        except Exception as e:
            print("[LuxIA] Worker loop error:", e)
            traceback.print_exc()
            time.sleep(5)

def render_one(sb, project_id, owner_id, concept_id, area_name, scene, camera, mood, quality, width, height):
    with tempfile.TemporaryDirectory() as td:
        td = os.path.abspath(td)
        scene_path = os.path.join(td, "scene.json")
        out_path = os.path.join(td, f"render_{camera}.png")

        scene_out = dict(scene)
        scene_out["render"] = {
            "camera": camera,
            "mood": mood,
            "quality": quality,
            "width": width,
            "height": height
        }

        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(scene_out, f, ensure_ascii=False, indent=2)

        # Run blender headless
        script_path = os.path.join(os.path.dirname(__file__), "render_blender_scene.py")
        cmd = [BLENDER_BIN, "-b", "--python", script_path, "--", scene_path, out_path]
        print("[LuxIA] Running:", " ".join(cmd))
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Blender failed ({camera}): {p.stderr[-800:]}")

        # Upload to storage
        storage_path = f"{project_id}/{concept_id or 'concept'}/{int(time.time())}_{camera}.png"
        with open(out_path, "rb") as f:
            sb.storage.from_(BUCKET).upload(storage_path, f, {"content-type":"image/png", "upsert": True})

        # Insert DB row
        sb.table("project_renders").insert({
            "project_id": project_id,
            "owner_id": owner_id,
            "concept_id": concept_id,
            "concept_type": concept_type,
            "area_name": area_name,
            "camera": camera,
            "mood": mood,
            "quality": quality,
            "width": width,
            "height": height,
            "storage_path": storage_path
        }).execute()

        print(f"[LuxIA] Uploaded {camera} render:", storage_path)

if __name__ == "__main__":
    main()
