# apps/engine/engine_api.py
# LuxIA Engine vNext (CLEAN + RUNNABLE)
# - Ingest: PDF, images, DXF (+DWG via optional converter)
# - "Prova del nove": deterministic checks + overlay + Vision verifier (Gemini/OpenRouter/legacy)
# - Concepts endpoint (Qwen/Groq/Ollama/dummy)

from __future__ import annotations

import os
import io
import re
import json
import math
import uuid
import base64
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==========================================================
# App
# ==========================================================
app = FastAPI(title="LuxIA Engine", version="22.0.0-vnext-clean")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod metti il dominio Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Small utils
# ==========================================================
def _require(module: str):
    try:
        __import__(module)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependency '{module}'. Install it in requirements.txt. Error: {e}",
        )


def _ext(name: str) -> str:
    name = (name or "").lower().strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1]


def _guess_type(filename: str, content_type: str) -> str:
    ext = _ext(filename)
    ct = (content_type or "").lower()

    if ext == "pdf" or "pdf" in ct:
        return "pdf"
    if ext in ("jpg", "jpeg", "png", "webp", "tif", "tiff") or ct.startswith("image/"):
        return "image"
    if ext == "dxf":
        return "dxf"
    if ext == "dwg":
        return "dwg"
    return "unknown"


# ==========================================================
# LLM Router (concepts orchestrator)
# ==========================================================
class LLMResult(BaseModel):
    text: str
    raw: Optional[Dict[str, Any]] = None
    provider: str


class LLMRouter:
    def __init__(self):
        self.provider = (os.getenv("LLM_PROVIDER") or "dummy").strip().lower()

        # Qwen (OpenAI-compatible)
        self.qwen_base_url = (os.getenv("QWEN_BASE_URL") or "").rstrip("/")
        self.qwen_api_key = os.getenv("QWEN_API_KEY") or ""
        self.qwen_model = os.getenv("QWEN_MODEL") or "qwen2.5-instruct"

        # Ollama
        self.ollama_url = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL") or "qwen2.5"

        # Groq (OpenAI-compatible)
        self.groq_base_url = (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        self.groq_api_key = os.getenv("GROQ_API_KEY") or ""
        self.groq_model = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1200) -> LLMResult:
        p = self.provider

        if p == "qwen":
            return self._openai_compat_chat(
                base_url=self.qwen_base_url,
                api_key=self.qwen_api_key,
                model=self.qwen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                provider="qwen",
            )

        if p == "groq":
            return self._openai_compat_chat(
                base_url=self.groq_base_url,
                api_key=self.groq_api_key,
                model=self.groq_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                provider="groq",
            )

        if p == "ollama":
            return self._ollama_chat(messages, temperature=temperature, provider="ollama")

        return LLMResult(
            text="(dummy) LLM not configured. Using heuristic concepts.",
            raw=None,
            provider="dummy",
        )

    def _openai_compat_chat(
        self,
        base_url: str,
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        provider: str,
    ) -> LLMResult:
        if not base_url:
            return LLMResult(
                text=f"({provider}) Missing BASE_URL. Using heuristic concepts.",
                raw=None,
                provider=provider,
            )
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            return LLMResult(text=text, raw=data, provider=provider)
        except Exception as e:
            return LLMResult(
                text=f"({provider}) Call failed: {e}. Using heuristic concepts.",
                raw={"error": str(e)},
                provider=provider,
            )

    def _ollama_chat(self, messages: List[Dict[str, str]], temperature: float, provider: str) -> LLMResult:
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
        url = f"{self.ollama_url}/api/generate"
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return LLMResult(text=data.get("response", ""), raw=data, provider=provider)
        except Exception as e:
            return LLMResult(text=f"(ollama) Call failed: {e}. Using heuristic concepts.", raw={"error": str(e)}, provider=provider)


llm = LLMRouter()


# ==========================================================
# Vision verifier (Gemini -> OpenRouter -> legacy OpenAI-compat)
# ==========================================================
class MultiVisionVerifier:
    def __init__(self):
        # Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or ""
        self.gemini_model = (os.getenv("GEMINI_MODEL") or "gemini-1.5-pro").strip()
        self.gemini_base_url = (os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

        # OpenRouter (OpenAI-compatible)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or ""
        self.openrouter_model = (os.getenv("OPENROUTER_MODEL") or "").strip()
        self.openrouter_base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.openrouter_http_referer = os.getenv("OPENROUTER_HTTP_REFERER") or ""
        self.openrouter_app_title = os.getenv("OPENROUTER_APP_TITLE") or "LuxIA"

        # Legacy OpenAI-compatible vision provider
        self.legacy_base_url = (os.getenv("VISION_BASE_URL") or "").rstrip("/")
        self.legacy_api_key = os.getenv("VISION_API_KEY") or ""
        self.legacy_model = (os.getenv("VISION_MODEL") or "").strip()

        self.min_confidence = float(os.getenv("VISION_MIN_CONFIDENCE") or "0.80")

    @property
    def enabled(self) -> bool:
        return bool(
            self.gemini_api_key
            or (self.openrouter_api_key and self.openrouter_model)
            or (self.legacy_base_url and self.legacy_model)
        )

    def verify(self, raster_png: bytes, extracted: Dict[str, Any], overlay_png: Optional[bytes] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "provider": None,
                "pass": False,
                "confidence": 0.0,
                "issues": [{"code": "VISION_NOT_CONFIGURED", "detail": "Vision verifier not configured."}],
                "suggested_fixes": ["Set GEMINI_API_KEY (recommended) or OPENROUTER_API_KEY + OPENROUTER_MODEL."],
                "observations": {},
            }

        # Gemini
        if self.gemini_api_key:
            out = self._verify_gemini(raster_png, extracted, overlay_png)
            if out.get("ok"):
                return self._postprocess(out, provider="gemini")

        # OpenRouter
        if self.openrouter_api_key and self.openrouter_model:
            out = self._verify_openai_compat(
                base_url=self.openrouter_base_url,
                api_key=self.openrouter_api_key,
                model=self.openrouter_model,
                raster_png=raster_png,
                extracted=extracted,
                overlay_png=overlay_png,
                extra_headers=self._openrouter_headers(),
            )
            if out.get("ok"):
                return self._postprocess(out, provider="openrouter")

        # Legacy
        if self.legacy_base_url and self.legacy_model:
            out = self._verify_openai_compat(
                base_url=self.legacy_base_url,
                api_key=self.legacy_api_key,
                model=self.legacy_model,
                raster_png=raster_png,
                extracted=extracted,
                overlay_png=overlay_png,
                extra_headers={},
            )
            if out.get("ok"):
                return self._postprocess(out, provider="legacy")

        return {
            "enabled": True,
            "provider": "none",
            "pass": False,
            "confidence": 0.0,
            "issues": [{"code": "VISION_ALL_FAILED", "detail": "All configured Vision providers failed."}],
            "suggested_fixes": [],
            "observations": {},
        }

    def _verify_gemini(self, raster_png: bytes, extracted: Dict[str, Any], overlay_png: Optional[bytes]) -> Dict[str, Any]:
        url = f"{self.gemini_base_url}/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        prompt = self._strict_prompt(extracted)

        parts: List[Dict[str, Any]] = [{"text": prompt}]
        parts.append({
            "inline_data": {"mime_type": "image/png", "data": base64.b64encode(raster_png).decode("ascii")}
        })
        if overlay_png:
            parts.append({"text": "OVERLAY IMAGE (predicted polygons/labels) for cross-check:"})
            parts.append({
                "inline_data": {"mime_type": "image/png", "data": base64.b64encode(overlay_png).decode("ascii")}
            })

        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1200}}

        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()

            text = ""
            cands = data.get("candidates") or []
            if cands:
                content = (cands[0].get("content") or {})
                parts_out = content.get("parts") or []
                for p in parts_out:
                    if "text" in p:
                        text += p["text"]

            out = self._extract_json(text)
            if not out:
                return {"ok": False, "error": "Gemini returned non-JSON.", "raw": (text or "")[:1200]}
            return {"ok": True, "out": out, "raw": (text or "")[:1200]}
        except Exception as e:
            return {"ok": False, "error": f"Gemini verify call failed: {e}", "raw": str(e)}

    def _verify_openai_compat(
        self,
        base_url: str,
        api_key: str,
        model: str,
        raster_png: bytes,
        extracted: Dict[str, Any],
        overlay_png: Optional[bytes],
        extra_headers: Dict[str, str],
    ) -> Dict[str, Any]:
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        b64a = base64.b64encode(raster_png).decode("ascii")
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self._strict_prompt(extracted)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64a}"}},
        ]
        if overlay_png:
            b64b = base64.b64encode(overlay_png).decode("ascii")
            content += [
                {"type": "text", "text": "OVERLAY IMAGE (predicted polygons/labels) for cross-check:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64b}"}},
            ]

        payload = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": 1200}

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            out = self._extract_json(text)
            if not out:
                return {"ok": False, "error": "Vision returned non-JSON.", "raw": text[:1200]}
            return {"ok": True, "out": out, "raw": text[:1200]}
        except Exception as e:
            return {"ok": False, "error": f"Vision verify call failed: {e}", "raw": str(e)}

    def _openrouter_headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.openrouter_http_referer:
            h["HTTP-Referer"] = self.openrouter_http_referer
        if self.openrouter_app_title:
            h["X-Title"] = self.openrouter_app_title
        return h

    def _strict_prompt(self, extracted: Dict[str, Any]) -> str:
        ex = json.dumps(extracted, ensure_ascii=False)
        if len(ex) > 12000:
            ex = ex[:12000] + "…(truncated)"
        return (
            "You are LuxIA Vision Verifier. You MUST be extremely strict.\n"
            "Task: Compare the planimetry image with the extracted JSON (and overlay image if provided).\n"
            "Return STRICT JSON ONLY, no markdown, schema:\n"
            "{\"confidence\":0..1,\"pass\":true|false,"
            "\"issues\":[{\"code\":string,\"detail\":string}],"
            "\"suggested_fixes\":[string],"
            "\"observations\":{}}.\n"
            "Rules:\n"
            "- pass=false if any issue is present OR if extraction seems incomplete.\n"
            "- confidence must reflect certainty.\n\n"
            f"EXTRACTED_JSON:\n{ex}\n"
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _postprocess(self, provider_out: Dict[str, Any], provider: str) -> Dict[str, Any]:
        out = provider_out.get("out") or {}
        conf = float(out.get("confidence") or 0.0)
        issues = out.get("issues") or []
        norm_issues: List[Dict[str, str]] = []
        for it in issues:
            if isinstance(it, str):
                norm_issues.append({"code": "ISSUE", "detail": it})
            elif isinstance(it, dict):
                norm_issues.append({"code": it.get("code") or "ISSUE", "detail": it.get("detail") or json.dumps(it)})
            else:
                norm_issues.append({"code": "ISSUE", "detail": str(it)})

        passed = bool(out.get("pass")) if "pass" in out else (conf >= self.min_confidence and len(norm_issues) == 0)
        if norm_issues:
            passed = False

        return {
            "enabled": True,
            "provider": provider,
            "pass": passed,
            "confidence": conf,
            "issues": norm_issues,
            "suggested_fixes": out.get("suggested_fixes") or [],
            "observations": out.get("observations") or {},
            "raw": provider_out.get("raw"),
            "min_confidence": self.min_confidence,
        }


vision = MultiVisionVerifier()


# ==========================================================
# Raster conversions
# ==========================================================
def _pdf_to_png_first_page(pdf_bytes: bytes, dpi: int = 260) -> Tuple[bytes, int]:
    _require("pypdfium2")
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_bytes)
    if len(pdf) < 1:
        raise HTTPException(status_code=400, detail="PDF has no pages")
    page = pdf.get_page(0)
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil()
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    out = io.BytesIO()
    pil.save(out, format="PNG", optimize=True)
    return out.getvalue(), int(len(pdf))


def _image_to_png(img_bytes: bytes) -> bytes:
    _require("PIL")
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes))
    im = im.convert("RGB")
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _dxf_to_png(dxf_bytes: bytes) -> bytes:
    _require("ezdxf")
    _require("matplotlib")
    import ezdxf
    from ezdxf import recover
    from io import BytesIO

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    # robust read
    try:
        doc, _auditor = recover.read(BytesIO(dxf_bytes))
    except Exception:
        doc = ezdxf.read(dxf_bytes.decode("utf-8", errors="ignore"))

    msp = doc.modelspace()
    fig = plt.figure(figsize=(10, 10), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()


# ==========================================================
# DWG -> DXF (optional microservice)
# ==========================================================
def _dwg_to_dxf(dwg_bytes: bytes, filename: str) -> bytes:
    url = (os.getenv("DWG2DXF_URL") or "").rstrip("/")
    if not url:
        raise HTTPException(
            status_code=422,
            detail=(
                "DWG uploaded but no converter configured. "
                "Set DWG2DXF_URL to a service that returns DXF bytes, or convert DWG to DXF before upload."
            ),
        )
    headers = {"Content-Type": "application/octet-stream", "X-Filename": filename}
    api_key = os.getenv("DWG2DXF_API_KEY") or ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.post(url, headers=headers, data=dwg_bytes, timeout=180)
        r.raise_for_status()
        return r.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DWG2DXF conversion failed: {e}")


# ==========================================================
# "Prova del nove" overlay
# ==========================================================
def _make_overlay_png(base_png: bytes, extracted: Dict[str, Any]) -> bytes:
    _require("PIL")
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(base_png)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    rooms = (extracted or {}).get("rooms") or []
    for i, room in enumerate(rooms):
        poly = room.get("polygon_px") or []
        if isinstance(poly, list) and len(poly) >= 3:
            pts = [(float(x), float(y)) for x, y in poly]
            draw.line(pts + [pts[0]], width=3, fill=(255, 0, 0, 180))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            label = room.get("label") or room.get("id") or f"room_{i+1}"
            area = room.get("area_m2")
            if isinstance(area, (int, float)):
                label = f"{label} ({area:.1f} m²)"
            draw.text((cx + 4, cy + 4), label, fill=(255, 0, 0, 220))

    lights = (extracted or {}).get("lights") or []
    for lt in lights:
        x = lt.get("x_px")
        y = lt.get("y_px")
        if x is None or y is None:
            continue
        x = float(x)
        y = float(y)
        r = 6
        draw.rectangle((x - r, y - r, x + r, y + r), outline=(255, 165, 0, 230), width=3)

    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


# ==========================================================
# Deterministic checks (must pass)
# ==========================================================
def _deterministic_checks(extracted: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    warnings: List[str] = []

    rooms = extracted.get("rooms") or []
    if not isinstance(rooms, list):
        issues.append("rooms is not a list")
        rooms = []
    if len(rooms) == 0:
        issues.append("No rooms detected/extracted")

    for i, r in enumerate(rooms[:200]):
        poly = r.get("polygon_px")
        if poly is None:
            warnings.append(f"Room[{i}] missing polygon_px")
            continue
        if not isinstance(poly, list) or len(poly) < 3:
            issues.append(f"Room[{i}] polygon_px invalid")

    ok = len(issues) == 0
    confidence = 0.4
    if ok:
        confidence += 0.2
    if len(rooms) >= 3:
        confidence += 0.1
    if len(warnings) == 0 and ok:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence))

    return {"ok": ok, "confidence": confidence, "issues": issues, "warnings": warnings}


# ==========================================================
# DXF extractor (rooms via vector->raster->floodfill->contours)
# ==========================================================
def _extract_from_dxf_real(dxf_bytes: bytes, opts: Dict[str, Any]) -> Dict[str, Any]:
    _require("ezdxf")
    _require("numpy")
    _require("opencv-python")

    import ezdxf
    from ezdxf import recover
    import numpy as np
    import cv2
    from io import BytesIO

    # read dxf
    try:
        doc, _auditor = recover.read(BytesIO(dxf_bytes))
    except Exception:
        doc = ezdxf.read(dxf_bytes.decode("utf-8", errors="ignore"))
    msp = doc.modelspace()

    segs: List[Tuple[float, float, float, float]] = []

    def add_seg(x1, y1, x2, y2):
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 1e-8:
            return
        segs.append((float(x1), float(y1), float(x2), float(y2)))

    # LINE
    for e in msp.query("LINE"):
        try:
            s = e.dxf.start
            t = e.dxf.end
            add_seg(s.x, s.y, t.x, t.y)
        except Exception:
            continue

    # LWPOLYLINE / POLYLINE
    for e in msp.query("LWPOLYLINE POLYLINE"):
        try:
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
            else:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices()]
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    add_seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                try:
                    closed = bool(getattr(e, "closed", False)) or bool(getattr(e, "is_closed", False))
                except Exception:
                    closed = False
                if closed:
                    add_seg(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1])
        except Exception:
            continue

    # ARC approx
    for e in msp.query("ARC"):
        try:
            c = e.dxf.center
            r = float(e.dxf.radius)
            a0 = float(e.dxf.start_angle) * math.pi / 180.0
            a1 = float(e.dxf.end_angle) * math.pi / 180.0
            n = 24
            angles = np.linspace(a0, a1, n)
            pts = [(c.x + r * math.cos(a), c.y + r * math.sin(a)) for a in angles]
            for i in range(len(pts) - 1):
                add_seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        except Exception:
            continue

    if len(segs) < 20:
        return {
            "meta": {"source": "dxf", "note": "too little geometry to detect rooms"},
            "rooms": [],
            "lights": [],
            "openings": [],
            "warnings": ["DXF geometry seems empty or not a floorplan (too few segments)."],
        }

    xs = [s[0] for s in segs] + [s[2] for s in segs]
    ys = [s[1] for s in segs] + [s[3] for s in segs]
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))
    w = maxx - minx
    h = maxy - miny
    maxdim = max(w, h)

    # units heuristic: mm if huge extents
    unit_to_m = 1.0
    units_guess = "m"
    if maxdim > 800:
        unit_to_m = 0.001
        units_guess = "mm"

    target = int(opts.get("dxf_raster_size") or 3000)
    target = max(1800, min(8000, target))
    pad = 0.03 * maxdim + 1e-6
    minx2, maxx2 = minx - pad, maxx + pad
    miny2, maxy2 = miny - pad, maxy + pad
    w2, h2 = maxx2 - minx2, maxy2 - miny2

    scale = target / max(w2, h2)
    img_w = int(w2 * scale)
    img_h = int(h2 * scale)

    def cad_to_px(x, y):
        px = (x - minx2) * scale
        py = (maxy2 - y) * scale
        return int(round(px)), int(round(py))

    def px_to_cad(px, py):
        x = (px / scale) + minx2
        y = maxy2 - (py / scale)
        return float(x), float(y)

    wall = np.zeros((img_h, img_w), dtype=np.uint8)
    thickness = int(opts.get("wall_thickness_px") or 4)
    thickness = max(2, min(12, thickness))

    for (x1, y1, x2, y2) in segs:
        x1p, y1p = cad_to_px(x1, y1)
        x2p, y2p = cad_to_px(x2, y2)
        cv2.line(wall, (x1p, y1p), (x2p, y2p), 255, thickness=thickness, lineType=cv2.LINE_AA)

    k = int(opts.get("close_kernel_px") or 7)
    k = max(3, min(25, k))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    wall2 = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, kernel, iterations=1)

    space = cv2.bitwise_not(wall2)
    ff = space.copy()
    mask = np.zeros((img_h + 2, img_w + 2), dtype=np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 0)

    rooms_mask = ff
    num, labels, stats, _ = cv2.connectedComponentsWithStats((rooms_mask > 0).astype(np.uint8), connectivity=8)

    min_area_px = int(opts.get("min_room_area_px") or 2500)
    areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, num)]
    areas.sort(reverse=True)

    def poly_area(poly: List[Tuple[float, float]]) -> float:
        s = 0.0
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    rooms: List[Dict[str, Any]] = []
    for area_px, i in areas:
        if area_px < min_area_px:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area_px:
            continue
        eps = float(opts.get("polygon_simplify_eps") or 3.0)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts_px = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts_px) < 3:
            continue

        pts_cad = [px_to_cad(x, y) for x, y in pts_px]
        pts_m = [(x * unit_to_m, y * unit_to_m) for x, y in pts_cad]
        area_m2 = poly_area(pts_m)

        rid = f"R{len(rooms)+1:03d}"
        rooms.append(
            {
                "id": rid,
                "label": None,
                "area_m2": float(area_m2),
                "polygon_px": [[int(x), int(y)] for x, y in pts_px],
                "polygon_m": [[float(x), float(y)] for x, y in pts_m],
            }
        )

    # lights placement minimal (centroid per room) – safe default
    lights: List[Dict[str, Any]] = []
    for r in rooms:
        poly = r.get("polygon_m") or []
        if not poly:
            continue
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        lid = f"L_{r['id']}_01"
        # convert centroid cad->px for overlay
        x_cad = cx / unit_to_m
        y_cad = cy / unit_to_m
        xpx, ypx = cad_to_px(x_cad, y_cad)
        lights.append(
            {
                "id": lid,
                "room_id": r["id"],
                "x_m": float(cx),
                "y_m": float(cy),
                "x_px": int(xpx),
                "y_px": int(ypx),
                "fixture_type": "DOWNLIGHT",
                "lumens": int(os.getenv("LIGHT_DEFAULT_LUMENS") or 800),
                "watt": float(os.getenv("LIGHT_DEFAULT_WATT") or 8),
                "cct_k": int(os.getenv("LIGHT_DEFAULT_CCT") or 3000),
                "cri": int(os.getenv("LIGHT_DEFAULT_CRI") or 80),
                "placement_rule": "CENTROID",
            }
        )

    return {
        "meta": {
            "source": "dxf",
            "units_guess": units_guess,
            "unit_to_m": float(unit_to_m),
            "bbox": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
        },
        "rooms": rooms,
        "lights": lights,
        "openings": [],
        "warnings": [],
    }


def _extract_from_raster_stub(raster_png: bytes, input_type: str) -> Dict[str, Any]:
    # Raster extractor non implementato: così non crasha ma Vision verifier sarà severo.
    return {
        "meta": {"source": input_type, "note": "raster extractor not implemented"},
        "rooms": [],
        "openings": [],
        "lights": [],
        "warnings": ["Raster extractor not implemented yet."],
    }


# ==========================================================
# API: Concepts
# ==========================================================
class ConceptReq(BaseModel):
    project_id: str
    brief: str = ""
    areas: List[Dict[str, Any]] = []
    n: int = 3


def heuristic_concepts(brief: str, areas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    styles = [
        {"name": "Concept A — Comfort Office", "cct": "4000K", "strategy": "UGR low, uniform task lighting + accent"},
        {"name": "Concept B — Warm Hospitality", "cct": "3000K", "strategy": "layered lighting, warm ambience + highlights"},
        {"name": "Concept C — Retail Contrast", "cct": "3500K", "strategy": "higher vertical illuminance, accents on focal zones"},
    ]
    out = []
    for s in styles:
        out.append(
            {
                "title": s["name"],
                "cct": s["cct"],
                "notes": f"{s['strategy']}. Brief: {brief[:240]}",
                "areas": [
                    {"name": a.get("name") or a.get("nome") or "Area", "type": a.get("type") or a.get("tipo_locale") or "Ufficio VDT"}
                    for a in (areas or [])
                ][:30],
            }
        )
    return out


@app.post("/concepts")
def generate_concepts(req: ConceptReq):
    system = (
        "You are LuxIA, a professional lighting design agent. "
        "Generate 3 distinct lighting concepts for the given areas. "
        "Return STRICT JSON only with schema: "
        "{\"concepts\":[{\"title\":...,\"cct\":...,\"style\":...,\"strategy\":...,\"per_area\":[{\"area\":...,\"target_lux\":...,\"notes\":...}]}]} "
        "Do not include markdown."
    )
    user = {"brief": req.brief, "areas": req.areas, "n": req.n, "constraints": {"standards": ["EN 12464-1"], "focus": "interiors"}}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    res = llm.chat(messages, temperature=0.2, max_tokens=1600)

    concepts = None
    try:
        m = re.search(r"\{[\s\S]*\}", res.text)
        if m:
            payload = json.loads(m.group(0))
            concepts = payload.get("concepts")
    except Exception:
        concepts = None

    if not concepts:
        concepts = heuristic_concepts(req.brief, req.areas)

    return {"ok": True, "provider": res.provider, "concepts": concepts, "llm_note": res.text[:400]}


# ==========================================================
# API: Planimetry analyze + verification
# ==========================================================
class PlanimetryResponse(BaseModel):
    ok: bool
    input_type: str
    page_count: int
    extracted: Dict[str, Any]
    checks: Dict[str, Any]
    vision: Dict[str, Any]
    overlay_png_base64: Optional[str] = None


@app.post("/planimetry/analyze", response_model=PlanimetryResponse)
async def planimetry_analyze(file: UploadFile = File(...), options: str = Form("{}")):
    raw = await file.read()
    if not raw or len(raw) < 16:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    input_type = _guess_type(file.filename or "", file.content_type or "")
    try:
        opts = json.loads(options or "{}")
    except Exception:
        opts = {}

    page_count = 1
    dxf_bytes: Optional[bytes] = None

    if input_type == "pdf":
        raster_png, page_count = _pdf_to_png_first_page(raw)
        extracted = _extract_from_raster_stub(raster_png, "pdf")

    elif input_type == "image":
        raster_png = _image_to_png(raw)
        extracted = _extract_from_raster_stub(raster_png, "image")

    elif input_type == "dxf":
        dxf_bytes = raw
        raster_png = _dxf_to_png(dxf_bytes)
        extracted = _extract_from_dxf_real(dxf_bytes, opts)

    elif input_type == "dwg":
        dxf_bytes = _dwg_to_dxf(raw, file.filename or "drawing.dwg")
        raster_png = _dxf_to_png(dxf_bytes)
        extracted = _extract_from_dxf_real(dxf_bytes, opts)

    else:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, JPG/PNG, DXF, or DWG.")

    checks = _deterministic_checks(extracted)

    overlay_png = _make_overlay_png(raster_png, extracted)
    vres = vision.verify(raster_png, extracted, overlay_png=overlay_png)

    # overlay base64 opzionale (utile alla UI)
    overlay_b64 = base64.b64encode(overlay_png).decode("ascii") if bool(opts.get("return_overlay_base64", True)) else None

    return {
        "ok": True,
        "input_type": input_type,
        "page_count": int(page_count),
        "extracted": extracted,
        "checks": checks,
        "vision": vres,
        "overlay_png_base64": overlay_b64,
    }


# ==========================================================
# Health
# ==========================================================
@app.get("/health")
def health():
    return {"ok": True, "version": app.version, "llm_provider": llm.provider, "vision_enabled": vision.enabled}


# ==========================================================
# Global error (nice)
# ==========================================================
@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": str(exc)})
