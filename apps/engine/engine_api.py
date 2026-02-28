# apps/engine/engine_api.py
# LuxIA Engine — Stable (Render/Docker friendly)
# - Planimetry ingestion: PDF, JPG/PNG, DXF (+DWG via conversion service)
# - "Prova del nove": deterministic checks + Vision cross-check (Gemini/OpenRouter/legacy)
# - Lighting placement (2D CAD blocks layer ILLUMINAZIONE) + basic lux grid + UGR approx
# - UNI target mapping (configurable mapping, not standard text)
# - PDF report generator + BOM/spec exports
#
# This file is self-contained and avoids indentation/return-scope issues.
#
# ENV (LLM orchestrator)
#   LLM_PROVIDER=qwen|groq|ollama|dummy
#   QWEN_BASE_URL=... (OpenAI-compatible)
#   QWEN_API_KEY=...
#   QWEN_MODEL=...
#
# ENV (Vision verifier)
#   GEMINI_API_KEY=...
#   GEMINI_MODEL=gemini-1.5-pro
#   OPENROUTER_API_KEY=...
#   OPENROUTER_MODEL=...
#   VISION_BASE_URL=... (OpenAI-compatible)
#   VISION_API_KEY=...
#   VISION_MODEL=...
#
# ENV (DWG conversion - optional)
#   DWG2DXF_URL=...  (microservice: POST DWG bytes => DXF bytes)
#   DWG2DXF_API_KEY=...
#
# CAD standard
#   LIGHT_LAYER_NAME=ILLUMINAZIONE
#   LIGHT_BLOCK_NAME=LUXIA_LIGHT_POINT

from __future__ import annotations

import os
import re
import io
import json
import math
import base64
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response
from pydantic import BaseModel

# ReportLab (PDF report)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# Pillow (overlay/image normalization)
from PIL import Image, ImageDraw

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _require(module: str):
    try:
        __import__(module)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency '{module}'. Install it in requirements.txt. Error: {e}")

def _ext(name: str) -> str:
    name = (name or "").lower().strip()
    if "." not in name:
        return ""
    return name.split(".")[-1]

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

# ------------------------------------------------------------
# Defaults (requested)
# ------------------------------------------------------------
CEILING_REFLECTANCE_DEFAULT = float(os.getenv("CEILING_REFLECTANCE", "0.8"))
WALL_REFLECTANCE_DEFAULT = float(os.getenv("WALL_REFLECTANCE", "0.6"))
FLOOR_REFLECTANCE_DEFAULT = float(os.getenv("FLOOR_REFLECTANCE", "0.4"))
MAINTENANCE_FACTOR_DEFAULT = float(os.getenv("MAINTENANCE_FACTOR", "0.8"))
WORKPLANE_HEIGHT_M_DEFAULT = float(os.getenv("WORKPLANE_HEIGHT_M", "0.8"))
ROOM_HEIGHT_M_DEFAULT = float(os.getenv("ROOM_HEIGHT_M", "2.7"))

# Photometry defaults
LIGHT_DEFAULT_LUMENS = float(os.getenv("LIGHT_DEFAULT_LUMENS", "800"))
LIGHT_DEFAULT_WATT = float(os.getenv("LIGHT_DEFAULT_WATT", "8"))
LIGHT_DEFAULT_CCT = int(float(os.getenv("LIGHT_DEFAULT_CCT", "3000")))
LIGHT_DEFAULT_CRI = int(float(os.getenv("LIGHT_DEFAULT_CRI", "80")))

# ------------------------------------------------------------
# UNI Targets (mapping only — not standard text)
# ------------------------------------------------------------
UNI_12464_1_TARGETS = {
    "UFFICIO": {"lux": 500, "u0": 0.60},
    "OPENSPACE": {"lux": 500, "u0": 0.60},
    "SALA_RIUNIONI": {"lux": 500, "u0": 0.60},
    "CUCINA": {"lux": 300, "u0": 0.60},
    "BAGNO": {"lux": 200, "u0": 0.40},
    "CAMERA": {"lux": 200, "u0": 0.40},
    "SOGGIORNO": {"lux": 200, "u0": 0.40},
    "CORRIDOIO": {"lux": 150, "u0": 0.40},
    "SCALA": {"lux": 150, "u0": 0.40},
    "MAGAZZINO": {"lux": 200, "u0": 0.40},
    "DEFAULT": {"lux": 300, "u0": 0.40},
}
UNI_12464_2_TARGETS = {
    "PARCHEGGIO": {"lux": 20, "u0": 0.25},
    "PASSAGGIO": {"lux": 20, "u0": 0.25},
    "CORTILE": {"lux": 30, "u0": 0.25},
    "GIARDINO": {"lux": 30, "u0": 0.25},
    "AREA_LAVORO": {"lux": 100, "u0": 0.40},
    "DEFAULT": {"lux": 30, "u0": 0.25},
}

def _norm_key(label: Optional[str]) -> str:
    if not label:
        return "DEFAULT"
    s = label.upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if "UFFIC" in s:
        return "UFFICIO"
    if "OPEN" in s and "SPACE" in s:
        return "OPENSPACE"
    if "RIUN" in s or "MEETING" in s:
        return "SALA_RIUNIONI"
    if "CUC" in s:
        return "CUCINA"
    if "BAG" in s or "WC" in s or "TOILET" in s:
        return "BAGNO"
    if "CAM" in s or "BED" in s:
        return "CAMERA"
    if "SOGG" in s or "LIVING" in s:
        return "SOGGIORNO"
    if "CORR" in s or "HALL" in s:
        return "CORRIDOIO"
    if "SCAL" in s or "STAIR" in s:
        return "SCALA"
    if "MAG" in s or "STORE" in s or "DEP" in s:
        return "MAGAZZINO"
    if "PARK" in s or "PARCHEGG" in s:
        return "PARCHEGGIO"
    if "PASS" in s:
        return "PASSAGGIO"
    if "CORT" in s:
        return "CORTILE"
    if "GIARD" in s or "GARDEN" in s:
        return "GIARDINO"
    return "DEFAULT"

def _is_outdoor_room(room: Dict[str, Any]) -> bool:
    if room.get("is_outdoor") is True:
        return True
    key = _norm_key(room.get("label") or "")
    return key in ("PARCHEGGIO", "PASSAGGIO", "CORTILE", "GIARDINO", "AREA_LAVORO")

def _ugr_target_for_key(key: str, is_outdoor: bool) -> Optional[float]:
    if is_outdoor:
        return None
    if key in ("UFFICIO", "OPENSPACE", "SALA_RIUNIONI"):
        return 19.0
    if key in ("CORRIDOIO", "SCALA"):
        return 22.0
    return 22.0

def _get_uni_targets(room: Dict[str, Any]) -> Dict[str, Any]:
    key = _norm_key(room.get("label"))
    if _is_outdoor_room(room):
        t = UNI_12464_2_TARGETS.get(key) or UNI_12464_2_TARGETS["DEFAULT"]
        return {"standard": "UNI EN 12464-2", "key": key, "ugr_max": _ugr_target_for_key(key, True), **t}
    t = UNI_12464_1_TARGETS.get(key) or UNI_12464_1_TARGETS["DEFAULT"]
    return {"standard": "UNI EN 12464-1", "key": key, "ugr_max": _ugr_target_for_key(key, False), **t}

# ------------------------------------------------------------
# LLM Router (kept)
# ------------------------------------------------------------

class LLMResult(BaseModel):
    text: str
    raw: Optional[Dict[str, Any]] = None
    provider: str

class LLMRouter:
    def __init__(self):
        self.provider = (os.getenv("LLM_PROVIDER") or "dummy").strip().lower()

        self.qwen_base_url = (os.getenv("QWEN_BASE_URL") or "").rstrip("/")
        self.qwen_api_key = os.getenv("QWEN_API_KEY") or ""
        self.qwen_model = os.getenv("QWEN_MODEL") or "qwen2.5-instruct"

        self.ollama_url = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL") or "qwen2.5"

        self.groq_base_url = (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        self.groq_api_key = os.getenv("GROQ_API_KEY") or ""
        self.groq_model = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: int = 1200) -> LLMResult:
        p = self.provider
        if p == "qwen":
            return self._openai_compat_chat(self.qwen_base_url, self.qwen_api_key, self.qwen_model, messages, temperature, max_tokens, "qwen")
        if p == "groq":
            return self._openai_compat_chat(self.groq_base_url, self.groq_api_key, self.groq_model, messages, temperature, max_tokens, "groq")
        if p == "ollama":
            return self._ollama_chat(messages, temperature=temperature, provider="ollama")
        return LLMResult(text="(dummy) LLM not configured. Using heuristic concepts.", raw=None, provider="dummy")

    def _openai_compat_chat(self, base_url: str, api_key: str, model: str, messages: List[Dict[str, Any]],
                           temperature: float, max_tokens: int, provider: str) -> LLMResult:
        if not base_url:
            return LLMResult(text=f"({provider}) Missing BASE_URL. Using heuristic concepts.", raw=None, provider=provider)
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "messages": messages, "temperature": float(temperature), "max_tokens": int(max_tokens)}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            return LLMResult(text=text, raw=data, provider=provider)
        except Exception as e:
            return LLMResult(text=f"({provider}) Call failed: {e}. Using heuristic concepts.", raw={"error": str(e)}, provider=provider)

    def _ollama_chat(self, messages: List[Dict[str, Any]], temperature: float, provider: str) -> LLMResult:
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

# ------------------------------------------------------------
# Vision verifier (Gemini -> OpenRouter -> legacy)
# ------------------------------------------------------------

class MultiVisionVerifier:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or ""
        self.gemini_model = (os.getenv("GEMINI_MODEL") or "gemini-1.5-pro").strip()
        self.gemini_base_url = (os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or ""
        self.openrouter_model = (os.getenv("OPENROUTER_MODEL") or "").strip()
        self.openrouter_base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.openrouter_http_referer = os.getenv("OPENROUTER_HTTP_REFERER") or ""
        self.openrouter_app_title = os.getenv("OPENROUTER_APP_TITLE") or "LuxIA"

        self.legacy_base_url = (os.getenv("VISION_BASE_URL") or "").rstrip("/")
        self.legacy_api_key = os.getenv("VISION_API_KEY") or ""
        self.legacy_model = (os.getenv("VISION_MODEL") or "").strip()

        self.min_confidence = float(os.getenv("VISION_MIN_CONFIDENCE") or "0.80")

    @property
    def enabled(self) -> bool:
        return bool(self.gemini_api_key or (self.openrouter_api_key and self.openrouter_model) or (self.legacy_base_url and self.legacy_model))

    def verify(self, raster_png: bytes, extracted: Dict[str, Any], overlay_png: Optional[bytes] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False, "provider": None, "pass": False, "confidence": 0.0,
                "issues": [{"code": "VISION_OFF", "detail": "Vision verifier not configured."}],
                "suggested_fixes": [], "observations": {}
            }

        if self.gemini_api_key:
            out = self._verify_gemini(raster_png, extracted, overlay_png)
            if out.get("ok"):
                return self._postprocess(out, provider="gemini")

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
            "enabled": True, "provider": "none", "pass": False, "confidence": 0.0,
            "issues": [{"code": "VISION_FAIL", "detail": "All configured Vision providers failed."}],
            "suggested_fixes": [], "observations": {}
        }

    def _strict_prompt(self, extracted: Dict[str, Any]) -> str:
        ex = json.dumps(extracted, ensure_ascii=False)
        if len(ex) > 12000:
            ex = ex[:12000] + "…(truncated)"
        return (
            "You are LuxIA Vision Verifier. Be extremely strict.\n"
            "Compare the planimetry image with EXTRACTED_JSON and optional overlay.\n"
            "Return STRICT JSON ONLY (no markdown):\n"
            "{\"confidence\":0..1,\"pass\":true|false,"
            "\"issues\":[{\"code\":string,\"detail\":string}],"
            "\"suggested_fixes\":[string],\"observations\":{}}.\n"
            "Rules: pass=false if any issue or if extraction seems incomplete.\n\n"
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
        norm_issues = []
        for it in issues:
            if isinstance(it, str):
                norm_issues.append({"code": "ISSUE", "detail": it})
            elif isinstance(it, dict):
                norm_issues.append({"code": it.get("code") or "ISSUE", "detail": it.get("detail") or json.dumps(it)})
            else:
                norm_issues.append({"code": "ISSUE", "detail": str(it)})

        passed = bool(out.get("pass")) if "pass" in out else (conf >= self.min_confidence and len(norm_issues) == 0)
        if len(norm_issues) > 0:
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

    def _verify_gemini(self, raster_png: bytes, extracted: Dict[str, Any], overlay_png: Optional[bytes]) -> Dict[str, Any]:
        url = f"{self.gemini_base_url}/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        prompt = self._strict_prompt(extracted)
        parts = [{"text": prompt}]
        parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(raster_png).decode("ascii")}})
        if overlay_png:
            parts.append({"text": "OVERLAY IMAGE for cross-check:"})
            parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(overlay_png).decode("ascii")}})
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1200}}
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            text = ""
            cands = data.get("candidates") or []
            if cands:
                content = (cands[0].get("content") or {})
                for p in (content.get("parts") or []):
                    if "text" in p:
                        text += p["text"]
            out = self._extract_json(text)
            if not out:
                return {"ok": False, "error": "Gemini returned non-JSON.", "raw": (text or "")[:1200]}
            return {"ok": True, "out": out, "raw": (text or "")[:1200]}
        except Exception as e:
            return {"ok": False, "error": f"Gemini verify failed: {e}", "raw": str(e)}

    def _openrouter_headers(self) -> Dict[str, str]:
        h = {}
        if self.openrouter_http_referer:
            h["HTTP-Referer"] = self.openrouter_http_referer
        if self.openrouter_app_title:
            h["X-Title"] = self.openrouter_app_title
        return h

    def _verify_openai_compat(self, base_url: str, api_key: str, model: str,
                             raster_png: bytes, extracted: Dict[str, Any], overlay_png: Optional[bytes],
                             extra_headers: Dict[str, str]) -> Dict[str, Any]:
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        b64a = base64.b64encode(raster_png).decode("ascii")
        content = [
            {"type": "text", "text": self._strict_prompt(extracted)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64a}"}},
        ]
        if overlay_png:
            b64b = base64.b64encode(overlay_png).decode("ascii")
            content += [{"type": "text", "text": "OVERLAY IMAGE for cross-check:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64b}"}}]
        messages = [{"role": "user", "content": content}]
        payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 1200}
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
            return {"ok": False, "error": f"Vision verify failed: {e}", "raw": str(e)}

vision = MultiVisionVerifier()

# ------------------------------------------------------------
# Photometry library (IES/LDT parsing subset)
# ------------------------------------------------------------

_LUMINAIRE_LIB: Dict[str, Dict[str, Any]] = {}
_UGR_TABLES: Dict[str, Any] = {}

def _parse_ies_lm63(raw_bytes: bytes) -> Dict[str, Any]:
    txt = raw_bytes.decode("latin1", errors="ignore").splitlines()
    while txt and not txt[0].strip():
        txt.pop(0)
    if not txt or "IESNA" not in txt[0].upper():
        raise ValueError("Not an IES LM-63 file")
    meta = {"raw_header": []}
    i = 0
    while i < len(txt):
        line = txt[i].strip()
        meta["raw_header"].append(line)
        if line.upper().startswith("TILT="):
            meta["tilt"] = line.split("=", 1)[1].strip().upper()
            i += 1
            break
        i += 1
    if meta.get("tilt", "NONE") not in ("NONE", "INCLUDE"):
        raise ValueError(f"Unsupported TILT={meta.get('tilt')}")
    nums: List[float] = []

    def push_nums(line: str):
        for t in line.replace(",", " ").split():
            try:
                nums.append(float(t))
            except Exception:
                pass

    while i < len(txt) and len(nums) < 13:
        push_nums(txt[i])
        i += 1
    if len(nums) < 13:
        raise ValueError("IES numeric header incomplete")

    n_lamps = int(nums[0])
    lumens_per_lamp = float(nums[1])
    candela_mult = float(nums[2])
    n_v = int(nums[3])
    n_h = int(nums[4])
    phot_type = int(nums[5])
    units_type = int(nums[6])
    width = float(nums[7])
    length = float(nums[8])
    height = float(nums[9])
    input_watts = float(nums[12])

    v_angles: List[float] = []
    while i < len(txt) and len(v_angles) < n_v:
        for t in txt[i].replace(",", " ").split():
            if len(v_angles) < n_v:
                v_angles.append(float(t))
        i += 1
    h_angles: List[float] = []
    while i < len(txt) and len(h_angles) < n_h:
        for t in txt[i].replace(",", " ").split():
            if len(h_angles) < n_h:
                h_angles.append(float(t))
        i += 1

    candela: List[List[float]] = []
    for _ in range(n_h):
        row: List[float] = []
        while i < len(txt) and len(row) < n_v:
            for t in txt[i].replace(",", " ").split():
                if len(row) < n_v:
                    row.append(float(t) * candela_mult)
            i += 1
        if len(row) != n_v:
            raise ValueError("Candela matrix incomplete")
        candela.append(row)

    return {
        "type": "ies_lm63",
        "n_lamps": n_lamps,
        "lumens_per_lamp": lumens_per_lamp,
        "input_watts": input_watts,
        "units_type": units_type,
        "dimensions": {"W": width, "L": length, "H": height},
        "photometric_type": phot_type,
        "v_angles": v_angles,
        "h_angles": h_angles,
        "candela": candela,
        "meta": meta,
    }

def _candela_at(phot: Dict[str, Any], gamma_deg: float, c_deg: float) -> float:
    v = phot["v_angles"]
    h = phot["h_angles"]
    mat = phot["candela"]

    gamma = max(min(gamma_deg, v[-1]), v[0])
    c = max(min(c_deg, h[-1]), h[0])

    def interp1(x: float, xs: List[float], ys: List[float]) -> float:
        if x <= xs[0]:
            return float(ys[0])
        if x >= xs[-1]:
            return float(ys[-1])
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    # interpolate between two C planes
    if c <= h[0]:
        row0 = row1 = 0
        tc = 0.0
    elif c >= h[-1]:
        row0 = row1 = len(h) - 1
        tc = 0.0
    else:
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if h[mid] <= c:
                lo = mid
            else:
                hi = mid
        row0, row1 = lo, hi
        tc = 0.0 if h[row1] == h[row0] else (c - h[row0]) / (h[row1] - h[row0])

    i0 = interp1(gamma, v, mat[row0])
    i1 = interp1(gamma, v, mat[row1])
    return float(i0 + tc * (i1 - i0))

def _ugr_approx(phot: Optional[Dict[str, Any]]) -> Optional[float]:
    if not phot:
        return None
    I65 = _candela_at(phot, 65.0, 0.0)
    lum = float(phot.get("lumens_per_lamp") or 1000.0)
    g = I65 / max(lum, 1.0)
    ugr = 16.0 + 30.0 * min(1.0, max(0.0, g * 8.0))
    return float(max(12.0, min(29.0, ugr)))

app = FastAPI(title="LuxIA Engine", version="v13.2A-stable")

# ------------------------------------------------------------
# Concepts API
# ------------------------------------------------------------
class ConceptReq(BaseModel):
    project_id: str
    brief: str = ""
    areas: List[Dict[str, Any]] = []
    n: int = 3

def _heuristic_concepts(brief: str, areas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    styles = [
        {"name": "Concept A — Comfort Office", "cct": "4000K", "strategy": "UGR low, uniform task lighting + accent"},
        {"name": "Concept B — Warm Hospitality", "cct": "3000K", "strategy": "layered lighting, warm ambience + highlights"},
        {"name": "Concept C — Retail Contrast", "cct": "3500K", "strategy": "higher vertical illuminance, accents on focal zones"},
    ]
    out = []
    for s in styles[: max(1, int(len(styles)))]:
        out.append({
            "title": s["name"],
            "cct": s["cct"],
            "notes": f"{s['strategy']}. Brief: {brief[:240]}",
            "areas": [{"name": a.get("name") or a.get("nome") or "Area",
                       "type": a.get("type") or a.get("tipo_locale") or "Ufficio VDT"} for a in (areas or [])][:30]
        })
    return out

@app.post("/concepts")
def generate_concepts(req: ConceptReq):
    system = (
        "You are LuxIA, a professional lighting design agent. "
        "Generate concepts for the given areas. Return STRICT JSON only with schema: "
        "{\"concepts\":[{\"title\":...,\"cct\":...,\"style\":...,\"strategy\":...,"
        "\"per_area\":[{\"area\":...,\"target_lux\":...,\"notes\":...}]}]} "
        "Do not include markdown."
    )
    user = {"brief": req.brief, "areas": req.areas, "n": req.n, "constraints": {"standards": ["EN 12464-1"], "focus": "interiors"}}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    res = llm.chat(messages, temperature=0.2, max_tokens=1400)

    concepts = None
    try:
        m = re.search(r"\{[\s\S]*\}", res.text)
        if m:
            payload = json.loads(m.group(0))
            concepts = payload.get("concepts")
    except Exception:
        concepts = None

    if not concepts:
        concepts = _heuristic_concepts(req.brief, req.areas)

    return {"ok": True, "provider": res.provider, "concepts": concepts, "llm_note": res.text[:400]}

# ------------------------------------------------------------
# Planimetry ingest: normalize to raster PNG
# ------------------------------------------------------------

def _pdf_to_png_first_page(pdf_bytes: bytes, dpi: int = 260) -> bytes:
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
    return out.getvalue()

def _image_to_png(img_bytes: bytes) -> bytes:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()

def _dxf_to_png(dxf_bytes: bytes) -> bytes:
    _require("ezdxf")
    _require("matplotlib")
    import ezdxf
    from ezdxf import recover
    from io import BytesIO

    try:
        doc, _auditor = recover.read(BytesIO(dxf_bytes))
    except Exception:
        doc = ezdxf.read(dxf_bytes.decode("utf-8", errors="ignore"))
    msp = doc.modelspace()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    fig = plt.figure(figsize=(10, 10), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()

def _dwg_to_dxf(dwg_bytes: bytes, filename: str) -> bytes:
    url = (os.getenv("DWG2DXF_URL") or "").rstrip("/")
    if not url:
        raise HTTPException(status_code=422, detail="DWG uploaded but no converter configured. Set DWG2DXF_URL.")
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

# ------------------------------------------------------------
# Deterministic checks + overlay (prova del nove)
# ------------------------------------------------------------

def _deterministic_checks(extracted: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    warnings: List[str] = []
    rooms = extracted.get("rooms") or []
    openings = extracted.get("openings") or []

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

    for i, o in enumerate(openings[:500]):
        if o.get("type") not in ("door", "window"):
            warnings.append(f"Opening[{i}] unknown type")

    scale = extracted.get("scale") or {}
    ppm = scale.get("pxPerMeter")
    if ppm is not None:
        try:
            ppm = float(ppm)
            if ppm <= 0:
                issues.append("pxPerMeter must be > 0")
            if ppm < 20 or ppm > 3000:
                warnings.append("pxPerMeter unusual; verify scale")
        except Exception:
            issues.append("pxPerMeter is not numeric")

    ok = len(issues) == 0
    confidence = 0.4 + (0.2 if ok else 0.0) + (0.1 if len(rooms) >= 3 else 0.0) + (0.1 if ppm is not None else 0.0) + (0.1 if ok and len(warnings) == 0 else 0.0)
    confidence = max(0.0, min(1.0, confidence))

    return {"ok": ok, "confidence": confidence, "issues": issues, "warnings": warnings}

def _make_overlay_png(base_png: bytes, extracted: Dict[str, Any]) -> bytes:
    img = Image.open(io.BytesIO(base_png)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    rooms = (extracted or {}).get("rooms") or []
    for i, room in enumerate(rooms):
        poly = room.get("polygon_px") or []
        if len(poly) >= 3:
            pts = [(float(x), float(y)) for x, y in poly]
            draw.line(pts + [pts[0]], width=3, fill=(255, 0, 0, 180))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            label = room.get("label") or room.get("id") or f"room_{i+1}"
            area = room.get("area_m2")
            if isinstance(area, (int, float)):
                label = f"{label} ({area:.1f} m²)"
            draw.text((cx + 4, cy + 4), label, fill=(255, 0, 0, 220))

    openings = (extracted or {}).get("openings") or []
    for op in openings:
        c = op.get("center_px")
        if isinstance(c, (list, tuple)) and len(c) == 2:
            x, y = float(c[0]), float(c[1])
            r = 8
            kind = (op.get("type") or "").lower()
            color = (0, 128, 255, 220) if kind == "window" else (0, 200, 0, 220)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)

    lights = (extracted or {}).get("lights") or []
    for lt in lights:
        x = lt.get("x_px"); y = lt.get("y_px")
        if x is None or y is None:
            continue
        x = float(x); y = float(y)
        r = 6
        draw.rectangle((x - r, y - r, x + r, y + r), outline=(255, 165, 0, 230), width=3)
        draw.text((x + r + 2, y - r - 2), str(lt.get("id") or "L"), fill=(255, 165, 0, 230))

    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()

# ------------------------------------------------------------
# DXF extraction (robust, vector->raster segmentation)
# ------------------------------------------------------------

def _extract_from_raster_stub(raster_png: bytes, input_type: str) -> Dict[str, Any]:
    # For PDF/JPEG/PNG: you can plug a real extractor later.
    return {
        "meta": {"source": input_type, "note": "Raster extractor not implemented yet"},
        "scale": {"detected": False, "pxPerMeter": None, "method": None, "units_guess": None},
        "rooms": [],
        "openings": [],
        "lights": [],
        "warnings": ["Raster extractor not implemented yet."],
    }

def _pip(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside

def _poly_area(poly: List[Tuple[float, float]]) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def _place_lights_from_rooms(rooms: List[Dict[str, Any]], opts: Dict[str, Any]) -> List[Dict[str, Any]]:
    _require("numpy")
    import numpy as np

    default_type = (opts.get("fixture_type") or os.getenv("LIGHT_DEFAULT_FIXTURE_TYPE") or "DOWNLIGHT").strip()
    lumens = int(opts.get("lumens") or os.getenv("LIGHT_DEFAULT_LUMENS") or LIGHT_DEFAULT_LUMENS)
    watt = float(opts.get("watt") or os.getenv("LIGHT_DEFAULT_WATT") or LIGHT_DEFAULT_WATT)
    cct = int(opts.get("cct_k") or os.getenv("LIGHT_DEFAULT_CCT") or LIGHT_DEFAULT_CCT)
    cri = int(opts.get("cri") or os.getenv("LIGHT_DEFAULT_CRI") or LIGHT_DEFAULT_CRI)
    min_off = float(opts.get("min_wall_offset_m") or os.getenv("LIGHT_MIN_WALL_OFFSET_M") or 0.6)
    grid = float(opts.get("grid_spacing_m") or os.getenv("LIGHT_GRID_SPACING_M") or 3.0)

    def centroid(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def bbox(poly):
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    lights: List[Dict[str, Any]] = []
    for r in rooms:
        poly = r.get("polygon_m") or []
        if len(poly) < 3:
            continue
        area = float(r.get("area_m2") or 0.0)
        rid = r.get("id")

        xmin, ymin, xmax, ymax = bbox(poly)
        cx, cy = centroid(poly)

        pts: List[Tuple[float, float]] = []
        rule = "GRID"

        if area < 12.0:
            pts = [(cx, cy)]
            rule = "CENTROID"
        elif area < 25.0:
            dx = xmax - xmin; dy = ymax - ymin
            if dx >= dy:
                pts = [(xmin + dx * 0.33, cy), (xmin + dx * 0.67, cy)]
            else:
                pts = [(cx, ymin + dy * 0.33), (cx, ymin + dy * 0.67)]
            rule = "AXIS2"
        else:
            xs = np.arange(xmin + min_off, xmax - min_off + 1e-9, grid)
            ys = np.arange(ymin + min_off, ymax - min_off + 1e-9, grid)
            for x in xs:
                for y in ys:
                    if _pip(float(x), float(y), [(float(a), float(b)) for a, b in poly]):
                        pts.append((float(x), float(y)))
            if not pts:
                pts = [(cx, cy)]
                rule = "CENTROID"

        for j, (x, y) in enumerate(pts):
            lights.append({
                "id": f"L_{rid}_{j+1:02d}",
                "room_id": rid,
                "x_m": float(x),
                "y_m": float(y),
                "fixture_type": default_type,
                "lumens": lumens,
                "watt": float(watt),
                "cct_k": cct,
                "cri": cri,
                "beam_deg": int(opts.get("beam_deg") or 60),
                "placement_rule": rule,
                "spacing_m": grid if rule == "GRID" else None,
                "offset_wall_m": min_off,
                "layer": "AMBIENT",
                "notes": "",
            })
    return lights

def _write_lights_to_dxf(doc, lights: List[Dict[str, Any]], units_guess: str = "m") -> bytes:
    _require("ezdxf")
    import ezdxf
    from io import StringIO

    layer_default = (os.getenv("LIGHT_LAYER_NAME") or "ILLUMINAZIONE").strip()
    block_name = (os.getenv("LIGHT_BLOCK_NAME") or "LUXIA_LIGHT_POINT").strip()

    if layer_default not in doc.layers:
        doc.layers.new(layer_default)

    if block_name not in doc.blocks:
        blk = doc.blocks.new(name=block_name)
        blk.add_circle((0, 0), radius=0.05)
        blk.add_line((-0.08, 0), (0.08, 0))
        blk.add_line((0, -0.08), (0, 0.08))
        blk.add_attdef("LUXIA_ID", (0.10, 0.10), height=0.08)
        blk.add_attdef("ROOM_ID", (0.10, 0.00), height=0.08)
        blk.add_attdef("FIXTURE_TYPE", (0.10, -0.10), height=0.08)
        blk.add_attdef("LUMENS", (0.10, -0.20), height=0.08)
        blk.add_attdef("WATT", (0.10, -0.30), height=0.08)
        blk.add_attdef("CCT_K", (0.10, -0.40), height=0.08)
        blk.add_attdef("CRI", (0.10, -0.50), height=0.08)
        blk.add_attdef("BEAM_DEG", (0.10, -0.60), height=0.08)
        blk.add_attdef("PLACEMENT_RULE", (0.10, -0.70), height=0.08)
        blk.add_attdef("NOTES", (0.10, -0.80), height=0.08)

    m_to_unit = 1.0 if units_guess == "m" else 1000.0
    msp = doc.modelspace()

    def _layer_for_light(light: Dict[str, Any]) -> str:
        lyr = str(light.get("layer") or "").upper()
        if lyr == "ACCENT":
            return "ILLUMINAZIONE_ACCENT"
        if lyr == "VERTICAL":
            return "ILLUMINAZIONE_VERTICAL"
        if lyr == "PERIMETER":
            return "ILLUMINAZIONE_PERIMETER"
        return layer_default

    for lt in lights:
        x = float(lt.get("x_m", 0.0)) * m_to_unit
        y = float(lt.get("y_m", 0.0)) * m_to_unit
        ins = msp.add_blockref(block_name, (x, y), dxfattribs={"layer": _layer_for_light(lt)})
        attrs = {
            "LUXIA_ID": str(lt.get("id") or ""),
            "ROOM_ID": str(lt.get("room_id") or ""),
            "FIXTURE_TYPE": str(lt.get("fixture_type") or ""),
            "LUMENS": str(lt.get("lumens") or ""),
            "WATT": str(lt.get("watt") or ""),
            "CCT_K": str(lt.get("cct_k") or ""),
            "CRI": str(lt.get("cri") or ""),
            "BEAM_DEG": str(lt.get("beam_deg") or ""),
            "PLACEMENT_RULE": str(lt.get("placement_rule") or ""),
            "NOTES": str(lt.get("notes") or ""),
        }
        try:
            ins.add_auto_attribs(attrs)
        except Exception:
            for tag, val in attrs.items():
                try:
                    ins.add_attrib(tag, val, insert=(x, y))
                except Exception:
                    pass

    sio = StringIO()
    doc.write(sio)
    return sio.getvalue().encode("utf-8", errors="ignore")

def _extract_from_dxf_real(dxf_bytes: bytes, opts: Dict[str, Any]) -> Dict[str, Any]:
    _require("ezdxf")
    _require("numpy")
    _require("opencv-python")
    import ezdxf
    from ezdxf import recover
    import numpy as np
    import cv2
    from io import BytesIO

    try:
        doc, _auditor = recover.read(BytesIO(dxf_bytes))
    except Exception:
        doc = ezdxf.read(dxf_bytes.decode("utf-8", errors="ignore"))

    msp = doc.modelspace()

    segs: List[Tuple[float, float, float, float]] = []

    def add_seg(x1, y1, x2, y2):
        if (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) < 1e-8:
            return
        segs.append((float(x1), float(y1), float(x2), float(y2)))

    for e in msp.query("LINE"):
        try:
            s = e.dxf.start
            t = e.dxf.end
            add_seg(s.x, s.y, t.x, t.y)
        except Exception:
            pass

    for e in msp.query("LWPOLYLINE POLYLINE"):
        try:
            pts = []
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
            else:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices()]
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    add_seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                closed = False
                try:
                    closed = bool(getattr(e, "closed", False)) or bool(getattr(e, "is_closed", False))
                except Exception:
                    pass
                if closed:
                    add_seg(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1])
        except Exception:
            pass

    if len(segs) < 20:
        return {
            "meta": {"source": "dxf", "note": "too little geometry to detect rooms"},
            "scale": {"detected": False, "pxPerMeter": None, "method": None, "units_guess": None},
            "rooms": [],
            "openings": [],
            "lights": [],
            "warnings": ["DXF geometry seems empty or not a floorplan (too few segments)."],
        }

    xs = [s[0] for s in segs] + [s[2] for s in segs]
    ys = [s[1] for s in segs] + [s[3] for s in segs]
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))
    w = maxx - minx
    h = maxy - miny
    maxdim = max(w, h)

    units_guess = "m"
    unit_to_m = 1.0
    if maxdim > 800:  # typical mm drawings
        units_guess = "mm"
        unit_to_m = 0.001

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
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats((rooms_mask > 0).astype(np.uint8), connectivity=8)

    areas = []
    for i in range(1, num):
        areas.append((int(stats[i, cv2.CC_STAT_AREA]), i))
    areas.sort(reverse=True)

    min_area_px = int(opts.get("min_room_area_px") or 2500)
    room_polys_px: List[List[Tuple[int, int]]] = []
    room_polys_m: List[List[Tuple[float, float]]] = []

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
        room_polys_px.append(pts_px)
        pts_cad = [px_to_cad(x, y) for x, y in pts_px]
        pts_m = [(x * unit_to_m, y * unit_to_m) for x, y in pts_cad]
        room_polys_m.append(pts_m)

    rooms: List[Dict[str, Any]] = []
    for idx, (px_poly, m_poly) in enumerate(zip(room_polys_px, room_polys_m)):
        rooms.append({
            "id": f"R{idx+1:03d}",
            "label": None,
            "area_m2": float(_poly_area(m_poly)),
            "polygon_px": [[int(x), int(y)] for x, y in px_poly],
            "polygon_m": [[float(x), float(y)] for x, y in m_poly],
        })

    lights = _place_lights_from_rooms(rooms, opts)

    # add x_px/y_px for overlay
    for lt in lights:
        x_cad = float(lt.get("x_m", 0.0)) / unit_to_m
        y_cad = float(lt.get("y_m", 0.0)) / unit_to_m
        xpx, ypx = cad_to_px(x_cad, y_cad)
        lt["x_px"] = int(xpx)
        lt["y_px"] = int(ypx)

    dxf_out_b64 = None
    if bool(opts.get("write_lights_dxf", True)):
        try:
            dxf_out = _write_lights_to_dxf(doc, lights, units_guess=units_guess)
            dxf_out_b64 = base64.b64encode(dxf_out).decode("ascii")
        except Exception:
            pass

    out = {
        "meta": {"source": "dxf", "units_guess": units_guess,
                 "bbox": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}},
        "scale": {"detected": True, "pxPerMeter": float(scale / unit_to_m), "method": "dxf_units_guess", "units_guess": units_guess},
        "rooms": rooms,
        "openings": [],
        "lights": lights,
        "warnings": [],
    }
    if dxf_out_b64:
        out["dxf_out_base64"] = dxf_out_b64
    return out

# ------------------------------------------------------------
# Simple grid lux metrics + BOM/spec + report
# ------------------------------------------------------------

def _grid_points_in_polygon(poly_m: List[List[float]], step_m: float = 0.5):
    _require("numpy")
    import numpy as np
    poly = np.array(poly_m, dtype=float)
    if poly.shape[0] < 3:
        return np.zeros((0, 2), dtype=float)

    minx, miny = poly[:, 0].min(), poly[:, 1].min()
    maxx, maxy = poly[:, 0].max(), poly[:, 1].max()
    xs = np.arange(minx, maxx + 1e-6, step_m)
    ys = np.arange(miny, maxy + 1e-6, step_m)
    pts = np.array([(x, y) for x in xs for y in ys], dtype=float)

    x = pts[:, 0]
    y = pts[:, 1]
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    inside = np.zeros(len(pts), dtype=bool)
    for i in range(len(poly)):
        cond = ((y0[i] > y) != (y1[i] > y)) & (x < (x1[i] - x0[i]) * (y - y0[i]) / (y1[i] - y0[i] + 1e-12) + x0[i])
        inside ^= cond
    return pts[inside]

def _radiosity_multiplier_pro(room: Dict[str, Any]) -> float:
    try:
        rho_c = float(room.get("rho_c") or CEILING_REFLECTANCE_DEFAULT)
        rho_w = float(room.get("rho_w") or WALL_REFLECTANCE_DEFAULT)
        rho_f = float(room.get("rho_f") or FLOOR_REFLECTANCE_DEFAULT)
        # conservative: 1.0..1.35
        rho_avg = max(0.0, min(0.95, 0.45 * rho_c + 0.45 * rho_w + 0.10 * rho_f))
        indirect = 0.15 + 0.35 * rho_avg
        return float(max(1.00, min(1.35, 1.0 + indirect)))
    except Exception:
        return 1.10

def _point_lux_iso(points_xy, lum_xyzh, workplane_z: float = WORKPLANE_HEIGHT_M_DEFAULT):
    _require("numpy")
    import numpy as np
    E = np.zeros((points_xy.shape[0],), dtype=float)
    for lx, ly, lz, lf in lum_xyzh:
        dx = points_xy[:, 0] - lx
        dy = points_xy[:, 1] - ly
        dz = workplane_z - lz
        r2 = dx * dx + dy * dy + dz * dz
        r = np.sqrt(np.maximum(r2, 1e-9))
        cos_theta = np.abs(dz) / r
        Iiso = (lf if lf > 0 else LIGHT_DEFAULT_LUMENS) / (4.0 * math.pi)
        E += Iiso * cos_theta / np.maximum(r2, 1e-9)
    return E

def _calc_room_grid_metrics(room: Dict[str, Any], lights: List[Dict[str, Any]], mount_z: float = ROOM_HEIGHT_M_DEFAULT):
    _require("numpy")
    import numpy as np
    poly_m = room.get("polygon_m")
    if not poly_m or len(poly_m) < 3:
        return {"Eavg": 0.0, "Emin": 0.0, "U0": 0.0, "points": 0}

    step = float(os.getenv("GRID_STEP_M", "0.5"))
    pts = _grid_points_in_polygon(poly_m, step_m=step)
    if pts.shape[0] == 0:
        return {"Eavg": 0.0, "Emin": 0.0, "U0": 0.0, "points": 0}

    rid = room.get("id")
    lum_xyzh = []
    for l in (lights or []):
        if str(l.get("room_id") or "") != str(rid or ""):
            continue
        lum_xyzh.append([float(l.get("x_m") or 0.0), float(l.get("y_m") or 0.0), float(mount_z), float(l.get("lumens") or LIGHT_DEFAULT_LUMENS)])

    if not lum_xyzh:
        return {"Eavg": 0.0, "Emin": 0.0, "U0": 0.0, "points": int(pts.shape[0]), "grid_step_m": step}

    lum_xyzh = np.array(lum_xyzh, dtype=float)
    E = _point_lux_iso(pts, lum_xyzh, workplane_z=WORKPLANE_HEIGHT_M_DEFAULT)
    E *= _radiosity_multiplier_pro(room)

    Eavg = float(E.mean())
    Emin = float(E.min())
    U0 = float(Emin / Eavg) if Eavg > 1e-6 else 0.0
    return {"Eavg": Eavg, "Emin": Emin, "U0": U0, "points": int(pts.shape[0]), "grid_step_m": step}

def _status_vs_targets(metrics: Dict[str, Any], targets: Dict[str, Any], ugr: Optional[float]) -> Dict[str, Any]:
    reasons: List[str] = []
    tgt_lux = float(targets.get("lux") or 0.0)
    tgt_u0 = float(targets.get("u0") or 0.0)
    tgt_ugr = targets.get("ugr_max", None)
    eavg = float(metrics.get("Eavg") or 0.0)
    u0 = float(metrics.get("U0") or 0.0)
    if tgt_lux > 0 and eavg + 1e-9 < tgt_lux:
        reasons.append("Eavg")
    if tgt_u0 > 0 and u0 + 1e-9 < tgt_u0:
        reasons.append("U0")
    if tgt_ugr is not None and ugr is not None:
        try:
            if float(ugr) > float(tgt_ugr) + 1e-9:
                reasons.append("UGR")
        except Exception:
            pass
    return {"status": ("OK" if not reasons else "KO"), "reasons": reasons}

def _build_bom(extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: Dict[str, Dict[str, Any]] = {}
    for l in (extracted.get("lights") or []):
        typ = (l.get("fixture_type") or "DOWNLIGHT").upper()
        cct = str(l.get("cct_k") or LIGHT_DEFAULT_CCT)
        cri = str(l.get("cri") or LIGHT_DEFAULT_CRI)
        watt = float(l.get("watt") or LIGHT_DEFAULT_WATT)
        lum = float(l.get("lumens") or LIGHT_DEFAULT_LUMENS)
        key = f"{typ}|{cct}|{cri}|{watt:.1f}|{lum:.0f}"
        items.setdefault(key, {"type": typ, "cct_k": int(float(cct)), "cri": int(float(cri)), "watt": watt, "lumens": lum, "qty": 0})
        items[key]["qty"] += 1
    return list(items.values())

def _bom_csv_bytes(extracted: Dict[str, Any]) -> bytes:
    bom = _build_bom(extracted)
    out = ["type,qty,watt,lumens,cct_k,cri"]
    for it in bom:
        out.append(f"{it['type']},{it['qty']},{it['watt']:.1f},{it['lumens']:.0f},{it['cct_k']},{it['cri']}")
    return ("\n".join(out) + "\n").encode("utf-8")

def _build_spec_text(extracted: Dict[str, Any]) -> str:
    bom = _build_bom(extracted)
    lines = []
    lines.append("CAPITOLATO – IMPIANTO ILLUMINAZIONE (LuxIA)")
    lines.append("")
    lines.append("1. Oggetto")
    lines.append("   Fornitura e posa di apparecchi di illuminazione come da elaborati e abaco.")
    lines.append("")
    lines.append("2. Prestazioni")
    lines.append("   Dimensionamento conforme a UNI EN 12464-1 (interni) e UNI EN 12464-2 (esterni).")
    lines.append(f"   Fattore di manutenzione (MF): {MAINTENANCE_FACTOR_DEFAULT:.2f}.")
    lines.append(f"   Riflettanze: soffitto {CEILING_REFLECTANCE_DEFAULT:.2f}, pareti {WALL_REFLECTANCE_DEFAULT:.2f}, pavimento {FLOOR_REFLECTANCE_DEFAULT:.2f}.")
    lines.append("")
    lines.append("3. Abaco apparecchi (sintesi)")
    if not bom:
        lines.append("   - n/d")
    else:
        for it in bom:
            lines.append(f"   - {it['qty']}x {it['type']}  {it['watt']:.0f}W  {it['lumens']:.0f}lm  {it['cct_k']}K  CRI {it['cri']}")
    lines.append("")
    lines.append("4. Note")
    lines.append("   Verifiche e overlay automatici eseguiti con controllo di coerenza geometrica (prova del nove).")
    return "\n".join(lines)

def _generate_pdf_report(project_title: str, analyzed: Dict[str, Any], overlay_png: Optional[bytes] = None,
                         verification: Optional[Dict[str, Any]] = None, checks: Optional[Dict[str, Any]] = None) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    def hr(y, gray=0.86):
        c.setStrokeGray(gray); c.setLineWidth(0.6)
        c.line(24 * mm, y, W - 24 * mm, y)
        c.setStrokeGray(0.0)

    def badge(x, y, text, ok=True):
        if ok: c.setFillColorRGB(0.10, 0.55, 0.25)
        else: c.setFillColorRGB(0.75, 0.15, 0.15)
        c.roundRect(x, y - 10, 56, 14, 6, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(x + 28, y - 6, text)
        c.setFillColorRGB(0, 0, 0)

    def header(title: str, subtitle: str = ""):
        c.setFont("Helvetica-Bold", 22)
        c.drawString(24 * mm, H - 28 * mm, title)
        if subtitle:
            c.setFont("Helvetica", 10); c.setFillGray(0.35)
            c.drawString(24 * mm, H - 34 * mm, subtitle)
            c.setFillGray(0.0)
        hr(H - 38 * mm, 0.80)

    def footer():
        c.setFont("Helvetica", 8); c.setFillGray(0.45)
        c.drawString(24 * mm, 14 * mm, "LuxIA • Report automatico")
        c.drawRightString(W - 24 * mm, 14 * mm, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
        c.setFillGray(0.0)

    rooms = analyzed.get("rooms") or []
    lights = analyzed.get("lights") or []
    lux_results = analyzed.get("lux_results") or []

    by_id = {r.get("room_id"): r for r in lux_results if r.get("room_id")}
    ok_n = sum(1 for rm in rooms if (by_id.get(rm.get("id")) or {}).get("status") == "OK")
    ko_n = sum(1 for rm in rooms if (by_id.get(rm.get("id")) or {}).get("status") == "KO")
    total_power = sum(float(l.get("watt") or 0.0) for l in lights) if lights else 0.0

    header("LuxIA", project_title or "Report Illuminotecnico")
    c.setFont("Helvetica", 11); c.setFillGray(0.25)
    c.drawString(24 * mm, H - 48 * mm, "Analisi multilivello • UNI EN 12464-1/2 • Interriflessioni (modello pro)")
    c.setFillGray(0.0)
    if verification and isinstance(verification, dict):
        badge(W - 24 * mm - 56, H - 34 * mm, "PASS" if verification.get("pass") else "FAIL", ok=bool(verification.get("pass")))

    y = H - 62 * mm
    if overlay_png:
        try:
            img = ImageReader(io.BytesIO(overlay_png))
            iw, ih = img.getSize()
            maxw = W - 48 * mm
            maxh = 118 * mm
            scale = min(maxw / iw, maxh / ih)
            dw, dh = iw * scale, ih * scale
            c.drawImage(img, 24 * mm, y - dh, width=dw, height=dh, preserveAspectRatio=True, mask="auto")
            y = y - dh - 10 * mm
        except Exception:
            pass

    c.setFont("Helvetica-Bold", 12)
    c.drawString(24 * mm, y, "Sintesi progetto")
    y -= 7 * mm
    hr(y + 2 * mm)
    c.setFont("Helvetica", 10)
    c.drawString(24 * mm, y, f"Ambienti: {len(rooms)}    •    Apparecchi: {len(lights)}    •    Potenza installata: {total_power:.0f} W")
    y -= 6 * mm
    c.drawString(24 * mm, y, f"Conformità (pre-check): OK {ok_n} • KO {ko_n}")
    y -= 6 * mm
    if checks:
        c.setFillGray(0.25)
        c.drawString(24 * mm, y, f"Checks: ok={checks.get('ok')} • conf={float(checks.get('confidence') or 0.0):.2f} • issues={len(checks.get('issues') or [])}")
        c.setFillGray(0.0)
        y -= 6 * mm

    footer()
    c.showPage()

    header("Abaco apparecchi", "BOM e note di capitolato")
    y = H - 52 * mm
    bom = _build_bom(analyzed)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(24 * mm, y, "BOM (Bill of Materials)")
    y -= 7 * mm
    hr(y + 2 * mm)

    c.setFont("Helvetica-Bold", 9)
    cols = [24 * mm, 74 * mm, 100 * mm, 124 * mm, 150 * mm]
    for i, t in enumerate(["Tipo", "Q.tà", "W", "lm", "CCT/CRI"]):
        c.drawString(cols[i], y, t)
    y -= 4 * mm
    hr(y, 0.85)
    y -= 6 * mm
    c.setFont("Helvetica", 9)

    if not bom:
        c.setFillGray(0.35); c.drawString(24 * mm, y, "n/d"); c.setFillGray(0.0)
    else:
        for it in bom:
            if y < 24 * mm:
                footer(); c.showPage()
                header("Abaco apparecchi", "continua")
                y = H - 52 * mm
                c.setFont("Helvetica", 9)
            c.drawString(cols[0], y, str(it["type"])[:22])
            c.drawRightString(cols[2] - 2 * mm, y, str(int(it["qty"])))
            c.drawRightString(cols[3] - 2 * mm, y, f"{float(it['watt']):.0f}")
            c.drawRightString(cols[4] - 2 * mm, y, f"{float(it['lumens']):.0f}")
            c.drawString(cols[4], y, f"{int(it['cct_k'])}K / CRI {int(it['cri'])}")
            y -= 6 * mm

    y -= 10 * mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(24 * mm, y, "Capitolato (estratto)")
    y -= 7 * mm
    hr(y + 2 * mm)
    c.setFont("Helvetica", 9)
    spec = _build_spec_text(analyzed)
    for ln in spec.splitlines():
        if y < 24 * mm:
            footer(); c.showPage()
            header("Capitolato", "continua")
            y = H - 52 * mm
            c.setFont("Helvetica", 9)
        c.setFillGray(0.15)
        c.drawString(24 * mm, y, ln[:110])
        c.setFillGray(0.0)
        y -= 4.5 * mm

    footer()
    c.showPage()
    c.save()
    return buf.getvalue()

# ------------------------------------------------------------
# API: planimetry/analyze
# ------------------------------------------------------------

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

    try:
        opts = json.loads(options or "{}")
        if not isinstance(opts, dict):
            opts = {}
    except Exception:
        opts = {}

    input_type = _guess_type(file.filename or "", file.content_type or "")
    page_count = 1

    dxf_bytes: Optional[bytes] = None
    if input_type == "pdf":
        raster_png = _pdf_to_png_first_page(raw)
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
        input_type = "dwg"
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, JPG/PNG, DXF, or DWG.")

    checks = _deterministic_checks(extracted)
    overlay_png = _make_overlay_png(raster_png, extracted)
    vres = vision.verify(raster_png, extracted, overlay_png=overlay_png)

    # add lux_results (for DXF flows)
    if extracted.get("rooms") and extracted.get("lights"):
        lux_results = []
        for room in extracted.get("rooms") or []:
            targets = _get_uni_targets(room)
            metrics = _calc_room_grid_metrics(room, extracted.get("lights") or [], mount_z=float(opts.get("mount_z_m") or ROOM_HEIGHT_M_DEFAULT))
            ugr = _ugr_approx(_LUMINAIRE_LIB.get(str(opts.get("luminaire_key") or "default")))
            st = _status_vs_targets(metrics, targets, ugr)
            lux_results.append({
                "room_id": room.get("id"),
                "label": room.get("label"),
                "targets": targets,
                "metrics": metrics,
                "status": st.get("status"),
                "reasons": st.get("reasons"),
                "ugr_approx": ugr,
            })
        extracted["lux_results"] = lux_results

    return {
        "ok": True,
        "input_type": input_type,
        "page_count": page_count,
        "extracted": extracted,
        "checks": checks,
        "vision": vres,
        "overlay_png_base64": base64.b64encode(overlay_png).decode("ascii"),
    }

# ------------------------------------------------------------
# API: report/pdf
# ------------------------------------------------------------
@app.post("/report/pdf")
async def report_pdf(req: Request):
    payload = await req.json()
    analyzed = payload.get("analyzed") or {}
    overlay_b64 = payload.get("overlay_png_base64")
    overlay_png = base64.b64decode(overlay_b64) if overlay_b64 else None
    title = payload.get("title") or (analyzed.get("meta", {}).get("project_title") if isinstance(analyzed, dict) else None) or "Report Illuminotecnico"
    pdf = _generate_pdf_report(title, analyzed, overlay_png=overlay_png, verification=payload.get("verification"), checks=payload.get("checks"))
    return {"ok": True, "pdf_base64": base64.b64encode(pdf).decode("ascii")}

# ------------------------------------------------------------
# API: luminaires
# ------------------------------------------------------------
@app.post("/luminaires/upload")
async def luminaires_upload(file: UploadFile = File(...), key: str = Form("default")):
    raw = await file.read()
    if not raw or len(raw) < 64:
        raise HTTPException(status_code=400, detail="Empty luminaire file")
    name = (key or "default").strip()[:64]
    phot = _parse_ies_lm63(raw)
    _LUMINAIRE_LIB[name] = phot
    return {"ok": True, "key": name, "type": phot.get("type"), "input_watts": phot.get("input_watts")}

@app.get("/luminaires/list")
async def luminaires_list():
    return {"ok": True, "items": [{"key": k, "type": v.get("type"), "input_watts": v.get("input_watts")} for k, v in _LUMINAIRE_LIB.items()]}

@app.post("/luminaires/ugr/upload")
async def ugr_upload(req: Request):
    payload = await req.json()
    key = str(payload.get("key") or "default")
    if not payload.get("K") or not payload.get("rho") or not payload.get("ugr"):
        raise HTTPException(status_code=400, detail="Missing K/rho/ugr")
    _UGR_TABLES[key] = {"K": payload["K"], "rho": payload["rho"], "ugr": payload["ugr"]}
    return {"ok": True, "key": key}

# ------------------------------------------------------------
# API: exports
# ------------------------------------------------------------
@app.post("/export/bom.csv")
async def export_bom(req: Request):
    payload = await req.json()
    extracted = payload.get("extracted") or {}
    data = _bom_csv_bytes(extracted)
    return Response(content=data, media_type="text/csv")

@app.post("/export/spec")
async def export_spec(req: Request):
    payload = await req.json()
    extracted = payload.get("extracted") or {}
    txt = _build_spec_text(extracted)
    return {"ok": True, "spec": txt}

# ------------------------------------------------------------
# Health
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "llm_provider": llm.provider, "vision_enabled": vision.enabled}
