# apps/engine/engine_api.py
# Patch: Qwen as LLM orchestrator (Vision remains parser/OCR)
# - Adds LLM_PROVIDER router: qwen | groq | ollama | dummy
# - Qwen uses OpenAI-compatible Chat Completions endpoint (self-host or provider)
#   Env:
#     LLM_PROVIDER=qwen
#     QWEN_BASE_URL=https://<your-openai-compatible-host>/v1
#     QWEN_API_KEY=...
#     QWEN_MODEL=qwen2.5-instruct   (example; set to your served model id)
#
# NOTE: This file is meant to replace your existing engine_api.py ONLY if you already
# have the LuxIA v22 engine with FastAPI. If your file name differs, copy the
# functions LLMRouter + llm_call below into your current engine_api.py.

from __future__ import annotations

import os, json, time, uuid
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LuxIA Engine", version="22.x-qwen-patch")

# --------------------------
# Models
# --------------------------
class Msg(BaseModel):
    role: str
    content: str

class LLMResult(BaseModel):
    text: str
    raw: Optional[Dict[str, Any]] = None
    provider: str

# --------------------------
# LLM Router
# --------------------------
class LLMRouter:
    def __init__(self):
        self.provider = (os.getenv("LLM_PROVIDER") or "dummy").strip().lower()

        # Qwen (OpenAI-compatible)
        self.qwen_base_url = (os.getenv("QWEN_BASE_URL") or "").rstrip("/")
        self.qwen_api_key = os.getenv("QWEN_API_KEY") or ""
        self.qwen_model = os.getenv("QWEN_MODEL") or "qwen2.5-instruct"

        # Ollama (OpenAI-like-ish or native)
        self.ollama_url = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL") or "qwen2.5"

        # Groq (optional; OpenAI-compatible endpoint)
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
            # Use native Ollama generate endpoint (works without API key).
            return self._ollama_chat(messages, temperature=temperature, provider="ollama")

        # dummy fallback (always works)
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
                text=f"({provider}) Missing {provider.upper()}_BASE_URL. Using heuristic concepts.",
                raw=None,
                provider=provider,
            )
        # Many OpenAI-compatible servers expose /chat/completions
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
        # Convert chat messages to a single prompt (simple and robust).
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

# --------------------------
# Heuristic fallback for concepts (works even without LLM)
# --------------------------
def heuristic_concepts(brief: str, areas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Minimal but useful: produce 3 variants with different style/CCT/UGR strategy.
    styles = [
        {"name":"Concept A — Comfort Office","cct":"4000K","strategy":"UGR low, uniform task lighting + accent"},
        {"name":"Concept B — Warm Hospitality","cct":"3000K","strategy":"layered lighting, warm ambience + highlights"},
        {"name":"Concept C — Retail Contrast","cct":"3500K","strategy":"higher vertical illuminance, accents on focal zones"},
    ]
    out=[]
    for s in styles:
        out.append({
            "title": s["name"],
            "cct": s["cct"],
            "notes": f"{s['strategy']}. Brief: {brief[:240]}",
            "areas": [{"name":a.get("name") or a.get("nome") or "Area", "type": a.get("type") or a.get("tipo_locale") or "Ufficio VDT"} for a in areas][:30]
        })
    return out

# --------------------------
# API: generate concepts (called by UI Autopilot)
# --------------------------
class ConceptReq(BaseModel):
    project_id: str
    brief: str = ""
    areas: List[Dict[str, Any]] = []
    n: int = 3

@app.post("/concepts")
def generate_concepts(req: ConceptReq):
    # Build prompt for orchestrator LLM (Qwen recommended).
    system = (
        "You are LuxIA, a professional lighting design agent. "
        "Generate 3 distinct lighting concepts for the given areas. "
        "Return STRICT JSON only with schema: "
        "{\"concepts\":[{\"title\":...,\"cct\":...,\"style\":...,\"strategy\":...,\"per_area\":[{\"area\":...,\"target_lux\":...,\"notes\":...}]}]} "
        "Do not include markdown."
    )
    user = {
        "brief": req.brief,
        "areas": req.areas,
        "n": req.n,
        "constraints": {"standards":["EN 12464-1"], "focus":"interiors"},
    }
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":json.dumps(user, ensure_ascii=False)},
    ]
    res = llm.chat(messages, temperature=0.2, max_tokens=1600)

    # Parse JSON if possible; fallback to heuristic
    concepts = None
    try:
        m = re.search(r"\{.*\}", res.text, flags=re.S)
        if m:
            payload = json.loads(m.group(0))
            concepts = payload.get("concepts")
    except Exception:
        concepts = None

    if not concepts:
        concepts = heuristic_concepts(req.brief, req.areas)

    return {"ok": True, "provider": res.provider, "concepts": concepts, "llm_note": res.text[:400]}

@app.get("/health")
def health():
    return {"ok": True, "llm_provider": llm.provider}
