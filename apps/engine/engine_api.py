# apps/engine/engine_api.py
from __future__ import annotations

import os
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="LuxIA Engine API (MVP)",
    version="1.0",
)

# CORS: per chiamate da Vercel (e in generale webapp)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # per MVP; quando vai in prod metti dominio Vercel specifico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Auth (semplice)
# -----------------------------
def auth(x_luxia_token: Optional[str]):
    # Se non imposti LUXIA_ENGINE_TOKEN, auth è “open” (utile MVP)
    expected = os.getenv("LUXIA_ENGINE_TOKEN", "").strip()
    if expected:
        if not x_luxia_token or x_luxia_token.strip() != expected:
            raise HTTPException(status_code=401, detail="Unauthorized (bad X-LuxIA-Token)")


# -----------------------------
# Requirements base (MVP)
# -----------------------------
REQUISITI: Dict[str, Dict[str, Any]] = {
    "Ufficio VDT": {"lux": 500, "ugr_max": 19, "ra_min": 80},
    "Sala riunioni": {"lux": 500, "ugr_max": 19, "ra_min": 80},
    "Corridoio": {"lux": 100, "ugr_max": 22, "ra_min": 80},
    "Ufficio": {"lux": 300, "ugr_max": 22, "ra_min": 80},
    "Open space": {"lux": 500, "ugr_max": 19, "ra_min": 80},
}

# -----------------------------
# Style helpers
# -----------------------------
def _pack_params(project_style_pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uniforma i parametri dello style pack.
    """
    return {
        "cct_default": int(project_style_pack.get("cct_default") or 3000),
        "density_bias": float(project_style_pack.get("density_bias") or 1.0),
        "uniformity_target": float(project_style_pack.get("uniformity_target") or 0.60),
        "accent_ratio": float(project_style_pack.get("accent_ratio") or 0.20),
        "contrast": str(project_style_pack.get("contrast") or "Medium"),
        "theme": str(project_style_pack.get("theme") or "Clean Office"),
    }

def _style_bias(style_tokens: Dict[str, Any]) -> Dict[str, float]:
    # MVP: se non hai tokens → neutro
    # puoi “spingere” un concept in base a token/tag futuri
    return {
        "comfort": float(style_tokens.get("comfort_bias") or 1.0),
        "efficiency": float(style_tokens.get("efficiency_bias") or 1.0),
        "architectural": float(style_tokens.get("architectural_bias") or 1.0),
    }

def _team_bias(designer_stats: Dict[str, Any]) -> Dict[str, float]:
    # MVP: se non hai stats → neutro
    return {
        "comfort": float(designer_stats.get("comfort") or 1.0),
        "efficiency": float(designer_stats.get("efficiency") or 1.0),
        "architectural": float(designer_stats.get("architectural") or 1.0),
    }

def _area_bias(designer_area_bias: Dict[str, Any]) -> Dict[str, float]:
    # designer_area_bias può arrivare vuoto
    return {
        "comfort": float(designer_area_bias.get("comfort") or 1.0),
        "efficiency": float(designer_area_bias.get("efficiency") or 1.0),
        "architectural": float(designer_area_bias.get("architectural") or 1.0),
    }

# -----------------------------
# Catalog filtering
# -----------------------------
def _filter_candidates(
    catalog: List[Dict[str, Any]],
    allowed_brands: List[str],
    spec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Filtra catalogo in base a brand e minimi requisiti.
    Catalog item atteso (MVP): {brand, model, flux_lm, watt, cri, ugr, optics, ...}
    """
    out = []
    lux = float(spec.get("required_lux") or 300)
    ra_min = int(spec.get("min_ra") or 80)
    ugr_max = float(spec.get("max_ugr") or 22)

    for it in (catalog or []):
        b = (it.get("brand") or "").strip()
        if allowed_brands and b and b not in allowed_brands:
            continue

        cri = int(it.get("cri") or it.get("ra") or 80)
        ugr = float(it.get("ugr") or 22)
        flux = float(it.get("flux_lm") or 0)
        if cri < ra_min:
            continue
        if ugr > ugr_max:
            continue
        # non imposto una soglia flux rigida, ma evito “zero”
        if flux <= 0:
            continue

        out.append(it)

    # se non hai catalogo / non matcha nulla → modalità brand-neutral (out vuoto)
    return out

# -----------------------------
# Layout & calc proxy
# -----------------------------
def _dims_from_area(area_m2: float) -> Tuple[float, float]:
    # MVP: supponiamo locale ~rettangolare
    W = math.sqrt(max(area_m2, 1.0))
    L = max(area_m2, 1.0) / W
    return (round(L, 2), round(W, 2))

def _grid_coords(L: float, W: float, n: int) -> List[List[float]]:
    if n <= 0:
        return []
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    dx = L / (cols + 1)
    dy = W / (rows + 1)
    coords = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= n:
                break
            coords.append([round((c + 1) * dx, 3), round((r + 1) * dy, 3)])
            k += 1
    return coords

def _estimate_n(area_m2: float, target_lux: float, lum_flux: float, util: float = 0.55, maint: float = 0.80) -> int:
    # Lumen method proxy (molto semplificato)
    needed_lm = target_lux * area_m2
    per_lum = max(lum_flux * util * maint, 1.0)
    return max(1, int(math.ceil(needed_lm / per_lum)))

def _uniformity_proxy(n: int, area_m2: float) -> float:
    # Proxy: più apparecchi, più uniformità (semplificatissimo)
    base = 0.35 + 0.08 * math.log(max(n, 1))
    penalty = 0.05 * math.log(max(area_m2, 1.0))
    return float(max(0.30, min(0.85, base - penalty)))

def _wm2_proxy(n: int, watt: float, area_m2: float) -> float:
    return round((n * max(watt, 0.0)) / max(area_m2, 1.0), 2)

# -----------------------------
# Spec builder (FIX pack_params)
# -----------------------------
def _ideal_spec(
    area: Dict[str, Any],
    concept_type: str,
    req: Dict[str, Any],
    constraints: str,
    pack_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Crea una “spec” target per concept.
    FIX: usa pack_params passato esplicitamente.
    """
    required_lux = float(req.get("lux") or 300)
    max_ugr = float(req.get("ugr_max") or 22)
    min_ra = int(req.get("ra_min") or 80)

    # bias per concept
    density_bias = float(pack_params.get("density_bias") or 1.0)
    uniformity_target = float(pack_params.get("uniformity_target") or 0.60)

    if concept_type == "comfort":
        required_lux *= 1.05
        uniformity_target = min(0.80, uniformity_target + 0.08)
    elif concept_type == "efficiency":
        required_lux *= 0.95
        density_bias *= 0.92
    elif concept_type == "architectural":
        required_lux *= 1.00
        # qui potresti spingere accent ratio, scene ecc.

    return {
        "required_lux": round(required_lux, 1),
        "max_ugr": max_ugr,
        "min_ra": min_ra,
        "density_bias": round(density_bias, 3),
        "uniformity_target": round(uniformity_target, 3),
        "cct_default": int(pack_params.get("cct_default") or 3000),
        "constraints": constraints or "",
        "area_name": area.get("name") or "Area",
        "tipo_locale": area.get("tipo_locale") or "Ufficio VDT",
    }

# -----------------------------
# Optimizer (MVP)
# -----------------------------
def _pick_best_luminaire(candidates: List[Dict[str, Any]], concept_type: str) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    # MVP ranking
    def score(it: Dict[str, Any]) -> float:
        flux = float(it.get("flux_lm") or 0)
        watt = float(it.get("watt") or 1)
        cri = float(it.get("cri") or 80)
        ugr = float(it.get("ugr") or 22)
        eff = flux / max(watt, 1e-6)

        s = 0.0
        if concept_type == "efficiency":
            s += 1.5 * eff
            s += 0.2 * cri
            s -= 0.3 * ugr
        elif concept_type == "comfort":
            s += 0.5 * eff
            s += 0.5 * cri
            s -= 1.0 * ugr
        else:  # architectural
            s += 0.9 * eff
            s += 0.4 * cri
            s -= 0.5 * ugr
        return s

    return sorted(candidates, key=score, reverse=True)[0]

def _solve(area_m2: float, h_m: float, spec: Dict[str, Any], lum: Optional[Dict[str, Any]], req: Dict[str, Any], concept_type: str) -> Dict[str, Any]:
    target_lux = float(spec.get("required_lux") or req.get("lux") or 300)
    density_bias = float(spec.get("density_bias") or 1.0)
    uniformity_target = float(spec.get("uniformity_target") or 0.60)

    L, W = _dims_from_area(area_m2)

    if lum:
        flux = float(lum.get("flux_lm") or 1500)
        watt = float(lum.get("watt") or 15)
        n0 = _estimate_n(area_m2, target_lux, flux)
        n = max(1, int(math.ceil(n0 * density_bias)))
        Em = round((n * flux * 0.55 * 0.80) / max(area_m2, 1.0), 1)
        u0 = round(_uniformity_proxy(n, area_m2), 2)
        wm2 = _wm2_proxy(n, watt, area_m2)
    else:
        # brand-neutral fallback
        n = max(4, int(math.ceil(area_m2 / 4.0)))
        Em = round(target_lux * 0.85, 1)
        u0 = round(_uniformity_proxy(n, area_m2), 2)
        wm2 = round(7.0, 2)

    coords = _grid_coords(L, W, n)
    return {
        "n": n,
        "Em": Em,
        "u0": u0,
        "wm2": wm2,
        "layout": "grid",
        "coords": coords,
        "dims": {"L": L, "W": W, "H": float(h_m)},
        "targets": {"lux": target_lux, "u0": uniformity_target},
        "autopilot": {"priority": "mix", "iterations": 1, "status": "mvp"},
    }

def _optimize(
    area_m2: float,
    tipo: str,
    h_m: float,
    req: Dict[str, Any],
    spec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    concept_type: str,
    priority: str,
) -> Dict[str, Any]:
    lum = _pick_best_luminaire(candidates, concept_type)
    calc = _solve(area_m2, h_m, spec, lum, req, concept_type)
    calc["autopilot"]["priority"] = priority
    return {"luminaire": lum, "calc": calc}

def _notes(concept_type: str, calc: Dict[str, Any], lum: Optional[Dict[str, Any]], spec: Dict[str, Any]) -> str:
    base = {
        "comfort": "Priorità comfort visivo (UGR, uniformità, riflessi VDT).",
        "efficiency": "Priorità efficienza energetica (W/m²), ottimizzazione quantità e flusso.",
        "architectural": "Priorità qualità percettiva (gerarchie, accenti, scene/dimming).",
    }[concept_type]
    pr = (calc.get("autopilot") or {}).get("priority", "mix")
    it = (calc.get("autopilot") or {}).get("iterations", 1)
    lay = calc.get("layout", "grid")
    u0 = calc.get("u0", "—")
    s = f" Autopilot: {pr} • tentativi={it} • layout={lay} • U0≈{u0}."
    if lum:
        part = f" Apparecchio: {lum.get('brand')} — {lum.get('model')} ({lum.get('flux_lm')} lm, {lum.get('watt')} W, Ra {lum.get('cri')}, UGR {lum.get('ugr')})."
    else:
        part = " Modalità brand-neutral: nessun brand selezionato o nessun match conforme."
    nums = f" Stima: N={calc.get('n')} • Em≈{calc.get('Em')} lux (target {spec.get('required_lux')}) • W/m²≈{calc.get('wm2')}."
    if spec.get("constraints"):
        nums += " Vincoli: " + spec.get("constraints")
    return base + s + part + nums

# -----------------------------
# Export helpers
# -----------------------------
def _dxf_from_solution(result: List[Dict[str, Any]]) -> str:
    lines = ["0","SECTION","2","ENTITIES"]
    x_off = 0.0
    spacing = 2.0
    for area_block in result:
        area_name = area_block.get("area","Area")
        concepts = area_block.get("concepts", [])
        if not concepts:
            continue
        c = concepts[0]
        calc = (c.get("calc") or {})
        dims = (calc.get("dims") or {})
        L = float(dims.get("L") or 5.0)
        W = float(dims.get("W") or 5.0)
        coords = calc.get("coords") or []

        # perimetro area
        lines += ["0","LWPOLYLINE","8","AREE","90","4","70","1",
                  "10",str(x_off),"20","0.0",
                  "10",str(x_off+L),"20","0.0",
                  "10",str(x_off+L),"20",str(W),
                  "10",str(x_off),"20",str(W)]
        # punti
        for p in coords:
            px, py = float(p[0]), float(p[1])
            lines += ["0","POINT","8","LAMPADE","10",str(x_off+px),"20",str(py),"30","0.0"]

        em = calc.get("Em","—")
        u0 = calc.get("u0","—")
        lines += ["0","TEXT","8","TESTI",
                  "10",str(x_off + L/2.0),"20",str(W/2.0),"30","0.0",
                  "40","0.35","1", f"{area_name} Em≈{em} U0≈{u0}"]

        x_off += L + spacing

    lines += ["0","ENDSEC","0","EOF"]
    return "\n".join(lines)

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "LuxIA Engine API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"ok": True, "features": ["mvp-concepts", "layout-grid", "uniformity-proxy", "export-dxf", "export-json"]}

@app.post("/projects/{project_id}/concepts")
def generate_concepts(project_id: str, payload: Dict[str, Any], x_luxia_token: Optional[str] = Header(None)):
    auth(x_luxia_token)

    areas: List[Dict[str, Any]] = payload.get("areas", []) or []
    style_tokens: Dict[str, Any] = payload.get("style_tokens", {}) or {}
    designer_stats: Dict[str, Any] = payload.get("designer_stats", {}) or {}
    project_style_pack: Dict[str, Any] = payload.get("project_style_pack", {}) or {}
    pack_params = _pack_params(project_style_pack)

    # IMPORTANT: sempre default sicuro
    designer_area_bias: Dict[str, Any] = payload.get("designer_area_bias", {}) or {}

    allowed_brands: List[str] = payload.get("allowed_brands", []) or []
    catalog: List[Dict[str, Any]] = payload.get("catalog", []) or []
    priority: str = (payload.get("priority") or "mix").lower()
    constraints: str = payload.get("constraints") or ""

    token_bias = _style_bias(style_tokens)
    team_bias = _team_bias(designer_stats)
    area_bias = _area_bias(designer_area_bias)

    # blend: tokens 0.35 + team 0.40 + area 0.25
    bias = {
        k: (0.35*token_bias.get(k,1.0) + 0.40*team_bias.get(k,1.0) + 0.25*area_bias.get(k,1.0))
        for k in ["comfort","efficiency","architectural"]
    }
    order = sorted(["comfort", "efficiency", "architectural"], key=lambda k: bias.get(k, 1.0), reverse=True)

    out = []
    for area in areas:
        name = area.get("name", "Area")
        tipo = area.get("tipo_locale", "Ufficio VDT")
        req = REQUISITI.get(tipo, REQUISITI["Ufficio VDT"])
        area_m2 = float(area.get("superficie_m2") or 20.0)
        h_m = float(area.get("height_m") or area.get("altezza_m") or 2.7)

        block = {"area": name, "concepts": []}
        for idx, ctype in enumerate(order):
            # FIX: passiamo pack_params qui (prima era rotto)
            spec = _ideal_spec(area, ctype, req, constraints, pack_params)
            candidates = _filter_candidates(catalog, allowed_brands, spec)
            sol = _optimize(area_m2, tipo, h_m, req, spec, candidates, ctype, priority)

            lum = sol.get("luminaire")
            calc = sol.get("calc", {})

            cname = "Comfort" if ctype == "comfort" else ("Efficienza" if ctype == "efficiency" else "Architetturale")
            cid = f"c{idx+1}"

            block["concepts"].append({
                "id": cid,
                "name": cname,
                "concept_type": ctype,
                "ideal_spec": spec,
                "luminaire": lum,
                "calc": calc,
                "notes": _notes(ctype, calc, lum, spec),
                "style_bias": round(bias.get(ctype, 1.0), 2),
            })

        out.append(block)

    return {
        "ok": True,
        "project_id": project_id,
        "result": out,
        "style_used": {
            "bias": bias,
            "token_bias": token_bias,
            "team_bias": team_bias,
            "area_bias": area_bias,
            "order": order,
            "pack_params": pack_params,
        },
        "brands_used": allowed_brands,
        "autopilot": {"priority": priority, "constraints": constraints},
    }

@app.post("/exports/layout.json")
def export_layout_json(payload: Dict[str, Any], x_luxia_token: Optional[str] = Header(None)):
    auth(x_luxia_token)
    result = payload.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="Missing result")
    content = json.dumps({"type":"luxia_layout", "version":"1.0", "result": result}, ensure_ascii=False, indent=2)
    return Response(content, media_type="application/json")

@app.post("/exports/layout.dxf")
def export_layout_dxf(payload: Dict[str, Any], x_luxia_token: Optional[str] = Header(None)):
    auth(x_luxia_token)
    result = payload.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="Missing result")
    dxf = _dxf_from_solution(result)
    return Response(dxf, media_type="application/dxf")
