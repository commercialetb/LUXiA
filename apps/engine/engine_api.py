import os
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import math
import json


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

APP_TOKEN = os.getenv("LUXIA_TOKEN", "dev-local-token")

app = FastAPI(title="LuxIA Engine API (MVP)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    # MVP: allow calls from Vercel/localhost without fighting CORS during setup.
    # Tighten this (specific domains) before production.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def auth(x_luxia_token: Optional[str]):
    if x_luxia_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Minimal normative targets (extend later)
REQUISITI = {
    "Ufficio VDT":   {"lux": 500, "ugr_max": 19, "ra_min": 80, "uni_min": 0.60, "CU": 0.60, "MF": 0.80},
    "Sala riunioni": {"lux": 500, "ugr_max": 19, "ra_min": 80, "uni_min": 0.60, "CU": 0.60, "MF": 0.80},
    "Corridoio":     {"lux": 100, "ugr_max": 28, "ra_min": 40, "uni_min": 0.40, "CU": 0.50, "MF": 0.80},
    "Reception":     {"lux": 300, "ugr_max": 22, "ra_min": 80, "uni_min": 0.60, "CU": 0.60, "MF": 0.80},
    "Archivio":      {"lux": 200, "ugr_max": 25, "ra_min": 80, "uni_min": 0.40, "CU": 0.55, "MF": 0.80},
    "Bagno/WC":      {"lux": 200, "ugr_max": 25, "ra_min": 80, "uni_min": 0.40, "CU": 0.55, "MF": 0.80},
    "Ingresso":      {"lux": 200, "ugr_max": 22, "ra_min": 80, "uni_min": 0.40, "CU": 0.55, "MF": 0.80},
    "Mensa/Ristoro": {"lux": 200, "ugr_max": 22, "ra_min": 80, "uni_min": 0.40, "CU": 0.55, "MF": 0.80},
    "Locale tecnico":{"lux": 200, "ugr_max": 25, "ra_min": 60, "uni_min": 0.40, "CU": 0.50, "MF": 0.80},
}

def _pack_bias(style_pack: Dict[str, Any]) -> Dict[str, float]:
    sp = style_pack or {}
    contrast = (sp.get('contrast_level') or 'medium')
    accent = float(sp.get('accent_ratio') or 0)
    b = {'comfort': 1.0, 'efficiency': 1.0, 'architectural': 1.0}
    if contrast == 'high' or accent >= 0.25:
        b['architectural'] *= 1.18
    if contrast == 'low':
        b['comfort'] *= 1.10
    if float(sp.get('uniformity_target') or 0) >= 0.6:
        b['efficiency'] *= 1.08
    return b



def _pack_params(style_pack: Dict[str, Any]) -> Dict[str, Any]:
    sp = style_pack or {}
    def f(key, default):
        try:
            v = sp.get(key, default)
            if v is None: return default
            return float(v)
        except Exception:
            return float(default)
    def i(key, default):
        try:
            v = sp.get(key, default)
            if v is None: return default
            return int(v)
        except Exception:
            return int(default)

    return {
        "cct_default": i("cct_default", 0) or None,
        "density_bias": max(0.6, min(1.5, f("density_bias", 1.0))),
        "uniformity_target": max(0.2, min(0.9, f("uniformity_target", 0.6))),
        "accent_ratio": max(0.0, min(0.6, f("accent_ratio", 0.2))),
        "contrast_level": (sp.get("contrast_level") or "medium"),
        "presentation_theme": (sp.get("presentation_theme") or "clean_office"),
        "mood": (sp.get("mood") or None),
    }

def _team_bias(designer_stats: Dict[str, Any]) -> Dict[str, float]:
    concept = (designer_stats or {}).get('concept', {}) if isinstance(designer_stats, dict) else {}
    c = float(concept.get('comfort', 0) or 0)
    e = float(concept.get('efficiency', 0) or 0)
    a = float(concept.get('architectural', 0) or 0)
    if (c + e + a) <= 0:
        return {'comfort': 1.0, 'efficiency': 1.0, 'architectural': 1.0}
    mx = max(c, e, a, 1.0)
    return {'comfort': 0.8 + 0.7*(c/mx), 'efficiency': 0.8 + 0.7*(e/mx), 'architectural': 0.8 + 0.7*(a/mx)}


def _area_bias(designer_area_bias: Dict[str, Any]) -> Dict[str, float]:
    """Bias based on concept choices for this specific area type."""
    dab = designer_area_bias or {}
    concept = dab.get("concept_area", {}) if isinstance(dab, dict) else {}
    c = float(concept.get("comfort", 0) or 0)
    e = float(concept.get("efficiency", 0) or 0)
    a = float(concept.get("architectural", 0) or 0)
    if (c + e + a) <= 0:
        return {"comfort": 1.0, "efficiency": 1.0, "architectural": 1.0}
    mx = max(c, e, a, 1.0)
    return {"comfort": 0.85 + 0.6*(c/mx), "efficiency": 0.85 + 0.6*(e/mx), "architectural": 0.85 + 0.6*(a/mx)}

def _style_bias(style_tokens: Dict[str, Any]) -> Dict[str, float]:
    mood = (style_tokens or {}).get('mood_hint')

    votes = (style_tokens or {}).get("concept_votes", {}) if isinstance(style_tokens, dict) else {}
    c = float(votes.get("comfort", 0) or 0)
    e = float(votes.get("efficiency", 0) or 0)
    a = float(votes.get("architectural", 0) or 0)
    if (c + e + a) <= 0:
        return {"comfort": 1.0, "efficiency": 1.0, "architectural": 1.0}
    mx = max(c, e, a, 1.0)
    return {"comfort": 0.8 + 0.7*(c/mx), "efficiency": 0.8 + 0.7*(e/mx), "architectural": 0.8 + 0.7*(a/mx)}

def _ideal_spec(
    area: Dict[str, Any],
    concept_type: str,
    req: Dict[str, Any],
    pack_params: Optional[Dict[str, Any]] = None,
    constraints: str = "",
) -> Dict[str, Any]:
    tipo = (area.get("tipo_locale") or "Ufficio VDT").lower()
    c_low = (constraints or "").lower()

    if "4000" in c_low:
        cct = "4000K"
    elif "3000" in c_low:
        cct = "3000K"
    else:
        cct = "3000K" if concept_type == "comfort" else ("4000K" if concept_type == "efficiency" else "3000K/4000K")

    if "solo incasso" in c_low or "incasso" in c_low:
        mounting = "recessed"
    elif "sospensione" in c_low:
        mounting = "pendant"
    else:
        mounting = "recessed" if ("ufficio" in tipo or "corridoio" in tipo) else "mixed"

    distribution = "wide" if concept_type != "architectural" else "mixed (ambient+accent)"
    # v22: style pack influences optics/distribution
    pack_params = pack_params or {}
    cl = str(pack_params.get('contrast_level') or 'medium')
    ar = float(pack_params.get('accent_ratio') or 0.2)
    if cl == 'high' or ar >= 0.25:
        distribution = "mixed (ambient+accent)"
    elif cl == 'low' and ar <= 0.15:
        distribution = "wide"
    if "accent" in c_low or "vetrina" in c_low:
        distribution = "mixed (ambient+accent)"

    target_ugr = int(req["ugr_max"])
    if "ugr16" in c_low or "ugr 16" in c_low:
        target_ugr = min(target_ugr, 16)

    return {
        "required_lux": float(req["lux"]),
        "target_ugr": target_ugr,
        "min_ra": int(req["ra_min"]),
        "uni_min": float(req.get("uni_min") or 0.4),
        "preferred_cct": cct,
        "distribution": distribution,
        "mounting": mounting,
        "constraints": constraints or "",
    }

def _lumen_method(area_m2: float, target_lux: float, CU: float, MF: float, flux_lm: float) -> Tuple[int, float]:
    area_m2 = max(float(area_m2 or 1.0), 1.0)
    CU = max(float(CU or 0.1), 0.1)
    MF = max(float(MF or 0.1), 0.1)
    flux_lm = max(float(flux_lm or 1.0), 1.0)
    n = int((target_lux * area_m2) / (CU * MF * flux_lm) + 0.999999)
    n = max(n, 1)
    em = (n * flux_lm * CU * MF) / area_m2
    return n, round(em, 1)


def _approx_interreflection_factor(
    area_m2: float,
    h_m: float,
    rho_avg: float,
    workplane_h_m: float = 0.8,
) -> float:
    """Fast interreflection approximation (MVP "A").

    Not Radiance. Pragmatic scalar boost:
        E_total ~= E_direct * 1/(1 - rho_avg * F)
    with F from a crude room-cavity ratio.
    """
    rho = _clamp(float(rho_avg or 0.0), 0.05, 0.90)
    # Approx perimeter assuming near-square
    side = max(1e-6, math.sqrt(max(1e-6, float(area_m2))))
    perimeter = 4.0 * side
    cavity_h = max(0.2, float(h_m) - float(workplane_h_m))
    rcr = 5.0 * cavity_h * perimeter / max(1e-6, float(area_m2))
    # Map RCR (0..10+) to coupling factor (0.35..0.70)
    F = 0.70 - 0.035 * _clamp(rcr, 0.0, 10.0)
    F = _clamp(F, 0.35, 0.70)
    return 1.0 / max(1e-3, (1.0 - rho * F))


def _make_isolux_grid(area_m2: float, nx: int = 14, ny: int = 10) -> Dict[str, Any]:
    """Create a simple rectangular grid payload to be filled with lux values."""
    w = math.sqrt(max(1e-6, float(area_m2)))
    h = w
    return {"nx": int(nx), "ny": int(ny), "width_m": w, "height_m": h, "values": []}

def _filter_candidates(catalog: List[Dict[str, Any]], allowed_brands: List[str], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not allowed_brands:
        return []
    req_ra = int(spec.get("min_ra") or 80)
    req_ugr = int(spec.get("target_ugr") or 19)

    def ok(x):
        if x.get("brand") not in allowed_brands:
            return False
        if int(x.get("cri") or 80) < req_ra:
            return False
        ugr = x.get("ugr")
        if ugr is not None and int(ugr) > req_ugr:
            return False
        return True

    return [x for x in catalog if ok(x)]

def _score_candidate(x: Dict[str, Any], concept_type: str, priority: str) -> float:
    ugr = x.get("ugr")
    cri = float(x.get("cri") or 80)
    flux = float(x.get("flux_lm") or 1)
    watt = float(x.get("watt") or 1)
    eff = flux / max(watt, 1.0)
    dim = 1.0 if x.get("dimmable", True) else 0.0

    if priority == "norms":
        s = (100 - (ugr if ugr is not None else 25)) * 2.2 + cri * 0.8 + eff * 0.2 + dim * 5.0
    elif priority == "efficiency":
        s = eff * 2.2 + dim * 5.0 + cri * 0.2 + (100 - (ugr if ugr is not None else 25)) * 0.3
    else:
        s = eff * 1.0 + cri * 0.4 + (100 - (ugr if ugr is not None else 25)) * 0.8 + dim * 8.0

    if concept_type == "comfort":
        s += (100 - (ugr if ugr is not None else 25)) * 0.8 + cri * 0.2
    elif concept_type == "architectural":
        s += dim * 10.0 + flux * 0.01
    return s

def _dims_from_area(area_m2: float, tipo_locale: str) -> Tuple[float, float]:
    a = max(float(area_m2 or 1.0), 1.0)
    t = (tipo_locale or "").lower()
    if "corridoio" in t:
        w = max(1.5, math.sqrt(a / 4.0))
        l = a / w
        return (l, w)
    s = math.sqrt(a)
    return (s, s)

def _layout_coords(n: int, L: float, W: float, layout: str = "grid", uniformity_target: float = 0.6) -> List[List[float]]:
    n = max(int(n), 1)
    L = max(float(L), 1.0)
    W = max(float(W), 1.0)
    coords: List[List[float]] = []

    if layout == "linear":
        u = max(0.2, min(0.9, float(uniformity_target or 0.6)))
        margin = max(0.5, min(1.6, W * 0.22 * (1.0 + (u-0.5)*0.9)))
        y = W / 2.0
        if n == 1:
            coords = [[L/2.0, y]]
        else:
            x0 = margin
            x1 = L - margin
            step = (x1 - x0) / (n - 1)
            coords = [[x0 + i * step, y] for i in range(n)]
        return [[round(x, 2), round(y, 2)] for x, y in coords]

    if layout == "perimeter+center" and n >= 5:
        u = max(0.2, min(0.9, float(uniformity_target or 0.6)))
        mx = max(0.6, min(1.8, L*0.10 * (1.0 + (u-0.5)*1.0)))
        my = max(0.6, min(1.8, W*0.10 * (1.0 + (u-0.5)*1.0)))
        base = [
            [mx, my],
            [L-mx, my],
            [L-mx, W-my],
            [mx, W-my],
            [L/2.0, W/2.0],
        ]
        coords.extend(base)
        remaining = n - len(coords)
        if remaining > 0:
            g = int(math.ceil(math.sqrt(remaining)))
            gx = max(1, g); gy = max(1, g)
            xs = [mx + (L-2*mx) * (i+0.5)/gx for i in range(gx)]
            ys = [my + (W-2*my) * (j+0.5)/gy for j in range(gy)]
            for x in xs:
                for y in ys:
                    if len(coords) < n:
                        coords.append([x, y])
        return [[round(x, 2), round(y, 2)] for x, y in coords[:n]]

    gx = int(math.ceil(math.sqrt(n * (L / W)))) if W > 0 else int(math.ceil(math.sqrt(n)))
    gx = max(1, gx)
    gy = int(math.ceil(n / gx))
    gy = max(1, gy)

    u = max(0.2, min(0.9, float(uniformity_target or 0.6)))
    mx = max(0.7, min(2.0, min(L, W) * 0.10 * (1.0 + (u-0.5)*1.2)))
    my = mx
    xs = [mx + (L - 2*mx) * (i+0.5)/gx for i in range(gx)]
    ys = [my + (W - 2*my) * (j+0.5)/gy for j in range(gy)]
    for y in ys:
        for x in xs:
            if len(coords) < n:
                coords.append([x, y])

    return [[round(x, 2), round(y, 2)] for x, y in coords]

def _uniformity_proxy(coords: List[List[float]], L: float, W: float, h_m: float, flux_lm: float, CU: float, MF: float) -> Dict[str, float]:
    if not coords:
        return {"emin": 0.0, "emax": 0.0, "emean": 0.0, "u0": 0.0}

    L = max(L, 1.0); W = max(W, 1.0)
    h = max(float(h_m or 2.7), 2.2)

    nx, ny = (12, 8) if (L / max(W, 1e-6)) > 2.0 else (10, 10)
    xs = [L * (i + 0.5) / nx for i in range(nx)]
    ys = [W * (j + 0.5) / ny for j in range(ny)]

    vals = []
    K = (flux_lm / (2.0 * math.pi)) * CU * MF * 0.65

    for y in ys:
        for x in xs:
            e = 0.0
            for lx, ly in coords:
                dx = x - lx
                dy = y - ly
                d = math.sqrt(dx*dx + dy*dy + h*h)
                ct = h / d
                e += K * (ct / (d*d))
            vals.append(e)

    emin = float(min(vals))
    emax = float(max(vals))
    emean = float(sum(vals) / len(vals)) if vals else 0.0
    u0 = float(emin / emean) if emean > 0 else 0.0
    return {"emin": round(emin, 2), "emax": round(emax, 2), "emean": round(emean, 2), "u0": round(u0, 2)}


def _compute_isolux_values(
    coords: List[List[float]],
    L: float,
    W: float,
    h_m: float,
    flux_lm: float,
    CU: float,
    MF: float,
    grid: Dict[str, Any],
    workplane_h_m: float = 0.8,
) -> Dict[str, Any]:
    """Crude point-source isolux grid (fast, MVP).

    Assumptions:
    - uniform luminous intensity over downward hemisphere (I ≈ flux / (2π))
    - workplane at 0.8 m
    - luminaires at ceiling
    """
    nx = int(grid.get("nx", 14))
    ny = int(grid.get("ny", 10))

    Lm = float(L) if L else float(grid.get("width_m", 1.0))
    Wm = float(W) if W else float(grid.get("height_m", 1.0))

    z = max(0.2, float(h_m) - float(workplane_h_m))
    I = float(flux_lm) / (2.0 * math.pi)  # cd

    vals: List[float] = []
    for iy in range(ny):
        y = (iy + 0.5) * Wm / ny
        for ix in range(nx):
            x = (ix + 0.5) * Lm / nx
            e = 0.0
            for (lx, ly) in coords:
                dx = x - lx
                dy = y - ly
                d2 = dx * dx + dy * dy + z * z
                d = math.sqrt(d2)
                cos_t = z / max(1e-6, d)
                e += I * cos_t / max(1e-6, d2)
            e *= float(CU) * float(MF)
            vals.append(round(e, 1))

    grid["values"] = vals
    grid["nx"] = nx
    grid["ny"] = ny
    grid["width_m"] = Lm
    grid["height_m"] = Wm
    return grid

def _choose_layout(tipo_locale: str, concept_type: str, constraints: str) -> str:
    t = (tipo_locale or "").lower()
    c = (constraints or "").lower()
    if "corridoio" in t:
        return "linear"
    if "perimetr" in c:
        return "perimeter+center"
    if concept_type == "architectural" and ("reception" in t):
        return "perimeter+center"
    return "grid"

def _scene_presets(tipo_locale: str) -> List[Dict[str, Any]]:
    t = (tipo_locale or "").lower()
    if "ufficio" in t:
        return [
            {"id":"work", "name":"Work", "channels":{"ambient":1.0, "accent":0.0}},
            {"id":"focus", "name":"Focus", "channels":{"ambient":1.0, "accent":0.3}},
            {"id":"soft", "name":"Soft", "channels":{"ambient":0.45, "accent":0.10}},
        ]
    if "riunioni" in t or "meeting" in t:
        return [
            {"id":"meeting", "name":"Meeting", "channels":{"ambient":0.9, "accent":0.2}},
            {"id":"presentation", "name":"Presentation", "channels":{"ambient":0.35, "accent":0.75}},
            {"id":"cleaning", "name":"Cleaning", "channels":{"ambient":1.0, "accent":0.3}},
        ]
    if "corridoio" in t:
        return [
            {"id":"normal", "name":"Normal", "channels":{"ambient":0.7}},
            {"id":"night", "name":"Night", "channels":{"ambient":0.25}},
            {"id":"emergency", "name":"Emergency", "channels":{"ambient":1.0}},
        ]
    return [
        {"id":"standard", "name":"Standard", "channels":{"ambient":0.8, "accent":0.2}},
        {"id":"accent", "name":"Accent", "channels":{"ambient":0.35, "accent":0.85}},
        {"id":"soft", "name":"Soft", "channels":{"ambient":0.4, "accent":0.1}},
    ]

def _optimize(
    area_m2: float,
    tipo_locale: str,
    h_m: float,
    req: Dict[str, Any],
    spec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    concept_type: str,
    priority: str,
    pack_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_lux = float(spec["required_lux"])
    CU, MF = float(req["CU"]), float(req["MF"])
    uni_min = float(spec.get("uni_min") or req.get("uni_min") or 0.4)

    L, W = _dims_from_area(area_m2, tipo_locale)
    layout = _choose_layout(tipo_locale, concept_type, spec.get("constraints") or "")

    ranked = sorted(candidates, key=lambda x: _score_candidate(x, concept_type, priority), reverse=True)

    pack_params = pack_params or {}

    def pack_solution(lum, flux, watt, ugr, cri, status, iterations):
        eff_target = float(target_lux) * float(pack_params.get('density_bias') or 1.0)
        n, Em = _lumen_method(area_m2, eff_target, CU, MF, flux)
        coords = _layout_coords(n, L, W, layout, uniformity_target=float(pack_params.get('uniformity_target') or 0.6))
        uni = _uniformity_proxy(coords, L, W, h_m, flux, CU, MF)

        # --- Interreflection (fast approximation)
        rho_avg = float(pack_params.get('rho_avg') or 0.50)
        ir_factor = _approx_interreflection_factor(area_m2=area_m2, h_m=h_m, rho_avg=rho_avg)
        Em_total = round(float(Em) * float(ir_factor), 1)

        # --- Isolux grid (fast point-source approximation)
        iso = _compute_isolux_values(
            coords=coords,
            L=L,
            W=W,
            h_m=h_m,
            flux_lm=flux,
            CU=CU,
            MF=MF,
            grid=_make_isolux_grid(area_m2),
        )

        Wt = round(n * watt, 1)
        wm2 = round(Wt / max(area_m2, 1.0), 2)

        ok_lux = Em_total >= target_lux * 0.95
        ok_ra  = int(cri) >= int(spec.get("min_ra") or req["ra_min"])
        ok_ugr = True if ugr is None else (int(ugr) <= int(spec.get("target_ugr") or req["ugr_max"]))
        ok_uni = float(uni.get("u0") or 0.0) >= uni_min

        return {
            "luminaire": lum,
            "calc": {
                "n": n, "Em": Em, "Em_total": Em_total, "Et": target_lux,
                "rho_avg": round(rho_avg, 2), "interreflection_factor": round(ir_factor, 3),
                "W": Wt, "wm2": wm2,
                "ok_lux": ok_lux, "ok_ra": ok_ra, "ok_ugr": ok_ugr, "ok_uni": ok_uni,
                "target_ugr": int(spec.get("target_ugr") or req["ugr_max"]),
                "min_ra": int(spec.get("min_ra") or req["ra_min"]),
                "uni_min": round(uni_min, 2),
                "u0": uni.get("u0"),
                "emin_proxy": uni.get("emin"),
                "emean_proxy": uni.get("emean"),
                "layout": layout,
                "dims": {"L": round(L, 2), "W": round(W, 2), "h": round(float(h_m or 2.7), 2)},
                "coords": coords,
                "isolux": iso,
                "scenes": _scene_presets(tipo_locale),
                "autopilot": {"priority": priority, "iterations": iterations, "status": status},
            }
        }

    if not ranked:
        return pack_solution(None, 3000.0, 25.0, None, int(spec.get("min_ra") or req["ra_min"]), "brand-neutral", 1)

    best = None
    for i, lum in enumerate(ranked[:12], start=1):
        flux = float(lum.get("flux_lm") or 3000)
        watt = float(lum.get("watt") or 25)
        ugr = lum.get("ugr")
        cri = int(lum.get("cri") or 80)

        sol = pack_solution(lum, flux, watt, ugr, cri, "tried", i)
        calc = sol["calc"]
        feasible = bool(calc["ok_lux"] and calc["ok_ra"] and calc["ok_ugr"] and calc["ok_uni"])

        if priority == "efficiency":
            obj = -float(calc["wm2"])
        elif priority == "norms":
            u0 = float(calc.get("u0") or 0.0)
            obj = (1000.0 if feasible else 0.0) + u0 * 100.0 + (100 - (ugr if ugr is not None else 25)) * 1.5 + cri * 0.2
        else:
            u0 = float(calc.get("u0") or 0.0)
            obj = _score_candidate(lum, concept_type, priority) + u0 * 50.0 - float(calc["wm2"])

        if best is None:
            best = (obj, feasible, sol)
        else:
            if feasible and not best[1]:
                best = (obj, feasible, sol)
            elif feasible == best[1] and obj > best[0]:
                best = (obj, feasible, sol)

        if priority == "norms" and feasible and i <= 3:
            break

    if best:
        best_sol = best[2]
        best_sol["calc"]["autopilot"]["status"] = "feasible" if best[1] else "best-effort"
        return best_sol

    return pack_solution(ranked[0], 3000.0, 25.0, None, int(spec.get("min_ra") or req["ra_min"]), "best-effort", 1)

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

        lines += ["0","LWPOLYLINE","8","AREE","90","4","70","1",
                  "10",str(x_off),"20","0.0",
                  "10",str(x_off+L),"20","0.0",
                  "10",str(x_off+L),"20",str(W),
                  "10",str(x_off),"20",str(W)]
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
@app.get("/health")
def health():
    return {"ok": True, "features": ["autopilot-optimizer", "layout-generator", "uniformity-proxy", "coords", "scenes", "export-dxf", "export-json"]}


@app.get("/")
def root():
    # Some platforms probe GET/HEAD /; returning JSON avoids a confusing 404.
    return {"ok": True, "name": "LuxIA Engine", "hint": "Open /docs for Swagger UI"}

@app.post("/projects/{project_id}/concepts")
def generate_concepts(
    project_id: str,
    payload: Dict[str, Any],
    mode: str = Query(default="fast", description="fast | radiance"),
    x_luxia_token: Optional[str] = Header(None),
):
    auth(x_luxia_token)

    mode = (mode or "fast").lower().strip()
    if mode == "radiance":
        # MVP: we expose the switch in UI, but Radiance execution requires extra setup.
        # When you're ready we can add: Docker image with radiance + materials + IFC/import.
        raise HTTPException(
            status_code=501,
            detail="Radiance mode not configured yet on this engine. Use fast mode or enable Radiance deployment.",
        )

    areas: List[Dict[str, Any]] = payload.get("areas", []) or []
    style_tokens: Dict[str, Any] = payload.get("style_tokens", {}) or {}
    designer_stats: Dict[str, Any] = payload.get("designer_stats", {}) or {}
    project_style_pack: Dict[str, Any] = payload.get("project_style_pack", {}) or {}
    pack_params = _pack_params(project_style_pack)

    designer_area_bias: Dict[str, Any] = payload.get("designer_area_bias", {}) or {}
    allowed_brands: List[str] = payload.get("allowed_brands", []) or []
    catalog: List[Dict[str, Any]] = payload.get("catalog", []) or []
    priority: str = (payload.get("priority") or "mix").lower()
    constraints: str = payload.get("constraints") or ""

    token_bias = _style_bias(style_tokens)
    team_bias = _team_bias(designer_stats)
    area_bias = _area_bias(designer_area_bias)
    # blend: tokens 0.35 + team DNA 0.40 + area-type DNA 0.25
    bias = {k: (0.35*token_bias.get(k,1.0) + 0.40*team_bias.get(k,1.0) + 0.25*area_bias.get(k,1.0)) for k in ['comfort','efficiency','architectural']}
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
            # Let the Style Pack influence the spec (CCT, contrast, accent ratio, etc.).
            spec = _ideal_spec(area, ctype, req, pack_params, constraints)
            candidates = _filter_candidates(catalog, allowed_brands, spec)
            sol = _optimize(area_m2, tipo, h_m, req, spec, candidates, ctype, priority, pack_params)

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
        "style_used": {"bias": bias, "token_bias": token_bias, "team_bias": team_bias, "area_bias": area_bias, "order": order},
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
