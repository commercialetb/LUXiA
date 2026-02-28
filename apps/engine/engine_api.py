# apps/engine/engine_api.py
# LuxIA Engine vNext Patch
# - Qwen as LLM orchestrator (existing)
# - Adds planimetry ingestion pipeline for: PDF, JPG/PNG, DXF (+DWG via conversion)
# - Adds "prova del nove" verification: deterministic geometry checks + Vision cross-check
#
# ENV (LLM orchestrator)
#   LLM_PROVIDER=qwen|groq|ollama|dummy
#   QWEN_BASE_URL=... (OpenAI-compatible)
#   QWEN_API_KEY=...
#   QWEN_MODEL=...
#
# ENV (Vision verifier)
#   VISION_BASE_URL=... (OpenAI-compatible)
#   VISION_API_KEY=...
#   VISION_MODEL=... (must support image inputs)
#
# ENV (DWG conversion - optional)
#   DWG2DXF_URL=...  (an internal microservice endpoint that accepts DWG bytes and returns DXF bytes)
#   DWG2DXF_API_KEY=...
#
# Recommended deps (engine requirements.txt):
#   pillow
#   pypdfium2
#   ezdxf
#   matplotlib
#   shapely
#
# Notes:
# - DWG is proprietary; parsing reliably requires a converter (ODA/Autodesk/Cloud).
# - This patch keeps a robust failure mode: DWG is accepted only if converter is configured.

from __future__ import annotations

import os, json, time, uuid, base64
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel


# ==========================================================
# LuxIA PRO - DXF units inference (mm/cm/m) + auto-fix
# ==========================================================
def _infer_dxf_units_from_header(doc) -> Tuple[float, float, str]:
    """Return meters_per_unit, confidence, hint."""
    mpu = None
    conf = 0.0
    hint = ""
    try:
        ins = int(doc.header.get("$INSUNITS", 0) or 0)
    except Exception:
        ins = 0
    # AutoCAD INSUNITS common codes: 1=in, 2=ft, 4=mm, 5=cm, 6=m
    if ins == 4:
        mpu, conf, hint = 0.001, 0.98, "INSUNITS=mm"
    elif ins == 5:
        mpu, conf, hint = 0.01, 0.98, "INSUNITS=cm"
    elif ins == 6:
        mpu, conf, hint = 1.0, 0.98, "INSUNITS=m"
    elif ins == 2:
        mpu, conf, hint = 0.3048, 0.92, "INSUNITS=ft"
    elif ins == 1:
        mpu, conf, hint = 0.0254, 0.92, "INSUNITS=in"
    return (mpu if mpu is not None else 1.0), conf, hint

def _infer_dxf_units_from_extents(segs: List[Tuple[float,float,float,float]]) -> Tuple[float, float, str]:
    if not segs:
        return 1.0, 0.0, "no_segs"
    xs=[]; ys=[]
    for x1,y1,x2,y2 in segs:
        xs.extend([x1,x2]); ys.extend([y1,y2])
    dx=max(xs)-min(xs); dy=max(ys)-min(ys)
    span=max(dx,dy)
    if span > 20000:
        return 0.001, 0.90, f"extents~{span:.0f}->mm"
    if span > 2000:
        return 0.01, 0.75, f"extents~{span:.0f}->cm"
    return 1.0, 0.60, f"extents~{span:.0f}->m"

def _infer_dxf_units(doc, segs: List[Tuple[float,float,float,float]]) -> Dict[str, Any]:
    mpu_h, conf_h, hint_h = _infer_dxf_units_from_header(doc)
    if conf_h >= 0.95:
        return {"meters_per_unit": float(mpu_h), "confidence": float(conf_h), "hint": hint_h}
    mpu_e, conf_e, hint_e = _infer_dxf_units_from_extents(segs)
    if hint_h and "INSUNITS" in hint_h and conf_h > conf_e:
        return {"meters_per_unit": float(mpu_h), "confidence": float(conf_h), "hint": hint_h}
    return {"meters_per_unit": float(mpu_e), "confidence": float(conf_e), "hint": hint_e}

def _apply_units_to_segs(segs: List[Tuple[float,float,float,float]], mpu: float) -> List[Tuple[float,float,float,float]]:
    if abs(mpu-1.0) < 1e-12:
        return segs
    return [(x1*mpu,y1*mpu,x2*mpu,y2*mpu) for (x1,y1,x2,y2) in segs]


# ==========================================================
# LuxIA PRO - Scale sanity gate
# ==========================================================
def _sanity_check_scale(extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    areas = [float(r.get("area_m2") or 0.0) for r in (extracted.get("rooms") or []) if (r.get("area_m2") or 0) > 0]
    if areas:
        med = statistics.median(areas)
        if med < 1.0:
            issues.append({"code":"SCALE_TOO_SMALL","detail":f"Mediana area stanza troppo piccola ({med:.2f} m²). Probabile unità mm/cm non convertite."})
        if med > 500.0:
            issues.append({"code":"SCALE_TOO_LARGE","detail":f"Mediana area stanza troppo grande ({med:.1f} m²). Probabile unità errate (m vs mm)."})
    return issues

# ==========================================================
# LuxIA PRO - Luminaire library (IES) + Grid Lux + UGR (approx)
# ==========================================================
def _load_default_luminaire_from_env():
    """Optionally preload a default luminaire photometry from ENV to keep the UI minimal.
    Set LUMINAIRE_IES_BASE64 to the base64 of an IES/LDT file.
    """
    b64 = os.getenv("LUMINAIRE_IES_BASE64")
    if not b64:
        # also allow shipping a default IES inside engine/assets/default.ies
        try:
            here = os.path.dirname(__file__)
            p = os.path.join(here, "assets", "default.ies")
            if os.path.exists(p):
                raw = open(p, "rb").read()
                phot = _parse_ies_lm63(raw)
                _LUMINAIRE_LIB["default"] = phot
        except Exception:
            pass
        return

    try:
        raw = base64.b64decode(b64)
        try:
            phot = _parse_ies_lm63(raw)
        except Exception:
            phot = _parse_ldt_eulumdat(raw)
        _LUMINAIRE_LIB["default"] = phot
    except Exception:
        return


_LUMINAIRE_LIB = {}  # key -> parsed photometry
_UGR_TABLES: Dict[str, Any] = {}

def _ugr_from_table(lum_key: str, K: float, rho: float) -> Optional[float]:
    tab = _UGR_TABLES.get(lum_key)
    if not tab:
        return None
    Ks = tab.get("K") or []
    rhos = tab.get("rho") or []
    grid = tab.get("ugr") or []
    if not Ks or not rhos or not grid:
        return None
    Kc = max(min(float(K), max(Ks)), min(Ks))
    rc = max(min(float(rho), max(rhos)), min(rhos))

    def seg(val, arr):
        if val <= arr[0]:
            return 0, 0, 0.0
        if val >= arr[-1]:
            return len(arr)-1, len(arr)-1, 0.0
        lo, hi = 0, len(arr)-1
        while hi-lo > 1:
            mid = (lo+hi)//2
            if arr[mid] <= val:
                lo = mid
            else:
                hi = mid
        t = 0.0 if arr[hi]==arr[lo] else (val-arr[lo])/(arr[hi]-arr[lo])
        return lo, hi, t

    i0,i1,tK = seg(Kc, Ks)
    j0,j1,tr = seg(rc, rhos)

    def g(i,j):
        try:
            return float(grid[i][j])
        except Exception:
            return None

    q00=g(i0,j0); q01=g(i0,j1); q10=g(i1,j0); q11=g(i1,j1)
    if None in (q00,q01,q10,q11):
        return None
    a = q00 + tr*(q01-q00)
    b = q10 + tr*(q11-q10)
    return float(a + tK*(b-a))

@app.post("/luminaires/ugr/upload")
async def ugr_upload(req: Request):
    payload = await req.json()
    key = str(payload.get("key") or "default")
    if not payload.get("K") or not payload.get("rho") or not payload.get("ugr"):
        raise HTTPException(status_code=400, detail="Missing K/rho/ugr")
    _UGR_TABLES[key] = {"K": payload["K"], "rho": payload["rho"], "ugr": payload["ugr"]}
    return {"ok": True, "key": key}


_load_default_luminaire_from_env()

def _parse_ldt_eulumdat(raw_bytes: bytes) -> Dict[str, Any]:
    """Parse a subset of EULUMDAT (.ldt) in a robust, heuristic way."""
    lines = raw_bytes.decode("latin1", errors="ignore").splitlines()
    if len(lines) < 40:
        raise ValueError("Not an LDT file")
    nums = []
    for l in lines:
        for t in l.replace(",", ".").split():
            try:
                nums.append(float(t))
            except Exception:
                continue
    if len(nums) < 200:
        raise ValueError("LDT numeric content insufficient")

    best = None
    for idx in range(0, min(400, len(nums)-10)):
        nC = int(nums[idx]); nG = int(nums[idx+1])
        if 1 <= nC <= 72 and 1 <= nG <= 181:
            start = idx+2
            end = start + nC + nG + nC*nG
            if end <= len(nums):
                C = nums[start:start+nC]
                G = nums[start+nC:start+nC+nG]
                if C and G and min(C) >= -1e-6 and max(C) <= 360.0+1e-6 and min(G) >= -1e-6 and max(G) <= 180.0+1e-6:
                    if all(C[i] <= C[i+1]+1e-6 for i in range(len(C)-1)) and all(G[i] <= G[i+1]+1e-6 for i in range(len(G)-1)):
                        best = (idx, nC, nG, C, G)
                        break
    if best is None:
        raise ValueError("Unable to locate C/G blocks in LDT")

    idx, nC, nG, C, G = best
    start = idx + 2 + nC + nG
    vals = nums[start:start+nC*nG]
    candela = []
    k = 0
    for _ in range(nC):
        candela.append([float(vals[k+j]) for j in range(nG)])
        k += nG

    lumens = float(os.getenv("LIGHT_DEFAULT_LUMENS","800"))
    watts = float(os.getenv("LIGHT_DEFAULT_WATT","8"))

    return {
        "type": "ldt_eulumdat",
        "v_angles": list(map(float, G)),
        "h_angles": list(map(float, C)),
        "candela": candela,
        "lumens_per_lamp": lumens,
        "input_watts": watts,
        "meta": {"parser": "heuristic"},
    }

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
            meta["tilt"] = line.split("=",1)[1].strip().upper()
            i += 1
            break
        i += 1
    if meta.get("tilt","NONE") not in ("NONE", "INCLUDE"):
        raise ValueError(f"Unsupported TILT={meta.get('tilt')}")
    nums = []
    def push_nums(line):
        for t in line.replace(",", " ").split():
            try: nums.append(float(t))
            except Exception: pass
    while i < len(txt) and len(nums) < 13:
        push_nums(txt[i]); i += 1
    if len(nums) < 13:
        raise ValueError("IES numeric header incomplete")
    n_lamps = int(nums[0]); lumens_per_lamp = float(nums[1]); candela_mult = float(nums[2])
    n_v = int(nums[3]); n_h = int(nums[4]); phot_type = int(nums[5]); units_type = int(nums[6])
    width = float(nums[7]); length = float(nums[8]); height = float(nums[9]); input_watts = float(nums[12])

    v_angles = []
    while i < len(txt) and len(v_angles) < n_v:
        for t in txt[i].replace(",", " ").split():
            if len(v_angles) < n_v: v_angles.append(float(t))
        i += 1

    h_angles = []
    while i < len(txt) and len(h_angles) < n_h:
        for t in txt[i].replace(",", " ").split():
            if len(h_angles) < n_h: h_angles.append(float(t))
        i += 1

    candela = []
    for _ in range(n_h):
        row = []
        while i < len(txt) and len(row) < n_v:
            for t in txt[i].replace(",", " ").split():
                if len(row) < n_v: row.append(float(t) * candela_mult)
            i += 1
        if len(row) != n_v:
            raise ValueError("Candela matrix incomplete")
        candela.append(row)

    return {"type":"ies_lm63","n_lamps":n_lamps,"lumens_per_lamp":lumens_per_lamp,"input_watts":input_watts,
            "units_type":units_type,"dimensions":{"W":width,"L":length,"H":height},
            "photometric_type":phot_type,"v_angles":v_angles,"h_angles":h_angles,"candela":candela,"meta":meta}

def _interp1(x: float, xs: List[float], ys: List[float]) -> float:
    if not xs: return 0.0
    if x <= xs[0]: return float(ys[0])
    if x >= xs[-1]: return float(ys[-1])
    lo, hi = 0, len(xs)-1
    while hi-lo>1:
        mid=(lo+hi)//2
        if xs[mid] <= x: lo=mid
        else: hi=mid
    x0,x1=xs[lo],xs[hi]; y0,y1=ys[lo],ys[hi]
    if x1==x0: return float(y0)
    t=(x-x0)/(x1-x0)
    return float(y0 + t*(y1-y0))

def _candela_at(phot: Dict[str, Any], gamma_deg: float, c_deg: float) -> float:
    v=phot["v_angles"]; h=phot["h_angles"]; mat=phot["candela"]
    gamma=max(min(gamma_deg, v[-1]), v[0])
    c=max(min(c_deg,h[-1]),h[0])
    if c <= h[0]:
        row0=row1=0; tc=0.0
    elif c >= h[-1]:
        row0=row1=len(h)-1; tc=0.0
    else:
        lo,hi=0,len(h)-1
        while hi-lo>1:
            mid=(lo+hi)//2
            if h[mid] <= c: lo=mid
            else: hi=mid
        row0,row1=lo,hi
        tc=0.0 if h[row1]==h[row0] else (c-h[row0])/(h[row1]-h[row0])
    i0=_interp1(gamma,v,mat[row0]); i1=_interp1(gamma,v,mat[row1])
    return float(i0 + tc*(i1-i0))

def _point_lux_from_luminaires(points_xy: np.ndarray, lum_xyzh: np.ndarray,
                               phot: Optional[Dict[str, Any]], lumens_fallback: float,
                               workplane_z: float = 0.8) -> np.ndarray:
    """Compute illuminance at points; uses photometry when available."""
    E = np.zeros((points_xy.shape[0],), dtype=float)
    for lx, ly, lz, lf in lum_xyzh:
        dx = points_xy[:, 0] - lx
        dy = points_xy[:, 1] - ly
        dz = workplane_z - lz
        r2 = dx*dx + dy*dy + dz*dz
        r = np.sqrt(np.maximum(r2, 1e-9))
        cos_theta = np.abs(dz) / r
        gamma = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
        if phot:
            c_val = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
            I = np.array([_candela_at(phot, g, c_val[i]) for i, g in enumerate(gamma)], dtype=float)
            E += I * cos_theta / np.maximum(r2, 1e-9)
        else:
            Iiso = (lf if lf > 0 else lumens_fallback) / (4.0 * math.pi)
            E += Iiso * cos_theta / np.maximum(r2, 1e-9)
    return E


def _ugr_approx(room: Dict[str, Any], phot: Optional[Dict[str, Any]]) -> Optional[float]:
    if phot is None: return None
    I65=_candela_at(phot,65.0,0.0)
    lum=float(phot.get("lumens_per_lamp") or 1000.0)
    g=(I65/max(lum,1.0))
    ugr=16.0 + 30.0*min(1.0,max(0.0,g*8.0))
    return float(max(12.0,min(29.0,ugr)))

def _grid_points_in_polygon(room_poly_m: List[List[float]], step_m: float = 0.5) -> np.ndarray:
    poly=np.array(room_poly_m,dtype=float)
    if poly.shape[0] < 3: return np.zeros((0,2),dtype=float)
    minx,miny=poly[:,0].min(),poly[:,1].min()
    maxx,maxy=poly[:,0].max(),poly[:,1].max()
    xs=np.arange(minx,maxx+1e-6,step_m)
    ys=np.arange(miny,maxy+1e-6,step_m)
    pts=np.array([(x,y) for x in xs for y in ys],dtype=float)
    x=pts[:,0]; y=pts[:,1]
    x0=poly[:,0]; y0=poly[:,1]
    x1=np.roll(x0,-1); y1=np.roll(y0,-1)
    inside=np.zeros(len(pts),dtype=bool)
    for i in range(len(poly)):
        cond=((y0[i]>y)!=(y1[i]>y)) & (x < (x1[i]-x0[i])*(y-y0[i])/(y1[i]-y0[i]+1e-12) + x0[i])
        inside ^= cond
    return pts[inside]

def _status_vs_targets(metrics: Dict[str, Any], targets: Dict[str, Any], ugr: Optional[float] = None) -> Dict[str, Any]:
    """Return status + reasons based on UNI targets."""
    reasons=[]
    try:
        tgt_lux=float(targets.get("lux") or 0.0)
        tgt_u0=float(targets.get("u0") or 0.0)
        tgt_ugr=targets.get("ugr_max", None)
        eavg=float(metrics.get("Eavg") or metrics.get("E_avg") or 0.0)
        u0=float(metrics.get("U0") or 0.0)
        if tgt_lux>0 and eavg+1e-9 < tgt_lux:
            reasons.append("Eavg")
        if tgt_u0>0 and u0+1e-9 < tgt_u0:
            reasons.append("U0")
        if tgt_ugr is not None and ugr is not None:
            try:
                if float(ugr) > float(tgt_ugr) + 1e-9:
                    reasons.append("UGR")
            except Exception:
                pass
    except Exception:
        pass
    return {"status": ("OK" if not reasons else "KO"), "reasons": reasons}

def _calc_room_grid_metrics(room: Dict[str, Any], lights: List[Dict[str, Any]], phot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    poly_m=room.get("polygon_m")
    if not poly_m or len(poly_m) < 3:
        return {"Eavg":0.0,"Emin":0.0,"U0":0.0,"points":0}
    step=float(os.getenv("GRID_STEP_M","0.5"))
    pts=_grid_points_in_polygon(poly_m,step_m=step)
    if pts.shape[0] == 0:
        return {"Eavg":0.0,"Emin":0.0,"U0":0.0,"points":0}
    lumens_default=float(os.getenv("LIGHT_DEFAULT_LUMENS","800"))
    z_mount=float(os.getenv("LUMINAIRE_MOUNT_Z_M", str(ROOM_HEIGHT_M_DEFAULT)))
    rid=room.get("id")
    lum_xyzh=[]
    for l in lights:
        if (l.get("ROOM_ID") or l.get("room_id")) != rid:
            continue
        x=float(l.get("x_m") or 0.0); y=float(l.get("y_m") or 0.0)
        lum=float(l.get("LUMENS") or l.get("lumens") or lumens_default)
        lum_xyzh.append([x,y,z_mount,lum])
    if not lum_xyzh:
        return {"Eavg":0.0,"Emin":0.0,"U0":0.0,"points":int(pts.shape[0]),"grid_step_m":step}
    lum_xyzh=np.array(lum_xyzh,dtype=float)
    E=_point_lux_from_luminaires(pts, lum_xyzh, phot, lumens_default, workplane_z=WORKPLANE_HEIGHT_M_DEFAULT)
    rho_c=float(room.get("rho_c") or CEILING_REFLECTANCE_DEFAULT)
    rho_w=float(room.get("rho_w") or WALL_REFLECTANCE_DEFAULT)
    rho_f=float(room.get("rho_f") or FLOOR_REFLECTANCE_DEFAULT)
    mult=_radiosity_multiplier_pro(room)
    E *= mult
    Eavg=float(E.mean()); Emin=float(E.min()); U0=float(Emin/Eavg) if Eavg>1e-6 else 0.0
    return {"Eavg":Eavg,"E_avg":Eavg,"Emin":Emin,"E_min":Emin,"U0":U0,"points":int(pts.shape[0]),"grid_step_m":step,"radiosity_mult":mult}

# ==========================================================
# LuxIA PRO - Luminaire selection policy (autonomous, simple)
# ==========================================================
def _select_luminaire_key(room: Dict[str, Any]) -> str:
    """Select a luminaire key from the library based on room type/targets.
    Keeps it simple: if library has 'office', 'outdoor' etc. prefer them; else 'default'.
    """
    key = _norm_key(room.get("label"))
    is_out = _is_outdoor_room(room)
    # user may provide overrides via env mapping
    # e.g. LUMINAIRE_MAP_OFFICE=office_ugr19
    if not is_out and key in ("UFFICIO","OPENSPACE","SALA_RIUNIONI"):
        envk = os.getenv("LUMINAIRE_MAP_OFFICE")
        if envk and envk in _LUMINAIRE_LIB:
            return envk
        for k in ("office","ugr19","work","default"):
            if k in _LUMINAIRE_LIB:
                return k
    if is_out:
        envk = os.getenv("LUMINAIRE_MAP_OUTDOOR")
        if envk and envk in _LUMINAIRE_LIB:
            return envk
        for k in ("outdoor","external","default"):
            if k in _LUMINAIRE_LIB:
                return k
    # fallback
    return "default" if "default" in _LUMINAIRE_LIB else (next(iter(_LUMINAIRE_LIB.keys())) if _LUMINAIRE_LIB else "default")

# ==========================================================
# LuxIA PRO - PDF Report Generator (server-side)
# ==========================================================
# ==========================================================
# LuxIA PRO - BOM + Spec (capitolato) generation
# ==========================================================
def _build_bom(extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = {}
    for l in (extracted.get("lights") or []):
        typ = (l.get("FIXTURE_TYPE") or l.get("fixture_type") or "DOWNLIGHT").upper()
        cct = str(l.get("CCT_K") or l.get("cct_k") or os.getenv("LIGHT_DEFAULT_CCT","3000"))
        cri = str(l.get("CRI") or l.get("cri") or os.getenv("LIGHT_DEFAULT_CRI","80"))
        watt = float(l.get("WATT") or l.get("watt") or os.getenv("LIGHT_DEFAULT_WATT","8"))
        lum = float(l.get("LUMENS") or l.get("lumens") or os.getenv("LIGHT_DEFAULT_LUMENS","800"))
        key = f"{typ}|{cct}|{cri}|{watt:.1f}|{lum:.0f}"
        items.setdefault(key, {"type": typ, "cct_k": int(float(cct)), "cri": int(float(cri)), "watt": watt, "lumens": lum, "qty": 0})
        items[key]["qty"] += 1
    return list(items.values())

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

def _bom_csv_bytes(extracted: Dict[str, Any]) -> bytes:
    bom = _build_bom(extracted)
    out = ["type,qty,watt,lumens,cct_k,cri"]
    for it in bom:
        out.append(f"{it['type']},{it['qty']},{it['watt']:.1f},{it['lumens']:.0f},{it['cct_k']},{it['cri']}")
    return ("\n".join(out) + "\n").encode("utf-8")

def _generate_pdf_report(project_title: str,
                         analyzed: Dict[str, Any],
                         overlay_png: Optional[bytes] = None,
                         verification: Optional[Dict[str, Any]] = None,
                         checks: Optional[List[Dict[str, Any]]] = None) -> bytes:
    """Generate a professional PDF report (A4) with a clean, high-impact layout."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    def hr(y, gray=0.86):
        c.setStrokeGray(gray)
        c.setLineWidth(0.6)
        c.line(24*mm, y, W-24*mm, y)
        c.setStrokeGray(0.0)

    def badge(x, y, text, ok=True):
        if ok:
            c.setFillColorRGB(0.10, 0.55, 0.25)
        else:
            c.setFillColorRGB(0.75, 0.15, 0.15)
        c.roundRect(x, y-10, 56, 14, 6, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(x+28, y-6, text)
        c.setFillColorRGB(0, 0, 0)

    def header(title: str, subtitle: str = ""):
        c.setFont("Helvetica-Bold", 22)
        c.drawString(24*mm, H-28*mm, title)
        if subtitle:
            c.setFont("Helvetica", 10)
            c.setFillGray(0.35)
            c.drawString(24*mm, H-34*mm, subtitle)
            c.setFillGray(0.0)
        hr(H-38*mm, 0.80)

    def footer():
        c.setFont("Helvetica", 8)
        c.setFillGray(0.45)
        c.drawString(24*mm, 14*mm, "LuxIA • Report automatico")
        c.drawRightString(W-24*mm, 14*mm, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
        c.setFillGray(0.0)

    meta = analyzed.get("meta", {}) if isinstance(analyzed, dict) else {}
    rooms = analyzed.get("rooms") or []
    lights = analyzed.get("lights") or []
    lux_results = analyzed.get("lux_results") or []
    unit_info = analyzed.get("unit_info") or {}

    # map room_id -> lux result
    by_id = {}
    for r in lux_results:
        rid = r.get("room_id")
        if rid:
            by_id[rid] = r

    # Compliance counts
    ok_n = 0
    ko_n = 0
    for rm in rooms:
        rid = rm.get("id") or ""
        rr = by_id.get(rid) or {}
        status = (rr.get("metrics") or {}).get("status") or rr.get("status")
        if status == "OK":
            ok_n += 1
        elif status:
            ko_n += 1

    total_power = 0.0
    for l in lights:
        try:
            total_power += float(l.get("WATT") or l.get("watt") or 0.0)
        except Exception:
            pass

    # -------------------------
    # COVER
    # -------------------------
    header("LuxIA", project_title or "Report Illuminotecnico")
    c.setFont("Helvetica", 11)
    c.setFillGray(0.25)
    c.drawString(24*mm, H-48*mm, "Analisi multilivello • UNI EN 12464-1/2 • Interriflessioni (modello pro)")
    c.setFillGray(0.0)

    # Top right badge
    if verification and isinstance(verification, dict):
        passed = bool(verification.get("pass"))
        badge(W-24*mm-56, H-34*mm, "PASS" if passed else "FAIL", ok=passed)

    y = H-62*mm
    if overlay_png:
        try:
            img = ImageReader(io.BytesIO(overlay_png))
            iw, ih = img.getSize()
            maxw = W-48*mm
            maxh = 118*mm
            scale = min(maxw/iw, maxh/ih)
            dw, dh = iw*scale, ih*scale
            c.drawImage(img, 24*mm, y-dh, width=dw, height=dh, preserveAspectRatio=True, mask='auto')
            y = y-dh-10*mm
        except Exception:
            pass

    # Summary cards
    c.setFont("Helvetica-Bold", 12)
    c.drawString(24*mm, y, "Sintesi progetto")
    y -= 7*mm
    hr(y+2*mm)
    c.setFont("Helvetica", 10)
    lines = [
        f"Ambienti: {len(rooms)}    •    Apparecchi: {len(lights)}    •    Potenza installata: {total_power:.0f} W" if total_power else f"Ambienti: {len(rooms)}    •    Apparecchi: {len(lights)}",
    ]
    if unit_info:
        lines.append(f"Unità DXF: {unit_info.get('hint','n/d')}  (m/unit={float(unit_info.get('meters_per_unit') or 1.0):g}, conf={float(unit_info.get('confidence') or 0.0):.2f})")
    if ok_n or ko_n:
        lines.append(f"Conformità (pre-check): OK {ok_n} • KO {ko_n}")
    for ln in lines:
        c.drawString(24*mm, y, ln)
        y -= 6*mm

    y -= 2*mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(24*mm, y, "Parametri di calcolo")
    y -= 7*mm
    c.setFont("Helvetica", 9)
    rho_c = float(meta.get("rho_c", CEILING_REFLECTANCE_DEFAULT))
    rho_w = float(meta.get("rho_w", WALL_REFLECTANCE_DEFAULT))
    rho_f = float(meta.get("rho_f", FLOOR_REFLECTANCE_DEFAULT))
    mf = float(meta.get("mf", MAINTENANCE_FACTOR_DEFAULT))
    c.drawString(24*mm, y, f"Riflettanze: soffitto {rho_c:.2f} • pareti {rho_w:.2f} • pavimento {rho_f:.2f}   |   MF: {mf:.2f}")
    y -= 5*mm
    c.setFillGray(0.35)
    c.drawString(24*mm, y, "Riferimenti: UNI EN 12464-1 (interni) • UNI EN 12464-2 (esterni)")
    c.setFillGray(0.0)

    # Vision/Checks
    y -= 9*mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(24*mm, y, "Verifica planimetria (prova del nove)")
    y -= 7*mm
    c.setFont("Helvetica", 9)
    if verification and isinstance(verification, dict):
        conf = float(verification.get("confidence") or 0.0)
        issues = verification.get("issues") or []
        c.drawString(24*mm, y, f"Esito: {'PASS' if verification.get('pass') else 'FAIL'}   •   Confidenza: {conf:.2f}   •   Issue: {len(issues)}")
        y -= 5*mm
        # show top 3 issues (short)
        for it in issues[:3]:
            code = it.get("code") or "ISSUE"
            detail = (it.get("detail") or "")[:90]
            c.setFillGray(0.25)
            c.drawString(24*mm, y, f"- {code}: {detail}")
            c.setFillGray(0.0)
            y -= 4.5*mm
    if checks:
        for ck in checks[:3]:
            c.setFillGray(0.25)
            c.drawString(24*mm, y, f"- {ck.get('code','CHECK')}: {(ck.get('detail') or '')[:90]}")
            c.setFillGray(0.0)
            y -= 4.5*mm

    footer()
    c.showPage()

    # -------------------------
    # ROOMS TABLE
    # -------------------------
    header("Risultati per ambiente", "Target UNI, lux calcolati, uniformità, UGR")
    y = H-52*mm

    cols = [24*mm, 72*mm, 92*mm, 118*mm, 136*mm, 158*mm, W-24*mm]
    c.setFont("Helvetica-Bold", 9)
    headers = ["Ambiente", "m²", "Target", "Em (lux)", "U0", "UGR", "Esito"]
    for i, htxt in enumerate(headers):
        c.drawString(cols[i], y, htxt)
    y -= 4*mm
    hr(y, 0.85)
    y -= 6*mm
    c.setFont("Helvetica", 9)

    def page_break():
        nonlocal y
        footer()
        c.showPage()
        header("Risultati per ambiente", "continua")
        y = H-52*mm
        c.setFont("Helvetica-Bold", 9)
        for i, htxt in enumerate(headers):
            c.drawString(cols[i], y, htxt)
        y -= 4*mm
        hr(y, 0.85)
        y -= 6*mm
        c.setFont("Helvetica", 9)

    for rm in rooms:
        if y < 24*mm:
            page_break()

        rid = rm.get("id") or ""
        label = rm.get("label") or rid
        area = rm.get("area_m2") or 0.0
        rr = by_id.get(rid) or {}
        targets = rr.get("targets") or rr.get("target") or {}
        metrics = rr.get("metrics") or {}
        tgt_lux = targets.get("lux") or rr.get("target_lux") or ""
        std = targets.get("standard") or rr.get("standard") or ""
        Em = metrics.get("E_avg") or rr.get("lux") or ""
        U0 = metrics.get("U0") or rr.get("uniformity") or ""
        status = metrics.get("status") or rr.get("status") or ""
        ugr = rr.get("ugr_approx") or ""

        c.drawString(cols[0], y, str(label)[:30])
        c.drawRightString(cols[2]-2*mm, y, f"{float(area):.1f}" if area else "n/d")
        c.drawString(cols[2], y, f"{tgt_lux} ({std.split()[-1] if std else ''})" if tgt_lux else "n/d")
        c.drawString(cols[3], y, f"{float(Em):.0f}" if isinstance(Em, (int,float)) else "n/d")
        c.drawString(cols[4], y, f"{float(U0):.2f}" if isinstance(U0, (int,float)) else "n/d")
        c.drawString(cols[5], y, f"{float(ugr):.0f}" if isinstance(ugr, (int,float)) else ("n/d" if ugr=="" else str(ugr)))

        if status == "OK":
            c.setFillColorRGB(0.10, 0.55, 0.25)
            c.drawString(cols[6]-22*mm, y, "Conforme")
        else:
            c.setFillColorRGB(0.75, 0.15, 0.15)
            c.drawString(cols[6]-24*mm, y, "Non conf.")
        c.setFillColorRGB(0,0,0)
        y -= 6*mm

    footer()
    c.showPage()

    # -------------------------
    # BOM + SPEC
    # -------------------------
    header("Abaco apparecchi", "BOM e note di capitolato")
    y = H-52*mm

    bom = _build_bom(analyzed if isinstance(analyzed, dict) else {})
    c.setFont("Helvetica-Bold", 11)
    c.drawString(24*mm, y, "BOM (Bill of Materials)")
    y -= 7*mm
    hr(y+2*mm)

    c.setFont("Helvetica-Bold", 9)
    cols2 = [24*mm, 74*mm, 100*mm, 124*mm, 150*mm, W-24*mm]
    h2 = ["Tipo", "Q.tà", "W", "lm", "CCT/CRI", "Note"]
    for i, t in enumerate(h2):
        c.drawString(cols2[i], y, t)
    y -= 4*mm
    hr(y, 0.85)
    y -= 6*mm
    c.setFont("Helvetica", 9)

    if not bom:
        c.setFillGray(0.35)
        c.drawString(24*mm, y, "n/d")
        c.setFillGray(0.0)
        y -= 6*mm
    else:
        for it in bom:
            if y < 24*mm:
                footer()
                c.showPage()
                header("Abaco apparecchi", "continua")
                y = H-52*mm
                c.setFont("Helvetica", 9)
            typ = it.get("type","").upper()
            qty = int(it.get("qty") or 0)
            watt = float(it.get("watt") or 0.0)
            lum = float(it.get("lumens") or 0.0)
            cct = int(it.get("cct_k") or int(float(os.getenv("LIGHT_DEFAULT_CCT","3000"))))
            cri = int(it.get("cri") or int(float(os.getenv("LIGHT_DEFAULT_CRI","80"))))
            note = "Layer: ILLUMINAZIONE • Blocchi"  # per tuo standard CAD
            c.drawString(cols2[0], y, str(typ)[:22])
            c.drawRightString(cols2[2]-2*mm, y, str(qty))
            c.drawRightString(cols2[3]-2*mm, y, f"{watt:.0f}")
            c.drawRightString(cols2[4]-2*mm, y, f"{lum:.0f}")
            c.drawString(cols2[4], y, f"{cct}K / CRI {cri}")
            c.drawString(cols2[5]-40*mm, y, note[:30])
            y -= 6*mm

    y -= 6*mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(24*mm, y, "Capitolato (estratto)")
    y -= 7*mm
    hr(y+2*mm)
    c.setFont("Helvetica", 9)
    spec = _build_spec_text(analyzed if isinstance(analyzed, dict) else {})
    for ln in spec.splitlines()[:24]:
        if y < 24*mm:
            footer()
            c.showPage()
            header("Capitolato", "continua")
            y = H-52*mm
            c.setFont("Helvetica", 9)
        c.setFillGray(0.15)
        c.drawString(24*mm, y, ln[:110])
        c.setFillGray(0.0)
        y -= 4.5*mm

    footer()
    c.showPage()

    # -------------------------
    # VISUAL HIERARCHY (Designer)
    # -------------------------
    header("Gerarchia visiva", "Allineamento assi • Verticale • Accent")
    y = H-52*mm

    layer_stats={}
    for lt in (analyzed.get("lights") or []):
        lyr = str(lt.get("layer") or "AMBIENT").upper()
        st = layer_stats.setdefault(lyr, {"qty":0,"w":0.0,"lm":0.0})
        st["qty"] += 1
        try: st["w"] += float(lt.get("watt") or 0.0)
        except Exception: pass
        try: st["lm"] += float(lt.get("lumens") or 0.0)
        except Exception: pass

    c.setFont("Helvetica-Bold", 11)
    c.drawString(24*mm, y, "Strati")
    y -= 7*mm
    hr(y+2*mm)
    c.setFont("Helvetica-Bold", 9)
    cols=[24*mm, 86*mm, 110*mm, 140*mm, W-24*mm]
    for i,t in enumerate(["Layer","Q.tà","W tot","lm tot"]):
        c.drawString(cols[i], y, t)
    y -= 4*mm
    hr(y, 0.85)
    y -= 6*mm
    c.setFont("Helvetica", 9)
    for lyr in ["AMBIENT","PERIMETER","VERTICAL","ACCENT"]:
        st = layer_stats.get(lyr)
        if not st: continue
        c.drawString(cols[0], y, lyr)
        c.drawRightString(cols[2]-2*mm, y, str(st["qty"]))
        c.drawRightString(cols[3]-2*mm, y, f"{st['w']:.0f}")
        c.drawRightString(cols[4]-2*mm, y, f"{st['lm']:.0f}")
        y -= 6*mm

    footer()
    c.showPage()

    c.save()
    return buf.getvalue()

app = FastAPI(title="LuxIA Engine", version="22.x-vnext")

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
# LLM Router (orchestrator)
# --------------------------
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
                text=f"({provider}) Missing {provider.upper()}_BASE_URL. Using heuristic concepts.",
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

# --------------------------
# Vision Router (verifier)
# --------------------------

class MultiVisionVerifier:
    """
    Vision verifier strategy:
      1) Gemini (primary) via Google Generative Language API
      2) OpenRouter (fallback) via OpenAI-compatible /chat/completions
      3) Legacy OpenAI-compatible provider (VISION_BASE_URL/VISION_MODEL) if configured
    It can verify using BOTH original raster and an overlay image (polygons/labels).
    """

    def __init__(self):
        # Primary: Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or ""
        self.gemini_model = (os.getenv("GEMINI_MODEL") or "gemini-1.5-pro").strip()
        self.gemini_base_url = (os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

        # Fallback: OpenRouter (OpenAI-compatible)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or ""
        self.openrouter_model = (os.getenv("OPENROUTER_MODEL") or "").strip()
        self.openrouter_base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.openrouter_http_referer = os.getenv("OPENROUTER_HTTP_REFERER") or ""
        self.openrouter_app_title = os.getenv("OPENROUTER_APP_TITLE") or "LuxIA"

        # Legacy / optional: any OpenAI-compatible vision provider
        self.legacy_base_url = (os.getenv("VISION_BASE_URL") or "").rstrip("/")
        self.legacy_api_key = os.getenv("VISION_API_KEY") or ""
        self.legacy_model = (os.getenv("VISION_MODEL") or "").strip()

        # Verification thresholds (gating)
        self.min_confidence = float(os.getenv("VISION_MIN_CONFIDENCE") or "0.80")

    @property
    def enabled(self) -> bool:
        return bool(self.gemini_api_key or (self.openrouter_api_key and self.openrouter_model) or (self.legacy_base_url and self.legacy_model))

    def verify(self, raster_png: bytes, extracted: Dict[str, Any], overlay_png: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Returns JSON:
          { enabled, provider, pass, confidence, issues, suggested_fixes, observations, raw? }
        """
        if not self.enabled:
            return {
                "enabled": False,
                "provider": None,
                "pass": False,
                "confidence": 0.0,
                "issues": ["Vision verifier not configured. Set GEMINI_API_KEY (recommended) or OPENROUTER_API_KEY+OPENROUTER_MODEL."],
                "suggested_fixes": [],
                "observations": {},
            }

        # Try Gemini first
        if self.gemini_api_key:
            out = self._verify_gemini(raster_png, extracted, overlay_png)
            if out.get("ok"):
                return self._postprocess(out, provider="gemini")
        # Fallback: OpenRouter
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

        # If all failed
        return {
            "enabled": True,
            "provider": "none",
            "pass": False,
            "confidence": 0.0,
            "issues": ["All configured Vision providers failed."],
            "suggested_fixes": [],
            "observations": {},
        }

    # --------------------------
    # Provider implementations
    # --------------------------
    def _verify_gemini(self, raster_png: bytes, extracted: Dict[str, Any], overlay_png: Optional[bytes]) -> Dict[str, Any]:
        """
        Gemini REST: POST /models/{model}:generateContent?key=...
        Multimodal via inlineData parts.
        """
        url = f"{self.gemini_base_url}/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        prompt = self._strict_prompt(extracted)

        parts = [{"text": prompt}]
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(raster_png).decode("ascii"),
            }
        })
        if overlay_png:
            parts.append({"text": "OVERLAY IMAGE (predicted polygons/labels) for cross-check:"})
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(overlay_png).decode("ascii"),
                }
            })

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1200},
        }

        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            # Extract text from candidates
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
            resp = {"ok": True, "out": out, "raw": (text or "")[:1200]}
        except Exception as e:
            return {"ok": False, "error": f"Gemini verify call failed: {e}", "raw": str(e)}

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
            content += [
                {"type": "text", "text": "OVERLAY IMAGE (predicted polygons/labels) for cross-check:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64b}"}},
            ]

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
            resp = {"ok": True, "out": out, "raw": text[:1200]}
        except Exception as e:
            return {"ok": False, "error": f"Vision verify call failed: {e}", "raw": str(e)}

    def _openrouter_headers(self) -> Dict[str, str]:
        h = {}
        # Optional but recommended by OpenRouter
        if self.openrouter_http_referer:
            h["HTTP-Referer"] = self.openrouter_http_referer
        if self.openrouter_app_title:
            h["X-Title"] = self.openrouter_app_title
        return h

    # --------------------------
    # Helpers
    # --------------------------
    def _strict_prompt(self, extracted: Dict[str, Any]) -> str:
        # Keep it short and strict; include extracted JSON (truncated) + explicit pass/fail
        ex = json.dumps(extracted, ensure_ascii=False)
        if len(ex) > 12000:
            ex = ex[:12000] + "…(truncated)"
        return (
            "You are LuxIA Vision Verifier. You MUST be extremely strict.\n"
            "Task: Compare the planimetry image with the extracted JSON (and the overlay image if provided).\n"
            "If uncertain, mark as an issue.\n"
            "Return STRICT JSON ONLY, no markdown, with schema:\n"
            "{\"confidence\":0..1,\"pass\":true|false,"
            "\"issues\":[{\"code\":string,\"detail\":string}],"
            "\"suggested_fixes\":[string],"
            "\"observations\":{}}.\n"
            "Rules:\n"
            "- pass=false if any issue is present or if extraction seems incomplete.\n"
            "- confidence must reflect certainty.\n"
            "- Only report verifiable issues (rooms missing, polygons outside walls, doors/windows mismatch).\n\n"
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
        # Normalize issues: allow strings or dicts
        norm_issues = []
        for it in issues:
            if isinstance(it, str):
                norm_issues.append({"code": "ISSUE", "detail": it})
            elif isinstance(it, dict):
                norm_issues.append({"code": it.get("code") or "ISSUE", "detail": it.get("detail") or json.dumps(it)})
            else:
                norm_issues.append({"code": "ISSUE", "detail": str(it)})

        passed = bool(out.get("pass")) if "pass" in out else (conf >= self.min_confidence and len(norm_issues) == 0)
        # Enforce strict gating: any issues => pass false
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

vision = MultiVisionVerifier()

# --------------------------
# Heuristic fallback for concepts
# --------------------------
def heuristic_concepts(brief: str, areas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
# Concepts API
# --------------------------
class ConceptReq(BaseModel):
    project_id: str
    brief: str = ""
    areas: List[Dict[str, Any]] = []
    n: int = 3

@app.post("/concepts")
def generate_concepts(req: ConceptReq):
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

    resp = {"ok": True, "provider": res.provider, "concepts": concepts, "llm_note": res.text[:400]}

# --------------------------
# Planimetry ingest + verification
# --------------------------

class PlanimetryResponse(BaseModel):
    ok: bool
    input_type: str
    page_count: int
    extracted: Dict[str, Any]
    checks: Dict[str, Any]
    vision: Dict[str, Any]


def _ext(name: str) -> str:
    name = (name or "").lower().strip()
    if "." not in name:
        return ""
    return name.split(".")[-1]


def _guess_type(filename: str, content_type: str) -> str:
    ext = _ext(filename)
    ct = (content_type or "").lower()
    if ext in ("pdf",) or "pdf" in ct:
        return "pdf"
    if ext in ("jpg","jpeg","png","webp","tif","tiff") or ct.startswith("image/"):
        return "image"
    if ext in ("dxf",):
        return "dxf"
    if ext in ("dwg",):
        return "dwg"
    return "unknown"


def _require(module: str):
    try:
        __import__(module)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency '{module}'. Install it in engine requirements. Error: {e}")


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
    # force RGB for consistent vision input
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    _require("PIL")
    from io import BytesIO
    out = BytesIO()
    pil.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _image_to_png(img_bytes: bytes) -> bytes:
    _require("PIL")
    from PIL import Image
    from io import BytesIO
    im = Image.open(BytesIO(img_bytes))
    im = im.convert("RGB")
    out = BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _dxf_to_png(dxf_bytes: bytes, size: int = 1800) -> bytes:
    """Render DXF to a raster PNG for Vision + keep vector data for geometry checks."""
    _require("ezdxf")
    _require("matplotlib")
    import ezdxf
    from ezdxf import recover
    from io import BytesIO

    # Robust DXF read from bytes (handles slightly corrupted files)
    try:
        doc, _auditor = recover.read(BytesIO(dxf_bytes))
    except Exception:
        # Fallback: text read
        doc = ezdxf.read(dxf_bytes.decode("utf-8", errors="ignore"))

    msp = doc.modelspace()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

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
        raise HTTPException(
            status_code=422,
            detail=(
                "DWG uploaded but no converter configured. "
                "Set DWG2DXF_URL to an internal service (ODA/Autodesk/Cloud) that returns DXF, "
                "or convert DWG to DXF client-side before upload."
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


def _score_extraction_units(extracted: Dict[str, Any]) -> float:
    """Heuristic score to pick the most plausible units scaling."""
    try:
        rooms = extracted.get("rooms") or []
        if not rooms:
            return -1e9
        areas=[]
        widths=[]
        for r in rooms:
            a=float(r.get("area_m2") or 0.0)
            if a>0: areas.append(a)
            dims=_room_bbox_dims_m(r, None)
            w=min(float(dims.get("L") or 0.0), float(dims.get("W") or 0.0))
            if w>0: widths.append(w)
        if not areas:
            return -1e9
        med_a=statistics.median(areas)
        med_w=statistics.median(widths) if widths else 0.0
        score=0.0
        # target median area (typical office/corridor mix): 6..80 m2
        if 6.0 <= med_a <= 80.0:
            score += 3.0
        else:
            score -= min(3.0, abs(math.log((med_a+1e-6)/25.0)))
        # target median width: 1.0..8.0 m
        if 1.0 <= med_w <= 8.0:
            score += 2.0
        else:
            score -= 1.5
        # penalize extreme sizes
        if med_a > 500 or med_a < 1.0:
            score -= 4.0
        return float(score)
    except Exception:
        return -1e9

def _deterministic_checks(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Hard checks that must pass before we trust the planimetry."""
    issues: List[str] = []
    warnings: List[str] = []

    rooms = extracted.get("rooms") or []
    openings = extracted.get("openings") or []

    # minimum sanity
    if not isinstance(rooms, list):
        issues.append("rooms is not a list")
        rooms = []
    if len(rooms) == 0:
        issues.append("No rooms detected/extracted")

    # polygon sanity (if provided)
    for i, r in enumerate(rooms[:200]):
        poly = r.get("polygon_px")
        if poly is None:
            warnings.append(f"Room[{i}] missing polygon_px")
            continue
        if not isinstance(poly, list) or len(poly) < 3:
            issues.append(f"Room[{i}] polygon_px invalid")

    # opening sanity
    for i, o in enumerate(openings[:500]):
        if o.get("type") not in ("door", "window"):
            warnings.append(f"Opening[{i}] unknown type")

    # scale plausibility
    scale = extracted.get("scale") or {}
    ppm = scale.get("pxPerMeter")
    if ppm is not None:
        try:
            ppm = float(ppm)
            if ppm <= 0:
                issues.append("pxPerMeter must be > 0")
            if ppm < 20 or ppm > 3000:
                warnings.append("pxPerMeter looks unusual; check scale")
        except Exception:
            issues.append("pxPerMeter is not numeric")

    ok = len(issues) == 0
    # Conservative confidence: start from 0.4, add signals
    confidence = 0.4
    if ok:
        confidence += 0.2
    if len(rooms) >= 3:
        confidence += 0.1
    if ppm is not None:
        confidence += 0.1
    if len(warnings) == 0 and ok:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence))

    return {
        "ok": ok,
        "confidence": confidence,
        "issues": issues,
        "warnings": warnings,
    }



# ==========================================================
# LuxIA PRO - Normative targets (UNI EN 12464-1/2) + Photometry
# NOTE: The full UNI standards text is proprietary; LuxIA uses
# a configurable mapping of common environments to maintained
# illuminance targets + minimum uniformity, and includes the
# normative reference in the generated report.
# ==========================================================

# Defaults (requested)
CEILING_REFLECTANCE_DEFAULT = float(os.getenv("CEILING_REFLECTANCE", "0.8"))
WALL_REFLECTANCE_DEFAULT = float(os.getenv("WALL_REFLECTANCE", "0.6"))
FLOOR_REFLECTANCE_DEFAULT = float(os.getenv("FLOOR_REFLECTANCE", "0.4"))
MAINTENANCE_FACTOR_DEFAULT = float(os.getenv("MAINTENANCE_FACTOR", "0.8"))
WORKPLANE_HEIGHT_M_DEFAULT = float(os.getenv("WORKPLANE_HEIGHT_M", "0.8"))
ROOM_HEIGHT_M_DEFAULT = float(os.getenv("ROOM_HEIGHT_M", "2.7"))

# Minimal built-in mapping (extend as needed)
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
    # normalize accents / separators minimally
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # keyword mapping
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
    # if room explicitly tagged
    if room.get("is_outdoor") is True:
        return True
    label = room.get("label") or ""
    key = _norm_key(label)
    return key in ("PARCHEGGIO","PASSAGGIO","CORTILE","GIARDINO","AREA_LAVORO")

def _ugr_target_for_key(key: str, is_outdoor: bool) -> Optional[float]:
    if is_outdoor:
        return None
    # common practice: offices 19, general areas 22
    if key in ("UFFICIO","OPENSPACE","SALA_RIUNIONI"):
        return 19.0
    if key in ("CORRIDOIO","SCALA"):
        return 22.0
    return 22.0

def _get_uni_targets(room: Dict[str, Any]) -> Dict[str, Any]:
    label = room.get("label")
    key = _norm_key(label)
    if _is_outdoor_room(room):
        t = UNI_12464_2_TARGETS.get(key) or UNI_12464_2_TARGETS["DEFAULT"]
        return {"standard": "UNI EN 12464-2", "key": key, "ugr_max": _ugr_target_for_key(key, True), **t}
    t = UNI_12464_1_TARGETS.get(key) or UNI_12464_1_TARGETS["DEFAULT"]
    return {"standard": "UNI EN 12464-1", "key": key, "ugr_max": _ugr_target_for_key(key, False), **t}


def _room_bbox_dims_m(room: Dict[str, Any], px_per_m: Optional[float]) -> Dict[str, float]:
    poly = room.get("polygon_px") or []
    if len(poly) < 3 or not px_per_m or px_per_m <= 0:
        return {"L": 0.0, "W": 0.0}
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    L = (max(xs) - min(xs)) / px_per_m
    W = (max(ys) - min(ys)) / px_per_m
    # ensure L >= W
    if W > L:
        L, W = W, L
    return {"L": float(L), "W": float(W)}

def _room_index_K(L: float, W: float, Hm: float) -> float:
    if Hm <= 0 or (L + W) <= 0:
        return 0.0
    return (L * W) / (Hm * (L + W))

def _cu_lookup(K: float, rho_c: float, rho_w: float, rho_f: float) -> float:
    # Simple CU approximation for downlight-wide distribution.
    # For production, replace with luminaire-specific CU tables / IES.
    # K in [0.5..5], reflectances influence modestly.
    Kc = max(0.2, min(6.0, K))
    base = 0.35 + 0.25 * (1 - pow(2.71828, -0.6 * Kc))  # 0.35..~0.58
    refl = (0.5 * rho_c + 0.4 * rho_w + 0.1 * rho_f)
    adj = 0.85 + 0.35 * (refl - 0.5)  # around +/- 0.175
    cu = base * adj
    return float(max(0.20, min(0.75, cu)))

def _radiosity_multiplier(rho_c: float, rho_w: float, rho_f: float) -> float:
    # Fast converging indirect light multiplier (iterative energy balance approximation)
    # Keeps results realistic without heavy geometry radiosity.
    rho_avg = max(0.0, min(0.95, 0.45 * rho_c + 0.45 * rho_w + 0.10 * rho_f))
    # indirect share: 0.15..0.45 typical
    indirect = 0.15 + 0.35 * rho_avg
    # clamp multiplier
    return float(max(1.00, min(1.35, 1.0 + indirect)))
def _radiosity_multiplier_pro(room: Dict[str, Any]) -> float:
    """Fast interreflection model based on room index + area-weighted reflectance."""
    try:
        rho_c=float(room.get("rho_c") or CEILING_REFLECTANCE_DEFAULT)
        rho_w=float(room.get("rho_w") or WALL_REFLECTANCE_DEFAULT)
        rho_f=float(room.get("rho_f") or FLOOR_REFLECTANCE_DEFAULT)

        dims=_room_bbox_dims_m(room, None)
        L=float(dims.get("L") or 0.0); W=float(dims.get("W") or 0.0)
        H=float(room.get("height_m") or ROOM_HEIGHT_M_DEFAULT)
        Hm=max(0.3, H - WORKPLANE_HEIGHT_M_DEFAULT)

        A_c=L*W; A_f=L*W; A_w=2*(L+W)*Hm
        At=A_c+A_f+A_w if (A_c+A_f+A_w)>1e-6 else 1.0
        rho_avg=(rho_c*A_c + rho_f*A_f + rho_w*A_w)/At

        K=_room_index_K(L,W,Hm)
        k_eff = 0.55 + 0.25*min(1.5, max(0.2, K)) / 1.5  # 0.55..0.80
        R = float(rho_avg) * k_eff
        R = max(0.0, min(R, 0.92))
        return float(1.0/(1.0-R))
    except Exception:
        return _radiosity_multiplier(
            float(room.get("rho_c") or CEILING_REFLECTANCE_DEFAULT),
            float(room.get("rho_w") or WALL_REFLECTANCE_DEFAULT),
            float(room.get("rho_f") or FLOOR_REFLECTANCE_DEFAULT),
        )



def _estimate_uniformity(room: Dict[str, Any], n_lights: int, L: float, W: float, Hm: float, spacing_m: float) -> float:
    # Simple spacing/height rule of thumb.
    if n_lights <= 0 or Hm <= 0:
        return 0.0
    # approximate grid spacing
    s = spacing_m if spacing_m > 0 else max(0.5, min(L, W))
    SHR = s / Hm  # spacing-to-height ratio
    # map SHR to uniformity; lower SHR => better
    u0 = 0.75 - 0.35 * max(0.0, min(1.0, (SHR - 0.8) / 1.2))
    # small penalty for very elongated rooms
    if W > 0 and L / W > 3.0:
        u0 -= 0.08
    return float(max(0.20, min(0.80, u0)))

def _photometry_calc(room: Dict[str, Any], lights_in_room: int, lumens_per_light: float, px_per_m: Optional[float]) -> Dict[str, Any]:
    area = float(room.get("area_m2") or 0.0)
    dims = _room_bbox_dims_m(room, px_per_m)
    L, W = dims["L"], dims["W"]
    H = float(room.get("height_m") or ROOM_HEIGHT_M_DEFAULT)
    Hm = max(0.3, H - WORKPLANE_HEIGHT_M_DEFAULT)

    targets = _get_uni_targets(room)
    K = _room_index_K(L, W, Hm)
    rho_c = float(room.get("rho_c") or CEILING_REFLECTANCE_DEFAULT)
    rho_w = float(room.get("rho_w") or WALL_REFLECTANCE_DEFAULT)
    rho_f = float(room.get("rho_f") or FLOOR_REFLECTANCE_DEFAULT)
    CU = _cu_lookup(K, rho_c, rho_w, rho_f)
    MF = float(room.get("mf") or MAINTENANCE_FACTOR_DEFAULT)

    if area <= 0:
        return {
            "target": targets,
            "lux": 0.0,
            "uniformity": 0.0,
            "K": K,
            "CU": CU,
            "MF": MF,
            "radiosity_mult": 1.0,
            "status": "INVALID_AREA",
        }

    E_direct = (lights_in_room * lumens_per_light * CU * MF) / area
    mult = _radiosity_multiplier(rho_c, rho_w, rho_f)
    E = E_direct * mult

    spacing_m = float(os.getenv("LIGHT_GRID_SPACING_M", "3.0"))
    u0 = _estimate_uniformity(room, lights_in_room, L, W, Hm, spacing_m)

    status = "OK" if (E >= targets["lux"] and u0 >= targets["u0"]) else "NOT_OK"
    return {
        "target": targets,
        "lux": float(E),
        "lux_direct": float(E_direct),
        "uniformity": float(u0),
        "K": float(K),
        "CU": float(CU),
        "MF": float(MF),
        "radiosity_mult": float(mult),
        "dims_m": {"L": L, "W": W, "H": H, "Hm": Hm},
        "status": status,
    }

def _optimize_lights_for_room(room: Dict[str, Any], base_lights: int, lumens_per_light: float, px_per_m: Optional[float], max_add: int = 30) -> Dict[str, Any]:
    n = max(0, int(base_lights))
    last = None
    for _ in range(max_add + 1):
        last = _photometry_calc(room, n, lumens_per_light, px_per_m)
        if last["status"] == "OK":
            break
        n += 1
    return {"lights_needed": n, "result": last}

def _make_overlay_png(base_png: bytes, extracted: Dict[str, Any]) -> bytes:
    """
    Draws predicted polygons/labels over the raster image as a 'prova del nove' overlay.
    This is deterministic and does not rely on Vision.
    """
    img = Image.open(io.BytesIO(base_png)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    rooms = (extracted or {}).get("rooms") or []
    # Semi-transparent strokes
    for i, room in enumerate(rooms):
        poly = room.get("polygon_px") or []
        if len(poly) >= 3:
            pts = [(float(x), float(y)) for x, y in poly]
            draw.line(pts + [pts[0]], width=3, fill=(255, 0, 0, 180))
            # Label
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            label = room.get("label") or room.get("id") or f"room_{i+1}"
            area = room.get("area_m2")
            if area is not None:
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
        # prefer pixel coordinates if provided; else project from meters not available here
        x = lt.get("x_px"); y = lt.get("y_px")
        if x is None or y is None:
            continue
        x = float(x); y = float(y)
        r = 6
        draw.rectangle((x - r, y - r, x + r, y + r), outline=(255, 165, 0, 230), width=3)
        draw.text((x + r + 2, y - r - 2), lt.get("id") or "L", fill=(255, 165, 0, 230))


    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()



def _extract_from_raster_stub(raster_png: bytes, input_type: str) -> Dict[str, Any]:
    """
    Placeholder for PDF/JPEG/PNG pipeline. Keep minimal and safe.
    The strict verifier will usually fail until a real CV extractor is plugged in.
    """
    return {
        "meta": {"source": input_type, "note": "raster extractor not yet implemented"},
        "scale": {"detected": False, "pxPerMeter": None, "method": None, "units_guess": None},
        "rooms": [],
        "openings": [],
        "lights": [],
        "geojson": {"type": "FeatureCollection", "features": []},
        "warnings": ["Raster extractor not implemented yet."],
    }



def _assign_room_labels_from_text(doc, rooms: List[Dict[str, Any]]):
    try:
        msp = doc.modelspace()
    except Exception:
        return
    texts=[]
    for e in msp:
        t=e.dxftype()
        if t in ("TEXT","MTEXT"):
            try:
                s = e.dxf.text if t=="TEXT" else e.text
                p = e.dxf.insert if t=="TEXT" else e.dxf.insert
                texts.append((float(p.x), float(p.y), str(s).strip()))
            except Exception:
                pass
    if not texts:
        return
    for r in rooms:
        poly=r.get("polygon_m") or []
        if len(poly)<3:
            continue
        cx=sum(p[0] for p in poly)/len(poly)
        cy=sum(p[1] for p in poly)/len(poly)
        best=None; bd=1e18
        for x,y,s in texts:
            d=(x-cx)*(x-cx)+(y-cy)*(y-cy)
            if d<bd:
                bd=d; best=s
        if best and len(best) <= 40:
            r["label"]=best

def _is_grid_layer(name: str) -> bool:
    n = (name or "").upper()
    return any(k in n for k in ("GRIGLIA","GRID","ASSI","AXIS","MODULO","MODULE"))

def _line_angle_rad(x1,y1,x2,y2) -> float:
    return math.atan2((y2-y1),(x2-x1))

def _quantize_angle(a: float, tol_deg: float = 7.0) -> Optional[float]:
    """Quantize to 0 or pi/2 if within tolerance."""
    tol = math.radians(tol_deg)
    # fold to [0, pi)
    while a < 0: a += math.pi
    while a >= math.pi: a -= math.pi
    if abs(a-0.0) < tol or abs(a-math.pi) < tol:
        return 0.0
    if abs(a-math.pi/2) < tol:
        return math.pi/2
    return None

def _robust_spacing(vals: List[float]) -> Optional[float]:
    if not vals or len(vals) < 4:
        return None
    v = sorted(set(round(x,3) for x in vals))
    diffs = [v[i+1]-v[i] for i in range(len(v)-1) if (v[i+1]-v[i]) > 1e-3]
    if len(diffs) < 3:
        return None
    # take median and clamp to sensible module range
    s = float(statistics.median(diffs))
    if s < 0.05 or s > 5.0:
        return None
    return s

def _detect_cad_grid(grid_lines: List[Tuple[float,float,float,float]], unit_to_m: float) -> Dict[str, Any]:
    """Detect orthogonal grid positions & spacing from candidate grid lines."""
    xs=[]; ys=[]
    for (x1,y1,x2,y2) in grid_lines:
        a=_quantize_angle(_line_angle_rad(x1,y1,x2,y2))
        if a is None:
            continue
        if abs(a) < 1e-6:  # horizontal -> y const
            ys.append(((y1+y2)/2.0)*unit_to_m)
        else:              # vertical -> x const
            xs.append(((x1+x2)/2.0)*unit_to_m)
    # de-dup with rounding
    xs = sorted(set(round(x,3) for x in xs))
    ys = sorted(set(round(y,3) for y in ys))
    sx = _robust_spacing(xs)
    sy = _robust_spacing(ys)
    spacing = None
    if sx and sy:
        spacing = float(statistics.median([sx,sy]))
    else:
        spacing = sx or sy
    return {"x": xs[:300], "y": ys[:300], "spacing_m": spacing, "detected": bool(xs or ys)}

def _extract_from_dxf_real(dxf_bytes: bytes, raster_png: bytes, opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robust DXF extractor that does NOT rely on layers:
      - Project all geometry to XY (handles 2D/3D sources)
      - Detect walls from geometry primitives
      - Build room polygons via vector->raster->floodfill->contours
      - Place lights deterministically and (optionally) write them back as INSERT blocks on layer ILLUMINAZIONE
    """
    _require("ezdxf")
    _require("numpy")
    _require("opencv-python")
    import ezdxf
    from ezdxf import recover
    import numpy as np
    import cv2
    from io import BytesIO

    # ----------------------------
    # Read DXF robustly
    # ----------------------------
    try:
        doc, _auditor = recover.read(BytesIO(dxf_bytes))
    except Exception:
        doc = ezdxf.read(dxf_bytes.decode("utf-8", errors="ignore"))
    msp = doc.modelspace()

    # ----------------------------
    # Collect geometry candidates (walls) — geometry-first
    # ----------------------------
    segs: List[Tuple[float,float,float,float]] = []
    grid_lines: List[Tuple[float,float,float,float]] = []

    def add_seg(x1,y1,x2,y2):
        # ignore tiny segments
        if (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) < 1e-8:
            return
        segs.append((float(x1), float(y1), float(x2), float(y2)))

    # LINE
    for e in msp.query("LINE"):
        try:
            s = e.dxf.start
            t = e.dxf.end
            add_seg(s.x, s.y, t.x, t.y)
            try:
                if _is_grid_layer(getattr(e.dxf, 'layer', '')):
                    grid_lines.append((s.x,s.y,t.x,t.y))
            except Exception:
                pass
        except Exception:
            continue

    # LWPOLYLINE / POLYLINE
    for e in msp.query("LWPOLYLINE POLYLINE"):
        try:
            pts = []
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
            else:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices()]
            if len(pts) >= 2:
                for i in range(len(pts)-1):
                    add_seg(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                    try:
                        if _is_grid_layer(getattr(e.dxf, 'layer', '')):
                            grid_lines.append((pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]))
                    except Exception:
                        pass
                # close if flagged/obvious
                try:
                    closed = bool(getattr(e, "closed", False)) or bool(getattr(e, "is_closed", False))
                except Exception:
                    closed = False
                if closed:
                    add_seg(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1])
                    try:
                        if _is_grid_layer(getattr(e.dxf, 'layer', '')):
                            grid_lines.append((pts[-1][0], pts[-1][1], pts[0][0], pts[0][1]))
                    except Exception:
                        pass
        except Exception:
            continue

    # ARC approximation
    for e in msp.query("ARC"):
        try:
            c = e.dxf.center
            r = float(e.dxf.radius)
            a0 = float(e.dxf.start_angle) * np.pi / 180.0
            a1 = float(e.dxf.end_angle) * np.pi / 180.0
            # sample
            n = 24
            angles = np.linspace(a0, a1, n)
            pts = [(c.x + r*np.cos(a), c.y + r*np.sin(a)) for a in angles]
            for i in range(len(pts)-1):
                add_seg(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
        except Exception:
            continue

    if len(segs) < 20:
        return {
            "meta": {"source": "dxf", "note": "too little geometry to detect rooms"},
            "scale": {"detected": False, "pxPerMeter": None, "method": None, "units_guess": None},
            "rooms": [],
            "openings": [],
            "lights": [],
            "geojson": {"type": "FeatureCollection", "features": []},
            "warnings": ["DXF geometry seems empty or not a floorplan (too few segments)."],
        }

    xs = [s[0] for s in segs] + [s[2] for s in segs]
    ys = [s[1] for s in segs] + [s[3] for s in segs]
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))
    w = maxx - minx
    h = maxy - miny
    maxdim = max(w, h)

    # Heuristic units guess (mm vs m)
    units_guess = "m"
    unit_to_m = 1.0
    if maxdim > 800:  # typical CAD in millimeters
        units_guess = "mm"
        unit_to_m = 0.001
    cad_grid = _detect_cad_grid(grid_lines, unit_to_m)


    # Raster size
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

    # Draw walls on binary image
    wall = np.zeros((img_h, img_w), dtype=np.uint8)
    thickness = int(opts.get("wall_thickness_px") or 4)
    thickness = max(2, min(12, thickness))
    for (x1,y1,x2,y2) in segs:
        x1p,y1p = cad_to_px(x1,y1)
        x2p,y2p = cad_to_px(x2,y2)
        cv2.line(wall, (x1p,y1p), (x2p,y2p), 255, thickness=thickness, lineType=cv2.LINE_AA)

    # Morphological close to seal tiny gaps
    k = int(opts.get("close_kernel_px") or 7)
    k = max(3, min(25, k))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    wall2 = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Space image
    space = cv2.bitwise_not(wall2)

    # Flood-fill outside from border
    ff = space.copy()
    mask = np.zeros((img_h+2, img_w+2), dtype=np.uint8)
    cv2.floodFill(ff, mask, (0,0), 0)

    # Remaining white regions are enclosed spaces (candidate rooms)
    rooms_mask = ff  # 255 where enclosed

    # Connected components
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats((rooms_mask > 0).astype(np.uint8), connectivity=8)
    room_polys_px: List[List[Tuple[int,int]]] = []
    room_polys_cad_m: List[List[Tuple[float,float]]] = []

    # Exclude background label 0
    areas = []
    for i in range(1, num):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        areas.append((area_px, i))
    # Sort by area descending to drop the biggest (often outside, but outside is removed by floodfill already)
    areas.sort(reverse=True)

    min_area_px = int(opts.get("min_room_area_px") or 2500)
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
        # Simplify
        eps = float(opts.get("polygon_simplify_eps") or 3.0)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts_px = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts_px) < 3:
            continue
        room_polys_px.append(pts_px)
        pts_cad = [px_to_cad(x,y) for x,y in pts_px]
        pts_m = [(x*unit_to_m, y*unit_to_m) for x,y in pts_cad]
        room_polys_cad_m.append(pts_m)

    def poly_area(poly: List[Tuple[float,float]]) -> float:
        s = 0.0
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]
            x2,y2 = poly[(i+1)%n]
            s += x1*y2 - x2*y1
        return abs(s) * 0.5

    rooms: List[Dict[str, Any]] = []
    for idx, (px_poly, m_poly) in enumerate(zip(room_polys_px, room_polys_cad_m)):
        area_m2 = poly_area(m_poly)
        rooms.append({
            "id": f"R{idx+1:03d}",
            "label": None,
            "area_m2": float(area_m2),
            "perimeter_m": None,
            "polygon_px": [[int(x), int(y)] for x,y in px_poly],
            "polygon_m": [[float(x), float(y)] for x,y in m_poly],
        })

    # ----------------------------
    # Place lights (deterministic)
    # ----------------------------
    lights = _place_lights_from_rooms(rooms, opts)
    # Add pixel coordinates for overlay
    for lt in lights:
        x_cad = (float(lt.get('x_m',0.0)) / unit_to_m)
        y_cad = (float(lt.get('y_m',0.0)) / unit_to_m)
        xpx, ypx = cad_to_px(x_cad, y_cad)
        lt['x_px'] = int(xpx)
        lt['y_px'] = int(ypx)

    # Optionally: write lights back into DXF and return dxf_out_base64
    dxf_out_b64 = None
    if bool(opts.get("write_lights_dxf", True)):
        try:
            dxf_out = _write_lights_to_dxf(doc, lights, units_guess=units_guess)
            dxf_out_b64 = base64.b64encode(dxf_out).decode("ascii")
        except Exception as e:
            # Don't crash extraction; report warning
            pass

    _assign_room_labels_from_text(doc, rooms)
    out = {
        "meta": {
            "source": "dxf",
            "units_guess": units_guess,
            "bbox": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
            "cad_grid": cad_grid,
        },
        "scale": {"detected": True, "pxPerMeter": float(scale / unit_to_m), "method": "dxf_units_guess", "units_guess": units_guess},
        "rooms": rooms,
        "openings": [],
        "lights": lights,
        "geojson": {"type": "FeatureCollection", "features": []},
        "warnings": [],
    }
    if dxf_out_b64:
        out["dxf_out_base64"] = dxf_out_b64
    return out


def _place_lights_from_rooms(rooms: List[Dict[str, Any]], opts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Place lights deterministically based on room area and polygon.
    Uses polygon_m coordinates (meters).
    """
    _require("numpy")
    import numpy as np

    # Defaults from ENV or opts
    default_type = (opts.get("fixture_type") or os.getenv("LIGHT_DEFAULT_FIXTURE_TYPE") or "DOWNLIGHT").strip()
    lumens = int(opts.get("lumens") or os.getenv("LIGHT_DEFAULT_LUMENS") or 800)
    watt = float(opts.get("watt") or os.getenv("LIGHT_DEFAULT_WATT") or 8)
    cct = int(opts.get("cct_k") or os.getenv("LIGHT_DEFAULT_CCT") or 3000)
    cri = int(opts.get("cri") or os.getenv("LIGHT_DEFAULT_CRI") or 80)
    min_off = float(opts.get("min_wall_offset_m") or os.getenv("LIGHT_MIN_WALL_OFFSET_M") or 0.6)
    grid = float(opts.get("grid_spacing_m") or os.getenv("LIGHT_GRID_SPACING_M") or 3.0)

    def centroid(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    def point_in_poly(x, y, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]
            x2,y2 = poly[(i+1)%n]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
                inside = not inside
        return inside

    def bbox(poly):
        xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    lights=[]
    for r in rooms:
        poly = r.get("polygon_m") or []
        if len(poly) < 3:
            continue
        area = float(r.get("area_m2") or 0.0)
        rid = r.get("id")
        # Determine count rule
        if area < 12.0:
            nlights = 1
            rule = "CENTROID"
        elif area < 25.0:
            nlights = 2
            rule = "AXIS2"
        else:
            # grid
            rule = "GRID"
            nlights = 0

        xmin,ymin,xmax,ymax = bbox(poly)
        cx,cy = centroid(poly)

        pts=[]
        if rule=="CENTROID":
            pts=[(cx,cy)]
        elif rule=="AXIS2":
            # place along major axis of bbox
            dx = xmax-xmin
            dy = ymax-ymin
            if dx >= dy:
                pts=[(xmin+dx*0.33, cy),(xmin+dx*0.67, cy)]
            else:
                pts=[(cx, ymin+dy*0.33),(cx, ymin+dy*0.67)]
        else:
            # grid sampling inside bbox
            xs = np.arange(xmin+min_off, xmax-min_off+1e-9, grid)
            ys = np.arange(ymin+min_off, ymax-min_off+1e-9, grid)
            for x in xs:
                for y in ys:
                    if point_in_poly(float(x), float(y), poly):
                        pts.append((float(x), float(y)))
            # fallback if none
            if not pts:
                pts=[(cx,cy)]
                rule="CENTROID"

        for j,(x,y) in enumerate(pts):
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
                "spacing_m": grid if rule=="GRID" else None,
                "offset_wall_m": min_off,
                "notes": "",
            })

    return lights

def _dominant_wall_segment(poly: List[Tuple[float,float]]) -> Optional[Tuple[Tuple[float,float], Tuple[float,float]]]:
    """Return the longest edge (p1,p2) of polygon as dominant wall segment."""
    if not poly or len(poly) < 3:
        return None
    best = None
    best_len = -1.0
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        L = math.hypot(x2-x1, y2-y1)
        if L > best_len:
            best_len = L
            best = ((float(x1),float(y1)),(float(x2),float(y2)))
    return best

def _points_along_segment(p1: Tuple[float,float], p2: Tuple[float,float], step: float) -> List[Tuple[float,float]]:
    x1,y1=p1; x2,y2=p2
    L = math.hypot(x2-x1, y2-y1)
    if L <= 1e-6 or step <= 0:
        return [p1]
    n = max(1, int(L/step))
    pts=[]
    for i in range(n+1):
        t=i/n
        pts.append((x1+(x2-x1)*t, y1+(y2-y1)*t))
    return pts

def _segment_normal_inward(poly: List[Tuple[float,float]], p1: Tuple[float,float], p2: Tuple[float,float]) -> Tuple[float,float]:
    x1,y1=p1; x2,y2=p2
    dx,dy = (x2-x1),(y2-y1)
    nx1,ny1 = -dy, dx
    nx2,ny2 = dy, -dx
    xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
    cx=sum(xs)/len(xs); cy=sum(ys)/len(ys)
    mx,my = (x1+x2)/2,(y1+y2)/2
    vcx, vcy = (cx-mx),(cy-my)
    if nx1*vcx + ny1*vcy >= nx2*vcx + ny2*vcy:
        nx,ny = nx1,ny1
    else:
        nx,ny = nx2,ny2
    norm = math.hypot(nx,ny) or 1.0
    return (nx/norm, ny/norm)

def _estimate_vertical_illuminance(room: Dict[str,Any], lights: List[Dict[str,Any]], wall_seg: Optional[Tuple[Tuple[float,float],Tuple[float,float]]]) -> Optional[Dict[str,Any]]:
    if not wall_seg:
        return None
    poly = room.get("polygon_m") or []
    if not poly or len(poly) < 3:
        return None
    (p1,p2) = wall_seg
    pts = _points_along_segment(p1,p2,0.8)
    nx,ny = _segment_normal_inward(poly,p1,p2)
    H = float(room.get("height_m") or ROOM_HEIGHT_M_DEFAULT)
    z_light = max(2.0, H-0.1)
    z_wall = 1.5
    evs=[]
    for (xw,yw) in pts:
        Ev=0.0
        for lt in lights:
            try:
                if str(lt.get("room_id") or "") != str(room.get("id") or ""):
                    continue
                xl=float(lt.get("x_m")); yl=float(lt.get("y_m"))
                lm=float(lt.get("lumens") or 0.0)
                if lm <= 0:
                    continue
                dx = xw-xl; dy = yw-yl; dz = z_wall - z_light
                r2 = dx*dx+dy*dy+dz*dz
                if r2 <= 1e-6:
                    continue
                r = math.sqrt(r2)
                ux,uy,uz = dx/r, dy/r, dz/r
                cos_theta = max(0.0, (ux*nx + uy*ny))
                cos_gamma = max(0.0, -uz)
                I = lm/(4*math.pi) * (0.6 + 0.4*cos_gamma)
                Ev += (I * cos_theta) / r2
            except Exception:
                continue
        evs.append(Ev)
    if not evs:
        return None
    return {"Ev_avg": float(sum(evs)/len(evs)), "samples": int(len(evs))}

def _dominant_axis_angle(poly: List[Tuple[float,float]]) -> float:
    if not poly or len(poly) < 3:
        return 0.0
    angles=[]
    n=len(poly)
    for i in range(n):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
        dx,dy=(x2-x1),(y2-y1)
        L=math.hypot(dx,dy)
        if L < 1e-6:
            continue
        a=math.atan2(dy,dx)
        while a < 0: a += math.pi
        while a >= math.pi: a -= math.pi
        angles.append((a,L))
    if not angles:
        return 0.0
    bins=18
    acc=[0.0]*bins
    for a,w in angles:
        b=int((a/math.pi)*bins) % bins
        acc[b]+=w
    best=max(range(bins), key=lambda i: acc[i])
    return (best+0.5)*(math.pi/bins)

def _rotate(x: float, y: float, ang: float) -> Tuple[float,float]:
    ca=math.cos(ang); sa=math.sin(ang)
    return (x*ca - y*sa, x*sa + y*ca)

def _snap_to_grid(points: List[Tuple[float,float]], grid: float) -> List[Tuple[float,float]]:
    if grid <= 0:
        return points
    return [(round(x/grid)*grid, round(y/grid)*grid) for (x,y) in points]

def _align_room_lights_to_axis(extracted: Dict[str,Any], opts: Dict[str,Any]) -> Dict[str,Any]:
    if not isinstance(extracted, dict):
        return extracted
    if not bool(opts.get("align_to_axis", True)):
        return extracted
    grid = float(opts.get("axis_grid_m") or 0.20)
    rooms = extracted.get("rooms") or []
    lights = extracted.get("lights") or []
    if not rooms or not lights:
        return extracted

    by_room={}
    for lt in lights:
        by_room.setdefault(str(lt.get("room_id") or ""), []).append(lt)

    for room in rooms:
        rid=str(room.get("id") or "")
        poly = room.get("polygon_m") or []
        if not poly or rid not in by_room:
            continue
        ang = _dominant_axis_angle(poly)
        for lt in by_room[rid]:
            try:
                x=float(lt.get("x_m")); y=float(lt.get("y_m"))
                xr,yr=_rotate(x,y,-ang)
                xr,yr=_snap_to_grid([(xr,yr)], grid)[0]
                x2,y2=_rotate(xr,yr,ang)
                if _pip(x2,y2,poly):
                    lt["x_m"]=float(x2); lt["y_m"]=float(y2)
                    lt["aligned_axis_rad"]=float(ang)
            except Exception:
                continue
        try:
            room["axis_angle_rad"]=float(ang)
        except Exception:
            pass
    return extracted

# ==========================================================
# LuxIA Designer Layer (controlled artistry)
# ==========================================================
def _lighting_profile(mode: str) -> Dict[str, Any]:
    m = (mode or "").strip().lower()
    if m in ("corporate_premium","premium","corporate"):
        return {
            "mode": "corporate_premium",
            "cct_k": 3000,
            "cri": 80,
            "perimeter_wash": True,
            "perimeter_offset_m": 0.55,
            "perimeter_spacing_m": 3.8,
            "perimeter_lumens_factor": 0.60,
            "accent": True,
            "accent_count": 2,
            "accent_lumens_factor": 0.55,
            "accent_beam_deg": 36,
            "contrast_target": 1.35,  # soft hierarchy
            "vertical_emphasis": True,
            "main_wall_only": True,
            "vertical_boost_factor": 1.35,
        }
    if m in ("architectural_minimal","minimal"):
        return {
            "mode": "architectural_minimal",
            "cct_k": 3000,
            "cri": 80,
            "perimeter_wash": True,
            "perimeter_offset_m": 0.60,
            "perimeter_spacing_m": 4.2,
            "perimeter_lumens_factor": 0.55,
            "accent": False,
            "accent_count": 0,
            "accent_lumens_factor": 0.0,
            "accent_beam_deg": 60,
            "contrast_target": 1.20,
            "vertical_emphasis": True,
            "main_wall_only": True,
            "vertical_boost_factor": 1.25,
        }
    if m in ("retail_contrast","retail"):
        return {
            "mode": "retail_contrast",
            "cct_k": 3500,
            "cri": 90,
            "perimeter_wash": True,
            "perimeter_offset_m": 0.50,
            "perimeter_spacing_m": 3.0,
            "perimeter_lumens_factor": 0.75,
            "accent": True,
            "accent_count": 3,
            "accent_lumens_factor": 0.90,
            "accent_beam_deg": 24,
            "contrast_target": 1.80,
            "vertical_emphasis": True,
            "main_wall_only": True,
            "vertical_boost_factor": 1.70,
        }
    if m in ("residential_soft","residential","home"):
        return {
            "mode": "residential_soft",
            "cct_k": 2700,
            "cri": 90,
            "perimeter_wash": False,
            "perimeter_offset_m": 0.0,
            "perimeter_spacing_m": 0.0,
            "perimeter_lumens_factor": 0.0,
            "accent": True,
            "accent_count": 1,
            "accent_lumens_factor": 0.55,
            "accent_beam_deg": 36,
            "contrast_target": 1.45,
        }
    return {"mode":"neutral","cct_k":3000,"cri":80,"perimeter_wash":False,"accent":False,"accent_count":0,"contrast_target":1.0,"perimeter_offset_m":0.0,"perimeter_spacing_m":0.0,"perimeter_lumens_factor":0.0,"accent_lumens_factor":0.0,"accent_beam_deg":60}

def _pip(x: float, y: float, poly: List[Tuple[float,float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1):
            inside = not inside
    return inside

def _perimeter_points(poly: List[Tuple[float,float]], offset: float, spacing: float) -> List[Tuple[float,float]]:
    """Generate interior points near the perimeter using bbox edges + point-in-poly."""
    if not poly or len(poly) < 3 or spacing <= 0:
        return []
    xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
    minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
    pts=[]
    # along bottom/top
    x=minx+offset
    while x <= maxx-offset+1e-9:
        for y in (miny+offset, maxy-offset):
            if _pip(x,y,poly):
                pts.append((float(x), float(y)))
        x += spacing
    # along left/right
    y=miny+offset
    while y <= maxy-offset+1e-9:
        for x in (minx+offset, maxx-offset):
            if _pip(x,y,poly):
                pts.append((float(x), float(y)))
        y += spacing
    # de-dup (grid tolerance)
    out=[]
    seen=set()
    for x,y in pts:
        k=(round(x,2), round(y,2))
        if k not in seen:
            seen.add(k); out.append((x,y))
    return out

def _feature_points(room: Dict[str, Any], count: int) -> List[Tuple[float,float]]:
    poly = room.get("polygon_m") or []
    if not poly or len(poly) < 3 or count <= 0:
        return []
    xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
    cx=sum(xs)/len(xs); cy=sum(ys)/len(ys)
    pts=[(float(cx), float(cy))]
    if count <= 1:
        return pts
    # add two points along longest bbox axis (soft hierarchy)
    minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
    if (maxx-minx) >= (maxy-miny):
        pts.append((float((minx+cx)/2), float(cy)))
        if count >= 3:
            pts.append((float((maxx+cx)/2), float(cy)))
    else:
        pts.append((float(cx), float((miny+cy)/2)))
        if count >= 3:
            pts.append((float(cx), float((maxy+cy)/2)))
    # ensure inside
    pts2=[]
    for x,y in pts:
        if _pip(x,y,poly):
            pts2.append((x,y))
    return pts2[:count]

def _accent_policy_for_room(room: Dict[str,Any], prof: Dict[str,Any]) -> Dict[str,Any]:
    label = str(room.get("label") or "").upper()
    if any(k in label for k in ("RECEPTION","INGRESSO","LOBBY")):
        return {"accent_count": max(2, int(prof.get("accent_count") or 2)), "beam_deg": 24}
    if any(k in label for k in ("SALA RIUNIONI","MEETING","CONFERENCE")):
        return {"accent_count": max(3, int(prof.get("accent_count") or 2)), "beam_deg": 36}
    if any(k in label for k in ("TAVOLO","ISOLA","BANCO")):
        return {"accent_count": max(2, int(prof.get("accent_count") or 1)), "beam_deg": 24}
    return {"accent_count": int(prof.get("accent_count") or 0), "beam_deg": int(prof.get("accent_beam_deg") or 36)}

def _rhythm_step(target_step: float, base_module: float, modules: List[int]) -> float:
    """Return step snapped to base_module * (3/5/7...) close to target_step."""
    try:
        base = float(base_module)
        if base <= 0:
            return float(target_step)
        cand=[]
        for m in modules:
            cand.append(base*float(m))
        best=min(cand, key=lambda s: abs(s-float(target_step))) if cand else float(target_step)
        return float(best)
    except Exception:
        return float(target_step)

def _apply_lighting_intent(extracted: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    """Add controlled designer layers (perimeter wash + accents) without breaking compliance."""
    if not isinstance(extracted, dict):
        return extracted
    rooms = extracted.get("rooms") or []
    lights = extracted.get("lights") or []
    if not rooms:
        return extracted

    mode = str(opts.get("lighting_mode") or os.getenv("LIGHTING_MODE") or "corporate_premium")
    prof = _lighting_profile(mode)

    # rhythm / module
    modules = list(opts.get("rhythm_modules") or [3,5,7])
    base_module = float(((extracted.get("meta") or {}).get("cad_grid") or {}).get("spacing_m") or 0.6)

    # default photometric defaults
    base_lm = int(opts.get("lumens") or os.getenv("LIGHT_DEFAULT_LUMENS") or 800)
    base_w = float(opts.get("watt") or os.getenv("LIGHT_DEFAULT_WATT") or 8)
    cct = int(prof.get("cct_k") or opts.get("cct_k") or os.getenv("LIGHT_DEFAULT_CCT") or 3000)
    cri = int(prof.get("cri") or opts.get("cri") or os.getenv("LIGHT_DEFAULT_CRI") or 80)

    added=[]
    for room in rooms:
        rid=str(room.get("id") or "")
        poly = room.get("polygon_m") or []
        if not poly or len(poly) < 3:
            continue

        # Perimeter wash
        # snap spacing to module rhythm (3/5/7) when grid spacing exists
        try:
            if base_module and float(base_module) > 0 and float(prof.get("perimeter_spacing_m") or 0) > 0:
                prof["perimeter_spacing_m"] = _rhythm_step(float(prof.get("perimeter_spacing_m")), base_module, modules)
        except Exception:
            pass

        if bool(prof.get("perimeter_wash")):
            wall_seg = _dominant_wall_segment(poly) if bool(prof.get("main_wall_only", False)) else None
            if wall_seg:
                (wp1,wp2)=wall_seg
                nx,ny = _segment_normal_inward(poly, wp1, wp2)
                pts = _points_along_segment(wp1, wp2, float(prof.get("perimeter_spacing_m")))
                off=float(prof.get("perimeter_offset_m"))
                pts=[(x+nx*off, y+ny*off) for (x,y) in pts if _pip(x+nx*off,y+ny*off,poly)]
            else:
                pts = _perimeter_points(poly, float(prof.get("perimeter_offset_m")), float(prof.get("perimeter_spacing_m")))
            for j,(x,y) in enumerate(pts):
                added.append({
                    "id": f"L_{rid}_P{j+1:02d}",
                    "room_id": rid,
                    "x_m": x, "y_m": y,
                    "fixture_type": "WALLWASH_MAIN" if bool(prof.get("main_wall_only", False)) else "WALLWASH",
                    "layer": "VERTICAL" if bool(prof.get("main_wall_only", False)) else "PERIMETER",
                    "lumens": int(base_lm * float(prof.get("perimeter_lumens_factor"))),
                    "watt": float(base_w * float(prof.get("perimeter_lumens_factor"))),
                    "cct_k": cct,
                    "cri": cri,
                    "beam_deg": 90,
                    "placement_rule": "PERIMETER",
                    "spacing_m": float(prof.get("perimeter_spacing_m")),
                    "offset_wall_m": float(prof.get("perimeter_offset_m")),
                    "notes": f"Designer:{prof.get('mode')}",
                })

        # Accents (feature points)
        apol = _accent_policy_for_room(room, prof)
        if bool(prof.get("accent")) and int(apol.get("accent_count") or 0) > 0:
            pts = _feature_points(room, int(apol.get("accent_count")))
            for j,(x,y) in enumerate(pts):
                added.append({
                    "id": f"L_{rid}_A{j+1:02d}",
                    "room_id": rid,
                    "x_m": x, "y_m": y,
                    "fixture_type": "SPOT",
                    "layer": "ACCENT",
                    "lumens": int(base_lm * float(prof.get("accent_lumens_factor"))),
                    "watt": float(base_w * float(prof.get("accent_lumens_factor"))),
                    "cct_k": cct,
                    "cri": cri,
                    "beam_deg": int(apol.get("beam_deg") or 36),
                    "placement_rule": "ACCENT",
                    "spacing_m": None,
                    "offset_wall_m": None,
                    "notes": f"Designer:{prof.get('mode')}",
                })

# Vertical emphasis: ensure dominant wall has presence (adds extra wallwash if needed)
if bool(prof.get("vertical_emphasis")):
    wall_seg2 = _dominant_wall_segment(poly)
    ev = _estimate_vertical_illuminance(room, (lights + added), wall_seg2) if wall_seg2 else None
    try:
        tgt = float((_get_uni_targets(room).get("lux") or 0.0))
    except Exception:
        tgt = 0.0
    ev_target = (tgt/float(prof.get("vertical_boost_factor") or 1.35)) if tgt>0 else 0.0
    if ev and ev_target>0 and float(ev.get("Ev_avg") or 0.0) < ev_target:
        (wp1,wp2)=wall_seg2
        nx,ny = _segment_normal_inward(poly, wp1, wp2)
        pts2 = _points_along_segment(wp1, wp2, max(1.6, float(prof.get("perimeter_spacing_m"))))
        off=float(prof.get("perimeter_offset_m"))
        pts2=[(x+nx*off, y+ny*off) for (x,y) in pts2 if _pip(x+nx*off,y+ny*off,poly)]
        for j,(x,y) in enumerate(pts2[:12]):
            added.append({
                "id": f"L_{rid}_V{j+1:02d}",
                "room_id": rid,
                "x_m": x, "y_m": y,
                "fixture_type": "WALLWASH_MAIN",
                "layer": "VERTICAL",
                "lumens": int(base_lm * 0.55),
                "watt": float(base_w * 0.55),
                "cct_k": cct,
                "cri": cri,
                "beam_deg": 90,
                "placement_rule": "VERTICAL_BOOST",
                "spacing_m": max(1.6, float(prof.get("perimeter_spacing_m"))),
                "offset_wall_m": off,
                "notes": f"Designer:{prof.get('mode')}",
            })
        try:
            room["vertical_ev"] = ev
            room["vertical_ev_target"] = ev_target
        except Exception:
            pass

        # store profile on room for reporting
        try:
            room["lighting_mode"] = prof.get("mode")
        except Exception:
            pass

    if added:
        extracted["lights"] = lights + added
        extracted.setdefault("meta", {})["lighting_mode"] = prof.get("mode")
        extracted.setdefault("meta", {})["designer_layers_added"] = True

    return extracted

def _rebuild_room_lights(room: Dict[str, Any], opts: Dict[str, Any], grid_spacing_m: float) -> List[Dict[str, Any]]:
    """Rebuild lights for a single room with a custom spacing."""
    _require("numpy")
    import numpy as np

    rid = str(room.get("id") or "R000")
    default_type = (opts.get("fixture_type") or os.getenv("LIGHT_DEFAULT_FIXTURE_TYPE") or "DOWNLIGHT").strip()
    lumens = int(opts.get("lumens") or os.getenv("LIGHT_DEFAULT_LUMENS") or 800)
    watt = float(opts.get("watt") or os.getenv("LIGHT_DEFAULT_WATT") or 8)
    cct = int(opts.get("cct_k") or os.getenv("LIGHT_DEFAULT_CCT") or 3000)
    cri = int(opts.get("cri") or os.getenv("LIGHT_DEFAULT_CRI") or 80)
    min_off = float(opts.get("min_wall_offset_m") or os.getenv("LIGHT_MIN_WALL_OFFSET_M") or 0.6)
    grid = float(grid_spacing_m)

    poly = room.get("polygon_m") or []
    if not poly or len(poly) < 3:
        return []

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))

    def pip(x, y, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]
            x2,y2 = poly[(i+1)%n]
            if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1):
                inside = not inside
        return inside

    gx = np.arange(minx + min_off, maxx - min_off + 1e-9, grid)
    gy = np.arange(miny + min_off, maxy - min_off + 1e-9, grid)
    pts = [(float(x), float(y)) for y in gy for x in gx if pip(float(x), float(y), poly)]
    rule = "GRID"
    if not pts:
        cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
        pts = [(float(cx), float(cy))]
        rule = "CENTER"

    lights=[]
    for j,(x,y) in enumerate(pts):
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
            "spacing_m": grid if rule=="GRID" else None,
            "offset_wall_m": min_off,
            "notes": "",
        })
    return lights


def _autofix_uni_to_ok(extracted: Dict[str, Any],
                       opts: Dict[str, Any],
                       phot: Optional[Dict[str, Any]],
                       checks: List[Dict[str, Any]],
                       max_iters: int = 6) -> Dict[str, Any]:
    """Autofix to reach UNI targets:
    - If Eavg < target: densify grid (reduce spacing)
    - If U0 < target: densify + reduce wall offset
    - If UGR > max: reduce lumens per fixture (and densify if needed) within caps
    """
    if not isinstance(extracted, dict):
        return extracted
    rooms = extracted.get("rooms") or []
    if not rooms:
        return extracted

    base_grid = float(opts.get("grid_spacing_m") or os.getenv("LIGHT_GRID_SPACING_M") or 3.0)
    min_grid = float(opts.get("min_grid_spacing_m") or 1.0)
    max_lights_room = int(opts.get("max_lights_per_room") or 160)
    base_lm = int(opts.get("lumens") or os.getenv("LIGHT_DEFAULT_LUMENS") or 800)
    base_w = float(opts.get("watt") or os.getenv("LIGHT_DEFAULT_WATT") or 8)
    min_lm = int(opts.get("min_lumens_per_fixture") or 350)

    any_changed = False

    for room in rooms:
        rid = str(room.get("id") or "")
        targets = _get_uni_targets(room)
        tgt_lux = float(targets.get("lux") or 0.0)
        tgt_u0 = float(targets.get("u0") or 0.0)
        tgt_ugr = targets.get("ugr_max", None)

        if tgt_lux <= 0:
            continue

        # start from current
        grid = base_grid
        wall_off = float(opts.get("min_wall_offset_m") or os.getenv("LIGHT_MIN_WALL_OFFSET_M") or 0.6)
        lumens_pf = base_lm
        watt_pf = base_w

        for _ in range(max_iters):
            # compute metrics + ugr for this room
            metrics = _calc_room_grid_metrics(room, extracted.get("lights", []), phot)
            # estimate UGR with current policy (table preferred)
            sel_key = room.get("luminaire_key") or _select_luminaire_key(room) or "default"
            phot_room = _LUMINAIRE_LIB.get(str(sel_key)) or phot
            dims = _room_bbox_dims_m(room, None)
            H = float(room.get("height_m") or ROOM_HEIGHT_M_DEFAULT)
            Hm = max(0.3, H - WORKPLANE_HEIGHT_M_DEFAULT)
            K = _room_index_K(float(dims.get("L") or 0.0), float(dims.get("W") or 0.0), float(Hm))
            rho_avg = float(0.5*(room.get('rho_c') or CEILING_REFLECTANCE_DEFAULT) + 0.5*(room.get('rho_w') or WALL_REFLECTANCE_DEFAULT))
            ugr = _ugr_from_table(str(sel_key), K, rho_avg) or _ugr_approx(room, phot_room)

            st = _status_vs_targets(metrics, targets, ugr)
            if st.get("status") == "OK":
                break

            eavg = float(metrics.get("Eavg") or 0.0)
            u0 = float(metrics.get("U0") or 0.0)

            need_e = ("Eavg" in st.get("reasons", []))
            need_u0 = ("U0" in st.get("reasons", []))
            need_ugr = ("UGR" in st.get("reasons", []))

            # 1) If UGR too high: reduce lumens per fixture (and power proportionally)
            if need_ugr and lumens_pf > min_lm:
                lumens_pf = max(min_lm, int(lumens_pf * 0.80))
                watt_pf = max(1.0, watt_pf * (lumens_pf / max(base_lm, 1)))
                opts["lumens"] = lumens_pf
                opts["watt"] = watt_pf

            # 2) Densify when Eavg is low or we reduced lumens
            ratio = 1.0
            if tgt_lux > 0:
                ratio = 4.0 if eavg <= 1e-6 else max(1.05, tgt_lux / max(eavg, 1e-6))
            if need_e or need_u0 or need_ugr:
                new_grid = max(min_grid, grid / math.sqrt(ratio))
                if abs(new_grid - grid) < 0.05:
                    new_grid = max(min_grid, grid * 0.88)
                grid = new_grid

            # 3) Improve uniformity by reducing wall offset a bit (helps perimeter points)
            if need_u0:
                wall_off = max(0.3, wall_off * 0.85)
                opts["min_wall_offset_m"] = wall_off

            new_lights = _rebuild_room_lights(room, opts, grid)
            if len(new_lights) > max_lights_room:
                new_lights = new_lights[:max_lights_room]

            extracted["lights"] = [lt for lt in extracted.get("lights", []) if str(lt.get("room_id") or "") != rid] + new_lights
            any_changed = True

        # attach per-room final targets/metrics status to room for later reporting
        try:
            room["uni_targets"] = targets
        except Exception:
            pass

    if any_changed:
        checks.append({"code":"UNI_AUTOFIX_PLUS", "detail":"UNI KO (lux/U0/UGR): LuxIA ha ottimizzato automaticamente (densità + offset + lumens) per raggiungere la conformità entro limiti."})
    return extracted
    rooms = extracted.get("rooms") or []
    if not rooms:
        return extracted

    base_grid = float(opts.get("grid_spacing_m") or os.getenv("LIGHT_GRID_SPACING_M") or 3.0)
    min_grid = float(opts.get("min_grid_spacing_m") or 1.2)
    max_lights_room = int(opts.get("max_lights_per_room") or 120)

    any_changed = False

    for room in rooms:
        rid = str(room.get("id") or "")
        targets = _get_uni_targets(room)
        tgt = float(targets.get("lux") or 0.0)
        if tgt <= 0:
            continue

        grid = base_grid
        for _ in range(max_iters):
            metrics = _calc_room_grid_metrics(room, extracted.get("lights", []), phot)
            status = str(metrics.get("status") or "")
            eavg = float(metrics.get("E_avg") or 0.0)
            if status == "OK":
                break

            ratio = 4.0 if eavg <= 1e-6 else max(1.05, tgt / max(eavg, 1e-6))
            new_grid = max(min_grid, grid / math.sqrt(ratio))
            if abs(new_grid - grid) < 0.05:
                new_grid = max(min_grid, grid * 0.85)
            grid = new_grid

            new_lights = _rebuild_room_lights(room, opts, grid)
            if len(new_lights) > max_lights_room:
                new_lights = new_lights[:max_lights_room]

            extracted["lights"] = [lt for lt in extracted.get("lights", []) if str(lt.get("room_id") or "") != rid] + new_lights
            any_changed = True

    if any_changed:
        checks.append({"code":"UNI_AUTOFIX", "detail":"UNI KO: LuxIA ha densificato automaticamente la griglia per raggiungere i target (entro limiti)."})
    return extracted
def _cad_layer_for_light(light: Dict[str,Any]) -> str:
    lyr = str(light.get("layer") or "").upper()
    if lyr == "ACCENT":
        return "ILLUMINAZIONE_ACCENT"
    if lyr == "VERTICAL":
        return "ILLUMINAZIONE_VERTICAL"
    if lyr == "PERIMETER":
        return "ILLUMINAZIONE_PERIMETER"
    return "ILLUMINAZIONE"



def _write_lights_to_dxf(doc, lights: List[Dict[str, Any]], units_guess: str = "m") -> bytes:
    """
    Writes lamp blocks with attributes into the DXF doc:
      - Ensures layer ILLUMINAZIONE exists
      - Ensures block LIGHT_BLOCK_NAME exists with ATTRIB defs
      - Inserts one block per light with populated attributes
    Returns DXF bytes.
    """
    _require("ezdxf")
    import ezdxf
    from io import StringIO

    layer_name = (os.getenv("LIGHT_LAYER_NAME") or "ILLUMINAZIONE").strip()
    block_name = (os.getenv("LIGHT_BLOCK_NAME") or "LUXIA_LIGHT_POINT").strip()

    # Ensure layer
    if layer_name not in doc.layers:
        doc.layers.new(layer_name)

    # Ensure block
    if block_name not in doc.blocks:
        blk = doc.blocks.new(name=block_name)
        # simple symbol: small circle cross
        blk.add_circle((0,0), radius=0.05)
        blk.add_line((-0.08,0), (0.08,0))
        blk.add_line((0,-0.08), (0,0.08))

        # Attribute definitions (positions relative to insert)
        # tag, insert, height
        blk.add_attdef("LUXIA_ID", (0.10, 0.10), height=0.08)
        blk.add_attdef("ROOM_ID", (0.10, 0.00), height=0.08)
        blk.add_attdef("ROOM_NAME", (0.10,-0.10), height=0.08)
        blk.add_attdef("FIXTURE_TYPE", (0.10,-0.20), height=0.08)
        blk.add_attdef("LUMENS", (0.10,-0.30), height=0.08)
        blk.add_attdef("WATT", (0.10,-0.40), height=0.08)
        blk.add_attdef("CCT_K", (0.10,-0.50), height=0.08)
        blk.add_attdef("CRI", (0.10,-0.60), height=0.08)
        blk.add_attdef("BEAM_DEG", (0.10,-0.70), height=0.08)
        blk.add_attdef("PLACEMENT_RULE", (0.10,-0.80), height=0.08)
        blk.add_attdef("SPACING_M", (0.10,-0.90), height=0.08)
        blk.add_attdef("OFFSET_WALL_M", (0.10,-1.00), height=0.08)
        blk.add_attdef("LEVEL", (0.10,-1.10), height=0.08)
        blk.add_attdef("NOTES", (0.10,-1.20), height=0.08)

    # Units conversion
    m_to_unit = 1.0 if units_guess == "m" else 1000.0  # meters->mm

    msp = doc.modelspace()
    for lt in lights:
        x = float(lt.get("x_m", 0.0)) * m_to_unit
        y = float(lt.get("y_m", 0.0)) * m_to_unit
        ins = msp.add_blockref(block_name, (x, y), dxfattribs={"layer": layer_name})
        # Populate attribs
        def sval(v):
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)

        attrs = {
            "LUXIA_ID": sval(lt.get("id")),
            "ROOM_ID": sval(lt.get("room_id")),
            "ROOM_NAME": sval(lt.get("room_name")),
            "FIXTURE_TYPE": sval(lt.get("fixture_type")),
            "LUMENS": sval(lt.get("lumens")),
            "WATT": sval(lt.get("watt")),
            "CCT_K": sval(lt.get("cct_k")),
            "CRI": sval(lt.get("cri")),
            "BEAM_DEG": sval(lt.get("beam_deg")),
            "PLACEMENT_RULE": sval(lt.get("placement_rule")),
            "SPACING_M": sval(lt.get("spacing_m")),
            "OFFSET_WALL_M": sval(lt.get("offset_wall_m")),
            "LEVEL": sval(lt.get("level") or 0),
            "NOTES": sval(lt.get("notes")),
        }
        try:
            ins.add_auto_attribs(attrs)
        except Exception:
            # Older ezdxf: fallback to add_attrib
            for tag, val in attrs.items():
                ins.add_attrib(tag, val, insert=(x, y))

    # Save DXF to bytes
    sio = StringIO()
    doc.write(sio)
    return sio.getvalue().encode("utf-8", errors="ignore")


def _extract_stub_for_now(raster_png: bytes, input_type: str, raw_bytes: Optional[bytes] = None, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    opts = opts or {}
    if input_type in ("dxf", "dwg") and raw_bytes:
        # For DWG, raw_bytes should already be DXF bytes.
        return _extract_from_dxf_real(raw_bytes, raster_png, opts)
    return _extract_from_raster_stub(raster_png, input_type)


@app.post("/planimetry/analyze", response_model=PlanimetryResponse)
async def planimetry_analyze(
    file: UploadFile = File(...),
    options: str = Form("{}"),
):
    """Accepts PDF/JPG/PNG/DXF/DWG and returns extracted geometry + strict verification."""

    raw = await file.read()
    if not raw or len(raw) < 16:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    input_type = _guess_type(file.filename or "", file.content_type or "")

    # Parse options
    try:
        _opts = json.loads(options or "{}")
    except Exception:
        _opts = {}

    dxf_bytes = None

    # Normalize to a single PNG for Vision verification
    page_count = 1
    if input_type == "pdf":
        raster_png = _pdf_to_png_first_page(raw)
    elif input_type == "image":
        raster_png = _image_to_png(raw)
    elif input_type == "dxf":
        dxf_bytes = raw
        raster_png = _dxf_to_png(dxf_bytes)
    elif input_type == "dwg":
        # Convert via configured converter (DWG2DXF_URL)
        dxf_bytes = _dwg_to_dxf(raw, file.filename or "drawing.dwg")
        raster_png = _dxf_to_png(dxf_bytes)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, JPG/PNG, DXF, or DWG.")

    # Extraction (current pipeline uses stub over raster; DXF bytes passed when available)
    extracted = _extract_stub_for_now(
        raster_png,
        input_type,
        raw_bytes=(dxf_bytes if dxf_bytes is not None else (raw if input_type == "dxf" else None)),
        opts=_opts,
    )
# QA_UNITS_LOOP: try alternative DXF units if confidence is low
if input_type in ("dxf", "dwg") and isinstance(extracted, dict):
    ui = extracted.get("unit_info") or {}
    conf = float(ui.get("confidence") or 0.0)
    if conf < 0.95:
        candidates = [0.001, 0.01, 1.0, 0.3048, 0.0254]
        best = extracted
        best_score = _score_extraction_units(extracted)
        cur_mpu = float(ui.get("meters_per_unit") or 1.0)
        for mpu in candidates:
            if abs(mpu - cur_mpu) < 1e-12:
                continue
            try:
                trial_opts = dict(_opts)
                trial_opts["meters_per_unit_override"] = mpu
                trial = _extract_stub_for_now(raster_png, input_type, raw_bytes=dxf_bytes, opts=trial_opts)
                s = _score_extraction_units(trial)
                if s > best_score + 0.5:
                    best, best_score = trial, s
            except Exception:
                continue
        if best is not extracted:
            extracted = best
            checks.append({"code":"UNITS_AUTOFIX", "detail":"Unità DXF autoselezionate tra ipotesi (mm/cm/m/ft/in) per massima plausibilità."})

    # Deterministic checks (must pass)
    checks = _deterministic_checks(extracted)

    # Vision "prova del nove": cross-check extracted JSON against the raster
    overlay_png = _make_overlay_png(raster_png, extracted)
    vres = vision.verify(raster_png, extracted, overlay_png=overlay_png)

    # LuxIA PRO: grid lux + UGR (if rooms + lights exist)
    lum_key = str(_opts.get("luminaire_key") or "default")
    phot = _LUMINAIRE_LIB.get(lum_key)

# Auto-fix UNI: se KO, corregge il layout (densifica la griglia) fino a target
if bool(_opts.get("autofix_uni", True)) and isinstance(extracted, dict) and extracted.get("rooms") and extracted.get("lights"):
    try:
        extracted = _autofix_uni_to_ok(extracted, _opts, phot, checks)
    except Exception:
        pass

# Designer layer: add controlled perimeter wash + accents, then re-check compliance
if bool(_opts.get("designer_mode", True)):
    try:
        extracted = _apply_lighting_intent(extracted, _opts)
        extracted = _align_room_lights_to_axis(extracted, _opts)
    except Exception:
        pass

# Post-intent safety: keep UNI OK after artistic layers
if bool(_opts.get("autofix_uni_post_intent", True)) and isinstance(extracted, dict) and extracted.get("rooms") and extracted.get("lights"):
    try:
        extracted = _autofix_uni_to_ok(extracted, _opts, phot, checks)
    except Exception:
        pass

    lux_results = []
    if isinstance(extracted, dict) and extracted.get("rooms") and extracted.get("lights"):
        for room in extracted.get("rooms", []):
            targets = _get_uni_targets(room)
            metrics = _calc_room_grid_metrics(room, extracted.get("lights", []), phot)
            sel_key = room.get('luminaire_key') or _select_luminaire_key(room) or lum_key
            phot_room = _LUMINAIRE_LIB.get(str(sel_key)) or phot
            dims = _room_bbox_dims_m(room, None)
            H = float(room.get('height_m') or ROOM_HEIGHT_M_DEFAULT)
            Hm = max(0.3, H - WORKPLANE_HEIGHT_M_DEFAULT)
            K = _room_index_K(float(dims.get('L') or 0.0), float(dims.get('W') or 0.0), float(Hm))
            rho_avg = float(0.5*(room.get('rho_c') or CEILING_REFLECTANCE_DEFAULT) + 0.5*(room.get('rho_w') or WALL_REFLECTANCE_DEFAULT))
            st = _status_vs_targets(metrics, targets, None)
            ugr = _ugr_from_table(str(sel_key), K, rho_avg) or _ugr_approx(room, phot_room)
            ev = _estimate_vertical_illuminance(room, extracted.get("lights", []), _dominant_wall_segment(room.get("polygon_m") or []))
            st = _status_vs_targets(metrics, targets, ugr)
            lux_results.append({
                "room_id": room.get("id"),
                "label": room.get("label"),
                "targets": targets,
                "metrics": metrics,
                "status": st.get("status"),
                "reasons": st.get("reasons"),
                "ugr_approx": ugr,
                "vertical": ev,
            })
        extracted["lux_results"] = lux_results

    return {
        "ok": True,
        "input_type": input_type,
        "page_count": page_count,
        "extracted": extracted,
        "checks": checks,
        "vision": vres,
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "llm_provider": llm.provider,
        "vision_enabled": vision.enabled,
    }
@app.post("/report/pdf")
async def report_pdf(req: Request):
    """Generate PDF report from analyzed JSON and optional overlay (base64)."""
    payload = await req.json()
    analyzed = payload.get("analyzed") or {}
    overlay_b64 = payload.get("overlay_png_base64")
    overlay_png = base64.b64decode(overlay_b64) if overlay_b64 else None
    title = payload.get("title") or (analyzed.get("meta", {}).get("project_title") if isinstance(analyzed, dict) else None) or "Report Illuminotecnico"
    pdf = _generate_pdf_report(title, analyzed, overlay_png=overlay_png, verification=payload.get("verification"), checks=payload.get("checks"))
    return {"ok": True, "pdf_base64": base64.b64encode(pdf).decode("ascii")}

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
    return {"ok": True, "items": [{"key": k, "type": v.get("type"), "input_watts": v.get("input_watts")} for k,v in _LUMINAIRE_LIB.items()]}

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
