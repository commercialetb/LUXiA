# ============================================================
# LIGHTING AGENT PRO v3.0
# UNI EN 12464-1:2021 | UNI EN 12464-2 | UNI EN 1838:2025
# UNI 11630:2016 | UNI 11248:2016 | UNI CEI 11222
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from datetime import datetime
import base64, json, os

# ---- installazione opzionale canvas ----
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False

try:
    import requests
    REQ_OK = True
except ImportError:
    REQ_OK = False

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================
st.set_page_config(
    page_title="Lighting Agent Pro v3.0",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.header-box{background:linear-gradient(135deg,#1a365d,#2b6cb0);color:white;
  padding:1.8rem 2rem;border-radius:12px;margin-bottom:1.5rem;}
.card{background:white;padding:1rem;border-radius:8px;
  border-left:4px solid #2b6cb0;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:.8rem;}
.em-card{background:#fef3c7;padding:1rem;border-radius:8px;
  border-left:4px solid #f59e0b;margin-bottom:.8rem;}
.ext-card{background:#f0fff4;padding:1rem;border-radius:8px;
  border-left:4px solid #38a169;margin-bottom:.8rem;}
.login-box{max-width:400px;margin:5rem auto;padding:2rem;
  background:white;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,.12);}
.stButton>button{background:#2b6cb0;color:white;border:none;
  border-radius:8px;font-weight:700;padding:.55rem 1.8rem;width:100%;}
.stButton>button:hover{background:#1a365d;}
</style>""", unsafe_allow_html=True)

# ============================================================
# LOGIN
# ============================================================
VALID_USERS = {
    "admin":   "admin2026",
    "demo":    "demo123",
    "progett": "luce2026",
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = ""

if not st.session_state.logged_in:
    st.markdown("""
    <div style="max-width:400px;margin:5rem auto;padding:2rem;
    background:white;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,.12)">
    <h2 style="text-align:center;color:#1a365d">💡 Lighting Agent Pro</h2>
    <p style="text-align:center;color:#666">v3.0 — Accesso riservato</p>
    </div>""", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown("#### Login")
        user_input = st.text_input("Username", key="li_user")
        pass_input = st.text_input("Password", type="password", key="li_pass")
        if st.button("Entra 🔐"):
            if user_input in VALID_USERS and VALID_USERS[user_input] == pass_input:
                st.session_state.logged_in = True
                st.session_state.username  = user_input
                st.rerun()
            else:
                st.error("❌ Credenziali non valide")
    st.stop()

# ============================================================
# DATABASE LAMPADE
# ============================================================
if "DB_LAMPADE" not in st.session_state:
    st.session_state.DB_LAMPADE = {
        "BEGA 12345 Downlight 3000lm/25W": {
            "produttore":"BEGA","flusso_lm":3000,"potenza_W":25,
            "efficienza":120,"ra":90,"temp_colore":"3000K","ugr":16,
            "prezzo":185,"installazione":45,"tipo":"Downlight","ip":"IP20",
            "dimmerabile":True,"classe_energ":"A+"},
        "BEGA 67890 Lineare 4500lm/35W": {
            "produttore":"BEGA","flusso_lm":4500,"potenza_W":35,
            "efficienza":128,"ra":90,"temp_colore":"4000K","ugr":18,
            "prezzo":245,"installazione":55,"tipo":"Lineare","ip":"IP20",
            "dimmerabile":True,"classe_energ":"A+"},
        "iGuzzini Laser Blade 3500lm/28W": {
            "produttore":"iGuzzini","flusso_lm":3500,"potenza_W":28,
            "efficienza":125,"ra":90,"temp_colore":"4000K","ugr":17,
            "prezzo":220,"installazione":50,"tipo":"Lineare","ip":"IP20",
            "dimmerabile":True,"classe_energ":"A+"},
        "Flos Aim Fix 2800lm/22W": {
            "produttore":"Flos","flusso_lm":2800,"potenza_W":22,
            "efficienza":127,"ra":90,"temp_colore":"3000K","ugr":15,
            "prezzo":195,"installazione":40,"tipo":"Sospensione","ip":"IP20",
            "dimmerabile":True,"classe_energ":"A++"},
        "Artemide Alphabet 4000lm/30W": {
            "produttore":"Artemide","flusso_lm":4000,"potenza_W":30,
            "efficienza":133,"ra":90,"temp_colore":"3000K","ugr":15,
            "prezzo":380,"installazione":60,"tipo":"Lineare","ip":"IP20",
            "dimmerabile":True,"classe_energ":"A++"},
        "Delta Light Tweeter 2500lm/20W": {
            "produttore":"Delta Light","flusso_lm":2500,"potenza_W":20,
            "efficienza":125,"ra":90,"temp_colore":"3000K","ugr":14,
            "prezzo":165,"installazione":35,"tipo":"Downlight","ip":"IP44",
            "dimmerabile":True,"classe_energ":"A+"},
        # Emergenza
        "Gewiss GW Emergenza 200lm/3W": {
            "produttore":"Gewiss","flusso_lm":200,"potenza_W":3,
            "efficienza":66,"ra":80,"temp_colore":"4000K","ugr":28,
            "prezzo":85,"installazione":25,"tipo":"Emergenza","ip":"IP20",
            "dimmerabile":False,"classe_energ":"A"},
        # Esterni
        "BEGA 77001 Proiettore Esterno 8000lm/60W": {
            "produttore":"BEGA","flusso_lm":8000,"potenza_W":60,
            "efficienza":133,"ra":80,"temp_colore":"4000K","ugr":55,
            "prezzo":420,"installazione":120,"tipo":"Proiettore","ip":"IP65",
            "dimmerabile":True,"classe_energ":"A+"},
        "Philips BRP080 Stradale 6500lm/50W": {
            "produttore":"Philips","flusso_lm":6500,"potenza_W":50,
            "efficienza":130,"ra":70,"temp_colore":"4000K","ugr":55,
            "prezzo":310,"installazione":200,"tipo":"Stradale","ip":"IP66",
            "dimmerabile":True,"classe_energ":"A+"},
        "iGuzzini iPro Parcheggio 5000lm/40W": {
            "produttore":"iGuzzini","flusso_lm":5000,"potenza_W":40,
            "efficienza":125,"ra":80,"temp_colore":"4000K","ugr":55,
            "prezzo":280,"installazione":150,"tipo":"Proiettore","ip":"IP65",
            "dimmerabile":True,"classe_energ":"A+"},
    }

DB_LAMPADE = st.session_state.DB_LAMPADE

# ============================================================
# REQUISITI ILLUMINOTECNICI NORMATIVI
# ============================================================
REQUISITI = {
    # INTERNI — UNI EN 12464-1:2021
    "Ufficio VDT":        {"lux":500,"ugr_max":19,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Reception":          {"lux":300,"ugr_max":22,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Corridoio":          {"lux":100,"ugr_max":28,"uni":0.40,"ra_min":40,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Sala riunioni":      {"lux":500,"ugr_max":19,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Archivio":           {"lux":200,"ugr_max":25,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Bagno/WC":           {"lux":200,"ugr_max":25,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Laboratorio":        {"lux":750,"ugr_max":16,"uni":0.70,"ra_min":90,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Ingresso":           {"lux":200,"ugr_max":22,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Mensa/Ristoro":      {"lux":200,"ugr_max":22,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Locale tecnico":     {"lux":200,"ugr_max":25,"uni":0.40,"ra_min":60,"norma":"UNI EN 12464-1:2021","area":"INT"},
    # Emergenza — UNI EN 1838:2025
    "Via di esodo":       {"lux":1,  "ugr_max":35,"uni":0.10,"ra_min":40,"norma":"UNI EN 1838:2025","area":"EM"},
    "Area antipanico":    {"lux":0.5,"ugr_max":35,"uni":0.10,"ra_min":40,"norma":"UNI EN 1838:2025","area":"EM"},
    # ESTERNI — UNI EN 12464-2:2025
    "Piazzale operativo":         {"lux":20, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Area carico/scarico":        {"lux":50, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Parcheggio esterno":         {"lux":10, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Percorso pedonale esterno":  {"lux":5,  "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    # STRADALE — UNI 11248:2016 / UNI EN 13201
    "Strada veicolare ME3a":      {"lux":7.5,"ugr_max":55,"uni":0.40,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
    "Strada residenziale CE2":    {"lux":7.5,"ugr_max":55,"uni":0.40,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
    "Zona pedonale S4":           {"lux":5,  "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
}


# ============================================================
# FUNZIONE CALCOLO ILLUMINOTECNICO
# ============================================================
def calcola_area(area: dict, modalita: str = "normale") -> dict:
    sup  = area["superficie_m2"]
    alt  = area.get("altezza_m", 2.70)
    req  = REQUISITI[area["tipo_locale"]]
    lamp = DB_LAMPADE[area["lampada"]]
    CU, MF = 0.60, 0.80

    if modalita == "emergenza":
        if req["area"] in ("EM",):
            E_t = req["lux"]
        elif "corridoio" in area["tipo_locale"].lower():
            E_t = 1.0
        else:
            E_t = 1.0
        lamp_em_key = next(
            (k for k, v in DB_LAMPADE.items() if v["tipo"] == "Emergenza"), area["lampada"]
        )
        lamp = DB_LAMPADE[lamp_em_key]
    else:
        E_t = req["lux"]

    n = max(1, int(np.ceil((E_t * sup) / (CU * MF * lamp["flusso_lm"]))))
    phi = n * lamp["flusso_lm"]
    E_m = round((phi * CU * MF) / sup, 1)
    W_t = n * lamp["potenza_W"]

    lato = np.sqrt(sup)
    ns   = max(1, int(np.ceil(np.sqrt(n))))
    mg   = max(0.8, lato / (ns * 3))
    ix   = max(0.5, (lato - 2 * mg) / max(ns - 1, 1))
    coords = []
    for i in range(ns):
        for j in range(ns):
            if len(coords) < n:
                coords.append((round(mg + i * ix, 2), round(mg + j * ix, 2)))

    k = round((lato * lato) / (alt * 2 * lato), 2) if alt > 0 else 1.0

    return {
        "n": n, "phi_lm": int(phi), "E_m": E_m, "E_t": E_t,
        "W_t": W_t, "wm2": round(W_t / sup, 2),
        "ix": round(ix, 2), "k": k, "CU": CU, "MF": MF,
        "coords": coords,
        "ugr_max": req["ugr_max"], "uni_min": req["uni"],
        "ok_lux": "✅" if E_m >= E_t * 0.95 else "❌",
        "ok_ugr": "✅" if lamp["ugr"] <= req["ugr_max"] else "❌",
        "ok_uni": "✅",
        "ok_ra":  "✅" if lamp["ra"] >= req["ra_min"] else "❌",
        "modalita": modalita,
        "lampada_usata": next(k for k, v in DB_LAMPADE.items() if v == lamp),
    }


# ============================================================
# RENDERING 3D FOTOREALISTICO (Matplotlib)
# ============================================================
def genera_rendering(area: dict, calc: dict) -> BytesIO:
    lato  = np.sqrt(area["superficie_m2"])
    alt   = area.get("altezza_m", 2.70)
    lamp  = DB_LAMPADE[area["lampada"]]
    coords = calc["coords"]
    is_ext = REQUISITI[area["tipo_locale"]]["area"] in ("EXT", "STR")

    fig = plt.figure(figsize=(14, 9), dpi=180, facecolor="#050816")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#050816")

    mat_pav    = (0.22, 0.22, 0.24) if not is_ext else (0.30, 0.30, 0.28)
    mat_pareti = (0.95, 0.93, 0.90, 0.12)
    mat_soff   = (0.97, 0.97, 0.97, 0.06)
    mat_legno  = (0.42, 0.28, 0.16)
    mat_sedia  = (0.15, 0.18, 0.22)
    mat_led    = (1.0, 0.95, 0.72)
    mat_emer   = (0.1, 0.9, 0.2)

    is_em = calc.get("modalita") == "emergenza"

    # Griglia pavimento
    Xg, Yg = np.meshgrid(np.linspace(0, lato, 12), np.linspace(0, lato, 12))
    Zg_base = np.zeros_like(Xg)
    ax.plot_wireframe(Xg, Yg, Zg_base, color="#2d3748", linewidth=0.4, alpha=0.5)

    # Pavimento pieno
    pav = Poly3DCollection(
        [[(0,0,0),(lato,0,0),(lato,lato,0),(0,lato,0)]], alpha=1.0)
    pav.set_facecolor(mat_pav)
    pav.set_edgecolor("#4b5563")
    ax.add_collection3d(pav)

    if not is_ext:
        # Soffitto
        soff = Poly3DCollection(
            [[(0,0,alt),(lato,0,alt),(lato,lato,alt),(0,lato,alt)]], alpha=0.8)
        soff.set_facecolor(mat_soff)
        soff.set_edgecolor("#9ca3af")
        ax.add_collection3d(soff)

        # Pareti
        for wall in [
            [(0,0,0),(lato,0,0),(lato,0,alt),(0,0,alt)],
            [(0,lato,0),(lato,lato,0),(lato,lato,alt),(0,lato,alt)],
            [(0,0,0),(0,lato,0),(0,lato,alt),(0,0,alt)],
            [(lato,0,0),(lato,lato,0),(lato,lato,alt),(lato,0,alt)],
        ]:
            pw = Poly3DCollection([wall], alpha=0.12)
            pw.set_facecolor(mat_pareti)
            pw.set_edgecolor("#9ca3af")
            ax.add_collection3d(pw)

    # Scrivanie + sedie (no esterni, no emergenza)
    if not is_ext and not is_em:
        n_rows = max(1, int(np.ceil(np.sqrt(len(coords)) / 2)))
        xs_desk = np.linspace(lato * 0.18, lato * 0.82, max(2, n_rows))
        ys_desk = np.linspace(lato * 0.18, lato * 0.82, max(2, n_rows))
        for xd in xs_desk:
            for yd in ys_desk:
                # Piano scrivania
                top = Poly3DCollection(
                    [[(xd-.65, yd-.38, 0.74),(xd+.65, yd-.38, 0.74),
                      (xd+.65, yd+.38, 0.74),(xd-.65, yd+.38, 0.74)]], alpha=1.0)
                top.set_facecolor(mat_legno)
                top.set_edgecolor("#1f2937")
                ax.add_collection3d(top)
                # Gambe
                for gx, gy in [(xd-.55, yd-.28),(xd+.55, yd-.28),
                                (xd+.55, yd+.28),(xd-.55, yd+.28)]:
                    ax.plot([gx,gx],[gy,gy],[0,0.74],
                            color="#6b7280", lw=1.5, alpha=0.9)
                # Monitor
                mon = Poly3DCollection(
                    [[(xd-.20, yd-.03, 0.74),(xd+.20, yd-.03, 0.74),
                      (xd+.20, yd-.03, 1.20),(xd-.20, yd-.03, 1.20)]], alpha=0.9)
                mon.set_facecolor((0.08, 0.10, 0.14))
                mon.set_edgecolor("#374151")
                ax.add_collection3d(mon)
                # Sedia (seduta + schienale)
                sed = Poly3DCollection(
                    [[(xd-.28, yd+.40, 0),(xd+.28, yd+.40, 0),
                      (xd+.28, yd+.65, 0),(xd-.28, yd+.65, 0)]], alpha=1.0)
                sed.set_facecolor(mat_sedia)
                sed.set_edgecolor("#0f172a")
                ax.add_collection3d(sed)
                sch = Poly3DCollection(
                    [[(xd-.25, yd+.62, 0),(xd+.25, yd+.62, 0),
                      (xd+.25, yd+.65, 0.75),(xd-.25, yd+.65, 0.75)]], alpha=0.9)
                sch.set_facecolor(mat_sedia)
                sch.set_edgecolor("#0f172a")
                ax.add_collection3d(sch)

    # Reception / bancone (solo reception)
    if "reception" in area["tipo_locale"].lower():
        ban = Poly3DCollection(
            [[(lato*.20, lato*.45, 0),(lato*.80, lato*.45, 0),
              (lato*.80, lato*.45, 1.1),(lato*.20, lato*.45, 1.1)]], alpha=0.95)
        ban.set_facecolor((0.50, 0.35, 0.20))
        ban.set_edgecolor("#292524")
        ax.add_collection3d(ban)

    # Uscita emergenza
    if is_em:
        for ex_pos in [(0.02, lato/2), (lato-0.02, lato/2)]:
            ex = Poly3DCollection(
                [[(ex_pos[0], ex_pos[1]-.5, 0.1),
                  (ex_pos[0], ex_pos[1]+.5, 0.1),
                  (ex_pos[0], ex_pos[1]+.5, 0.6),
                  (ex_pos[0], ex_pos[1]-.5, 0.6)]], alpha=0.98)
            ex.set_facecolor((0.1, 0.8, 0.1))
            ex.set_edgecolor("white")
            ax.add_collection3d(ex)

    # Lampade e coni luce
    c_led = mat_emer if is_em else mat_led
    theta = np.linspace(0, 2 * np.pi, 20)
    for (lx, ly) in coords:
        h = alt - 0.05 if not is_ext else 6.0
        ax.scatter([lx],[ly],[h], c=[c_led], s=300,
                   edgecolors="white", lw=1.5, zorder=10)
        for rr, alp in [(0.6,0.16),(1.2,0.09),(2.0,0.05)]:
            for ang in theta[::2]:
                ax.plot([lx, lx + rr*np.cos(ang)],
                        [ly, ly + rr*np.sin(ang)],
                        [h, 0.04], color="#fef3c7", alpha=alp, lw=0.7)

    # Distribuzione lux pavimento (heatmap)
    Xh, Yh = np.meshgrid(
        np.linspace(0.1, lato-0.1, 50), np.linspace(0.1, lato-0.1, 50))
    Zh = np.zeros_like(Xh)
    h_lamp = alt if not is_ext else 6.0
    for (lx2, ly2) in coords:
        d2 = np.sqrt((Xh-lx2)**2+(Yh-ly2)**2+h_lamp**2)
        Zh += (lamp["flusso_lm"]/(2*np.pi))*(h_lamp/d2**3)
    Zn = (Zh - Zh.min())/(Zh.max()-Zh.min()+1e-9)
    cmap_use = plt.cm.summer if is_em else plt.cm.inferno
    ax.plot_surface(Xh, Yh, np.full_like(Xh, 0.02),
                    facecolors=cmap_use(Zn), alpha=0.55, shade=False)

    ax.set_xlim(0, lato)
    ax.set_ylim(0, lato)
    ax.set_zlim(0, max(alt, 6.5) if is_ext else alt)
    ax.view_init(elev=30, azim=235)
    ax.axis("off")

    em_label = " [EMERGENZA]" if is_em else ""
    fig.text(.5, .97, f"RENDERING 3D — {area['nome']}{em_label}",
             fontsize=14, fontweight="bold", color="white", ha="center", va="top")
    fig.text(.5, .93,
             f"{calc['n']}x {area['lampada'][:38]}  |  {calc['E_m']} lux  |  "
             f"{calc['W_t']} W  |  {calc['wm2']} W/m²  |  Norma: {REQUISITI[area['tipo_locale']]['norma']}",
             fontsize=8, color="#a5b4fc", ha="center")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="#050816")
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================
# ISOLUX 2D
# ============================================================
def genera_isolux(ax, coords, lamp, sup, alt):
    lato = np.sqrt(sup)
    X, Y = np.meshgrid(np.linspace(0,lato,60), np.linspace(0,lato,60))
    Z = np.zeros_like(X)
    for (lx, ly) in coords:
        d = np.sqrt((X-lx)**2+(Y-ly)**2+alt**2)
        ct = alt/d
        Z += (lamp["flusso_lm"]/(2*np.pi))*(ct/d**2)*0.6
    cmap = LinearSegmentedColormap.from_list(
        "iso", ["#1a365d","#2b6cb0","#48bb78","#f6e05e","#fc8181","white"])
    cf = ax.contourf(X, Y, Z, levels=15, cmap=cmap, alpha=0.85)
    ax.contour(X, Y, Z, levels=[1, 5, 10, 50, 100, 200, 300, 500, 750],
               colors="black", linewidths=0.6, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="Lux", shrink=0.85)
    for (lx, ly) in coords:
        ax.plot(lx, ly, "o", color="#fbbf24", ms=7, mec="black", mew=1.2, zorder=5)
    ax.set_xlim(0, lato); ax.set_ylim(0, lato)
    ax.set_aspect("equal"); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")


# ============================================================
# EXPORT GLTF
# ============================================================
def export_gltf_scene(risultati: list) -> BytesIO:
    nodes = []
    for r in risultati:
        lato = float(np.sqrt(r["sup"]))
        alt  = float(r.get("altezza_m", 2.70))
        nodes.append({"name": r["nome"],
                      "translation": [float(r.get("offset_x",0)), 0.0, 0.0],
                      "scale": [lato, lato, alt]})
        for (lx, ly) in r["calc"]["coords"]:
            nodes.append({"name": f"Lamp_{r['nome']}",
                          "translation": [float(r.get("offset_x",0)+lx),
                                          float(ly), alt-0.05]})
    gltf = {
        "asset": {"version":"2.0","generator":"LightingAgentPro v3.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": [{"name":"RoomBox","primitives":[]}],
    }
    buf = BytesIO()
    buf.write(json.dumps(gltf, indent=2).encode("utf-8"))
    buf.seek(0)
    return buf


# ============================================================
# PREVENTIVO
# ============================================================
def calc_preventivo(risultati, mg_pct, sg_pct, os_pct, iva_pct):
    righe, tm, ti = [], 0, 0
    for r in risultati:
        lamp = DB_LAMPADE[r.get("lampada_calc", r["lampada"])]
        mat = r["calc"]["n"] * lamp["prezzo"]
        ins = r["calc"]["n"] * lamp["installazione"]
        righe.append({"area": r["nome"], "n": r["calc"]["n"],
                      "lampada": r["lampada"][:30],
                      "mat": mat, "ins": ins, "sub": mat+ins,
                      "modalita": r["calc"].get("modalita","normale")})
        tm += mat; ti += ins
    tn = tm + ti
    sg = tn * sg_pct/100
    os2= tn * os_pct/100
    tl = tn + sg + os2
    mg = tl * mg_pct/100
    to = tl + mg
    iva= to * iva_pct/100
    return {"righe":righe,"tm":tm,"ti":ti,"tn":tn,"sg":sg,
            "os":os2,"tl":tl,"mg":mg,"to":to,"iva":iva,"tf":to+iva}


# ============================================================
# DXF
# ============================================================
def genera_dxf(risultati):
    out = "0\nSECTION\n2\nENTITIES\n"
    lid = 1
    for r in risultati:
        ox = r.get("offset_x", 0); oy = r.get("offset_y", 0)
        layer = "EMERGENZA" if r["calc"].get("modalita")=="emergenza" else "LUCI"
        for (x, y) in r["calc"]["coords"]:
            out += (f"0\nCIRCLE\n8\n{layer}\n"
                    f"10\n{ox+x:.2f}\n20\n{oy+y:.2f}\n30\n0\n40\n0.25\n"
                    f"0\nTEXT\n8\nIDENTIF\n"
                    f"10\n{ox+x+0.3:.2f}\n20\n{oy+y+0.3:.2f}\n30\n0\n"
                    f"40\n0.20\n1\nL{lid:03d}\n"); lid += 1
    out += "0\nENDSEC\n0\nEOF\n"
    return out


# ============================================================
# PDF TAVOLA A3 + VERIFICHE
# ============================================================
def genera_pdf(progetto, risultati, logo_bytes=None):
    buf = BytesIO()
    with PdfPages(buf) as pdf:

        # --- PAGINA 1: TAVOLA ---
        fig = plt.figure(figsize=(42/2.54, 29.7/2.54), dpi=120)
        fig.patch.set_facecolor("white")

        ax_h = fig.add_axes([0.0, 0.93, 1.0, 0.07])
        ax_h.set_xlim(0,1); ax_h.set_ylim(0,1); ax_h.axis("off")
        ax_h.add_patch(mpatches.Rectangle((0,0),1,1,facecolor="#1a365d"))
        ax_h.text(.01,.62,"TAVOLA ILLUMINOTECNICA",color="white",
                  fontsize=18,fontweight="bold",va="center")
        ax_h.text(.01,.18,
            f"Progetto: {progetto['nome']}  |  Committente: {progetto['committente']}  |  "
            f"Data: {progetto['data']}  |  Tav. {progetto['num_tavola']}  |  Scala 1:100  |  "
            f"Norma: UNI 11630:2016 + UNI EN 12464-1:2021",
            color="#90cdf4", fontsize=7.5, va="center")
        tot_l = sum(r["calc"]["n"] for r in risultati)
        tot_W = sum(r["calc"]["W_t"] for r in risultati)
        tot_s = sum(r["sup"] for r in risultati)
        ax_h.text(.72,.62,
            f"Lamp: {tot_l}  |  Potenza: {tot_W}W  |  Sup: {tot_s:.0f}m²  |  {tot_W/max(tot_s,1):.1f}W/m²",
            color="white", fontsize=8, va="center")

        # Logo
        if logo_bytes:
            from PIL import Image as PILImg
            logo_img = PILImg.open(BytesIO(logo_bytes)).convert("RGBA")
            logo_arr = np.array(logo_img)
            ax_h.imshow(logo_arr, extent=(0.84,0.99,0.05,0.95),
                        aspect="auto", zorder=5)

        ax_p = fig.add_axes([0.01,0.10,0.68,0.81])
        ax_p.set_facecolor("#f8fafc")
        ax_p.grid(True, alpha=0.3, linewidth=0.4)
        ax_p.set_xlabel("X [m]", fontsize=8); ax_p.set_ylabel("Y [m]", fontsize=8)
        ax_p.set_title("PLANIMETRIA — POSIZIONAMENTO APPARECCHI",
                       fontsize=10, fontweight="bold", pad=6)

        COLORS = ["#3182ce","#e53e3e","#38a169","#d69e2e","#805ad5","#dd6b20",
                  "#0f766e","#be185d","#1d4ed8","#b45309"]
        lid = 1
        for idx, r in enumerate(risultati):
            ox = r.get("offset_x",0); oy = r.get("offset_y",0)
            lato = np.sqrt(r["sup"]); c = COLORS[idx % len(COLORS)]
            ax_p.add_patch(mpatches.Rectangle((ox,oy),lato,lato,
                fill=True,facecolor=c,alpha=0.07,edgecolor=c,linewidth=2))
            ax_p.text(ox+lato/2, oy+lato+0.25, r["nome"][:14],
                      fontsize=7, ha="center", color=c, fontweight="bold")
            sym_color = "#10b981" if r["calc"].get("modalita")=="emergenza" else "#fbbf24"
            for (x, y) in r["calc"]["coords"]:
                ax_p.add_patch(plt.Circle((ox+x, oy+y), 0.22,
                    color=sym_color, ec="black", lw=1.2, zorder=5))
                ax_p.text(ox+x, oy+y, f"L{lid}",
                    fontsize=4.5, ha="center", va="center", fontweight="bold", zorder=6)
                lid += 1
        ax_p.autoscale_view(); ax_p.set_aspect("equal")

        # Legenda
        ax_l = fig.add_axes([0.71, 0.60, 0.28, 0.32])
        ax_l.axis("off")
        ax_l.set_title("LEGENDA APPARECCHI", fontsize=8, fontweight="bold", loc="left")
        seen, yy = set(), 0.88
        for r in risultati:
            lk = r["calc"].get("lampada_usata", r["lampada"])
            if lk not in seen:
                seen.add(lk)
                lsp = DB_LAMPADE[lk]
                sc = "#10b981" if lsp["tipo"]=="Emergenza" else "#fbbf24"
                ax_l.add_patch(plt.Circle((0.04,yy),0.03,color=sc,ec="black",lw=1))
                ax_l.text(.10, yy+.03, lk[:30], fontsize=6.5, fontweight="bold", va="center")
                ax_l.text(.10, yy-.04,
                    f"{lsp['flusso_lm']}lm | {lsp['potenza_W']}W | Ra{lsp['ra']} | {lsp['temp_colore']}",
                    fontsize=5.5, color="#555", va="center")
                yy -= 0.18
        ax_l.set_xlim(0,1); ax_l.set_ylim(0,1)

        # Tabella riepilogo
        ax_t = fig.add_axes([0.71, 0.25, 0.28, 0.33])
        ax_t.axis("off")
        ax_t.set_title("RIEPILOGO CALCOLI", fontsize=8, fontweight="bold", loc="left")
        hdr = ["Area","m²","Lamp","Lux","W/m²","Norma"]
        rows = [[r["nome"][:9], str(int(r["sup"])), str(r["calc"]["n"]),
                 str(r["calc"]["E_m"]), str(r["calc"]["wm2"]),
                 REQUISITI[r["tipo_locale"]]["norma"][-14:]]
                for r in risultati]
        tbl = ax_t.table(cellText=rows, colLabels=hdr, cellLoc="center",
                         loc="upper center", colWidths=[0.22,0.10,0.10,0.10,0.10,0.22])
        tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
        for (row,col),cell in tbl.get_celld().items():
            if row==0:
                cell.set_facecolor("#1a365d"); cell.set_text_props(color="white",fontweight="bold")
            elif row%2==0: cell.set_facecolor("#ebf8ff")
            cell.set_edgecolor("#cbd5e0")
        ax_t.set_xlim(0,1); ax_t.set_ylim(0,1)

        # Note normative
        ax_n = fig.add_axes([0.71, 0.10, 0.28, 0.13])
        ax_n.axis("off")
        note = ("NORME DI RIFERIMENTO:\n"
                "• UNI 11630:2016 — Criteri progetto\n"
                "• UNI EN 12464-1:2021 — Interni\n"
                "• UNI EN 12464-2:2025 — Esterni\n"
                "• UNI 11248:2016 — Strade\n"
                "• UNI EN 1838:2025 — Emergenza\n"
                "• UNI CEI 11222 — Manutenzione")
        ax_n.text(0.03,0.95,note,fontsize=6,va="top",linespacing=1.6,
                  bbox=dict(boxstyle="round",facecolor="#f0fff4",
                            edgecolor="#68d391",lw=1))
        ax_n.set_xlim(0,1); ax_n.set_ylim(0,1)

        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # --- PAGINE VERIFICHE ---
        for r in risultati:
            calc = r["calc"]
            lk   = calc.get("lampada_usata", r["lampada"])
            lamp = DB_LAMPADE[lk]
            alt2 = r.get("altezza_m", 2.70)
            req  = REQUISITI[r["tipo_locale"]]

            fig2, axes = plt.subplots(2, 3, figsize=(42/2.54, 29.7/2.54), dpi=120)
            fig2.patch.set_facecolor("white")
            em_label = " [EMERGENZA]" if calc.get("modalita")=="emergenza" else ""
            fig2.suptitle(
                f"VERIFICHE — {r['nome'].upper()}{em_label} | {r['tipo_locale']} | {req['norma']}",
                fontsize=11, fontweight="bold", color="#1a365d", y=0.99)

            ax = axes[0,0]; ax.axis("off")
            ax.set_title("DATI DI CALCOLO", fontsize=9, fontweight="bold", color="#1a365d")
            dati = [["Tipo locale",r["tipo_locale"]],["Superficie",f"{r['sup']} m²"],
                    ["Altezza",f"{alt2} m"],["Indice k",str(calc["k"])],
                    ["CU",str(calc["CU"])],["MF",str(calc["MF"])],
                    ["Lux target",f"{calc['E_t']} lux"],["Lux ottenuto",f"{calc['E_m']} lux"],
                    ["N apparecchi",str(calc["n"])],["Lampada",lk[:28]],
                    ["Potenza totale",f"{calc['W_t']} W"],["W/m²",str(calc["wm2"])],
                    ["UGR app.",str(lamp["ugr"])],["UGR max",f"< {calc['ugr_max']}"],
                    ["Norma",req["norma"]]]
            tb = ax.table(cellText=dati, loc="center", cellLoc="left", colWidths=[0.50,0.50])
            tb.auto_set_font_size(False); tb.set_fontsize(7.5)
            for (row,col),cell in tb.get_celld().items():
                cell.set_edgecolor("#e2e8f0")
                if col==0: cell.set_facecolor("#ebf8ff"); cell.set_text_props(fontweight="bold")
                if row in [7,8] and col==1: cell.set_facecolor("#c6f6d5")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            ax = axes[0,1]
            ax.set_title("MAPPA ISOLUX 2D", fontsize=9, fontweight="bold", color="#1a365d")
            genera_isolux(ax, calc["coords"], lamp, r["sup"], alt2)

            ax = axes[0,2]; ax.axis("off")
            ax.set_title("VERIFICA CONFORMITÀ", fontsize=9, fontweight="bold", color="#1a365d")
            checks = [
                ("Illuminamento medio", f"{calc['E_m']} lux ≥ {calc['E_t']} lux", calc["ok_lux"]),
                ("UGR abbagliamento",   f"{lamp['ugr']} ≤ {calc['ugr_max']}",      calc["ok_ugr"]),
                ("Uniformità",          f"≥ {calc['uni_min']}",                    calc["ok_uni"]),
                ("Resa cromatica Ra",   f"{lamp['ra']} ≥ {req['ra_min']}",          calc["ok_ra"]),
            ]
            yp = 0.84
            for nm, vl, st2 in checks:
                cc = "#22c55e" if st2=="✅" else "#ef4444"
                bg = "#f0fff4" if st2=="✅" else "#fff5f5"
                ax.add_patch(mpatches.FancyBboxPatch((0.02,yp-.13),0.96,0.18,
                    boxstyle="round,pad=0.02",facecolor=bg,edgecolor=cc,lw=1.5))
                ax.text(0.08,yp-.02,st2,fontsize=16,va="center")
                ax.text(0.20,yp,nm,fontsize=8,fontweight="bold",va="center")
                ax.text(0.20,yp-.07,vl,fontsize=7.5,color="#555",va="center")
                yp -= 0.22
            ax.add_patch(mpatches.Rectangle((0.02,0.01),0.96,0.12,
                facecolor="#22c55e",edgecolor="none"))
            ax.text(.50,.07,f"CONFORME {req['norma']}",fontsize=10,fontweight="bold",
                    color="white",ha="center",va="center")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            ax = axes[1,0]
            ax.set_title("PROFILO ILLUMINAMENTO ASSE X",fontsize=9,fontweight="bold",color="#1a365d")
            lato2 = np.sqrt(r["sup"])
            xv = np.linspace(0,lato2,120)
            ym2 = lato2/2
            zp = np.zeros_like(xv)
            for (lx2,ly2) in calc["coords"]:
                d2 = np.sqrt((xv-lx2)**2+(ym2-ly2)**2+alt2**2)
                ct2 = alt2/d2
                zp += (lamp["flusso_lm"]/(2*np.pi))*(ct2/d2**2)*0.60
            ax.fill_between(xv, zp, alpha=0.25, color="#3182ce")
            ax.plot(xv, zp, color="#1a365d", lw=2)
            ax.axhline(calc["E_t"],color="#e53e3e",ls="--",lw=1.5,
                       label=f"Target {calc['E_t']} lux")
            ax.set_xlabel("X [m]",fontsize=8); ax.set_ylabel("Lux",fontsize=8)
            ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

            ax = axes[1,1]; ax.axis("off")
            ax.set_title("SCHEDA APPARECCHIO",fontsize=9,fontweight="bold",color="#1a365d")
            scheda = [["Produttore",lamp["produttore"]],["Modello",lk[:28]],
                      ["Tipo",lamp["tipo"]],["Flusso",f"{lamp['flusso_lm']} lm"],
                      ["Potenza",f"{lamp['potenza_W']} W"],
                      ["Efficienza",f"{lamp['efficienza']} lm/W"],
                      ["Ra",str(lamp["ra"])],["Temp.colore",lamp["temp_colore"]],
                      ["UGR",str(lamp["ugr"])],["IP",lamp["ip"]],
                      ["Dimmerabile","Sì" if lamp["dimmerabile"] else "No"],
                      ["Classe en.",lamp["classe_energ"]],
                      ["Prezzo",f"EUR {lamp['prezzo']}"],
                      ["Inst.",f"EUR {lamp['installazione']}"]]
            ts = ax.table(cellText=scheda,loc="center",cellLoc="left",colWidths=[0.48,0.52])
            ts.auto_set_font_size(False); ts.set_fontsize(7.5)
            for (row,col),cell in ts.get_celld().items():
                cell.set_edgecolor("#e2e8f0")
                if col==0: cell.set_facecolor("#fef3c7"); cell.set_text_props(fontweight="bold")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            ax = axes[1,2]; ax.axis("off")
            ax.set_title("PREVENTIVO AREA",fontsize=9,fontweight="bold",color="#1a365d")
            mat2 = calc["n"]*lamp["prezzo"]; ins2 = calc["n"]*lamp["installazione"]
            sub2 = mat2+ins2
            prows = [[f"{calc['n']}x {lk[:20]}",f"EUR {mat2:,.0f}"],
                     ["Installazione",f"EUR {ins2:,.0f}"],
                     ["Subtotale",f"EUR {sub2:,.0f}"],
                     ["IVA 22%",f"EUR {sub2*.22:,.0f}"],
                     ["TOTALE",f"EUR {sub2*1.22:,.0f}"]]
            tp = ax.table(cellText=prows,colLabels=["VOCE","IMPORTO"],
                          loc="center",cellLoc="center",colWidths=[0.68,0.32])
            tp.auto_set_font_size(False); tp.set_fontsize(7.5)
            for (row,col),cell in tp.get_celld().items():
                if row==0: cell.set_facecolor("#1a365d"); cell.set_text_props(color="white",fontweight="bold")
                elif row==5: cell.set_facecolor("#22c55e"); cell.set_text_props(fontweight="bold")
                elif row%2==0: cell.set_facecolor("#f7fafc")
                cell.set_edgecolor("#e2e8f0")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            plt.tight_layout(rect=[0,0,1,0.97])
            pdf.savefig(fig2, bbox_inches="tight"); plt.close(fig2)

    buf.seek(0)
    return buf


# ============================================================
# RELAZIONE COMPLETA UNI 11630:2016
# ============================================================
def genera_relazione_completa(progetto, risultati, prev, logo_bytes=None,
                               mg_pct=35, sg_pct=12, os_pct=4, iva_pct=22):
    buf = BytesIO()
    with PdfPages(buf) as pdf:

        # --- FRONTESPIZIO ---
        fig = plt.figure(figsize=(21/2.54, 29.7/2.54), dpi=120)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0,0,1,1]); ax.axis("off")

        ax.add_patch(mpatches.Rectangle((0,.85),1,.15,facecolor="#1a365d"))
        ax.add_patch(mpatches.Rectangle((0,.0),1,.08,facecolor="#1a365d"))
        ax.add_patch(mpatches.Rectangle((.0,.84),1,.002,facecolor="#2b6cb0"))

        if logo_bytes:
            from PIL import Image as PILImg
            logo_img = PILImg.open(BytesIO(logo_bytes)).convert("RGBA")
            logo_arr = np.array(logo_img)
            ax.imshow(logo_arr, extent=(.60,.96,.86,.99), aspect="auto", zorder=5)

        ax.text(.05,.92,"PROGETTO ILLUMINOTECNICO",fontsize=22,fontweight="bold",
                color="white",va="center")
        ax.text(.05,.87,"RELAZIONE TECNICA GENERALE",fontsize=14,color="#90cdf4",va="center")

        y = 0.78
        for label, val in [
            ("Progetto",       progetto["nome"]),
            ("Committente",    progetto["committente"]),
            ("Progettista",    progetto["progettista"]),
            ("Data",           progetto["data"]),
            ("N. Tavola",      progetto["num_tavola"]),
        ]:
            ax.text(.07,y,label+":",fontsize=11,fontweight="bold",color="#1a365d",va="center")
            ax.text(.38,y,val,fontsize=11,color="#1a365d",va="center")
            y -= 0.065

        ax.add_patch(mpatches.Rectangle((.05,.38),.90,.002,facecolor="#e2e8f0"))
        ax.text(.07,.35,"NORME DI RIFERIMENTO",fontsize=10,fontweight="bold",color="#1a365d")
        norme = [
            "UNI 11630:2016 — Luce e illuminazione: Criteri per la stesura del progetto illuminotecnico",
            "UNI EN 12464-1:2021 — Illuminazione dei posti di lavoro in interni",
            "UNI EN 12464-2:2025 — Illuminazione dei posti di lavoro in esterni",
            "UNI 11248:2016 — Illuminazione stradale: Selezione delle categorie",
            "UNI EN 13201-2/3/4 — Illuminazione stradale: Requisiti di prestazione",
            "UNI EN 1838:2025 — Applicazione dell'illuminotecnica: Illuminazione di emergenza",
            "UNI CEI 11222:2013 — Impianti di illuminazione di sicurezza negli edifici",
        ]
        yn = 0.30
        for n in norme:
            ax.text(.08, yn, f"• {n}", fontsize=8, color="#374151", va="center")
            yn -= 0.035

        tot_l = sum(r["calc"]["n"] for r in risultati)
        tot_W = sum(r["calc"]["W_t"] for r in risultati)
        tot_s = sum(r["sup"] for r in risultati)
        ax.text(.07,.07,
            f"Lampade totali: {tot_l}  |  Potenza: {tot_W} W  |  "
            f"Superficie: {tot_s:.0f} m²  |  W/m²: {tot_W/max(tot_s,1):.1f}",
            fontsize=9,color="white",va="center")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # --- RELAZIONE DESCRITTIVA ---
        fig = plt.figure(figsize=(21/2.54,29.7/2.54), dpi=120)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.08, 0.05, 0.84, 0.90]); ax.axis("off")

        righe_testo = [
            ("1. PREMESSA", True),
            (f"Il presente documento costituisce la Relazione Tecnica Illuminotecnica del progetto «{progetto['nome']}»,", False),
            (f"redatta in conformità alla norma UNI 11630:2016 che definisce i criteri per la stesura del progetto", False),
            ("illuminotecnico e i contenuti minimi della relazione tecnica.", False),
            ("", False),
            ("2. CRITERI GENERALI DI PROGETTAZIONE", True),
            ("La progettazione è stata sviluppata con l'obiettivo di garantire le condizioni ottimali di visibilità,", False),
            ("comfort visivo e sicurezza per gli occupanti, nel rispetto delle normative vigenti e dei criteri di", False),
            ("efficienza energetica. Il metodo di calcolo adottato è il Metodo del Flusso Luminoso (UNI 11630:2016,", False),
            ("§ 6.3), con coefficiente di utilizzazione (CU=0,60) e fattore di manutenzione (MF=0,80).", False),
            ("", False),
            ("3. DESCRIZIONE DEGLI AMBIENTI E APPARECCHI", True),
        ]
        for r in risultati:
            lk   = r["calc"].get("lampada_usata", r["lampada"])
            lamp = DB_LAMPADE[lk]
            req  = REQUISITI[r["tipo_locale"]]
            em_label = " — EMERGENZA" if r["calc"].get("modalita")=="emergenza" else ""
            righe_testo.append((f"   • {r['nome']}{em_label}: {r['tipo_locale']}, "
                                f"{r['sup']} m², {r['calc']['n']}x {lk[:35]}, "
                                f"{r['calc']['E_m']} lux (target {r['calc']['E_t']} lux), "
                                f"Norma: {req['norma']}", False))

        righe_testo += [
            ("", False),
            ("4. BILANCIO ENERGETICO", True),
            (f"Potenza installata totale: {tot_W} W  —  Superficie trattata: {tot_s:.0f} m²", False),
            (f"Densità di potenza media: {tot_W/max(tot_s,1):.2f} W/m²", False),
            ("Il sistema è progettato con apparecchi dimmerabili DALI per ottimizzare i consumi.", False),
            ("", False),
            ("5. CONFORMITÀ NORMATIVA", True),
            ("Tutti gli ambienti risultano conformi alle rispettive norme di riferimento come", False),
            ("verificato nelle schede di calcolo allegate (Tavole successive).", False),
        ]

        yy = 0.97
        for testo, bold in righe_testo:
            if testo == "":
                yy -= 0.018; continue
            fw = "bold" if bold else "normal"
            cl = "#1a365d" if bold else "#1f2937"
            fs = 10 if bold else 8.5
            ax.text(0, yy, testo, fontsize=fs, fontweight=fw, color=cl, va="top",
                    wrap=True, transform=ax.transAxes)
            yy -= 0.028 if bold else 0.022
            if yy < 0.03: break

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # --- TAVOLA + VERIFICHE (riuso funzione) ---
        pdf_inner = genera_pdf(progetto, risultati, logo_bytes)
        import PyPDF2
        try:
            reader = PyPDF2.PdfReader(pdf_inner)
            writer = PyPDF2.PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            tmp = BytesIO()
            writer.write(tmp)
        except Exception:
            pass

        # --- RENDERING 3D per ogni area ---
        for r in risultati:
            try:
                r_buf = genera_rendering(r, r["calc"])
                img_arr = plt.imread(r_buf)
                fig = plt.figure(figsize=(21/2.54, 29.7/2.54), dpi=100)
                fig.patch.set_facecolor("black")
                ax = fig.add_axes([0.02, 0.08, 0.96, 0.86])
                ax.imshow(img_arr)
                ax.axis("off")
                ax.set_title(f"Rendering 3D — {r['nome']}",
                             fontsize=13, fontweight="bold", color="white", pad=10)
                pdf.savefig(fig, bbox_inches="tight", facecolor="black")
                plt.close(fig)
            except Exception:
                pass

        # --- PREVENTIVO FINALE ---
        pv = calc_preventivo(risultati, mg_pct, sg_pct, os_pct, iva_pct)
        fig = plt.figure(figsize=(21/2.54, 29.7/2.54), dpi=120)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0,0,1,1]); ax.axis("off")
        ax.add_patch(mpatches.Rectangle((0,.92),1,.08,facecolor="#1a365d"))
        ax.text(.05,.96,"PREVENTIVO ECONOMICO",fontsize=16,fontweight="bold",
                color="white",va="center")
        ax.text(.05,.91,
            f"Progetto: {progetto['nome']}  |  Data: {progetto['data']}  |  "
            f"Progettista: {progetto['progettista']}",
            fontsize=8, color="#1a365d", va="top")

        righe_prev = [[r["area"],str(r["n"]),r["lampada"][:25],
                       f"EUR {r['mat']:,.0f}",f"EUR {r['ins']:,.0f}",
                       f"EUR {r['sub']:,.0f}"] for r in pv["righe"]]
        tab_ax = fig.add_axes([0.05, 0.50, 0.90, 0.38])
        tab_ax.axis("off")
        tp = tab_ax.table(
            cellText=righe_prev,
            colLabels=["Area","N","Lampada","Materiali","Installazione","Subtotale"],
            loc="center", cellLoc="center",
            colWidths=[0.18,0.06,0.28,0.14,0.14,0.14])
        tp.auto_set_font_size(False); tp.set_fontsize(7)
        for (row,col),cell in tp.get_celld().items():
            if row==0: cell.set_facecolor("#1a365d"); cell.set_text_props(color="white",fontweight="bold")
            elif row%2==0: cell.set_facecolor("#f7fafc")
            cell.set_edgecolor("#e2e8f0")

        ry = 0.46
        riepilogo = [
            ("Materiali",           f"EUR {pv['tm']:>12,.0f}"),
            ("Installazione",       f"EUR {pv['ti']:>12,.0f}"),
            ("Totale lavori netto", f"EUR {pv['tn']:>12,.0f}"),
            (f"Spese generali {sg_pct}%",f"EUR {pv['sg']:>12,.0f}"),
            (f"Oneri sicurezza {os_pct}%",f"EUR {pv['os']:>12,.0f}"),
            (f"Margine {mg_pct}%",  f"EUR {pv['mg']:>12,.0f}"),
            ("OFFERTA CLIENTE",     f"EUR {pv['to']:>12,.0f}"),
            (f"IVA {iva_pct}%",     f"EUR {pv['iva']:>12,.0f}"),
            ("TOTALE IVA INCLUSA",  f"EUR {pv['tf']:>12,.0f}"),
        ]
        for i,(label,val) in enumerate(riepilogo):
            bold_rows = [6,8]
            fg = "#22c55e" if i in bold_rows else ("#e53e3e" if i==7 else "#1f2937")
            fw = "bold" if i in bold_rows else "normal"
            fs = 11 if i in bold_rows else 9
            ax.text(.40, ry, label, fontsize=fs, fontweight=fw, color=fg, va="center")
            ax.text(.86, ry, val,   fontsize=fs, fontweight=fw, color=fg, va="center", ha="right")
            if i in bold_rows:
                ax.add_patch(mpatches.Rectangle((.38,ry-.018),.50,.032,
                    facecolor="#f0fff4" if i==8 else "#fef9c3",
                    edgecolor=fg, lw=1.2, alpha=0.7))
            ry -= 0.035

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    buf.seek(0)
    return buf

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"👤 **{st.session_state.username}**")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    st.markdown("---")

    # LOGO STUDIO
    st.markdown("### 🏢 Logo studio")
    logo_file = st.file_uploader("Carica logo (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_up")
    if logo_file:
        st.session_state.logo_bytes = logo_file.read()
    if "logo_bytes" in st.session_state:
        st.image(st.session_state.logo_bytes, use_column_width=True)

    st.markdown("---")
    st.markdown("### 📋 Progetto")
    nome_prog   = st.text_input("Nome progetto",  "UFFICI TELEDIFESA")
    committente = st.text_input("Committente",     "Teledifesa S.p.A.")
    progettista = st.text_input("Progettista",     "Ing. Mario Rossi")
    num_tav     = st.text_input("N. Tavola",       "26A3S001")

    st.markdown("---")
    st.markdown("### 🗺️ Planimetria")
    plan_file = st.file_uploader("Carica planimetria (PDF/PNG/JPG)",
                                  type=["pdf","png","jpg","jpeg"], key="plan_up")
    if plan_file and plan_file.type.startswith("image"):
        plan_bytes = plan_file.read()
        st.session_state.plan_bytes = plan_bytes
        st.image(plan_bytes, use_column_width=True)

    st.markdown("---")
    st.markdown("### 💡 Fotometrie personalizzate")
    ies_file = st.file_uploader("Carica IES/LDT", type=["ies","ldt","txt"], key="ies_up")
    if ies_file:
        content = ies_file.read()
        txt = content.decode(errors="ignore").upper()
        approx_flux = 2000.0
        for line in txt.splitlines():
            if "LUMEN" in line:
                nums = []
                for t in line.replace(","," ").split():
                    try: nums.append(float(t))
                    except: pass
                if nums: approx_flux = nums[-1]; break
        custom_key = f"CUSTOM — {ies_file.name}"
        DB_LAMPADE[custom_key] = {
            "produttore":"Custom IES","flusso_lm":approx_flux,
            "potenza_W":20,"efficienza":round(approx_flux/20,1),
            "ra":80,"temp_colore":"4000K","ugr":19,
            "prezzo":150,"installazione":50,
            "tipo":"Custom IES","ip":"IP65",
            "dimmerabile":False,"classe_energ":"A"}
        st.success(f"✅ {custom_key} caricata (φ≈{approx_flux:.0f} lm)")

    st.markdown("---")
    st.markdown("### 🔧 Filtro lampade")
    prod_filter = st.selectbox("Produttore",
        ["Tutti","BEGA","iGuzzini","Flos","Artemide","Delta Light","Gewiss","Philips","Custom IES"])

    st.markdown("---")
    st.markdown("### 📏 Scala planimetria")
    scala_mpp = st.number_input("Metri per 100px", min_value=0.1, max_value=50.0, value=2.5, step=0.1)

# ============================================================
# HEADER
# ============================================================
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if "logo_bytes" in st.session_state:
        st.image(st.session_state.logo_bytes, width=80)
with col_title:
    st.markdown("""
<div class="header-box">
<h1 style="margin:0;font-size:2rem">💡 Lighting Agent Pro v3.0</h1>
<p style="margin:.3rem 0 0;opacity:.85">
UNI 11630:2016 · UNI EN 12464-1:2021 · UNI EN 12464-2:2025 ·
UNI 11248:2016 · UNI EN 1838:2025 · UNI CEI 11222 · AI Vision · Rendering 3D
</p></div>""", unsafe_allow_html=True)

if "aree" not in st.session_state:
    st.session_state.aree = []

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️ Aree",
    "🧮 Calcoli",
    "📐 Tavola A3",
    "✅ Verifiche",
    "🎨 Rendering 3D",
    "💶 Preventivo",
    "📄 Relazione Completa",
])

# ============================================================
# TAB 1 — AREE
# ============================================================
with tab1:
    st.subheader("Definizione Aree")

    lamp_disp = {k:v for k,v in DB_LAMPADE.items()
                 if prod_filter=="Tutti" or v["produttore"]==prod_filter}

    st.markdown("#### ➕ Aggiungi area manualmente")
    with st.form("form_area", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            nome_area   = st.text_input("Nome area *", placeholder="es. Ufficio A")
            tipo_locale = st.selectbox("Tipo locale *", list(REQUISITI.keys()))
        with c2:
            sup  = st.number_input("Superficie m² *", 1.0, 5000.0, 35.0, 0.5)
            alt  = st.number_input("Altezza netta m", 2.0, 12.0, 2.70, 0.05)
        with c3:
            lamp_scelta = st.selectbox("Apparecchio *", list(lamp_disp.keys()))
            lsp = DB_LAMPADE[lamp_scelta]
            st.caption(f"{lsp['potenza_W']}W | {lsp['flusso_lm']}lm | Ra{lsp['ra']} | "
                       f"{lsp['temp_colore']} | UGR{lsp['ugr']} | {lsp['ip']}")
            emergenza = st.checkbox("🚨 Calcola anche illuminazione di emergenza (UNI EN 1838:2025)")

        if st.form_submit_button("➕ Aggiungi area", type="primary"):
            if nome_area.strip():
                st.session_state.aree.append({
                    "nome": nome_area.strip(), "tipo_locale": tipo_locale,
                    "superficie_m2": sup, "altezza_m": alt,
                    "lampada": lamp_scelta, "sup": sup, "emergenza": emergenza,
                })
                st.success(f"Area «{nome_area}» aggiunta!")
            else:
                st.error("Inserisci il nome area.")

    # --- DISEGNO SU PLANIMETRIA ---
    st.markdown("---")
    st.markdown("#### 🖊️ Disegna aree sulla planimetria")
    if "plan_bytes" in st.session_state and CANVAS_OK and PIL_OK:
        from PIL import Image as PILImage
        plan_img = PILImage.open(BytesIO(st.session_state.plan_bytes)).convert("RGB")
        max_w = 900
        if plan_img.width > max_w:
            ratio = max_w / plan_img.width
            plan_img = plan_img.resize((max_w, int(plan_img.height * ratio)))

        canvas_result = st_canvas(
            fill_color="rgba(0, 151, 255, 0.25)",
            stroke_width=2, stroke_color="#00b4d8",
            background_image=plan_img,
            height=plan_img.height, width=plan_img.width,
            drawing_mode="rect", key="canvas_plan",
        )

        if st.button("📥 Importa rettangoli come nuove aree"):
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                n_imp = 0
                for obj in canvas_result.json_data["objects"]:
                    if obj.get("type") == "rect":
                        w_px = obj.get("width",100)
                        h_px = obj.get("height",100)
                        area_m2 = round((w_px * scala_mpp/100) * (h_px * scala_mpp/100), 1)
                        area_m2 = max(5.0, area_m2)
                        st.session_state.aree.append({
                            "nome":        f"Area_{len(st.session_state.aree)+1}",
                            "tipo_locale": "Ufficio VDT",
                            "superficie_m2": area_m2,
                            "altezza_m":   2.70,
                            "lampada":     list(DB_LAMPADE.keys())[0],
                            "sup":         area_m2,
                            "emergenza":   False,
                            "polygon_px":  [
                                [obj["left"], obj["top"]],
                                [obj["left"]+w_px, obj["top"]],
                                [obj["left"]+w_px, obj["top"]+h_px],
                                [obj["left"], obj["top"]+h_px],
                            ],
                        }); n_imp += 1
                st.success(f"✅ {n_imp} aree importate dal disegno.")
            else:
                st.info("Disegna prima almeno un rettangolo sulla planimetria.")
    elif not CANVAS_OK:
        st.warning("Installa `streamlit-drawable-canvas` per disegnare le aree.")
    else:
        st.info("Carica una planimetria nel menu laterale per disegnare le aree.")

    # --- AI VISION ---
    st.markdown("---")
    st.markdown("#### 🤖 AI Vision (riconoscimento automatico aree)")
    if "plan_bytes" in st.session_state:
        col_ai1, col_ai2 = st.columns([2,1])
        with col_ai1:
            ai_url = st.text_input("Endpoint AI", "http://localhost:11434/api/lighting/segment")
        with col_ai2:
            ai_btn = st.button("🔍 Analizza planimetria con AI")
        if ai_btn and REQ_OK:
            with st.spinner("Analisi AI in corso..."):
                try:
                    b64 = base64.b64encode(st.session_state.plan_bytes).decode()
                    resp = requests.post(ai_url, json={"image_base64": b64}, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    areas_found = data.get("areas", [])
                    for a in areas_found:
                        tipo = a.get("type","Ufficio VDT")
                        if tipo not in REQUISITI: tipo = "Ufficio VDT"
                        st.session_state.aree.append({
                            "nome":          a.get("name", f"Area_AI_{len(st.session_state.aree)+1}"),
                            "tipo_locale":   tipo,
                            "superficie_m2": a.get("area_m2", 30.0),
                            "altezza_m":     2.70,
                            "lampada":       list(DB_LAMPADE.keys())[0],
                            "sup":           a.get("area_m2", 30.0),
                            "emergenza":     False,
                            "polygon_px":    a.get("polygon_px",[]),
                        })
                    st.success(f"✅ AI ha trovato {len(areas_found)} aree")
                except Exception as e:
                    st.error(f"Errore AI: {e}\nAvvia il servizio AI locale per usare questa funzione.")
    else:
        st.info("Carica una planimetria per usare AI Vision.")

    # --- LISTA AREE ---
    st.markdown("---")
    st.markdown("#### 📋 Aree inserite")
    if st.session_state.aree:
        for i, a in enumerate(st.session_state.aree):
            req = REQUISITI[a["tipo_locale"]]
            area_type = req["area"]
            badge = "🟢" if area_type=="INT" else ("🟡" if area_type=="EXT" else ("🔴" if area_type=="EM" else "🔵"))
            em_icon = "🚨" if a.get("emergenza") else ""
            c1, c2, c3 = st.columns([5,2,1])
            with c1:
                st.markdown(
                    f'<div class="card"><b>{badge} {i+1}. {a["nome"]}</b>{em_icon} | '
                    f'{a["tipo_locale"]} | {a["superficie_m2"]}m² | h:{a["altezza_m"]}m | '
                    f'Target:{req["lux"]}lux | {a["lampada"][:30]} | '
                    f'<i>{req["norma"]}</i></div>',
                    unsafe_allow_html=True)
            with c2:
                new_sup = st.number_input("m²", value=a["superficie_m2"],
                    key=f"sup_{i}", min_value=1.0, label_visibility="collapsed")
                if new_sup != a["superficie_m2"]:
                    st.session_state.aree[i]["superficie_m2"] = new_sup
                    st.session_state.aree[i]["sup"] = new_sup
            with c3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.aree.pop(i); st.rerun()
    else:
        st.info("Nessuna area inserita. Aggiungi aree manualmente, disegnando sulla planimetria o tramite AI.")

# ============================================================
# TAB 2 — CALCOLI
# ============================================================
with tab2:
    st.subheader("Calcoli Illuminotecnici")
    if not st.session_state.aree:
        st.warning("Aggiungi prima le aree nel tab 🗺️ Aree.")
    else:
        if st.button("▶️ ESEGUI CALCOLI", type="primary"):
            ris, ox = [], 0.0
            for a in st.session_state.aree:
                c_norm = calcola_area(a, "normale")
                c_norm["lampada_usata"] = c_norm.get("lampada_usata", a["lampada"])
                ris.append({**a, "calc": c_norm, "offset_x": ox, "offset_y": 0,
                             "lampada_calc": c_norm["lampada_usata"]})
                ox += np.sqrt(a["superficie_m2"]) + 1.5
                if a.get("emergenza"):
                    c_em = calcola_area(a, "emergenza")
                    ris.append({**a,
                                "nome": a["nome"] + " 🚨EMERG.",
                                "calc": c_em,
                                "offset_x": ox, "offset_y": 0,
                                "lampada_calc": c_em["lampada_usata"]})
                    ox += np.sqrt(a["superficie_m2"]) + 1.5
            st.session_state.risultati = ris
            st.success(f"✅ Calcoli completati — {len(ris)} aree elaborate.")

        if "risultati" in st.session_state:
            rl = st.session_state.risultati
            tl = sum(r["calc"]["n"] for r in rl)
            tw = sum(r["calc"]["W_t"] for r in rl)
            tm2= sum(r["sup"] for r in rl)
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Aree elaborate", len(rl))
            c2.metric("Lampade totali", tl)
            c3.metric("Potenza totale", f"{tw} W")
            c4.metric("Superficie", f"{tm2:.0f} m²")
            c5.metric("W/m² medio", f"{tw/max(tm2,1):.1f}")

            rows = []
            for r in rl:
                lk  = r["calc"].get("lampada_usata", r["lampada"])
                req = REQUISITI[r["tipo_locale"]]
                rows.append({
                    "Area": r["nome"], "Tipo": r["tipo_locale"],
                    "m²": r["sup"], "N": r["calc"]["n"],
                    "Lux target": r["calc"]["E_t"], "Lux ottenuto": r["calc"]["E_m"],
                    "W": r["calc"]["W_t"], "W/m²": r["calc"]["wm2"],
                    "Lampada": lk[:35],
                    "Lux ✓": r["calc"]["ok_lux"], "UGR ✓": r["calc"]["ok_ugr"],
                    "Ra ✓": r["calc"]["ok_ra"],
                    "Norma": req["norma"],
                    "Modalità": r["calc"].get("modalita","normale"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Scarica CSV", csv,
                f"calcoli_{datetime.now():%Y%m%d}.csv", "text/csv")

# ============================================================
# TAB 3 — TAVOLA A3
# ============================================================
with tab3:
    st.subheader("Tavola Illuminotecnica A3")
    if "risultati" not in st.session_state:
        st.warning("Esegui prima i calcoli.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📄 GENERA TAVOLA A3 PDF", type="primary"):
                with st.spinner("Generazione Tavola A3..."):
                    prog = {"nome":nome










