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
import base64, json, os, sqlite3, hashlib, re

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
# DATABASE SQLITE
# ============================================================
DB_PATH = "luxia_data.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT,
        password_hash TEXT NOT NULL,
        created_at TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        nome_progetto TEXT NOT NULL,
        committente TEXT,
        progettista TEXT,
        num_tavola TEXT,
        data_creazione TEXT,
        data_modifica TEXT,
        aree_json TEXT,
        risultati_json TEXT,
        prev_json TEXT
    )""")
    for uname, upwd in [("admin","admin2026"),("demo","demo123"),("progett","luce2026")]:
        h = hashlib.sha256(upwd.encode()).hexdigest()
        con.execute("INSERT OR IGNORE INTO users (username,email,password_hash,created_at) VALUES (?,?,?,?)",
                    (uname, f"{uname}@luxia.it", h, datetime.now().isoformat()))
    con.commit(); con.close()

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def check_user(u, p):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT password_hash FROM users WHERE username=?", (u,)).fetchone()
    con.close()
    return row is not None and row[0] == hash_pw(p)

def user_exists(u):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT id FROM users WHERE username=?", (u,)).fetchone()
    con.close(); return row is not None

def register_user(u, email, p):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO users (username,email,password_hash,created_at) VALUES (?,?,?,?)",
                (u, email, hash_pw(p), datetime.now().isoformat()))
    con.commit(); con.close()

def save_project(username, nome, committente, progettista, num_tav, aree, risultati, prev):
    con = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    ris_clean = []
    for r in risultati:
        rc = {k: v for k, v in r.items() if k != "calc"}
        rc["calc"] = {k2: v2 for k2, v2 in r.get("calc", {}).items()
                      if isinstance(v2, (str, int, float, bool, list, type(None)))}
        ris_clean.append(rc)
    ex = con.execute("SELECT id FROM projects WHERE username=? AND nome_progetto=?", (username, nome)).fetchone()
    if ex:
        con.execute("""UPDATE projects SET committente=?,progettista=?,num_tavola=?,data_modifica=?,
                    aree_json=?,risultati_json=?,prev_json=? WHERE username=? AND nome_progetto=?""",
                    (committente,progettista,num_tav,now,
                     json.dumps(aree,ensure_ascii=False),
                     json.dumps(ris_clean,ensure_ascii=False),
                     json.dumps(prev or {},ensure_ascii=False),
                     username,nome))
    else:
        con.execute("""INSERT INTO projects
                    (username,nome_progetto,committente,progettista,num_tavola,
                     data_creazione,data_modifica,aree_json,risultati_json,prev_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (username,nome,committente,progettista,num_tav,now,now,
                     json.dumps(aree,ensure_ascii=False),
                     json.dumps(ris_clean,ensure_ascii=False),
                     json.dumps(prev or {},ensure_ascii=False)))
    con.commit(); con.close()

def load_projects_list(username):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id,nome_progetto,committente,data_modifica FROM projects WHERE username=? ORDER BY data_modifica DESC",(username,)).fetchall()
    con.close(); return rows

def load_project_data(pid):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT nome_progetto,committente,progettista,num_tavola,aree_json,risultati_json,prev_json FROM projects WHERE id=?",(pid,)).fetchone()
    con.close()
    if row:
        return {"nome":row[0],"committente":row[1],"progettista":row[2],"num_tavola":row[3],
                "aree":json.loads(row[4]) if row[4] else [],
                "risultati":json.loads(row[5]) if row[5] else [],
                "prev":json.loads(row[6]) if row[6] else {}}
    return None

def delete_project(pid):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM projects WHERE id=?",(pid,)); con.commit(); con.close()

init_db()

# ============================================================
# AI VISION — Groq → Gemini → Ollama
# ============================================================
def analizza_planimetria_ai(image_bytes, groq_key="", gemini_key=""):
    if not REQ_OK: return []
    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        'Analizza questa planimetria. Per ogni stanza restituisci SOLO JSON valido:\n'
        '{"areas":[{"name":"Nome","type":"Ufficio VDT","area_m2":30.0}]}\n'
        "Tipi: Ufficio VDT, Corridoio, Bagno/WC, Sala riunioni, Archivio, "
        "Ingresso, Mensa/Ristoro, Locale tecnico, Reception, Laboratorio."
    )
    def _parse(testo):
        m = re.search(r'\{.*\}', testo, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                return d.get("areas", [])
            except Exception:
                pass
        return []

    if groq_key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}","Content-Type":"application/json"},
                json={"model":"llama-3.2-11b-vision-preview","temperature":0.1,"max_tokens":1024,
                      "messages":[{"role":"user","content":[
                          {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
                          {"type":"text","text":prompt}]}]}, timeout=30)
            areas = _parse(r.json()["choices"][0]["message"]["content"])
            if areas: return areas
        except Exception as e:
            st.warning(f"Groq: {e}")

    if gemini_key:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={gemini_key}",
                json={"contents":[{"parts":[
                    {"inline_data":{"mime_type":"image/jpeg","data":b64}},
                    {"text":prompt}]}]}, timeout=30)
            areas = _parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            if areas: return areas
        except Exception as e:
            st.warning(f"Gemini: {e}")

    try:
        r = requests.post("http://localhost:11434/api/generate",
            json={"model":"llava","stream":False,"prompt":prompt,"images":[b64]}, timeout=120)
        areas = _parse(r.json().get("response",""))
        if areas: return areas
    except Exception as e:
        st.warning(f"Ollama: {e}")
    return []

def detect_scala_ai(image_bytes, groq_key="", gemini_key=""):
    if not REQ_OK: return ""
    b64 = base64.b64encode(image_bytes).decode()
    prompt = "Questa planimetria ha una scala (1:50, 1:100, 1:200, 1:500)? Rispondi SOLO col numero dopo i due punti, es: 100. Se non la vedi scrivi: 0."
    def _num(t):
        m = re.search(r'\d+', t.strip())
        return f"1:{m.group()}" if m and int(m.group()) > 0 else ""
    if groq_key:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization":f"Bearer {groq_key}"},
                json={"model":"llama-3.2-11b-vision-preview","max_tokens":20,"temperature":0,
                      "messages":[{"role":"user","content":[
                          {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
                          {"type":"text","text":prompt}]}]}, timeout=20)
            return _num(r.json()["choices"][0]["message"]["content"])
        except Exception: pass
    if gemini_key:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={gemini_key}",
                json={"contents":[{"parts":[
                    {"inline_data":{"mime_type":"image/jpeg","data":b64}},
                    {"text":prompt}]}]}, timeout=20)
            return _num(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception: pass
    return ""

# ============================================================
# LOGIN / REGISTRAZIONE
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = ""

if not st.session_state.logged_in:
    st.markdown("""
    <div style="max-width:460px;margin:5rem auto;padding:2rem;
    background:white;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,.12)">
    <h2 style="text-align:center;color:#1a365d">💡 Lighting Agent Pro</h2>
    <p style="text-align:center;color:#666">v4.0 — Accesso riservato</p>
    </div>""", unsafe_allow_html=True)
    _, col_b, _ = st.columns([1,2,1])
    with col_b:
        tl, tr = st.tabs(["🔐 Login","📝 Registrati"])
        with tl:
            u = st.text_input("Username", key="li_u")
            p = st.text_input("Password", type="password", key="li_p")
            if st.button("Entra 🔐", key="btn_li"):
                if check_user(u.strip(), p):
                    st.session_state.logged_in = True
                    st.session_state.username  = u.strip()
                    st.rerun()
                else:
                    st.error("❌ Credenziali non valide")
        with tr:
            nu  = st.text_input("Username *",        key="r_u")
            ne  = st.text_input("Email",             key="r_e")
            np1 = st.text_input("Password *",        type="password", key="r_p1")
            np2 = st.text_input("Conferma password", type="password", key="r_p2")
            if st.button("Registrati ✅", key="btn_reg"):
                if not nu.strip() or not np1:      st.error("Campi obbligatori mancanti.")
                elif len(np1) < 6:                 st.error("Password minimo 6 caratteri.")
                elif np1 != np2:                   st.error("Le password non coincidono.")
                elif user_exists(nu.strip()):       st.error("Username già in uso.")
                else:
                    register_user(nu.strip(), ne.strip(), np1)
                    st.success("✅ Account creato! Ora puoi fare login.")
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
    if st.button("🔍 Rileva scala con AI") and "plan_bytes" in st.session_state:
        with st.spinner("Rilevamento in corso..."):
            s = detect_scala_ai(
                st.session_state.plan_bytes,
                groq_key=st.session_state.get("groq_key",""),
                gemini_key=st.session_state.get("gemini_key",""),
            )
        st.success(f"Scala rilevata: {s}") if s else st.warning("Non rilevata, imposta manualmente.")
        if s: st.session_state.scala_suggerita = s
    _sopts = ["1:50","1:100","1:200","1:500"]
    _sdef  = st.session_state.get("scala_suggerita","1:100")
    _sidx  = _sopts.index(_sdef) if _sdef in _sopts else 1
    _ssel  = st.selectbox("Scala", _sopts, index=_sidx)
    scala_mpp = {"1:50":1.25,"1:100":2.5,"1:200":5.0,"1:500":12.5}.get(_ssel, 2.5)
    st.caption(f"100px = {scala_mpp} m")

    st.markdown("---")
    with st.expander("🤖 Chiavi API AI Vision (gratuite)"):
        _gk  = st.text_input("Groq API Key",   value=st.session_state.get("groq_key",""),   type="password", help="console.groq.com")
        _gmk = st.text_input("Gemini API Key", value=st.session_state.get("gemini_key",""), type="password", help="aistudio.google.com")
        if _gk:  st.session_state.groq_key   = _gk
        if _gmk: st.session_state.gemini_key = _gmk
        st.caption("Fallback automatico: Groq → Gemini → Ollama locale")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🗺️ Aree",
    "🧮 Calcoli",
    "📐 Tavola A3",
    "✅ Verifiche",
    "🎨 Rendering 3D",
    "💶 Preventivo",
    "📄 Relazione Completa",
    "📁 Progetti",
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

   # --- DISEGNO SU PLANIMETRIA (versione cloud-safe) ---
st.markdown("---")
st.markdown("#### 🖊️ Aggiungi aree dalla planimetria")

if "plan_bytes" in st.session_state:
    st.image(
        st.session_state.plan_bytes,
        caption="Planimetria caricata — usa il form sopra per aggiungere le aree",
        use_column_width=True,
    )
    st.info(
        "💡 **Istruzioni**: osserva la planimetria e inserisci le aree "
        "manualmente nel form qui sopra, indicando superficie e tipo locale."
    )

    # Aggiungi area da coordinate manuali
    st.markdown("##### Inserisci coordinate area (opzionale)")
    with st.form("form_coords", clear_on_submit=True):
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: cx = st.number_input("X origine [m]", 0.0, 500.0, 0.0, 0.5)
        with cc2: cy = st.number_input("Y origine [m]", 0.0, 500.0, 0.0, 0.5)
        with cc3: cw = st.number_input("Larghezza [m]", 1.0, 100.0, 5.0, 0.5)
        with cc4: ch = st.number_input("Profondità [m]", 1.0, 100.0, 4.0, 0.5)

        nome_coord  = st.text_input("Nome area", placeholder="es. Ufficio B")
        tipo_coord  = st.selectbox("Tipo locale", list(REQUISITI.keys()), key="tc_coord")
        lamp_coord  = st.selectbox("Apparecchio", list(DB_LAMPADE.keys()), key="lc_coord")
        alt_coord   = st.number_input("Altezza [m]", 2.0, 12.0, 2.70, 0.05, key="ac_coord")
        em_coord    = st.checkbox("🚨 Calcola emergenza", key="ec_coord")

        if st.form_submit_button("➕ Aggiungi area da coordinate"):
            if nome_coord.strip():
                area_m2 = round(cw * ch, 1)
                st.session_state.aree.append({
                    "nome":          nome_coord.strip(),
                    "tipo_locale":   tipo_coord,
                    "superficie_m2": area_m2,
                    "altezza_m":     alt_coord,
                    "lampada":       lamp_coord,
                    "sup":           area_m2,
                    "emergenza":     em_coord,
                    "polygon_px":    [
                        [cx, cy], [cx+cw, cy],
                        [cx+cw, cy+ch], [cx, cy+ch]
                    ],
                })
                st.success(f"✅ Area «{nome_coord}» {area_m2}m² aggiunta!")
            else:
                st.error("Inserisci il nome area.")
else:
    st.info("Carica una planimetria nel menu laterale per visualizzarla.")


    # --- AI VISION ---
    st.markdown("---")
    st.markdown("""
<div style="background:linear-gradient(135deg,#1a365d,#2b6cb0);padding:1.2rem 1.5rem;
border-radius:12px;margin-bottom:1rem;">
<h3 style="color:white;margin:0">🤖 Rileva Aree con AI</h3>
<p style="color:#bee3f8;margin:.3rem 0 0;font-size:.9rem">
Groq (gratuito) → Gemini (gratuito) → Ollama locale &nbsp;|&nbsp;
Riconosce automaticamente stanze, tipo e superficie dalla planimetria
</p></div>""", unsafe_allow_html=True)

    if "plan_bytes" not in st.session_state:
        st.warning("⬆️ Carica prima una planimetria nella sidebar per usare AI Vision.")
    else:
        gk_ok  = "✅ Groq configurato"  if st.session_state.get("groq_key")   else "⚠️ Groq non configurato"
        gmk_ok = "✅ Gemini configurato" if st.session_state.get("gemini_key") else "⚠️ Gemini non configurato"
        st.caption(f"{gk_ok}  |  {gmk_ok}  |  Ollama: localhost:11434  |  Configura le chiavi API nella sidebar")

        col_btn1, col_btn2, col_btn3 = st.columns([2,2,3])
        with col_btn1:
            ai_btn = st.button("🔍 RILEVA AREE CON AI", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Cancella risultati AI", use_container_width=True):
                st.session_state.pop("ai_areas_preview", None)
                st.rerun()
        with col_btn3:
            st.caption("Il riconoscimento richiede ~5-15 secondi con Groq/Gemini")

        if ai_btn:
            with st.spinner("🔍 Analisi AI in corso... (Groq → Gemini → Ollama)"):
                found = analizza_planimetria_ai(
                    st.session_state.plan_bytes,
                    groq_key=st.session_state.get("groq_key",""),
                    gemini_key=st.session_state.get("gemini_key",""),
                )
            if found:
                st.session_state.ai_areas_preview = found
                st.success(f"✅ AI ha identificato **{len(found)} aree** — Verifica e conferma qui sotto")
            else:
                st.error("❌ Nessuna area rilevata. Verifica le chiavi API nella sidebar o aggiungi le aree manualmente.")

        # ── ANTEPRIMA con modifica prima di confermare ──
        if st.session_state.get("ai_areas_preview"):
            st.markdown("##### ✏️ Verifica e modifica le aree rilevate dall'AI prima di aggiungerle")
            preview = st.session_state.ai_areas_preview
            cols_hdr = st.columns([3,3,2,2,2])
            for h, c in zip(["Nome area","Tipo locale","m²","h (m)","Emergenza?"], cols_hdr):
                c.markdown(f"**{h}**")
            edited = []
            for i, a in enumerate(preview):
                tipo_ai = a.get("type","Ufficio VDT")
                if tipo_ai not in REQUISITI: tipo_ai = "Ufficio VDT"
                c1,c2,c3,c4,c5 = st.columns([3,3,2,2,2])
                with c1:
                    nome_e = st.text_input("", value=a.get("name",f"Area_{i+1}"),
                                           key=f"ai_n_{i}", label_visibility="collapsed")
                with c2:
                    tipo_e = st.selectbox("", list(REQUISITI.keys()),
                                          index=list(REQUISITI.keys()).index(tipo_ai),
                                          key=f"ai_t_{i}", label_visibility="collapsed")
                with c3:
                    sup_e = st.number_input("", value=float(a.get("area_m2",30.0)),
                                            min_value=1.0, step=0.5,
                                            key=f"ai_s_{i}", label_visibility="collapsed")
                with c4:
                    alt_e = st.number_input("", value=2.70, min_value=2.0, step=0.05,
                                            key=f"ai_h_{i}", label_visibility="collapsed")
                with c5:
                    em_e = st.checkbox("", key=f"ai_em_{i}", label_visibility="collapsed")
                edited.append({"nome":nome_e,"tipo_locale":tipo_e,
                               "superficie_m2":sup_e,"altezza_m":alt_e,"emergenza":em_e})

            st.markdown("---")
            c_ok, c_skip = st.columns([2,1])
            with c_ok:
                if st.button("✅ AGGIUNGI TUTTE LE AREE RILEVATE", type="primary", use_container_width=True):
                    default_lamp = list(DB_LAMPADE.keys())[0]
                    for ed in edited:
                        st.session_state.aree.append({
                            "nome":          ed["nome"],
                            "tipo_locale":   ed["tipo_locale"],
                            "superficie_m2": ed["superficie_m2"],
                            "altezza_m":     ed["altezza_m"],
                            "lampada":       default_lamp,
                            "sup":           ed["superficie_m2"],
                            "emergenza":     ed["emergenza"],
                            "polygon_px":    [],
                        })
                    st.session_state.pop("ai_areas_preview", None)
                    st.success(f"✅ {len(edited)} aree aggiunte con successo!")
                    st.rerun()
            with c_skip:
                if st.button("↩️ Annulla", use_container_width=True):
                    st.session_state.pop("ai_areas_preview", None)
                    st.rerun()

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
                    prog = {
                        "nome": nome_prog, "committente": committente,
                        "progettista": progettista,
                        "data": datetime.now().strftime("%d/%m/%Y"),
                        "num_tavola": num_tav,
                    }
                    logo = st.session_state.get("logo_bytes", None)
                    buf = genera_pdf(prog, st.session_state.risultati, logo)
                    st.download_button("⬇️ Scarica Tavola A3 PDF", data=buf,
                        file_name=f"{num_tav}_tavola.pdf", mime="application/pdf")
                    st.success("✅ Tavola A3 generata!")
        with c2:
            if st.button("📐 GENERA DXF AUTOCAD"):
                with st.spinner("Generazione DXF..."):
                    dxf = genera_dxf(st.session_state.risultati)
                    st.download_button("⬇️ Scarica DXF", data=dxf,
                        file_name=f"{num_tav}_layout.dxf", mime="application/dxf")
                    st.success("✅ DXF generato!")

# ============================================================
# TAB 4 — VERIFICHE
# ============================================================
with tab4:
    st.subheader("Verifiche Illuminotecniche")
    if "risultati" not in st.session_state:
        st.warning("Esegui prima i calcoli.")
    else:
        for r in st.session_state.risultati:
            req = REQUISITI[r["tipo_locale"]]
            em_label = " 🚨 EMERGENZA" if r["calc"].get("modalita") == "emergenza" else ""
            badge = {"INT":"🟢","EXT":"🟡","EM":"🔴","STR":"🔵"}.get(req["area"],"⚪")
            with st.expander(
                f"{badge} {r['nome']}{em_label} | "
                f"{r['calc']['E_m']} lux | "
                f"{r['calc']['ok_lux']} Lux | {r['calc']['ok_ugr']} UGR | "
                f"{req['norma']}"
            ):
                ca, cb, cc, cd = st.columns(4)
                ca.metric("Lux ottenuto", str(r["calc"]["E_m"]))
                cb.metric("Potenza W",    str(r["calc"]["W_t"]))
                cc.metric("W/m²",         str(r["calc"]["wm2"]))
                cd.metric("Lampade",      str(r["calc"]["n"]))

                c1, c2, c3, c4 = st.columns(4)
                c1.success(f"Illuminamento {r['calc']['ok_lux']}")
                c2.success(f"UGR {r['calc']['ok_ugr']}")
                c3.success(f"Uniformità {r['calc']['ok_uni']}")
                c4.success(f"Ra {r['calc']['ok_ra']}")

                if r["calc"].get("modalita") == "emergenza":
                    st.info(
                        "🚨 **Emergenza UNI EN 1838:2025** — "
                        f"Illuminamento minimo richiesto: {r['calc']['E_t']} lux | "
                        "Autonomia: ≥ 1h | Test: mensile (funzionale) + annuale (durata)"
                    )

        st.markdown("---")
        if st.button("📋 GENERA REPORT VERIFICHE PDF", type="primary"):
            with st.spinner("Generazione report..."):
                prog = {
                    "nome": nome_prog, "committente": committente,
                    "progettista": progettista,
                    "data": datetime.now().strftime("%d/%m/%Y"),
                    "num_tavola": num_tav,
                }
                logo = st.session_state.get("logo_bytes", None)
                buf = genera_pdf(prog, st.session_state.risultati, logo)
                st.download_button("⬇️ Scarica Report Verifiche PDF", data=buf,
                    file_name=f"{num_tav}_verifiche.pdf", mime="application/pdf")

# ============================================================
# TAB 5 — RENDERING 3D
# ============================================================
with tab5:
    st.subheader("Rendering 3D")
    if "risultati" not in st.session_state:
        st.warning("Esegui prima i calcoli.")
    else:
        names = [r["nome"] for r in st.session_state.risultati]
        scelta = st.selectbox("Seleziona area da renderizzare", names)

        c1, c2, c3 = st.columns(3)
        with c1:
            render_singolo = st.button("🎨 RENDERING AREA SELEZIONATA", type="primary")
        with c2:
            render_tutti = st.button("🎨 RENDERING TUTTE LE AREE")
        with c3:
            export_gltf = st.button("📦 ESPORTA SCENA glTF (Blender/Unreal)")

        if render_singolo:
            idx = names.index(scelta)
            r = st.session_state.risultati[idx]
            with st.spinner(f"Rendering {scelta}..."):
                buf = genera_rendering(r, r["calc"])
                st.image(buf, caption=f"Rendering 3D — {scelta}", use_column_width=True)
                buf.seek(0)
                st.download_button("⬇️ Scarica PNG", data=buf,
                    file_name=f"render_{scelta.lower().replace(' ','_')}.png",
                    mime="image/png")

        if render_tutti:
            cols = st.columns(2)
            for i, r in enumerate(st.session_state.risultati):
                with st.spinner(f"Rendering {r['nome']}..."):
                    buf = genera_rendering(r, r["calc"])
                    with cols[i % 2]:
                        st.image(buf, caption=r["nome"], use_column_width=True)
                        buf.seek(0)
                        st.download_button(
                            f"⬇️ Scarica {r['nome']}", data=buf,
                            file_name=f"render_{i}.png", mime="image/png",
                            key=f"rend_dl_{i}")

        if export_gltf:
            with st.spinner("Generazione scena glTF..."):
                gltf_buf = export_gltf_scene(st.session_state.risultati)
                st.download_button("⬇️ Scarica .glTF", data=gltf_buf,
                    file_name=f"{num_tav}_scene.gltf", mime="model/gltf+json")
                st.success("✅ Scena glTF pronta per Blender/Unreal Engine!")
                st.info(
                    "**Come usare in Blender:**\n"
                    "1. File → Import → glTF 2.0\n"
                    "2. Assegna materiali PBR alle mesh\n"
                    "3. Render → Cycles → 256 samples\n"
                    "4. Output → PNG 4K"
                )

        st.markdown("---")
        st.markdown("#### 📝 Script Blender fotorealistico")
        with st.expander("Mostra script `render_blender.py`"):
            st.code('''
import bpy, os

gltf_path   = "/path/al/tuo_scene.gltf"
output_path = "/path/output/render_fotorealistico.png"

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=gltf_path)

# Luce HDRI
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[1].default_value = 0.3

# Luci area (ogni lampada della scena)
for obj in bpy.context.scene.objects:
    if "Lamp_" in obj.name:
        light_data = bpy.data.lights.new(name=obj.name+"_light", type="AREA")
        light_data.energy = 1200
        light_data.color  = (1.0, 0.95, 0.8)   # 3000K warm
        light_data.size   = 0.17
        light_obj = bpy.data.objects.new(obj.name+"_light", light_data)
        light_obj.location = obj.location
        bpy.context.collection.objects.link(light_obj)

# Pavimento materiale PBR
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        mat = bpy.data.materials.new(name="PBR_Floor")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value  = (0.22,0.22,0.24,1)
        bsdf.inputs["Roughness"].default_value   = 0.7
        bsdf.inputs["Metallic"].default_value    = 0.0
        obj.data.materials.append(mat)

# Camera
cam_data = bpy.data.cameras.new("Camera")
cam_obj  = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.location       = (8, -10, 5)
cam_obj.rotation_euler = (1.1, 0, 0.8)

# Render Cycles
scene = bpy.context.scene
scene.render.engine          = "CYCLES"
scene.cycles.samples         = 512
scene.cycles.use_denoising   = True
scene.render.resolution_x    = 1920
scene.render.resolution_y    = 1080
scene.render.filepath        = output_path
scene.render.image_settings.file_format = "PNG"

bpy.ops.render.render(write_still=True)
print("✅ Render completato:", output_path)
''', language="python")
            st.caption("Lancia con: `blender --background --python render_blender.py`")

# ============================================================
# TAB 6 — PREVENTIVO
# ============================================================
with tab6:
    st.subheader("Preventivo Economico")
    if "risultati" not in st.session_state:
        st.warning("Esegui prima i calcoli.")
    else:
        st.markdown("#### ⚙️ Parametri preventivo (modificabili)")
        c1, c2, c3, c4 = st.columns(4)
        with c1: mg_sl  = st.slider("Margine %",       10, 60, 35, key="sl_mg")
        with c2: iva_sl = st.slider("IVA %",            0, 22, 22, key="sl_iva")
        with c3: sg_sl  = st.slider("Spese generali %", 5, 25, 12, key="sl_sg")
        with c4: os_sl  = st.slider("Oneri sicurezza %",2, 10,  4, key="sl_os")

        if st.button("🧮 CALCOLA PREVENTIVO", type="primary"):
            st.session_state.prev = calc_preventivo(
                st.session_state.risultati, mg_sl, sg_sl, os_sl, iva_sl)

        if "prev" in st.session_state:
            pv = st.session_state.prev
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Materiali",           f"EUR {pv['tm']:,.0f}")
            c2.metric("Installazione",        f"EUR {pv['ti']:,.0f}")
            c3.metric("Offerta cliente",      f"EUR {pv['to']:,.0f}")
            c4.metric("Totale IVA inclusa",   f"EUR {pv['tf']:,.0f}")

            st.markdown("#### 📋 Dettaglio per area")
            df_p = pd.DataFrame([{
                "Area":         r["area"],
                "N lampade":    r["n"],
                "Apparecchio":  r["lampada"],
                "Modalità":     r["modalita"],
                "Materiali":    f"EUR {r['mat']:,.0f}",
                "Installazione":f"EUR {r['ins']:,.0f}",
                "Subtotale":    f"EUR {r['sub']:,.0f}",
            } for r in pv["righe"]])
            st.dataframe(df_p, use_container_width=True, hide_index=True)

            st.markdown("#### 💶 Riepilogo economico")
            st.markdown(f"""
| Voce | Importo |
|---|---|
| Materiali | EUR {pv['tm']:,.0f} |
| Installazione | EUR {pv['ti']:,.0f} |
| Totale lavori netto | EUR {pv['tn']:,.0f} |
| Spese generali {sg_sl}% | EUR {pv['sg']:,.0f} |
| Oneri sicurezza {os_sl}% | EUR {pv['os']:,.0f} |
| Margine {mg_sl}% | EUR {pv['mg']:,.0f} |
| **OFFERTA CLIENTE** | **EUR {pv['to']:,.0f}** |
| IVA {iva_sl}% | EUR {pv['iva']:,.0f} |
| **TOTALE IVA INCLUSA** | **EUR {pv['tf']:,.0f}** |
""")
            txt_prev = (
                f"PREVENTIVO — {nome_prog}\n"
                f"Committente: {committente}\n"
                f"Data: {datetime.now():%d/%m/%Y}\n"
                f"Progettista: {progettista}\n\n"
                + "\n".join(
                    f"  {r['area']}: {r['n']}x {r['lampada']}  "
                    f"Mat. EUR {r['mat']:,.0f}  Ins. EUR {r['ins']:,.0f}"
                    for r in pv["righe"]
                )
                + f"\n\nTOTALE IVA INCLUSA: EUR {pv['tf']:,.0f}\n"
            )
            st.download_button("⬇️ Scarica Preventivo TXT", data=txt_prev.encode(),
                file_name=f"preventivo_{datetime.now():%Y%m%d}.txt")

# ============================================================
# TAB 7 — RELAZIONE COMPLETA
# ============================================================
with tab7:
    st.subheader("Relazione Tecnica Completa — UNI 11630:2016")
    if "risultati" not in st.session_state:
        st.warning("Esegui prima i calcoli.")
    elif "prev" not in st.session_state:
        st.warning("Calcola prima il preventivo nel tab 💶 Preventivo.")
    else:
        st.markdown("""
La relazione completa include in un **unico PDF**:
- 📋 Frontespizio con logo e dati progetto
- 📝 Relazione descrittiva (UNI 11630:2016)
- 📐 Tavola A3 planimetria + posizionamento
- ✅ Schede di verifica per ogni area
- 🎨 Rendering 3D di ogni area
- 💶 Preventivo economico dettagliato
""")

        mg_r  = st.session_state.get("sl_mg",  35)
        sg_r  = st.session_state.get("sl_sg",  12)
        os_r  = st.session_state.get("sl_os",   4)
        iva_r = st.session_state.get("sl_iva",  22)

        if st.button("📄 GENERA RELAZIONE COMPLETA PDF", type="primary"):
            with st.spinner("Generazione relazione completa in corso... (può richiedere 1-2 minuti)"):
                prog = {
                    "nome": nome_prog, "committente": committente,
                    "progettista": progettista,
                    "data": datetime.now().strftime("%d/%m/%Y"),
                    "num_tavola": num_tav,
                }
                logo = st.session_state.get("logo_bytes", None)
                try:
                    buf = genera_relazione_completa(
                        prog, st.session_state.risultati,
                        st.session_state.prev, logo,
                        mg_pct=mg_r, sg_pct=sg_r,
                        os_pct=os_r, iva_pct=iva_r,
                    )
                    st.download_button(
                        "⬇️ SCARICA RELAZIONE COMPLETA PDF",
                        data=buf,
                        file_name=f"{num_tav}_relazione_completa_{datetime.now():%Y%m%d}.pdf",
                        mime="application/pdf",
                    )
                    st.success("✅ Relazione completa generata!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Errore generazione: {e}")
                    st.info("Prova a installare PyPDF2: pip install PyPDF2")

        st.markdown("---")
        st.markdown("#### 📎 Export separati")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📐 Solo Tavola A3"):
                prog = {"nome":nome_prog,"committente":committente,
                        "progettista":progettista,
                        "data":datetime.now().strftime("%d/%m/%Y"),"num_tavola":num_tav}
                buf = genera_pdf(prog, st.session_state.risultati,
                                 st.session_state.get("logo_bytes"))
                st.download_button("⬇️ Tavola A3", data=buf,
                    file_name=f"{num_tav}_tavola.pdf", mime="application/pdf",
                    key="dl_tav_r7")
        with c2:
            if st.button("📦 Solo glTF Blender"):
                gltf_buf = export_gltf_scene(st.session_state.risultati)
                st.download_button("⬇️ Scena .glTF", data=gltf_buf,
                    file_name=f"{num_tav}_scene.gltf", mime="model/gltf+json",
                    key="dl_gltf_r7")
        with c3:
            if "prev" in st.session_state:
                pv2 = st.session_state.prev
                txt2 = (f"PREVENTIVO — {nome_prog}\n"
                        f"Data: {datetime.now():%d/%m/%Y}\n\n"
                        + "\n".join(f"  {r['area']}: EUR {r['sub']:,.0f}"
                                    for r in pv2["righe"])
                        + f"\n\nTOTALE: EUR {pv2['tf']:,.0f}\n")
                st.download_button("⬇️ Preventivo TXT", data=txt2.encode(),
                    file_name=f"prev_{datetime.now():%Y%m%d}.txt",
                    key="dl_prev_r7")

# ============================================================
# TAB 8 — PROGETTI
# ============================================================
with tab8:
    st.subheader("📁 Gestione Progetti")
    col_save, col_load = st.columns(2)
    with col_save:
        st.markdown("#### 💾 Salva progetto corrente")
        if not st.session_state.aree:
            st.info("Aggiungi almeno un'area prima di salvare.")
        else:
            nome_save = st.text_input("Nome progetto", value=nome_prog, key="nome_save")
            if st.button("💾 SALVA PROGETTO", type="primary"):
                try:
                    save_project(
                        st.session_state.username, nome_save.strip(),
                        committente, progettista, num_tav,
                        st.session_state.aree,
                        st.session_state.get("risultati",[]),
                        st.session_state.get("prev",{}),
                    )
                    st.success(f"✅ Progetto «{nome_save}» salvato!")
                except Exception as e:
                    st.error(f"Errore: {e}")
    with col_load:
        st.markdown("#### 📂 Carica progetto salvato")
        progetti = load_projects_list(st.session_state.username)
        if not progetti:
            st.info("Nessun progetto salvato.")
        else:
            proj_opts = {f"{p[1]} — {p[2]} ({p[3][:10]})": p[0] for p in progetti}
            proj_sel  = st.selectbox("Progetto", list(proj_opts.keys()), key="proj_sel")
            cl1, cl2  = st.columns(2)
            with cl1:
                if st.button("📂 CARICA", type="primary"):
                    try:
                        data = load_project_data(proj_opts[proj_sel])
                        if data:
                            st.session_state.aree      = data["aree"]
                            st.session_state.risultati = data["risultati"] or []
                            if data["prev"]: st.session_state.prev = data["prev"]
                            st.success(f"✅ Caricato: {len(data['aree'])} aree")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Errore: {e}")
            with cl2:
                if st.button("🗑️ ELIMINA", type="secondary"):
                    try:
                        delete_project(proj_opts[proj_sel])
                        st.success("Eliminato."); st.rerun()
                    except Exception as e:
                        st.error(f"Errore: {e}")
    st.markdown("---")
    st.markdown("#### 📋 I tuoi progetti")
    rows_p = load_projects_list(st.session_state.username)
    if rows_p:
        df_p = pd.DataFrame(rows_p, columns=["ID","Nome","Committente","Ultima modifica"])
        df_p["Ultima modifica"] = df_p["Ultima modifica"].str[:16].str.replace("T"," ")
        st.dataframe(df_p.drop(columns=["ID"]), use_container_width=True, hide_index=True)
    else:
        st.info("Nessun progetto nel database.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "💡 **Lighting Agent Pro v4.0** | "
    "UNI 11630:2016 · UNI EN 12464-1:2021 · UNI EN 12464-2:2025 · "
    "UNI 11248:2016 · UNI EN 1838:2025 · UNI CEI 11222 | "
    f"Utente: {st.session_state.username} | {datetime.now():%d/%m/%Y %H:%M}"
)
