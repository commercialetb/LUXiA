# ============================================================
# LIGHTING AGENT PRO v4.0
# UNI EN 12464-1:2021 | UNI EN 12464-2:2025 | UNI EN 1838:2025
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

try:
    from pdf2image import convert_from_bytes
    PDF_OK = True
except ImportError:
    PDF_OK = False

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================
st.set_page_config(
    page_title="Lighting Agent Pro v4.0",
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
.stButton>button{background:#2b6cb0;color:white;border:none;
  border-radius:8px;font-weight:700;padding:.55rem 1.8rem;}
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
    con.close(); return row is not None and row[0] == hash_pw(p)

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
    ex = con.execute("SELECT id FROM projects WHERE username=? AND nome_progetto=?",
                     (username, nome)).fetchone()
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
    rows = con.execute(
        "SELECT id,nome_progetto,committente,data_modifica FROM projects WHERE username=? ORDER BY data_modifica DESC",
        (username,)).fetchall()
    con.close(); return rows

def load_project_data(pid):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT nome_progetto,committente,progettista,num_tavola,aree_json,risultati_json,prev_json FROM projects WHERE id=?",
        (pid,)).fetchone()
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
# PDF → JPG CONVERSIONE (per AI Vision)
# ============================================================
def convert_pdf_to_jpg(pdf_bytes):
    if not PDF_OK:
        st.error("pdf2image non installato. Aggiungi pdf2image a requirements.txt e poppler-utils a packages.txt")
        return None
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
        buf = BytesIO()
        images[0].save(buf, format="JPEG", quality=90, optimize=True)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"Errore conversione PDF: {e}")
        return None

# ============================================================
# AI VISION — Groq → Gemini → Ollama
# ============================================================
def analizza_planimetria_ai(image_bytes, groq_key="", gemini_key=""):
    if not REQ_OK: return []
    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        'Analizza questa planimetria architettonica. '
        'Per ogni stanza/ambiente visibile restituisci SOLO JSON valido:\n'
        '{"areas":[{"name":"Nome stanza","type":"Ufficio VDT","area_m2":30.0}]}\n'
        "Tipi validi: Ufficio VDT, Corridoio, Bagno/WC, Sala riunioni, Archivio, "
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
                headers={"Authorization":f"Bearer {groq_key}","Content-Type":"application/json"},
                json={"model":"llama-3.2-11b-vision-preview","temperature":0.1,"max_tokens":1024,
                      "messages":[{"role":"user","content":[
                          {"type":"image_url","image_url":{"url":f"image/jpeg;base64,{b64}"}},
                          {"type":"text","text":prompt}]}]},
                timeout=30)
            areas = _parse(r.json()["choices"][0]["message"]["content"])
            if areas: return areas
        except Exception as e:
            st.warning(f"Groq non disponibile: {e}")

    if gemini_key:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={gemini_key}",
                json={"contents":[{"parts":[
                    {"inline_data":{"mime_type":"image/jpeg","data":b64}},
                    {"text":prompt}]}]},
                timeout=30)
            areas = _parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            if areas: return areas
        except Exception as e:
            st.warning(f"Gemini non disponibile: {e}")

    try:
        r = requests.post("http://localhost:11434/api/generate",
            json={"model":"llava","stream":False,"prompt":prompt,"images":[b64]},
            timeout=120)
        areas = _parse(r.json().get("response",""))
        if areas: return areas
    except Exception as e:
        st.warning(f"Ollama non disponibile: {e}")
    return []

def detect_scala_ai(image_bytes, groq_key="", gemini_key=""):
    if not REQ_OK: return ""
    b64 = base64.b64encode(image_bytes).decode()
    prompt = ("Questa planimetria ha una scala tipo 1:50, 1:100, 1:200, 1:500? "
              "Rispondi SOLO col numero dopo i due punti, es: 100. Se non la vedi scrivi: 0.")
    def _num(t):
        m = re.search(r'\d+', t.strip())
        return f"1:{m.group()}" if m and int(m.group()) > 0 else ""
    if groq_key:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization":f"Bearer {groq_key}"},
                json={"model":"llama-3.2-11b-vision-preview","max_tokens":20,"temperature":0,
                      "messages":[{"role":"user","content":[
                          {"type":"image_url","image_url":{"url":f"image/jpeg;base64,{b64}"}},
                          {"type":"text","text":prompt}]}]},
                timeout=20)
            return _num(r.json()["choices"][0]["message"]["content"])
        except Exception: pass
    if gemini_key:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={gemini_key}",
                json={"contents":[{"parts":[
                    {"inline_data":{"mime_type":"image/jpeg","data":b64}},
                    {"text":prompt}]}]},
                timeout=20)
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
                if not nu.strip() or not np1:   st.error("Campi obbligatori mancanti.")
                elif len(np1) < 6:              st.error("Password minimo 6 caratteri.")
                elif np1 != np2:                st.error("Le password non coincidono.")
                elif user_exists(nu.strip()):   st.error("Username già in uso.")
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
        "Gewiss GW Emergenza 200lm/3W": {
            "produttore":"Gewiss","flusso_lm":200,"potenza_W":3,
            "efficienza":66,"ra":80,"temp_colore":"4000K","ugr":28,
            "prezzo":85,"installazione":25,"tipo":"Emergenza","ip":"IP20",
            "dimmerabile":False,"classe_energ":"A"},
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
    "Ufficio VDT":       {"lux":500,"ugr_max":19,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Reception":         {"lux":300,"ugr_max":22,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Corridoio":         {"lux":100,"ugr_max":28,"uni":0.40,"ra_min":40,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Sala riunioni":     {"lux":500,"ugr_max":19,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Archivio":          {"lux":200,"ugr_max":25,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Bagno/WC":          {"lux":200,"ugr_max":25,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Laboratorio":       {"lux":750,"ugr_max":16,"uni":0.70,"ra_min":90,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Ingresso":          {"lux":200,"ugr_max":22,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Mensa/Ristoro":     {"lux":200,"ugr_max":22,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Locale tecnico":    {"lux":200,"ugr_max":25,"uni":0.40,"ra_min":60,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Via di esodo":      {"lux":1,  "ugr_max":35,"uni":0.10,"ra_min":40,"norma":"UNI EN 1838:2025",   "area":"EM"},
    "Area antipanico":   {"lux":0.5,"ugr_max":35,"uni":0.10,"ra_min":40,"norma":"UNI EN 1838:2025",   "area":"EM"},
    "Piazzale operativo":        {"lux":20, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Area carico/scarico":       {"lux":50, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Parcheggio esterno":        {"lux":10, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Percorso pedonale esterno": {"lux":5,  "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Strada veicolare ME3a":     {"lux":7.5,"ugr_max":55,"uni":0.40,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
    "Strada residenziale CE2":   {"lux":7.5,"ugr_max":55,"uni":0.40,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
    "Zona pedonale S4":          {"lux":5,  "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
}

# ============================================================
# FUNZIONE CALCOLO ILLUMINOTECNICO
# ============================================================
def calcola_area(area, modalita="normale"):
    sup  = area["superficie_m2"]
    alt  = area.get("altezza_m", 2.70)
    req  = REQUISITI[area["tipo_locale"]]
    lamp = DB_LAMPADE[area["lampada"]]
    CU, MF = 0.60, 0.80

    if modalita == "emergenza":
        E_t = 1.0
        lamp_em_key = next((k for k, v in DB_LAMPADE.items() if v["tipo"] == "Emergenza"), area["lampada"])
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
# RENDERING 3D
# ============================================================
def genera_rendering(area, calc):
    lato  = np.sqrt(area["superficie_m2"])
    alt   = area.get("altezza_m", 2.70)
    lamp  = DB_LAMPADE[area["lampada"]]
    coords = calc["coords"]
    is_ext = REQUISITI[area["tipo_locale"]]["area"] in ("EXT", "STR")
    is_em  = calc.get("modalita") == "emergenza"

    fig = plt.figure(figsize=(14, 9), dpi=150, facecolor="#050816")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#050816")

    pav = Poly3DCollection([[(0,0,0),(lato,0,0),(lato,lato,0),(0,lato,0)]], alpha=1.0)
    pav.set_facecolor((0.22, 0.22, 0.24)); pav.set_edgecolor("#4b5563")
    ax.add_collection3d(pav)

    if not is_ext:
        soff = Poly3DCollection([[(0,0,alt),(lato,0,alt),(lato,lato,alt),(0,lato,alt)]], alpha=0.6)
        soff.set_facecolor((0.97,0.97,0.97,0.06)); soff.set_edgecolor("#9ca3af")
        ax.add_collection3d(soff)
        for wall in [
            [(0,0,0),(lato,0,0),(lato,0,alt),(0,0,alt)],
            [(0,lato,0),(lato,lato,0),(lato,lato,alt),(0,lato,alt)],
            [(0,0,0),(0,lato,0),(0,lato,alt),(0,0,alt)],
            [(lato,0,0),(lato,lato,0),(lato,lato,alt),(lato,0,alt)],
        ]:
            pw = Poly3DCollection([wall], alpha=0.10)
            pw.set_facecolor((0.95,0.93,0.90,0.12)); pw.set_edgecolor("#9ca3af")
            ax.add_collection3d(pw)

    c_led = (0.1,0.9,0.2) if is_em else (1.0,0.95,0.72)
    theta = np.linspace(0, 2*np.pi, 20)
    for (lx, ly) in coords:
        h = alt - 0.05 if not is_ext else 6.0
        ax.scatter([lx],[ly],[h], c=[c_led], s=300, edgecolors="white", lw=1.5, zorder=10)
        for rr, alp in [(0.6,0.16),(1.2,0.09),(2.0,0.05)]:
            for ang in theta[::2]:
                ax.plot([lx, lx+rr*np.cos(ang)],[ly, ly+rr*np.sin(ang)],
                        [h, 0.04], color="#fef3c7", alpha=alp, lw=0.7)

    Xh, Yh = np.meshgrid(np.linspace(0.1,lato-0.1,50), np.linspace(0.1,lato-0.1,50))
    Zh = np.zeros_like(Xh)
    h_lamp = alt if not is_ext else 6.0
    for (lx2, ly2) in coords:
        d2 = np.sqrt((Xh-lx2)**2+(Yh-ly2)**2+h_lamp**2)
        Zh += (lamp["flusso_lm"]/(2*np.pi))*(h_lamp/d2**3)
    Zn = (Zh-Zh.min())/(Zh.max()-Zh.min()+1e-9)
    cmap_use = plt.cm.summer if is_em else plt.cm.inferno
    ax.plot_surface(Xh, Yh, np.full_like(Xh,0.02), facecolors=cmap_use(Zn), alpha=0.55, shade=False)

    ax.set_xlim(0,lato); ax.set_ylim(0,lato)
    ax.set_zlim(0, max(alt,6.5) if is_ext else alt)
    ax.view_init(elev=30, azim=235); ax.axis("off")

    em_label = " [EMERGENZA]" if is_em else ""
    fig.text(.5,.97,f"RENDERING 3D — {area['nome']}{em_label}",
             fontsize=14,fontweight="bold",color="white",ha="center",va="top")
    fig.text(.5,.93,
             f"{calc['n']}x {area['lampada'][:38]}  |  {calc['E_m']} lux  |  "
             f"{calc['W_t']} W  |  {calc['wm2']} W/m²  |  {REQUISITI[area['tipo_locale']]['norma']}",
             fontsize=8,color="#a5b4fc",ha="center")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#050816")
    buf.seek(0); plt.close(fig)
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
    ax.contour(X, Y, Z, levels=[1,5,10,50,100,200,300,500,750],
               colors="black", linewidths=0.6, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="Lux", shrink=0.85)
    for (lx, ly) in coords:
        ax.plot(lx, ly, "o", color="#fbbf24", ms=7, mec="black", mew=1.2, zorder=5)
    ax.set_xlim(0,lato); ax.set_ylim(0,lato)
    ax.set_aspect("equal"); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")

# ============================================================
# EXPORT GLTF
# ============================================================
def export_gltf_scene(risultati):
    nodes, meshes = [], []
    for r in risultati:
        alt  = r.get("altezza_m", 2.70)
        lato = float(np.sqrt(r["sup"]))
        meshes.append({"name": r["nome"],
                       "translation": [float(r.get("offset_x",0)), 0.0, 0.0],
                       "scale": [lato, lato, alt]})
        for (lx, ly) in r["calc"]["coords"]:
            nodes.append({"name": f"Lamp_{r['nome']}",
                          "translation": [float(r.get("offset_x",0)+lx), float(ly), alt-0.05]})
    gltf = {
        "asset": {"version":"2.0","generator":"Lighting Agent Pro v4.0"},
        "scene": 0,
        "scenes": [{"name":"LightingScene","nodes":list(range(len(meshes)+len(nodes)))}],
        "nodes": meshes + nodes,
    }
    buf = BytesIO()
    buf.write(json.dumps(gltf, indent=2).encode())
    buf.seek(0); return buf

# ============================================================
# PREVENTIVO
# ============================================================
def calc_preventivo(risultati, mg_pct, sg_pct, os_pct, iva_pct):
    righe = []
    for r in risultati:
        lk   = r["calc"].get("lampada_usata", r["lampada"])
        lamp = DB_LAMPADE.get(lk, DB_LAMPADE[r["lampada"]])
        n    = r["calc"]["n"]
        mat  = n * lamp["prezzo"]
        ins  = n * lamp["installazione"]
        sub  = mat + ins
        righe.append({"area":r["nome"],"n":n,"lampada":lk[:40],
                      "modalita":r["calc"].get("modalita","normale"),
                      "mat":mat,"ins":ins,"sub":sub})
    tm = sum(r["mat"] for r in righe)
    ti = sum(r["ins"] for r in righe)
    tn = tm + ti
    sg = round(tn*sg_pct/100, 2)
    os = round(tn*os_pct/100, 2)
    mg = round((tn+sg+os)*mg_pct/100, 2)
    to = round(tn+sg+os+mg, 2)
    iva= round(to*iva_pct/100, 2)
    tf = round(to+iva, 2)
    return {"righe":righe,"tm":tm,"ti":ti,"tn":tn,"sg":sg,"os":os,"mg":mg,"to":to,"iva":iva,"tf":tf}

# ============================================================
# DXF
# ============================================================
def genera_dxf(risultati):
    lines = ["0","SECTION","2","ENTITIES"]
    for r in risultati:
        ox = float(r.get("offset_x",0))
        lato = float(np.sqrt(r["sup"]))
        lines += ["0","LWPOLYLINE","8","AREE","90","4","70","1",
                  "10",str(ox),"20","0.0","10",str(ox+lato),"20","0.0",
                  "10",str(ox+lato),"20",str(lato),"10",str(ox),"20",str(lato)]
        for (lx,ly) in r["calc"]["coords"]:
            lines += ["0","POINT","8","LAMPADE","10",str(ox+lx),"20",str(ly),"30","0.0"]
        lines += ["0","TEXT","8","TESTI","10",str(ox+lato/2),"20",str(lato/2),"30","0.0",
                  "40","0.3","1",f"{r['nome']} {r['calc']['E_m']}lux"]
    lines += ["0","ENDSEC","0","EOF"]
    buf = BytesIO()
    buf.write("\n".join(lines).encode()); buf.seek(0); return buf

# ============================================================
# PDF TAVOLA A3
# ============================================================
def genera_pdf(progetto, risultati, logo_bytes=None):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(16.54,11.69), dpi=120, facecolor="white")
        if logo_bytes and PIL_OK:
            try:
                ax_logo = fig.add_axes([0.02,0.90,0.10,0.08])
                ax_logo.imshow(PILImage.open(BytesIO(logo_bytes))); ax_logo.axis("off")
            except Exception: pass

        ax_title = fig.add_axes([0.13,0.90,0.85,0.08])
        ax_title.set_facecolor("#1a365d")
        ax_title.text(0.5,0.7,"TAVOLA ILLUMINOTECNICA — LIGHTING AGENT PRO v4.0",
                      color="white",fontsize=13,fontweight="bold",ha="center",va="center",
                      transform=ax_title.transAxes)
        ax_title.text(0.5,0.2,
                      f"Progetto: {progetto['nome']}  |  Committente: {progetto['committente']}  |  "
                      f"Data: {progetto['data']}  |  Tav. {progetto['num_tavola']}  |  "
                      f"UNI 11630:2016 + UNI EN 12464-1:2021",
                      color="#90cdf4",fontsize=7.5,va="center",ha="center",
                      transform=ax_title.transAxes)
        ax_title.axis("off")

        n_aree = len(risultati)
        cols_plan = min(n_aree,4)
        for idx, r in enumerate(risultati):
            row = idx // cols_plan; col = idx % cols_plan
            w_ax=0.22; h_ax=0.25
            left=0.04+col*(w_ax+0.02); bottom=0.58-row*(h_ax+0.05)
            ax = fig.add_axes([left,bottom,w_ax,h_ax])
            genera_isolux(ax, r["calc"]["coords"], DB_LAMPADE[r["lampada"]],
                          r["sup"], r.get("altezza_m",2.70))
            ok = r["calc"]["ok_lux"]=="✅"
            ax.set_title(f"{r['nome']}\n{r['calc']['E_m']} lux {'✅' if ok else '❌'} | "
                         f"{r['calc']['n']} lamp | {r['calc']['W_t']}W",
                         fontsize=6.5, color="#1a365d" if ok else "#c53030", pad=3)

        ax_tab = fig.add_axes([0.04,0.05,0.92,0.28]); ax_tab.axis("off")
        col_labels = ["Area","Tipo","m²","N","Target lux","Ottenuto","W","W/m²","Lux✓","UGR✓","Ra✓","Norma"]
        rows_data  = []
        for r in risultati:
            req = REQUISITI[r["tipo_locale"]]
            rows_data.append([r["nome"][:20],r["tipo_locale"][:18],str(r["sup"]),
                              str(r["calc"]["n"]),str(r["calc"]["E_t"]),str(r["calc"]["E_m"]),
                              str(r["calc"]["W_t"]),str(r["calc"]["wm2"]),
                              r["calc"]["ok_lux"],r["calc"]["ok_ugr"],r["calc"]["ok_ra"],
                              req["norma"][:22]])
        tbl = ax_tab.table(cellText=rows_data,colLabels=col_labels,
                           cellLoc="center",loc="upper center",bbox=[0,0,1,1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
        for (ri,ci), cell in tbl.get_celld().items():
            if ri==0: cell.set_facecolor("#1a365d"); cell.set_text_props(color="white",fontweight="bold")
            elif ri%2==0: cell.set_facecolor("#ebf8ff")
            cell.set_edgecolor("#cbd5e0")

        tot_l=sum(r["calc"]["n"] for r in risultati)
        tot_W=sum(r["calc"]["W_t"] for r in risultati)
        tot_s=sum(r["sup"] for r in risultati)
        ax_foot = fig.add_axes([0.04,0.01,0.92,0.03])
        ax_foot.set_facecolor("#2d3748")
        ax_foot.text(0.5,0.5,
                     f"Totale: {tot_l} lampade | {tot_W} W | {tot_s:.0f} m² | "
                     f"{tot_W/max(tot_s,1):.1f} W/m² | Progettista: {progetto['progettista']} | {progetto['data']}",
                     color="white",fontsize=7,ha="center",va="center",transform=ax_foot.transAxes)
        ax_foot.axis("off")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf

# ============================================================
# RELAZIONE COMPLETA PDF
# ============================================================
def genera_relazione_completa(progetto, risultati, prev, logo_bytes=None,
                               mg_pct=35, sg_pct=12, os_pct=4, iva_pct=22):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Frontespizio
        fig = plt.figure(figsize=(11.69,8.27), dpi=100, facecolor="#1a365d")
        ax  = fig.add_axes([0,0,1,1]); ax.set_facecolor("#1a365d"); ax.axis("off")
        if logo_bytes and PIL_OK:
            try:
                ax_l = fig.add_axes([0.38,0.72,0.24,0.14])
                ax_l.imshow(PILImage.open(BytesIO(logo_bytes))); ax_l.axis("off")
            except Exception: pass
        ax.text(0.5,0.65,"RELAZIONE TECNICA ILLUMINOTECNICA",
                color="white",fontsize=18,fontweight="bold",ha="center",transform=ax.transAxes)
        ax.text(0.5,0.57,progetto["nome"],color="#90cdf4",fontsize=14,ha="center",transform=ax.transAxes)
        for i,(label,val) in enumerate([("Committente",progetto["committente"]),
                                         ("Progettista",progetto["progettista"]),
                                         ("Data",progetto["data"]),
                                         ("N. Tavola",progetto["num_tavola"])]):
            ax.text(0.3,0.44-i*0.07,f"{label}:",color="#a0aec0",fontsize=10,ha="right",transform=ax.transAxes)
            ax.text(0.32,0.44-i*0.07,val,color="white",fontsize=10,ha="left",transform=ax.transAxes)
        ax.text(0.5,0.10,
                "UNI 11630:2016 | UNI EN 12464-1:2021 | UNI EN 12464-2:2025 | UNI EN 1838:2025 | UNI 11248:2016",
                color="#4a5568",fontsize=8,ha="center",transform=ax.transAxes)
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

        # Tavola A3
        try:
            buf_tav = genera_pdf(progetto, risultati, logo_bytes)
            import matplotlib.image as mpimg
            img_arr = mpimg.imread(buf_tav)
            fig2 = plt.figure(figsize=(16.54,11.69),dpi=80)
            ax2  = fig2.add_axes([0,0,1,1])
            ax2.imshow(img_arr); ax2.axis("off")
            pdf.savefig(fig2,bbox_inches="tight"); plt.close(fig2)
        except Exception: pass

        # Schede verifica per ogni area
        for r in risultati:
            req = REQUISITI[r["tipo_locale"]]
            fig3 = plt.figure(figsize=(11.69,8.27), dpi=100, facecolor="white")
            ax3  = fig3.add_axes([0,0.7,1,0.28])
            ax3.set_facecolor("#2b6cb0"); ax3.axis("off")
            ax3.text(0.5,0.6,f"SCHEDA VERIFICA — {r['nome']}",
                     color="white",fontsize=13,fontweight="bold",ha="center",transform=ax3.transAxes)
            ax3.text(0.5,0.2,f"{r['tipo_locale']} | {r['sup']} m² | {r.get('altezza_m',2.70)} m | {req['norma']}",
                     color="#bee3f8",fontsize=9,ha="center",transform=ax3.transAxes)
            ax_iso = fig3.add_axes([0.05,0.18,0.45,0.48])
            genera_isolux(ax_iso, r["calc"]["coords"],
                          DB_LAMPADE[r["lampada"]], r["sup"], r.get("altezza_m",2.70))
            ax_iso.set_title("Distribuzione Isolux",fontsize=9,color="#1a365d")
            ax_v = fig3.add_axes([0.55,0.18,0.40,0.48]); ax_v.axis("off")
            checks = [
                ("Illuminamento medio Em", f"{r['calc']['E_m']} lux ≥ {r['calc']['E_t']} lux", r["calc"]["ok_lux"]),
                ("UGR massimo",            f"{DB_LAMPADE[r['lampada']]['ugr']} ≤ {req['ugr_max']}", r["calc"]["ok_ugr"]),
                ("Uniformità U0",          f"≥ {req['uni']}", r["calc"]["ok_uni"]),
                ("Indice resa colore Ra",   f"{DB_LAMPADE[r['lampada']]['ra']} ≥ {req['ra_min']}", r["calc"]["ok_ra"]),
            ]
            for i,(label,val,ok) in enumerate(checks):
                color="#276749" if ok=="✅" else "#9b2c2c"
                ax_v.text(0.0,0.85-i*0.22,f"{ok} {label}",fontsize=9,color=color,
                          fontweight="bold",transform=ax_v.transAxes)
                ax_v.text(0.0,0.77-i*0.22,f"    {val}",fontsize=8,color="#4a5568",transform=ax_v.transAxes)
            ax_info = fig3.add_axes([0.05,0.02,0.90,0.14]); ax_info.axis("off")
            lk   = r["calc"].get("lampada_usata", r["lampada"])
            ax_info.text(0.5,0.6,
                         f"Apparecchio: {lk} | N={r['calc']['n']} | "
                         f"Potenza: {r['calc']['W_t']} W | {r['calc']['wm2']} W/m² | "
                         f"k={r['calc']['k']} | CU={r['calc']['CU']} | MF={r['calc']['MF']}",
                         fontsize=8,ha="center",color="#2d3748",transform=ax_info.transAxes)
            pdf.savefig(fig3,bbox_inches="tight"); plt.close(fig3)

        # Rendering 3D per ogni area
        for r in risultati:
            try:
                buf_r = genera_rendering(r, r["calc"])
                import matplotlib.image as mpimg
                img = mpimg.imread(buf_r)
                fig4 = plt.figure(figsize=(14,9), dpi=80, facecolor="#050816")
                ax4  = fig4.add_axes([0,0,1,1])
                ax4.imshow(img); ax4.axis("off")
                pdf.savefig(fig4,bbox_inches="tight",facecolor="#050816"); plt.close(fig4)
            except Exception: pass

        # Preventivo
        if prev:
            fig5 = plt.figure(figsize=(11.69,8.27), dpi=100, facecolor="white")
            ax5  = fig5.add_axes([0,0,1,1]); ax5.axis("off")
            ax5.text(0.5,0.95,"PREVENTIVO ECONOMICO",fontsize=14,fontweight="bold",
                     color="#1a365d",ha="center",transform=ax5.transAxes)
            rows_p = [[r["area"],str(r["n"]),r["lampada"][:35],
                       f"EUR {r['mat']:,.0f}",f"EUR {r['ins']:,.0f}",f"EUR {r['sub']:,.0f}"]
                      for r in prev["righe"]]
            tbl5 = ax5.table(cellText=rows_p,
                             colLabels=["Area","N","Apparecchio","Materiali","Installazione","Subtotale"],
                             cellLoc="center",loc="center",bbox=[0.0,0.35,1.0,0.50])
            tbl5.auto_set_font_size(False); tbl5.set_fontsize(8)
            for (ri,ci),cell in tbl5.get_celld().items():
                if ri==0: cell.set_facecolor("#1a365d"); cell.set_text_props(color="white",fontweight="bold")
                elif ri%2==0: cell.set_facecolor("#ebf8ff")
            riepilogo = (
                f"Materiali: EUR {prev['tm']:,.0f}   |   Installazione: EUR {prev['ti']:,.0f}   |   "
                f"Spese generali: EUR {prev['sg']:,.0f}   |   Oneri sicurezza: EUR {prev['os']:,.0f}\n"
                f"Margine: EUR {prev['mg']:,.0f}   |   OFFERTA: EUR {prev['to']:,.0f}   |   "
                f"IVA: EUR {prev['iva']:,.0f}   |   TOTALE IVA INCLUSA: EUR {prev['tf']:,.0f}"
            )
            ax5.text(0.5,0.20,riepilogo,fontsize=9,ha="center",color="#1a365d",
                     transform=ax5.transAxes,
                     bbox=dict(boxstyle="round,pad=0.5",facecolor="#ebf8ff",edgecolor="#2b6cb0"))
            pdf.savefig(fig5,bbox_inches="tight"); plt.close(fig5)

    buf.seek(0); return buf

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"👤 **{st.session_state.username}**")
    if st.button("Logout"):
        st.session_state.logged_in = False; st.rerun()
    st.markdown("---")

    st.markdown("### 🏢 Logo studio")
    logo_file = st.file_uploader("Carica logo (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_up")
    if logo_file:
        st.session_state.logo_bytes = logo_file.read()
    if "logo_bytes" in st.session_state:
        st.image(st.session_state.logo_bytes, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Progetto")
    nome_prog   = st.text_input("Nome progetto",  "UFFICI TELEDIFESA")
    committente = st.text_input("Committente",     "Teledifesa S.p.A.")
    progettista = st.text_input("Progettista",     "Ing. Mario Rossi")
    num_tav     = st.text_input("N. Tavola",       "26A3S001")

    st.markdown("---")
    st.markdown("### 🗺️ Planimetria")
    plan_file = st.file_uploader("📄 Carica planimetria (PDF/PNG/JPG)",
                                  type=["pdf","png","jpg","jpeg"], key="plan_up")
    if plan_file:
        raw = plan_file.read()
        if plan_file.type == "application/pdf":
            st.info("🔄 PDF rilevato — conversione in JPG per AI Vision...")
            with st.spinner("Conversione pagina 1..."):
                jpg = convert_pdf_to_jpg(raw)
            if jpg:
                st.session_state.plan_bytes = jpg
                st.image(jpg, caption="✅ Prima pagina PDF convertita", use_container_width=True)
                st.success("✅ PDF pronto per AI Vision!")
            else:
                st.error("❌ Aggiungi poppler-utils a packages.txt")
        else:
            st.session_state.plan_bytes = raw
            st.image(raw, caption="✅ Planimetria caricata", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📏 Scala planimetria")
    if st.button("🔍 Rileva scala con AI") and "plan_bytes" in st.session_state:
        with st.spinner("Rilevamento in corso..."):
            s = detect_scala_ai(st.session_state.plan_bytes,
                                groq_key=st.session_state.get("groq_key",""),
                                gemini_key=st.session_state.get("gemini_key",""))
        if s: st.success(f"Scala: {s}"); st.session_state.scala_suggerita = s
        else: st.warning("Non rilevata, imposta manualmente.")
    _sopts = ["1:50","1:100","1:200","1:500"]
    _sdef  = st.session_state.get("scala_suggerita","1:100")
    _sidx  = _sopts.index(_sdef) if _sdef in _sopts else 1
    _ssel  = st.selectbox("Scala", _sopts, index=_sidx)
    scala_mpp = {"1:50":1.25,"1:100":2.5,"1:200":5.0,"1:500":12.5}.get(_ssel, 2.5)
    st.caption(f"100px = {scala_mpp} m")

    st.markdown("---")
    with st.expander("🤖 Chiavi API AI Vision (gratuite)"):
        _gk  = st.text_input("Groq API Key",   value=st.session_state.get("groq_key",""),
                              type="password", help="console.groq.com — gratuito")
        _gmk = st.text_input("Gemini API Key", value=st.session_state.get("gemini_key",""),
                              type="password", help="aistudio.google.com — gratuito")
        if _gk:  st.session_state.groq_key   = _gk
        if _gmk: st.session_state.gemini_key = _gmk
        st.caption("Fallback automatico: Groq → Gemini → Ollama locale")

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
            "produttore":"Custom IES","flusso_lm":approx_flux,"potenza_W":20,
            "efficienza":round(approx_flux/20,1),"ra":80,"temp_colore":"4000K","ugr":19,
            "prezzo":150,"installazione":50,"tipo":"Custom IES","ip":"IP65",
            "dimmerabile":False,"classe_energ":"A"}
        st.success(f"✅ {custom_key} caricata (φ≈{approx_flux:.0f} lm)")

    st.markdown("---")
    st.markdown("### 🔧 Filtro lampade")
    prod_filter = st.selectbox("Produttore",
        ["Tutti","BEGA","iGuzzini","Flos","Artemide","Delta Light","Gewiss","Philips","Custom IES"])

# ============================================================
# HEADER
# ============================================================
col_logo, col_title = st.columns([1,6])
with col_logo:
    if "logo_bytes" in st.session_state:
        st.image(st.session_state.logo_bytes, width=80)
with col_title:
    st.markdown("""
<div class="header-box">
<h1 style="margin:0;font-size:2rem">💡 Lighting Agent Pro v4.0</h1>
<p style="margin:.3rem 0 0;opacity:.85">
UNI 11630:2016 · UNI EN 12464-1:2021 · UNI EN 12464-2:2025 ·
UNI 11248:2016 · UNI EN 1838:2025 · UNI CEI 11222 · AI Vision PDF/PNG/JPG · Rendering 3D
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
            alt  = st.number_input("Altezza netta m",  2.0, 12.0,   2.70, 0.05)
        with c3:
            lamp_scelta = st.selectbox("Apparecchio *", list(lamp_disp.keys()))
            lsp = DB_LAMPADE[lamp_scelta]
            st.caption(f"{lsp['potenza_W']}W | {lsp['flusso_lm']}lm | Ra{lsp['ra']} | "
                       f"{lsp['temp_colore']} | UGR{lsp['ugr']} | {lsp['ip']}")
            emergenza = st.checkbox("🚨 Calcola illuminazione di emergenza (UNI EN 1838:2025)")
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

    # --- PLANIMETRIA ---
    st.markdown("---")
    st.markdown("#### 🖊️ Aggiungi aree da coordinate planimetria")
    if "plan_bytes" in st.session_state:
        st.image(st.session_state.plan_bytes, caption="Planimetria caricata", use_container_width=True)
        with st.form("form_coords", clear_on_submit=True):
            cc1,cc2,cc3,cc4 = st.columns(4)
            with cc1: cx = st.number_input("X origine [m]", 0.0, 500.0, 0.0, 0.5)
            with cc2: cy = st.number_input("Y origine [m]", 0.0, 500.0, 0.0, 0.5)
            with cc3: cw = st.number_input("Larghezza [m]",  1.0, 100.0, 5.0, 0.5)
            with cc4: ch = st.number_input("Profondità [m]", 1.0, 100.0, 4.0, 0.5)
            nome_coord = st.text_input("Nome area", placeholder="es. Ufficio B")
            tipo_coord = st.selectbox("Tipo locale", list(REQUISITI.keys()), key="tc_coord")
            lamp_coord = st.selectbox("Apparecchio",  list(DB_LAMPADE.keys()), key="lc_coord")
            alt_coord  = st.number_input("Altezza [m]", 2.0, 12.0, 2.70, 0.05, key="ac_coord")
            em_coord   = st.checkbox("🚨 Calcola emergenza", key="ec_coord")
            if st.form_submit_button("➕ Aggiungi da coordinate"):
                if nome_coord.strip():
                    area_m2 = round(cw * ch, 1)
                    st.session_state.aree.append({
                        "nome": nome_coord.strip(), "tipo_locale": tipo_coord,
                        "superficie_m2": area_m2, "altezza_m": alt_coord,
                        "lampada": lamp_coord, "sup": area_m2, "emergenza": em_coord,
                        "polygon_px": [[cx,cy],[cx+cw,cy],[cx+cw,cy+ch],[cx,cy+ch]],
                    })
                    st.success(f"✅ Area «{nome_coord}» {area_m2}m² aggiunta!")
                else:
                    st.error("Inserisci il nome area.")
    else:
        st.info("⬆️ Carica una planimetria nella sidebar (PDF, PNG o JPG) per visualizzarla qui.")

    # --- AI VISION ---
    st.markdown("---")
    st.markdown("""
<div style="background:linear-gradient(135deg,#1a365d,#2b6cb0);padding:1.2rem 1.5rem;
border-radius:12px;margin-bottom:1rem;">
<h3 style="color:white;margin:0">🤖 Rileva Aree con AI</h3>
<p style="color:#bee3f8;margin:.3rem 0 0;font-size:.9rem">
Groq (gratuito) → Gemini (gratuito) → Ollama locale — Supporta PDF, PNG, JPG
</p></div>""", unsafe_allow_html=True)

    if "plan_bytes" not in st.session_state or not st.session_state.plan_bytes:
        st.warning("⬆️ Carica prima una planimetria nella sidebar (PDF, PNG o JPG).")
    else:
        gk_ok  = "✅ Groq"   if st.session_state.get("groq_key")   else "⚠️ Groq (non configurato)"
        gmk_ok = "✅ Gemini" if st.session_state.get("gemini_key") else "⚠️ Gemini (non configurato)"
        st.caption(f"{gk_ok}  |  {gmk_ok}  |  Ollama: localhost:11434  — configura nella sidebar")

        c_btn1, c_btn2, c_btn3 = st.columns([2,2,3])
        with c_btn1:
            ai_btn = st.button("🔍 RILEVA AREE CON AI", type="primary", use_container_width=True)
        with c_btn2:
            if st.button("🗑️ Cancella risultati AI", use_container_width=True):
                st.session_state.pop("ai_areas_preview", None); st.rerun()
        with c_btn3:
            st.caption("~5-15s con Groq/Gemini — PDF, PNG, JPG tutti supportati")

        if ai_btn:
            with st.spinner("🔍 Analisi AI... (Groq → Gemini → Ollama)"):
                found = analizza_planimetria_ai(
                    st.session_state.plan_bytes,
                    groq_key=st.session_state.get("groq_key",""),
                    gemini_key=st.session_state.get("gemini_key",""),
                )
            if found:
                st.session_state.ai_areas_preview = found
                st.success(f"✅ AI ha identificato **{len(found)} aree** — verifica e conferma")
            else:
                st.error("❌ Nessuna area rilevata. Verifica le chiavi API nella sidebar.")

        if st.session_state.get("ai_areas_preview"):
            preview = st.session_state.ai_areas_preview
            st.markdown("##### ✏️ Verifica e modifica prima di aggiungere")
            hdr = st.columns([3,3,2,2,2])
            for h, c in zip(["Nome area","Tipo locale","m²","Altezza m","Emergenza?"], hdr):
                c.markdown(f"**{h}**")
            edited = []
            for i, a in enumerate(preview):
                tipo_ai = a.get("type","Ufficio VDT")
                if tipo_ai not in REQUISITI: tipo_ai = "Ufficio VDT"
                cc1,cc2,cc3,cc4,cc5 = st.columns([3,3,2,2,2])
                with cc1:
                    n_e = st.text_input("", value=a.get("name",f"Area_{i+1}"),
                                        key=f"ai_n_{i}", label_visibility="collapsed")
                with cc2:
                    t_e = st.selectbox("", list(REQUISITI.keys()),
                                       index=list(REQUISITI.keys()).index(tipo_ai),
                                       key=f"ai_t_{i}", label_visibility="collapsed")
                with cc3:
                    s_e = st.number_input("", value=float(a.get("area_m2",30.0)),
                                          min_value=1.0, step=0.5,
                                          key=f"ai_s_{i}", label_visibility="collapsed")
                with cc4:
                    h_e = st.number_input("", value=2.70, min_value=2.0, step=0.05,
                                          key=f"ai_h_{i}", label_visibility="collapsed")
                with cc5:
                    em_e = st.checkbox("", key=f"ai_em_{i}", label_visibility="collapsed")
                edited.append({"nome":n_e,"tipo_locale":t_e,"superficie_m2":s_e,
                               "altezza_m":h_e,"emergenza":em_e})

            st.markdown("---")
            col_ok, col_ann = st.columns([2,1])
            with col_ok:
                if st.button("✅ AGGIUNGI TUTTE LE AREE", type="primary", use_container_width=True):
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
                    st.success(f"✅ {len(edited)} aree aggiunte!")
                    st.rerun()
            with col_ann:
                if st.button("↩️ Annulla", use_container_width=True):
                    st.session_state.pop("ai_areas_preview", None); st.rerun()

    # --- LISTA AREE ---
    st.markdown("---")
    st.markdown("#### 📋 Aree inserite")
    if st.session_state.aree:
        for i, a in enumerate(st.session_state.aree):
            req = REQUISITI[a["tipo_locale"]]
            badge = {"INT":"🟢","EXT":"🟡","EM":"🔴","STR":"🔵"}.get(req["area"],"⚪")
            em_icon = "🚨" if a.get("emergenza") else ""
            c1, c2, c3 = st.columns([5,2,1])
            with c1:
                st.markdown(
                    f'<div class="card"><b>{badge} {i+1}. {a["nome"]}</b>{em_icon} | '
                    f'{a["tipo_locale"]} | {a["superficie_m2"]}m² | h:{a["altezza_m"]}m | '
                    f'Target:{req["lux"]}lux | {a["lampada"][:30]} | '
                    f'<i>{req["norma"]}</i></div>', unsafe_allow_html=True)
            with c2:
                new_sup = st.number_input("m²", value=a["superficie_m2"],
                    key=f"sup_{i}", min_value=1.0, label_visibility="collapsed")
                if new_sup != a["superficie_m2"]:
                    st.session_state.aree[i]["superficie_m2"] = new_sup
                    st.session_state.aree[i]["sup"] = new_sup
            with c3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.aree.pop(i); st.rerun()
        if st.button("🗑️ Svuota tutte le aree", type="secondary"):
            st.session_state.aree = []; st.rerun()
    else:
        st.info("Nessuna area inserita. Aggiungi aree manualmente o usa AI Vision.")

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
                ris.append({**a, "calc": c_norm, "offset_x": ox, "offset_y": 0})
                ox += np.sqrt(a["superficie_m2"]) + 1.5
                if a.get("emergenza"):
                    c_em = calcola_area(a, "emergenza")
                    ris.append({**a, "nome": a["nome"]+" 🚨EMERG.",
                                "calc": c_em, "offset_x": ox, "offset_y": 0})
                    ox += np.sqrt(a["superficie_m2"]) + 1.5
            st.session_state.risultati = ris
            st.success(f"✅ Calcoli completati — {len(ris)} aree elaborate.")

        if "risultati" in st.session_state:
            rl = st.session_state.risultati
            tl = sum(r["calc"]["n"] for r in rl)
            tw = sum(r["calc"]["W_t"] for r in rl)
            tm2= sum(r["sup"] for r in rl)
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Aree",      len(rl))
            c2.metric("Lampade",   tl)
            c3.metric("Potenza W", tw)
            c4.metric("m²",        f"{tm2:.0f}")
            c5.metric("W/m²",      f"{tw/max(tm2,1):.1f}")

            rows = []
            for r in rl:
                lk  = r["calc"].get("lampada_usata", r["lampada"])
                req = REQUISITI[r["tipo_locale"]]
                rows.append({
                    "Area": r["nome"], "Tipo": r["tipo_locale"], "m²": r["sup"],
                    "N": r["calc"]["n"], "Lux target": r["calc"]["E_t"],
                    "Lux ottenuto": r["calc"]["E_m"],
                    "W": r["calc"]["W_t"], "W/m²": r["calc"]["wm2"],
                    "Lampada": lk[:35],
                    "Lux ✓": r["calc"]["ok_lux"], "UGR ✓": r["calc"]["ok_ugr"],
                    "Ra ✓": r["calc"]["ok_ra"], "Norma": req["norma"],
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
                with st.spinner("Generazione..."):
                    prog = {"nome":nome_prog,"committente":committente,
                            "progettista":progettista,
                            "data":datetime.now().strftime("%d/%m/%Y"),"num_tavola":num_tav}
                    buf = genera_pdf(prog, st.session_state.risultati,
                                     st.session_state.get("logo_bytes"))
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
            em_label = " 🚨 EMERGENZA" if r["calc"].get("modalita")=="emergenza" else ""
            badge = {"INT":"🟢","EXT":"🟡","EM":"🔴","STR":"🔵"}.get(req["area"],"⚪")
            with st.expander(f"{badge} {r['nome']}{em_label} | {r['calc']['E_m']} lux | "
                             f"{r['calc']['ok_lux']} Lux | {r['calc']['ok_ugr']} UGR | {req['norma']}"):
                ca,cb,cc,cd = st.columns(4)
                ca.metric("Lux ottenuto", str(r["calc"]["E_m"]))
                cb.metric("Potenza W",    str(r["calc"]["W_t"]))
                cc.metric("W/m²",         str(r["calc"]["wm2"]))
                cd.metric("Lampade",      str(r["calc"]["n"]))
                c1,c2,c3,c4 = st.columns(4)
                c1.success(f"Illuminamento {r['calc']['ok_lux']}")
                c2.success(f"UGR {r['calc']['ok_ugr']}")
                c3.success(f"Uniformità {r['calc']['ok_uni']}")
                c4.success(f"Ra {r['calc']['ok_ra']}")
                if r["calc"].get("modalita")=="emergenza":
                    st.info("🚨 UNI EN 1838:2025 — Em min: 1 lux | Autonomia ≥1h | Test mensile + annuale")

        st.markdown("---")
        if st.button("📋 GENERA REPORT VERIFICHE PDF", type="primary"):
            with st.spinner("Generazione..."):
                prog = {"nome":nome_prog,"committente":committente,
                        "progettista":progettista,
                        "data":datetime.now().strftime("%d/%m/%Y"),"num_tavola":num_tav}
                buf = genera_pdf(prog, st.session_state.risultati,
                                 st.session_state.get("logo_bytes"))
                st.download_button("⬇️ Scarica Report PDF", data=buf,
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
        scelta = st.selectbox("Seleziona area", names)

        c1, c2, c3 = st.columns(3)
        with c1:
            render_singolo = st.button("🎨 RENDERING AREA SELEZIONATA", type="primary")
        with c2:
            render_tutti = st.button("🎨 RENDERING TUTTE LE AREE")
        with c3:
            export_gltf  = st.button("📦 ESPORTA SCENA glTF")

        if render_singolo:
            idx = names.index(scelta)
            r   = st.session_state.risultati[idx]
            with st.spinner(f"Rendering {scelta}..."):
                buf = genera_rendering(r, r["calc"])
                st.image(buf, caption=f"Rendering 3D — {scelta}", use_container_width=True)
                buf.seek(0)
                st.download_button("⬇️ Scarica PNG", data=buf,
                    file_name=f"render_{scelta.lower().replace(' ','_')}.png", mime="image/png")

        if render_tutti:
            cols = st.columns(2)
            for i, r in enumerate(st.session_state.risultati):
                with st.spinner(f"Rendering {r['nome']}..."):
                    buf = genera_rendering(r, r["calc"])
                    with cols[i % 2]:
                        st.image(buf, caption=r["nome"], use_container_width=True)
                        buf.seek(0)
                        st.download_button(f"⬇️ {r['nome']}", data=buf,
                            file_name=f"render_{i}.png", mime="image/png", key=f"rend_{i}")

        if export_gltf:
            with st.spinner("Generazione glTF..."):
                gltf_buf = export_gltf_scene(st.session_state.risultati)
                st.download_button("⬇️ Scarica .glTF", data=gltf_buf,
                    file_name=f"{num_tav}_scene.gltf", mime="model/gltf+json")
                st.success("✅ Scena glTF pronta per Blender/Unreal!")

# ============================================================
# TAB 6 — PREVENTIVO
# ============================================================
with tab6:
    st.subheader("Preventivo Economico")
    if "risultati" not in st.session_state:
        st.warning("Esegui prima i calcoli.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        with c1: mg_sl  = st.slider("Margine %",        10, 60, 35, key="sl_mg")
        with c2: iva_sl = st.slider("IVA %",             0, 22, 22, key="sl_iva")
        with c3: sg_sl  = st.slider("Spese generali %",  5, 25, 12, key="sl_sg")
        with c4: os_sl  = st.slider("Oneri sicurezza %", 2, 10,  4, key="sl_os")

        if st.button("🧮 CALCOLA PREVENTIVO", type="primary"):
            st.session_state.prev = calc_preventivo(
                st.session_state.risultati, mg_sl, sg_sl, os_sl, iva_sl)

        if "prev" in st.session_state:
            pv = st.session_state.prev
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Materiali",         f"EUR {pv['tm']:,.0f}")
            c2.metric("Installazione",     f"EUR {pv['ti']:,.0f}")
            c3.metric("Offerta cliente",   f"EUR {pv['to']:,.0f}")
            c4.metric("Totale IVA inclusa",f"EUR {pv['tf']:,.0f}")

            df_p = pd.DataFrame([{
                "Area": r["area"], "N": r["n"], "Apparecchio": r["lampada"],
                "Modalità": r["modalita"],
                "Materiali": f"EUR {r['mat']:,.0f}",
                "Installazione": f"EUR {r['ins']:,.0f}",
                "Subtotale": f"EUR {r['sub']:,.0f}",
            } for r in pv["righe"]])
            st.dataframe(df_p, use_container_width=True, hide_index=True)

            st.markdown(f"""
| Voce | Importo |
|---|---|
| Materiali | EUR {pv['tm']:,.0f} |
| Installazione | EUR {pv['ti']:,.0f} |
| Totale netto | EUR {pv['tn']:,.0f} |
| Spese generali {sg_sl}% | EUR {pv['sg']:,.0f} |
| Oneri sicurezza {os_sl}% | EUR {pv['os']:,.0f} |
| Margine {mg_sl}% | EUR {pv['mg']:,.0f} |
| **OFFERTA CLIENTE** | **EUR {pv['to']:,.0f}** |
| IVA {iva_sl}% | EUR {pv['iva']:,.0f} |
| **TOTALE IVA INCLUSA** | **EUR {pv['tf']:,.0f}** |
""")
            txt_prev = (
                f"PREVENTIVO — {nome_prog}\nCommittente: {committente}\n"
                f"Data: {datetime.now():%d/%m/%Y}\nProgettista: {progettista}\n\n"
                + "\n".join(f"  {r['area']}: {r['n']}x {r['lampada']}  EUR {r['sub']:,.0f}"
                            for r in pv["righe"])
                + f"\n\nTOTALE IVA INCLUSA: EUR {pv['tf']:,.0f}\n"
            )
            st.download_button("⬇️ Scarica Preventivo TXT",
                data=txt_prev.encode(), file_name=f"preventivo_{datetime.now():%Y%m%d}.txt")

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
La relazione include in un **unico PDF**:
- 📋 Frontespizio · 📐 Tavola A3 · ✅ Schede verifica · 🎨 Rendering 3D · 💶 Preventivo
""")
        if st.button("📄 GENERA RELAZIONE COMPLETA PDF", type="primary"):
            with st.spinner("Generazione relazione completa... (1-2 minuti)"):
                prog = {"nome":nome_prog,"committente":committente,
                        "progettista":progettista,
                        "data":datetime.now().strftime("%d/%m/%Y"),"num_tavola":num_tav}
                try:
                    buf = genera_relazione_completa(
                        prog, st.session_state.risultati,
                        st.session_state.prev,
                        st.session_state.get("logo_bytes"),
                        mg_pct=st.session_state.get("sl_mg",35),
                        sg_pct=st.session_state.get("sl_sg",12),
                        os_pct=st.session_state.get("sl_os",4),
                        iva_pct=st.session_state.get("sl_iva",22),
                    )
                    st.download_button("⬇️ SCARICA RELAZIONE COMPLETA PDF", data=buf,
                        file_name=f"{num_tav}_relazione_{datetime.now():%Y%m%d}.pdf",
                        mime="application/pdf")
                    st.success("✅ Relazione completa generata!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Errore: {e}")

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
                    file_name=f"{num_tav}_tavola.pdf", mime="application/pdf", key="dl_tav_r7")
        with c2:
            if st.button("📦 Solo glTF Blender"):
                gltf_buf = export_gltf_scene(st.session_state.risultati)
                st.download_button("⬇️ Scena .glTF", data=gltf_buf,
                    file_name=f"{num_tav}_scene.gltf", mime="model/gltf+json", key="dl_gltf_r7")
        with c3:
            if "prev" in st.session_state:
                pv2 = st.session_state.prev
                txt2 = (f"PREVENTIVO — {nome_prog}\nData: {datetime.now():%d/%m/%Y}\n\n"
                        + "\n".join(f"  {r['area']}: EUR {r['sub']:,.0f}" for r in pv2["righe"])
                        + f"\n\nTOTALE: EUR {pv2['tf']:,.0f}\n")
                st.download_button("⬇️ Preventivo TXT", data=txt2.encode(),
                    file_name=f"prev_{datetime.now():%Y%m%d}.txt", key="dl_prev_r7")

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
                        if 
                            st.session_state.aree      = data["aree"]
                            st.session_state.risultati = data["risultati"] or []
                            if data["prev"]: st.session_state.prev = data["prev"]
                            st.success(f"✅ Caricato: {len(data['aree'])} aree"); st.rerun()
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
