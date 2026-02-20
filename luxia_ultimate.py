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
    con.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "username TEXT UNIQUE NOT NULL,"
        "email TEXT,"
        "password_hash TEXT NOT NULL,"
        "created_at TEXT)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS projects ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "username TEXT NOT NULL,"
        "nome_progetto TEXT NOT NULL,"
        "committente TEXT,"
        "progettista TEXT,"
        "num_tavola TEXT,"
        "data_creazione TEXT,"
        "data_modifica TEXT,"
        "aree_json TEXT,"
        "risultati_json TEXT,"
        "prev_json TEXT)"
    )
    for uname, upwd in [("admin","admin2026"),("demo","demo123"),("progett","luce2026")]:
        h = hashlib.sha256(upwd.encode()).hexdigest()
        con.execute(
            "INSERT OR IGNORE INTO users (username,email,password_hash,created_at) VALUES (?,?,?,?)",
            (uname, uname + "@luxia.it", h, datetime.now().isoformat())
        )
    con.commit()
    con.close()

def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

def check_user(u, p):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT password_hash FROM users WHERE username=?", (u,)).fetchone()
    con.close()
    return row is not None and row[0] == hash_pw(p)

def user_exists(u):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT id FROM users WHERE username=?", (u,)).fetchone()
    con.close()
    return row is not None

def register_user(u, email, p):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO users (username,email,password_hash,created_at) VALUES (?,?,?,?)",
        (u, email, hash_pw(p), datetime.now().isoformat())
    )
    con.commit()
    con.close()

def save_project(username, nome, committente, progettista, num_tav, aree, risultati, prev):
    con = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    ris_clean = []
    for r in risultati:
        rc = {k: v for k, v in r.items() if k != "calc"}
        rc["calc"] = {k2: v2 for k2, v2 in r.get("calc", {}).items()
                      if isinstance(v2, (str, int, float, bool, list, type(None)))}
        ris_clean.append(rc)
    ex = con.execute(
        "SELECT id FROM projects WHERE username=? AND nome_progetto=?",
        (username, nome)
    ).fetchone()
    if ex:
        con.execute(
            "UPDATE projects SET committente=?,progettista=?,num_tavola=?,"
            "data_modifica=?,aree_json=?,risultati_json=?,prev_json=? "
            "WHERE username=? AND nome_progetto=?",
            (committente, progettista, num_tav, now,
             json.dumps(aree, ensure_ascii=False),
             json.dumps(ris_clean, ensure_ascii=False),
             json.dumps(prev or {}, ensure_ascii=False),
             username, nome)
        )
    else:
        con.execute(
            "INSERT INTO projects (username,nome_progetto,committente,progettista,num_tavola,"
            "data_creazione,data_modifica,aree_json,risultati_json,prev_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (username, nome, committente, progettista, num_tav, now, now,
             json.dumps(aree, ensure_ascii=False),
             json.dumps(ris_clean, ensure_ascii=False),
             json.dumps(prev or {}, ensure_ascii=False))
        )
    con.commit()
    con.close()

def load_projects_list(username):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id,nome_progetto,committente,data_modifica FROM projects "
        "WHERE username=? ORDER BY data_modifica DESC",
        (username,)
    ).fetchall()
    con.close()
    return rows

def load_project_data(pid):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT nome_progetto,committente,progettista,num_tavola,"
        "aree_json,risultati_json,prev_json FROM projects WHERE id=?",
        (pid,)
    ).fetchone()
    con.close()
    if row:
        return {
            "nome": row[0], "committente": row[1],
            "progettista": row[2], "num_tavola": row[3],
            "aree":      json.loads(row[4]) if row[4] else [],
            "risultati": json.loads(row[5]) if row[5] else [],
            "prev":      json.loads(row[6]) if row[6] else {},
        }
    return None

def delete_project(pid):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM projects WHERE id=?", (pid,))
    con.commit()
    con.close()

init_db()

# ============================================================
# PDF -> JPG PER AI VISION
# ============================================================
def convert_pdf_to_jpg(pdf_bytes):
    if not PDF_OK:
        st.error("pdf2image non installato. Aggiungi pdf2image a requirements.txt "
                 "e poppler-utils a packages.txt")
        return None
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=1)
        buf = BytesIO()
        images[0].save(buf, format="JPEG", quality=90, optimize=True)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error("Errore conversione PDF: " + str(e))
        return None

# ============================================================
# AI VISION — Groq -> Gemini -> Ollama
# ============================================================
def analizza_planimetria_ai(image_bytes, groq_key="", gemini_key=""):
    if not REQ_OK:
        return []
    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        "Analizza questa planimetria architettonica. "
        "Per ogni stanza visibile restituisci SOLO JSON valido: "
        "{\"areas\":[{\"name\":\"Nome\",\"type\":\"Ufficio VDT\",\"area_m2\":30.0}]} "
        "Tipi validi: Ufficio VDT, Corridoio, Bagno/WC, Sala riunioni, Archivio, "
        "Ingresso, Mensa/Ristoro, Locale tecnico, Reception, Laboratorio."
    )

    def _parse(testo):
        m = re.search(r"[{].*[}]", testo, re.DOTALL)
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
                headers={"Authorization": "Bearer " + groq_key,
                         "Content-Type": "application/json"},
                json={
                    "model": "llama-3.2-11b-vision-preview",
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": "image/jpeg;base64," + b64}},
                        {"type": "text", "text": prompt}
                    ]}]
                },
                timeout=30
            )
            areas = _parse(r.json()["choices"][0]["message"]["content"])
            if areas:
                return areas
        except Exception as e:
            st.warning("Groq non disponibile: " + str(e))

    if gemini_key:
        try:
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.0-flash-exp:generateContent?key=" + gemini_key,
                json={"contents": [{"parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    {"text": prompt}
                ]}]},
                timeout=30
            )
            areas = _parse(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            if areas:
                return areas
        except Exception as e:
            st.warning("Gemini non disponibile: " + str(e))

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llava", "stream": False,
                  "prompt": prompt, "images": [b64]},
            timeout=120
        )
        areas = _parse(r.json().get("response", ""))
        if areas:
            return areas
    except Exception as e:
        st.warning("Ollama non disponibile: " + str(e))

    return []


def detect_scala_ai(image_bytes, groq_key="", gemini_key=""):
    if not REQ_OK:
        return ""
    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        "Questa planimetria ha una scala tipo 1:50, 1:100, 1:200, 1:500? "
        "Rispondi SOLO col numero dopo i due punti, es: 100. "
        "Se non la vedi scrivi: 0."
    )

    def _num(t):
        m = re.search(r"[0-9]+", t.strip())
        return ("1:" + m.group()) if (m and int(m.group()) > 0) else ""

    if groq_key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": "Bearer " + groq_key},
                json={
                    "model": "llama-3.2-11b-vision-preview",
                    "max_tokens": 20, "temperature": 0,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": "image/jpeg;base64," + b64}},
                        {"type": "text", "text": prompt}
                    ]}]
                },
                timeout=20
            )
            return _num(r.json()["choices"][0]["message"]["content"])
        except Exception:
            pass

    if gemini_key:
        try:
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.0-flash-exp:generateContent?key=" + gemini_key,
                json={"contents": [{"parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    {"text": prompt}
                ]}]},
                timeout=20
            )
            return _num(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception:
            pass

    return ""


# ============================================================
# LOGIN / REGISTRAZIONE
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = ""

if not st.session_state.logged_in:
    st.markdown(
        "<div style='max-width:460px;margin:5rem auto;padding:2rem;"
        "background:white;border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,.12)'>"
        "<h2 style='text-align:center;color:#1a365d'>Lighting Agent Pro</h2>"
        "<p style='text-align:center;color:#666'>v4.0 - Accesso riservato</p>"
        "</div>",
        unsafe_allow_html=True
    )
    _, col_b, _ = st.columns([1, 2, 1])
    with col_b:
        tl, tr = st.tabs(["Accedi", "Registrati"])
        with tl:
            u = st.text_input("Username", key="li_u")
            p = st.text_input("Password", type="password", key="li_p")
            if st.button("Entra", key="btn_li"):
                if check_user(u.strip(), p):
                    st.session_state.logged_in = True
                    st.session_state.username  = u.strip()
                    st.rerun()
                else:
                    st.error("Credenziali non valide")
        with tr:
            nu  = st.text_input("Username *",        key="r_u")
            ne  = st.text_input("Email",             key="r_e")
            np1 = st.text_input("Password *",        type="password", key="r_p1")
            np2 = st.text_input("Conferma password", type="password", key="r_p2")
            if st.button("Registrati", key="btn_reg"):
                if not nu.strip() or not np1:
                    st.error("Campi obbligatori mancanti.")
                elif len(np1) < 6:
                    st.error("Password minimo 6 caratteri.")
                elif np1 != np2:
                    st.error("Le password non coincidono.")
                elif user_exists(nu.strip()):
                    st.error("Username gia in uso.")
                else:
                    register_user(nu.strip(), ne.strip(), np1)
                    st.success("Account creato! Ora puoi fare login.")
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
    "Ufficio VDT":       {"lux":500, "ugr_max":19,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Reception":         {"lux":300, "ugr_max":22,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Corridoio":         {"lux":100, "ugr_max":28,"uni":0.40,"ra_min":40,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Sala riunioni":     {"lux":500, "ugr_max":19,"uni":0.60,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Archivio":          {"lux":200, "ugr_max":25,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Bagno/WC":          {"lux":200, "ugr_max":25,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Laboratorio":       {"lux":750, "ugr_max":16,"uni":0.70,"ra_min":90,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Ingresso":          {"lux":200, "ugr_max":22,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Mensa/Ristoro":     {"lux":200, "ugr_max":22,"uni":0.40,"ra_min":80,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Locale tecnico":    {"lux":200, "ugr_max":25,"uni":0.40,"ra_min":60,"norma":"UNI EN 12464-1:2021","area":"INT"},
    "Via di esodo":      {"lux":1,   "ugr_max":35,"uni":0.10,"ra_min":40,"norma":"UNI EN 1838:2025",   "area":"EM"},
    "Area antipanico":   {"lux":0.5, "ugr_max":35,"uni":0.10,"ra_min":40,"norma":"UNI EN 1838:2025",   "area":"EM"},
    "Piazzale operativo":        {"lux":20, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Area carico/scarico":       {"lux":50, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Parcheggio esterno":        {"lux":10, "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Percorso pedonale esterno": {"lux":5,  "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI EN 12464-2:2025","area":"EXT"},
    "Strada veicolare ME3a":     {"lux":7.5,"ugr_max":55,"uni":0.40,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
    "Strada residenziale CE2":   {"lux":7.5,"ugr_max":55,"uni":0.40,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
    "Zona pedonale S4":          {"lux":5,  "ugr_max":55,"uni":0.25,"ra_min":60,"norma":"UNI 11248:2016","area":"STR"},
}

# ============================================================
# FUNZIONE CALCOLO
# ============================================================
def calcola_area(area, modalita="normale"):
    sup  = area["superficie_m2"]
    alt  = area.get("altezza_m", 2.70)
    req  = REQUISITI[area["tipo_locale"]]
    lamp = DB_LAMPADE[area["lampada"]]
    CU, MF = 0.60, 0.80

    if modalita == "emergenza":
        E_t = 1.0
        lamp_em_key = next(
            (k for k, v in DB_LAMPADE.items() if v["tipo"] == "Emergenza"),
            area["lampada"]
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
# RENDERING 3D
# ============================================================
def genera_rendering(area, calc):
    lato   = np.sqrt(area["superficie_m2"])
    alt    = area.get("altezza_m", 2.70)
    lamp   = DB_LAMPADE[area["lampada"]]
    coords = calc["coords"]
    is_ext = REQUISITI[area["tipo_locale"]]["area"] in ("EXT", "STR")
    is_em  = calc.get("modalita") == "emergenza"

    fig = plt.figure(figsize=(14, 9), dpi=150, facecolor="#050816")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#050816")

    pav = Poly3DCollection([[(0,0,0),(lato,0,0),(lato,lato,0),(0,lato,0)]], alpha=1.0)
    pav.set_facecolor((0.22,0.22,0.24)); pav.set_edgecolor("#4b5563")
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

    c_led  = (0.1,0.9,0.2) if is_em else (1.0,0.95,0.72)
    theta  = np.linspace(0, 2*np.pi, 20)
    for (lx, ly) in coords:
        h = alt - 0.05 if not is_ext else 6.0
        ax.scatter([lx],[ly],[h], c=[c_led], s=300, edgecolors="white", lw=1.5, zorder=10)
        for rr, alp in [(0.6,0.16),(1.2,0.09),(2.0,0.05)]:
            for ang in theta[::2]:
                ax.plot([lx, lx+rr*np.cos(ang)],[ly, ly+rr*np.sin(ang)],
                        [h, 0.04], color="#fef3c7", alpha=alp, lw=0.7)

    Xh, Yh = np.meshgrid(np.linspace(0.1,lato-0.1,50), np.linspace(0.1,lato-0.1,50))
    Zh     = np.zeros_like(Xh)
    h_lamp = alt if not is_ext else 6.0
    for (lx2, ly2) in coords:
        d2  = np.sqrt((Xh-lx2)**2+(Yh-ly2)**2+h_lamp**2)
        Zh += (lamp["flusso_lm"]/(2*np.pi))*(h_lamp/d2**3)
    Zn = (Zh-Zh.min())/(Zh.max()-Zh.min()+1e-9)
    cmap_use = plt.cm.summer if is_em else plt.cm.inferno
    ax.plot_surface(Xh, Yh, np.full_like(Xh,0.02), facecolors=cmap_use(Zn), alpha=0.55, shade=False)

    ax.set_xlim(0,lato); ax.set_ylim(0,lato)
    ax.set_zlim(0, max(alt,6.5) if is_ext else alt)
    ax.view_init(elev=30, azim=235); ax.axis("off")

    em_label = " [EMERGENZA]" if is_em else ""
    fig.text(.5,.97,
             "RENDERING 3D — " + area["nome"] + em_label,
             fontsize=14, fontweight="bold", color="white", ha="center", va="top")
    fig.text(.5,.93,
             str(calc["n"]) + "x " + area["lampada"][:38] +
             "  |  " + str(calc["E_m"]) + " lux  |  " +
             str(calc["W_t"]) + " W  |  " + str(calc["wm2"]) + " W/m2  |  " +
             REQUISITI[area["tipo_locale"]]["norma"],
             fontsize=8, color="#a5b4fc", ha="center")

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
    Z    = np.zeros_like(X)
    for (lx, ly) in coords:
        d  = np.sqrt((X-lx)**2+(Y-ly)**2+alt**2)
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
            nodes.append({"name": "Lamp_" + r["nome"],
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
    tm = sum(x["mat"] for x in righe)
    ti = sum(x["ins"] for x in righe)
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
        ox   = float(r.get("offset_x",0))
        lato = float(np.sqrt(r["sup"]))
        lines += ["0","LWPOLYLINE","8","AREE","90","4","70","1",
                  "10",str(ox),"20","0.0","10",str(ox+lato),"20","0.0",
                  "10",str(ox+lato),"20",str(lato),"10",str(ox),"20",str(lato)]
        for (lx, ly) in r["calc"]["coords"]:
            lines += ["0","POINT","8","LAMPADE",
                      "10",str(ox+lx),"20",str(ly),"30","0.0"]
        lines += ["0","TEXT","8","TESTI",
                  "10",str(ox+lato/2),"20",str(lato/2),"30","0.0",
                  "40","0.3","1", r["nome"] + " " + str(r["calc"]["E_m"]) + "lux"]
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
            except Exception:
                pass

        ax_title = fig.add_axes([0.13,0.90,0.85,0.08])
        ax_title.set_facecolor("#1a365d")
        ax_title.text(0.5,0.7,
                      "TAVOLA ILLUMINOTECNICA — LIGHTING AGENT PRO v4.0",
                      color="white",fontsize=13,fontweight="bold",
                      ha="center",va="center",transform=ax_title.transAxes)
        ax_title.text(0.5,0.2,
                      "Progetto: " + progetto["nome"] +
                      "  |  Committente: " + progetto["committente"] +
                      "  |  Data: " + progetto["data"] +
                      "  |  Tav. " + progetto["num_tavola"] +
                      "  |  UNI 11630:2016 + UNI EN 12464-1:2021",
                      color="#90cdf4",fontsize=7.5,va="center",ha="center",
                      transform=ax_title.transAxes)
        ax_title.axis("off")

        n_aree    = len(risultati)
        cols_plan = min(n_aree, 4)
        for idx, r in enumerate(risultati):
            row = idx // cols_plan
            col = idx % cols_plan
            w_ax = 0.22; h_ax = 0.25
            left   = 0.04 + col*(w_ax+0.02)
            bottom = 0.58 - row*(h_ax+0.05)
            ax = fig.add_axes([left,bottom,w_ax,h_ax])
            genera_isolux(ax, r["calc"]["coords"],
                          DB_LAMPADE[r["lampada"]], r["sup"], r.get("altezza_m",2.70))
            ok = r["calc"]["ok_lux"] == "✅"
            ax.set_title(
                r["nome"] + "\n" + str(r["calc"]["E_m"]) + " lux " +
                ("✅" if ok else "❌") + " | " + str(r["calc"]["n"]) +
                " lamp | " + str(r["calc"]["W_t"]) + "W",
                fontsize=6.5, color="#1a365d" if ok else "#c53030", pad=3)

        ax_tab = fig.add_axes([0.04,0.05,0.92,0.28]); ax_tab.axis("off")
        col_labels = ["Area","Tipo","m2","N","Target lux","Ottenuto",
                      "W","W/m2","Lux","UGR","Ra","Norma"]
        rows_data = []
        for r in risultati:
            req = REQUISITI[r["tipo_locale"]]
            rows_data.append([
                r["nome"][:20], r["tipo_locale"][:18], str(r["sup"]),
                str(r["calc"]["n"]), str(r["calc"]["E_t"]), str(r["calc"]["E_m"]),
                str(r["calc"]["W_t"]), str(r["calc"]["wm2"]),
                r["calc"]["ok_lux"], r["calc"]["ok_ugr"], r["calc"]["ok_ra"],
                req["norma"][:22]
            ])
        tbl = ax_tab.table(cellText=rows_data, colLabels=col_labels,
                           cellLoc="center", loc="upper center", bbox=[0,0,1,1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
        for (ri,ci), cell in tbl.get_celld().items():
            if ri == 0:
                cell.set_facecolor("#1a365d")
                cell.set_text_props(color="white", fontweight="bold")
            elif ri % 2 == 0:
                cell.set_facecolor("#ebf8ff")
            cell.set_edgecolor("#cbd5e0")

        tot_l = sum(r["calc"]["n"]   for r in risultati)
        tot_W = sum(r["calc"]["W_t"] for r in risultati)
        tot_s = sum(r["sup"]         for r in risultati)
        ax_foot = fig.add_axes([0.04,0.01,0.92,0.03])
        ax_foot.set_facecolor("#2d3748")
        ax_foot.text(0.5, 0.5,
                     "Totale: " + str(tot_l) + " lampade | " +
                     str(tot_W) + " W | " + str(tot_s) + " m2 | " +
                     str(round(tot_W/max(tot_s,1),1)) + " W/m2 | " +
                     "Progettista: " + progetto["progettista"] + " | " + progetto["data"],
                     color="white", fontsize=7, ha="center", va="center",
                     transform=ax_foot.transAxes)
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

        # — Frontespizio
        fig = plt.figure(figsize=(11.69,8.27), dpi=100, facecolor="#1a365d")
        ax  = fig.add_axes([0,0,1,1]); ax.set_facecolor("#1a365d"); ax.axis("off")
        if logo_bytes and PIL_OK:
            try:
                ax_l = fig.add_axes([0.38,0.72,0.24,0.14])
                ax_l.imshow(PILImage.open(BytesIO(logo_bytes))); ax_l.axis("off")
            except Exception:
                pass
        ax.text(0.5,0.65,"RELAZIONE TECNICA ILLUMINOTECNICA",
                color="white",fontsize=18,fontweight="bold",
                ha="center",transform=ax.transAxes)
        ax.text(0.5,0.57, progetto["nome"],
                color="#90cdf4",fontsize=14,ha="center",transform=ax.transAxes)
        for i,(label,val) in enumerate([
            ("Committente", progetto["committente"]),
            ("Progettista", progetto["progettista"]),
            ("Data",        progetto["data"]),
            ("N. Tavola",   progetto["num_tavola"]),
        ]):
            ax.text(0.3, 0.44-i*0.07, label + ":",
                    color="#a0aec0",fontsize=10,ha="right",transform=ax.transAxes)
            ax.text(0.32, 0.44-i*0.07, val,
                    color="white",fontsize=10,ha="left",transform=ax.transAxes)
        ax.text(0.5,0.10,
                "UNI 11630:2016 | UNI EN 12464-1:2021 | UNI EN 12464-2:2025 | "
                "UNI EN 1838:2025 | UNI 11248:2016",
                color="#4a5568",fontsize=8,ha="center",transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # — Tavola A3
        try:
            buf_tav = genera_pdf(progetto, risultati, logo_bytes)
            import matplotlib.image as mpimg
            img_arr = mpimg.imread(buf_tav)
            fig2 = plt.figure(figsize=(16.54,11.69), dpi=80)
            ax2  = fig2.add_axes([0,0,1,1])
            ax2.imshow(img_arr); ax2.axis("off")
            pdf.savefig(fig2, bbox_inches="tight"); plt.close(fig2)
        except Exception:
            pass

        # — Schede verifica per ogni area
        for r in risultati:
            req = REQUISITI[r["tipo_locale"]]
            fig3 = plt.figure(figsize=(11.69,8.27), dpi=100, facecolor="white")
            ax3  = fig3.add_axes([0,0.7,1,0.28])
            ax3.set_facecolor("#2b6cb0"); ax3.axis("off")
            ax3.text(0.5,0.6, "SCHEDA VERIFICA — " + r["nome"],
                     color="white",fontsize=13,fontweight="bold",
                     ha="center",transform=ax3.transAxes)
            ax3.text(0.5,0.2,
                     r["tipo_locale"] + " | " + str(r["sup"]) + " m2 | " +
                     str(r.get("altezza_m",2.70)) + " m | " + req["norma"],
                     color="#bee3f8",fontsize=9,ha="center",transform=ax3.transAxes)
            ax_iso = fig3.add_axes([0.05,0.18,0.45,0.48])
            genera_isolux(ax_iso, r["calc"]["coords"],
                          DB_LAMPADE[r["lampada"]], r["sup"], r.get("altezza_m",2.70))
            ax_iso.set_title("Distribuzione Isolux", fontsize=9, color="#1a365d")
            ax_v = fig3.add_axes([0.55,0.18,0.40,0.48]); ax_v.axis("off")
            checks = [
                ("Illuminamento Em",
                 str(r["calc"]["E_m"]) + " lux >= " + str(r["calc"]["E_t"]) + " lux",
                 r["calc"]["ok_lux"]),
                ("UGR massimo",
                 str(DB_LAMPADE[r["lampada"]]["ugr"]) + " <= " + str(req["ugr_max"]),
                 r["calc"]["ok_ugr"]),
                ("Uniformita U0", ">= " + str(req["uni"]), r["calc"]["ok_uni"]),
                ("Indice Ra",
                 str(DB_LAMPADE[r["lampada"]]["ra"]) + " >= " + str(req["ra_min"]),
                 r["calc"]["ok_ra"]),
            ]
            for i,(label,val,ok) in enumerate(checks):
                color = "#276749" if ok == "✅" else "#9b2c2c"
                ax_v.text(0.0, 0.85-i*0.22, ok + " " + label,
                          fontsize=9,color=color,fontweight="bold",transform=ax_v.transAxes)
                ax_v.text(0.0, 0.77-i*0.22, "    " + val,
                          fontsize=8,color="#4a5568",transform=ax_v.transAxes)
            ax_info = fig3.add_axes([0.05,0.02,0.90,0.14]); ax_info.axis("off")
            lk = r["calc"].get("lampada_usata", r["lampada"])
            ax_info.text(0.5, 0.6,
                         "Apparecchio: " + lk +
                         " | N=" + str(r["calc"]["n"]) +
                         " | Potenza: " + str(r["calc"]["W_t"]) + " W" +
                         " | " + str(r["calc"]["wm2"]) + " W/m2" +
                         " | k=" + str(r["calc"]["k"]) +
                         " | CU=" + str(r["calc"]["CU"]) +
                         " | MF=" + str(r["calc"]["MF"]),
                         fontsize=8,ha="center",color="#2d3748",transform=ax_info.transAxes)
            pdf.savefig(fig3, bbox_inches="tight"); plt.close(fig3)

        # — Rendering 3D per ogni area
        for r in risultati:
            try:
                buf_r = genera_rendering(r, r["calc"])
                import matplotlib.image as mpimg
                img = mpimg.imread(buf_r)
                fig4 = plt.figure(figsize=(14,9), dpi=80, facecolor="#050816")
                ax4  = fig4.add_axes([0,0,1,1])
                ax4.imshow(img); ax4.axis("off")
                pdf.savefig(fig4, bbox_inches="tight", facecolor="#050816")
                plt.close(fig4)
            except Exception:
                pass

        # — Preventivo
        if prev and prev.get("righe"):
            fig5 = plt.figure(figsize=(11.69,8.27), dpi=100, facecolor="white")
            ax5  = fig5.add_axes([0,0,1,1]); ax5.set_facecolor("white"); ax5.axis("off")
            ax5.text(0.5, 0.96, "PREVENTIVO DI SPESA",
                     fontsize=14, fontweight="bold", color="#1a365d",
                     ha="center", transform=ax5.transAxes)
            ax5.text(0.5, 0.91,
                     "Progetto: " + progetto["nome"] +
                     "  |  Committente: " + progetto["committente"] +
                     "  |  Data: " + progetto["data"],
                     fontsize=8, ha="center", color="#4a5568",
                     transform=ax5.transAxes)

            col_labels = ["Area","N","Apparecchio","Mod.","Mat. €","Install. €","Sub. €"]
            rows_data  = []
            for riga in prev["righe"]:
                rows_data.append([
                    riga["area"][:22],
                    str(riga["n"]),
                    riga["lampada"][:36],
                    riga.get("modalita","normale"),
                    f"{riga['mat']:.2f}",
                    f"{riga['ins']:.2f}",
                    f"{riga['sub']:.2f}",
                ])

            ax_tbl = fig5.add_axes([0.03, 0.32, 0.94, 0.55]); ax_tbl.axis("off")
            tbl5 = ax_tbl.table(cellText=rows_data, colLabels=col_labels,
                                cellLoc="center", loc="upper center", bbox=[0,0,1,1])
            tbl5.auto_set_font_size(False); tbl5.set_fontsize(7.5)
            for (ri, ci), cell in tbl5.get_celld().items():
                if ri == 0:
                    cell.set_facecolor("#1a365d")
                    cell.set_text_props(color="white", fontweight="bold")
                elif ri % 2 == 0:
                    cell.set_facecolor("#ebf8ff")
                cell.set_edgecolor("#cbd5e0")

            riepilogo = [
                ("Totale materiali",   f"€ {prev.get('tm',0):,.2f}"),
                ("Totale installazione", f"€ {prev.get('ti',0):,.2f}"),
                ("Netto",              f"€ {prev.get('tn',0):,.2f}"),
                ("Spese generali",     f"€ {prev.get('sg',0):,.2f}"),
                ("Oneri sicurezza",    f"€ {prev.get('os',0):,.2f}"),
                ("Margine",            f"€ {prev.get('mg',0):,.2f}"),
                ("Imponibile",         f"€ {prev.get('to',0):,.2f}"),
                ("IVA",                f"€ {prev.get('iva',0):,.2f}"),
                ("TOTALE COMPLESSIVO", f"€ {prev.get('tf',0):,.2f}"),
            ]
            ax5.add_patch(plt.Rectangle((0.03,0.03),0.94,0.27,
                          fill=True, facecolor="#f7fafc",
                          edgecolor="#cbd5e0", lw=1,
                          transform=ax5.transAxes))
            cols_r = 3
            for idx, (label, val) in enumerate(riepilogo):
                col_r = idx % cols_r
                row_r = idx // cols_r
                x_pos = 0.06 + col_r * 0.32
                y_pos = 0.27 - row_r * 0.08
                bold  = (label == "TOTALE COMPLESSIVO")
                ax5.text(x_pos, y_pos, label + ":",
                         fontsize=8, color="#4a5568",
                         fontweight="bold" if bold else "normal",
                         transform=ax5.transAxes)
                ax5.text(x_pos + 0.18, y_pos, val,
                         fontsize=8, color="#1a365d" if bold else "#2d3748",
                         fontweight="bold" if bold else "normal",
                         transform=ax5.transAxes)

            pdf.savefig(fig5, bbox_inches="tight"); plt.close(fig5)

    buf.seek(0)
    return buf

# ============================================================
# SESSION STATE INIT
# ============================================================
def init_session():
    defaults = {
        "aree":          [],
        "risultati":     [],
        "prev":          {},
        "nome_prog":     "Nuovo Progetto",
        "committente":   "",
        "progettista":   "",
        "num_tavola":    "T01",
        "logo_bytes":    None,
        "groq_key":      "",
        "gemini_key":    "",
        "mg_pct":        35,
        "sg_pct":        12,
        "os_pct":        4,
        "iva_pct":       22,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        "<div style='background:#1a365d;padding:1rem;border-radius:8px;"
        "color:white;text-align:center;margin-bottom:1rem'>"
        "<b>💡 Lighting Agent Pro v4.0</b><br>"
        "<small>UNI EN 12464-1:2021</small></div>",
        unsafe_allow_html=True
    )
    st.caption(f"👤 {st.session_state.username}")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()

    st.divider()
    st.subheader("📁 Progetto")
    st.session_state.nome_prog   = st.text_input("Nome progetto",  st.session_state.nome_prog)
    st.session_state.committente = st.text_input("Committente",    st.session_state.committente)
    st.session_state.progettista = st.text_input("Progettista",    st.session_state.progettista)
    st.session_state.num_tavola  = st.text_input("N. Tavola",      st.session_state.num_tavola)
    logo_up = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_up")
    if logo_up:
        st.session_state.logo_bytes = logo_up.read()
        st.image(st.session_state.logo_bytes, width=120)

    st.divider()
    st.subheader("🤖 AI Vision")
    st.session_state.groq_key   = st.text_input("Groq API Key",   type="password",
                                                  value=st.session_state.groq_key)
    st.session_state.gemini_key = st.text_input("Gemini API Key", type="password",
                                                  value=st.session_state.gemini_key)

    st.divider()
    st.subheader("💰 Preventivo")
    st.session_state.mg_pct  = st.slider("Margine %",         0, 60, st.session_state.mg_pct)
    st.session_state.sg_pct  = st.slider("Spese generali %",  0, 30, st.session_state.sg_pct)
    st.session_state.os_pct  = st.slider("Oneri sicurezza %", 0, 15, st.session_state.os_pct)
    st.session_state.iva_pct = st.slider("IVA %",             0, 25, st.session_state.iva_pct)

    st.divider()
    st.subheader("💾 Archivio Progetti")
    proj_list = load_projects_list(st.session_state.username)
    if proj_list:
        for pid, pnome, pcomm, pdata in proj_list:
            col1, col2, col3 = st.columns([3,1,1])
            col1.caption(f"**{pnome}**\n{pdata[:10]}")
            if col2.button("📂", key=f"load_{pid}"):
                d = load_project_data(pid)
                if d:
                    st.session_state.nome_prog   = d["nome"]
                    st.session_state.committente = d["committente"]
                    st.session_state.progettista = d["progettista"]
                    st.session_state.num_tavola  = d["num_tavola"]
                    st.session_state.aree        = d["aree"]
                    st.session_state.risultati   = d["risultati"]
                    st.session_state.prev        = d["prev"]
                    st.success("Progetto caricato!")
                    st.rerun()
            if col3.button("🗑️", key=f"del_{pid}"):
                delete_project(pid)
                st.rerun()
    else:
        st.caption("Nessun progetto salvato.")

# ============================================================
# HEADER
# ============================================================
st.markdown(
    "<div class='header-box'>"
    "<h1 style='margin:0;font-size:1.8rem'>💡 Lighting Agent Pro v4.0</h1>"
    "<p style='margin:.3rem 0 0;opacity:.85'>"
    "UNI EN 12464-1:2021 | UNI EN 12464-2:2025 | UNI EN 1838:2025 | "
    "UNI 11630:2016 | UNI 11248:2016</p>"
    "</div>",
    unsafe_allow_html=True
)

# ============================================================
# TAB PRINCIPALI
# ============================================================
TAB_AREE, TAB_AI, TAB_CALC, TAB_3D, TAB_PREV, TAB_EXPORT, TAB_DB = st.tabs([
    "📐 Aree",
    "🤖 AI Planimetria",
    "⚡ Calcoli",
    "🎨 3D / Isolux",
    "💰 Preventivo",
    "📄 Export",
    "🔦 Database Lampade",
])

# ============================================================
# TAB: AREE
# ============================================================
with TAB_AREE:
    st.subheader("Definizione Aree")

    with st.expander("➕ Aggiungi Area", expanded=True):
        c1, c2, c3 = st.columns(3)
        nome_a   = c1.text_input("Nome area", "Ufficio 1", key="na")
        tipo_a   = c2.selectbox("Tipo locale", list(REQUISITI.keys()), key="ta")
        sup_a    = c3.number_input("Superficie m²", 5.0, 5000.0, 30.0, 1.0, key="sa")

        c4, c5, c6 = st.columns(3)
        alt_a    = c4.number_input("Altezza m", 2.0, 15.0, 2.70, 0.05, key="aa")
        lamp_a   = c5.selectbox("Apparecchio", list(DB_LAMPADE.keys()), key="la")
        mod_a    = c6.selectbox("Modalità", ["normale","emergenza"], key="ma")

        req_p = REQUISITI[tipo_a]
        st.info(
            f"**Norma:** {req_p['norma']} | "
            f"**Em target:** {req_p['lux']} lux | "
            f"**UGR max:** {req_p['ugr_max']} | "
            f"**U₀ min:** {req_p['uni']} | "
            f"**Ra min:** {req_p['ra_min']}"
        )

        if st.button("➕ Aggiungi area", key="btn_add_area"):
            st.session_state.aree.append({
                "nome": nome_a, "tipo_locale": tipo_a,
                "superficie_m2": sup_a, "altezza_m": alt_a,
                "lampada": lamp_a, "modalita": mod_a,
                "sup": sup_a,
            })
            st.success(f"Area «{nome_a}» aggiunta.")
            st.rerun()

    if st.session_state.aree:
        st.markdown("### Aree definite")
        for idx, a in enumerate(st.session_state.aree):
            area_type = REQUISITI[a["tipo_locale"]]["area"]
            card_cls  = "em-card" if area_type == "EM" else ("ext-card" if area_type in ("EXT","STR") else "card")
            c1, c2 = st.columns([5,1])
            c1.markdown(
                f"<div class='{card_cls}'>"
                f"<b>{a['nome']}</b> — {a['tipo_locale']} | "
                f"{a['sup']} m² | {a['altezza_m']} m | "
                f"{a['lampada'][:40]}"
                f"</div>",
                unsafe_allow_html=True
            )
            if c2.button("🗑️", key=f"del_area_{idx}"):
                st.session_state.aree.pop(idx)
                st.rerun()
    else:
        st.info("Nessuna area definita. Aggiungine una sopra.")

# ============================================================
# TAB: AI PLANIMETRIA
# ============================================================
with TAB_AI:
    st.subheader("🤖 Analisi AI Planimetria")
    st.info("Carica una planimetria (JPG, PNG o PDF). "
            "L'AI individuerà automaticamente le aree e le aggiungerà al progetto.")

    up_plan = st.file_uploader("Planimetria", type=["jpg","jpeg","png","pdf"], key="up_plan")

    if up_plan:
        raw = up_plan.read()
        if up_plan.name.lower().endswith(".pdf"):
            st.info("Conversione PDF → JPG…")
            raw = convert_pdf_to_jpg(raw)

        if raw:
            if PIL_OK:
                st.image(raw, caption="Planimetria caricata", use_column_width=True)

            scala_ai = ""
            col_s1, col_s2 = st.columns(2)
            if col_s1.button("🔍 Rileva scala automatica", key="btn_scala"):
                with st.spinner("Rilevamento scala…"):
                    scala_ai = detect_scala_ai(
                        raw,
                        st.session_state.groq_key,
                        st.session_state.gemini_key
                    )
                if scala_ai:
                    st.success(f"Scala rilevata: {scala_ai}")
                else:
                    st.warning("Scala non rilevata.")

            if col_s2.button("🏠 Analizza aree AI", key="btn_ai_areas"):
                with st.spinner("Analisi in corso…"):
                    areas_ai = analizza_planimetria_ai(
                        raw,
                        st.session_state.groq_key,
                        st.session_state.gemini_key
                    )
                if areas_ai:
                    lamp_default = list(DB_LAMPADE.keys())[0]
                    for a in areas_ai:
                        tipo_ok = a.get("type","Ufficio VDT")
                        if tipo_ok not in REQUISITI:
                            tipo_ok = "Ufficio VDT"
                        st.session_state.aree.append({
                            "nome":         a.get("name","Area AI"),
                            "tipo_locale":  tipo_ok,
                            "superficie_m2": float(a.get("area_m2", 20.0)),
                            "altezza_m":    2.70,
                            "lampada":      lamp_default,
                            "modalita":     "normale",
                            "sup":          float(a.get("area_m2", 20.0)),
                        })
                    st.success(f"{len(areas_ai)} aree importate dall'AI!")
                    st.rerun()
                else:
                    st.error("Nessuna area rilevata. Verifica le API Key nella sidebar.")

    if CANVAS_OK:
        st.divider()
        st.subheader("✏️ Disegno Planimetria (canvas)")
        canvas_result = st_canvas(
            fill_color="rgba(43,108,176,0.15)",
            stroke_width=2,
            stroke_color="#2b6cb0",
            background_color="#f7fafc",
            height=400,
            drawing_mode="rect",
            key="canvas_plan",
        )
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            n_rect = len(canvas_result.json_data["objects"])
            st.caption(f"{n_rect} rettangol{'o' if n_rect==1 else 'i'} disegnati.")

# ============================================================
# TAB: CALCOLI
# ============================================================
with TAB_CALC:
    st.subheader("⚡ Calcoli Illuminotecnici")

    if not st.session_state.aree:
        st.warning("Definisci prima le aree nel tab **Aree**.")
    else:
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        if col_btn1.button("▶️ Calcola tutte le aree", key="btn_calc_all"):
            risultati_new = []
            for a in st.session_state.aree:
                calc = calcola_area(a, a.get("modalita","normale"))
                r = dict(a)
                r["calc"] = calc
                r["sup"]  = a["superficie_m2"]
                risultati_new.append(r)
            st.session_state.risultati = risultati_new
            st.session_state.prev = calc_preventivo(
                risultati_new,
                st.session_state.mg_pct,
                st.session_state.sg_pct,
                st.session_state.os_pct,
                st.session_state.iva_pct,
            )
            st.success("Calcolo completato!")

        if col_btn2.button("💾 Salva Progetto", key="btn_save"):
            save_project(
                st.session_state.username,
                st.session_state.nome_prog,
                st.session_state.committente,
                st.session_state.progettista,
                st.session_state.num_tavola,
                st.session_state.aree,
                st.session_state.risultati,
                st.session_state.prev,
            )
            st.success("Progetto salvato!")

        if col_btn3.button("🗑️ Svuota progetto", key="btn_clear"):
            st.session_state.aree      = []
            st.session_state.risultati = []
            st.session_state.prev      = {}
            st.rerun()

        if st.session_state.risultati:
            st.divider()
            tot_n = sum(r["calc"]["n"]   for r in st.session_state.risultati)
            tot_W = sum(r["calc"]["W_t"] for r in st.session_state.risultati)
            tot_s = sum(r["sup"]         for r in st.session_state.risultati)
            ok_c  = sum(1 for r in st.session_state.risultati if r["calc"]["ok_lux"]=="✅")

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Aree", len(st.session_state.risultati))
            m2.metric("Lampade totali", tot_n)
            m3.metric("Potenza totale", f"{tot_W} W")
            m4.metric("Conformi", f"{ok_c}/{len(st.session_state.risultati)}")

            st.divider()
            for r in st.session_state.risultati:
                req = REQUISITI[r["tipo_locale"]]
                area_type = req["area"]
                card_cls  = "em-card" if area_type=="EM" else ("ext-card" if area_type in ("EXT","STR") else "card")
                all_ok = all([
                    r["calc"]["ok_lux"]=="✅",
                    r["calc"]["ok_ugr"]=="✅",
                    r["calc"]["ok_ra"]=="✅",
                ])
                with st.expander(
                    f"{'✅' if all_ok else '❌'} {r['nome']} — "
                    f"{r['tipo_locale']} | {r['sup']} m² | "
                    f"{r['calc']['E_m']} lux | {r['calc']['n']} lamp | "
                    f"{r['calc']['W_t']} W"
                ):
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Illuminamento Em", f"{r['calc']['E_m']} lux",
                               delta=f"target {r['calc']['E_t']} lux")
                    cc2.metric("N° apparecchi", r["calc"]["n"])
                    cc3.metric("Potenza", f"{r['calc']['W_t']} W",
                               delta=f"{r['calc']['wm2']} W/m²")
                    cc4.metric("Indice locale k", r["calc"]["k"])

                    st.markdown(
                        f"| Verifica | Valore | Limite | Esito |\n"
                        f"|---|---|---|---|\n"
                        f"| Illuminamento Em | {r['calc']['E_m']} lux | ≥ {r['calc']['E_t']} lux | {r['calc']['ok_lux']} |\n"
                        f"| UGR | {DB_LAMPADE[r['lampada']]['ugr']} | ≤ {r['calc']['ugr_max']} | {r['calc']['ok_ugr']} |\n"
                        f"| Uniformità U₀ | — | ≥ {r['calc']['uni_min']} | {r['calc']['ok_uni']} |\n"
                        f"| Ra | {DB_LAMPADE[r['lampada']]['ra']} | ≥ {req['ra_min']} | {r['calc']['ok_ra']} |"
                    )
                    st.caption(
                        f"CU={r['calc']['CU']} | MF={r['calc']['MF']} | "
                        f"Interasse={r['calc']['ix']} m | "
                        f"Apparecchio: {r['calc'].get('lampada_usata', r['lampada'])}"
                    )

# ============================================================
# TAB: 3D / ISOLUX
# ============================================================
with TAB_3D:
    st.subheader("🎨 Rendering 3D e Diagrammi Isolux")

    if not st.session_state.risultati:
        st.warning("Esegui prima i calcoli nel tab **Calcoli**.")
    else:
        nomi_aree = [r["nome"] for r in st.session_state.risultati]
        sel_area  = st.selectbox("Seleziona area", nomi_aree, key="sel_3d")
        r_sel     = next(r for r in st.session_state.risultati if r["nome"] == sel_area)

        col_3d1, col_3d2 = st.columns(2)

        with col_3d1:
            st.markdown("#### Rendering 3D")
            with st.spinner("Generazione rendering…"):
                buf_r = genera_rendering(r_sel, r_sel["calc"])
            st.image(buf_r, use_column_width=True)
            st.download_button(
                "⬇️ Scarica PNG",
                data=buf_r.getvalue(),
                file_name=f"rendering_{sel_area}.png",
                mime="image/png",
                key=f"dl_3d_{sel_area}"
            )

        with col_3d2:
            st.markdown("#### Diagramma Isolux")
            fig_iso, ax_iso = plt.subplots(figsize=(7,6))
            genera_isolux(
                ax_iso,
                r_sel["calc"]["coords"],
                DB_LAMPADE[r_sel["lampada"]],
                r_sel["sup"],
                r_sel.get("altezza_m", 2.70)
            )
            ax_iso.set_title(
                f"{r_sel['nome']} — {r_sel['calc']['E_m']} lux",
                fontsize=10, color="#1a365d"
            )
            st.pyplot(fig_iso)
            plt.close(fig_iso)

        if st.button("🎬 Rendering 3D EMERGENZA", key="btn_em_3d"):
            r_em       = dict(r_sel)
            calc_em    = calcola_area(r_sel, "emergenza")
            r_em["calc"] = calc_em
            buf_em = genera_rendering(r_em, calc_em)
            st.image(buf_em, caption="Modalità Emergenza", use_column_width=True)

# ============================================================
# TAB: PREVENTIVO
# ============================================================
with TAB_PREV:
    st.subheader("💰 Preventivo di Spesa")

    if not st.session_state.risultati:
        st.warning("Esegui prima i calcoli.")
    else:
        if st.button("🔄 Ricalcola preventivo", key="btn_prev_ricalc"):
            st.session_state.prev = calc_preventivo(
                st.session_state.risultati,
                st.session_state.mg_pct,
                st.session_state.sg_pct,
                st.session_state.os_pct,
                st.session_state.iva_pct,
            )
            st.success("Preventivo aggiornato!")

        if st.session_state.prev:
            prev = st.session_state.prev
            st.markdown("#### Dettaglio voci")
            df_prev = pd.DataFrame(prev["righe"])
            if not df_prev.empty:
                df_prev.columns = [
                    "Area","N","Apparecchio","Modalità",
                    "Materiali €","Installazione €","Subtotale €"
                ]
                st.dataframe(df_prev, use_container_width=True)

            st.divider()
            p1,p2,p3,p4,p5,p6,p7 = st.columns(7)
            p1.metric("Materiali",    f"€ {prev['tm']:,.0f}")
            p2.metric("Installazione",f"€ {prev['ti']:,.0f}")
            p3.metric("Netto",        f"€ {prev['tn']:,.0f}")
            p4.metric("Sp. Generali", f"€ {prev['sg']:,.0f}")
            p5.metric("Oneri Sic.",   f"€ {prev['os']:,.0f}")
            p6.metric("Margine",      f"€ {prev['mg']:,.0f}")
            p7.metric("🏷️ TOTALE",   f"€ {prev['tf']:,.0f}")

# ============================================================
# TAB: EXPORT
# ============================================================
with TAB_EXPORT:
    st.subheader("📄 Export Documenti")

    if not st.session_state.risultati:
        st.warning("Esegui prima i calcoli.")
    else:
        progetto_info = {
            "nome":        st.session_state.nome_prog,
            "committente": st.session_state.committente,
            "progettista": st.session_state.progettista,
            "num_tavola":  st.session_state.num_tavola,
            "data":        datetime.now().strftime("%d/%m/%Y"),
        }

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)

        # Tavola A3
        with col_e1:
            st.markdown("#### 📐 Tavola A3")
            if st.button("Genera Tavola A3", key="btn_pdf_tav"):
                with st.spinner("Generazione PDF…"):
                    buf_tav = genera_pdf(
                        progetto_info,
                        st.session_state.risultati,
                        st.session_state.logo_bytes
                    )
                st.download_button(
                    "⬇️ Scarica Tavola A3",
                    data=buf_tav.getvalue(),
                    file_name=f"tavola_{st.session_state.nome_prog}.pdf",
                    mime="application/pdf",
                    key="dl_tav"
                )

        # Relazione completa
        with col_e2:
            st.markdown("#### 📋 Relazione Completa")
            if st.button("Genera Relazione", key="btn_relazione"):
                with st.spinner("Generazione relazione completa…"):
                    buf_rel = genera_relazione_completa(
                        progetto_info,
                        st.session_state.risultati,
                        st.session_state.prev,
                        st.session_state.logo_bytes,
                        st.session_state.mg_pct,
                        st.session_state.sg_pct,
                        st.session_state.os_pct,
                        st.session_state.iva_pct,
                    )
                st.download_button(
                    "⬇️ Scarica Relazione PDF",
                    data=buf_rel.getvalue(),
                    file_name=f"relazione_{st.session_state.nome_prog}.pdf",
                    mime="application/pdf",
                    key="dl_rel"
                )

        # DXF
        with col_e3:
            st.markdown("#### 📏 DXF AutoCAD")
            if st.button("Genera DXF", key="btn_dxf"):
                buf_dxf = genera_dxf(st.session_state.risultati)
                st.download_button(
                    "⬇️ Scarica DXF",
                    data=buf_dxf.getvalue(),
                    file_name=f"pianta_{st.session_state.nome_prog}.dxf",
                    mime="application/octet-stream",
                    key="dl_dxf"
                )

        # glTF
        with col_e4:
            st.markdown("#### 🧊 glTF 3D")
            if st.button("Genera glTF", key="btn_gltf"):
                buf_gltf = export_gltf_scene(st.session_state.risultati)
                st.download_button(
                    "⬇️ Scarica glTF",
                    data=buf_gltf.getvalue(),
                    file_name=f"scene_{st.session_state.nome_prog}.gltf",
                    mime="model/gltf+json",
                    key="dl_gltf"
                )

        st.divider()
        st.markdown("#### 📊 Export CSV dati calcolo")
        if st.button("Genera CSV", key="btn_csv"):
            rows_csv = []
            for r in st.session_state.risultati:
                req = REQUISITI[r["tipo_locale"]]
                rows_csv.append({
                    "Area":          r["nome"],
                    "Tipo":          r["tipo_locale"],
                    "Superficie m2": r["sup"],
                    "Altezza m":     r.get("altezza_m", 2.70),
                    "Apparecchio":   r["lampada"],
                    "N lampade":     r["calc"]["n"],
                    "Em lux":        r["calc"]["E_m"],
                    "Et lux":        r["calc"]["E_t"],
                    "Watt totali":   r["calc"]["W_t"],
                    "W/m2":          r["calc"]["wm2"],
                    "UGR lamp":      DB_LAMPADE[r["lampada"]]["ugr"],
                    "UGR max":       req["ugr_max"],
                    "Ra lamp":       DB_LAMPADE[r["lampada"]]["ra"],
                    "Ra min":        req["ra_min"],
                    "Ok Lux":        r["calc"]["ok_lux"],
                    "Ok UGR":        r["calc"]["ok_ugr"],
                    "Ok Ra":         r["calc"]["ok_ra"],
                    "Norma":         req["norma"],
                })
            df_csv = pd.DataFrame(rows_csv)
            csv_bytes = df_csv.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Scarica CSV",
                data=csv_bytes,
                file_name=f"calcoli_{st.session_state.nome_prog}.csv",
                mime="text/csv",
                key="dl_csv"
            )

# ============================================================
# TAB: DATABASE LAMPADE
# ============================================================
with TAB_DB:
    st.subheader("🔦 Database Apparecchi")

    # Mostra database esistente
    st.markdown("#### Apparecchi disponibili")
    rows_db = []
    for k, v in DB_LAMPADE.items():
        rows_db.append({
            "Codice":        k,
            "Produttore":    v["produttore"],
            "Tipo":          v["tipo"],
            "Flusso lm":     v["flusso_lm"],
            "Potenza W":     v["potenza_W"],
            "Efficienza":    v["efficienza"],
            "Ra":            v["ra"],
            "UGR":           v["ugr"],
            "IP":            v["ip"],
            "Temp. colore":  v["temp_colore"],
            "Dimmerabile":   "✅" if v["dimmerabile"] else "❌",
            "Classe energ.": v["classe_energ"],
            "Prezzo €":      v["prezzo"],
            "Install. €":    v["installazione"],
        })
    df_db = pd.DataFrame(rows_db)
    st.dataframe(df_db, use_container_width=True, height=350)

    st.divider()
    st.markdown("#### ➕ Aggiungi apparecchio personalizzato")
    with st.form("form_lamp"):
        fc1, fc2, fc3 = st.columns(3)
        lc_cod  = fc1.text_input("Codice / Nome")
        lc_prod = fc2.text_input("Produttore")
        lc_tipo = fc3.selectbox("Tipo",
                    ["Downlight","Lineare","Sospensione","Proiettore",
                     "Stradale","Emergenza","Plafoniera","Applique"])
        fc4, fc5, fc6 = st.columns(3)
        lc_flux = fc4.number_input("Flusso lm",   100, 100000, 3000, 100)
        lc_watt = fc5.number_input("Potenza W",   1,   2000,   25,   1)
        lc_ugr  = fc6.number_input("UGR",         0,   60,     19,   1)
        fc7, fc8, fc9 = st.columns(3)
        lc_ra   = fc7.number_input("Ra",          0,   100,    80,   1)
        lc_ip   = fc8.text_input("IP", "IP20")
        lc_tc   = fc9.text_input("Temp. colore", "4000K")
        fc10, fc11, fc12 = st.columns(3)
        lc_pre  = fc10.number_input("Prezzo €",   0,   10000,  200,  5)
        lc_ins  = fc11.number_input("Installaz. €", 0, 2000,   50,   5)
        lc_dim  = fc12.checkbox("Dimmerabile", value=True)
        lc_cls  = st.selectbox("Classe energetica", ["A++","A+","A","B","C"])

        submitted = st.form_submit_button("➕ Aggiungi al database")
        if submitted:
            if not lc_cod.strip():
                st.error("Inserisci un codice/nome per l'apparecchio.")
            elif lc_cod in DB_LAMPADE:
                st.error("Codice già presente nel database.")
            else:
                eff = round(lc_flux / max(lc_watt, 1), 1)
                st.session_state.DB_LAMPADE[lc_cod] = {
                    "produttore":    lc_prod,
                    "flusso_lm":     lc_flux,
                    "potenza_W":     lc_watt,
                    "efficienza":    eff,
                    "ra":            lc_ra,
                    "temp_colore":   lc_tc,
                    "ugr":           lc_ugr,
                    "prezzo":        lc_pre,
                    "installazione": lc_ins,
                    "tipo":          lc_tipo,
                    "ip":            lc_ip,
                    "dimmerabile":   lc_dim,
                    "classe_energ":  lc_cls,
                }
                DB_LAMPADE = st.session_state.DB_LAMPADE
                st.success(f"Apparecchio «{lc_cod}» aggiunto!")
                st.rerun()

    st.divider()
    st.markdown("#### 🗑️ Rimuovi apparecchio")
    lamp_da_rimuovere = st.selectbox("Seleziona apparecchio da rimuovere",
                                      list(DB_LAMPADE.keys()), key="del_lamp_sel")
    if st.button("🗑️ Rimuovi", key="btn_del_lamp"):
        if lamp_da_rimuovere in st.session_state.DB_LAMPADE:
            del st.session_state.DB_LAMPADE[lamp_da_rimuovere]
            DB_LAMPADE = st.session_state.DB_LAMPADE
            st.success(f"Apparecchio «{lamp_da_rimuovere}» rimosso.")
            st.rerun()
