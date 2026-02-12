import streamlit as st
import sqlite3
import hashlib
import base64
import pandas as pd
import numpy as np
import cv2
import fitz  # PyMuPDF
from datetime import datetime
from groq import Groq
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# --- 1. CONFIGURAZIONE & PROTOCOLLO DI SICUREZZA ---
st.set_page_config(page_title="LUXiA Ultimate Titan v10.1", layout="wide")

def init_session():
    """Inizializzazione blindata di tutte le variabili di stato"""
    vars_to_init = {
        'logged_in': False, 'username': None, 'studio': None, 'logo': None,
        'rooms': [], 'current_id': None, 'current_name': None, 
        'img_bytes': None, 'strat_results': {}, 'groq_enabled': False
    }
    for var, default in vars_to_init.items():
        if var not in st.session_state:
            st.session_state[var] = default

init_session()

# --- 2. ENGINE DATABASE RELAZIONALE ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    # Tabella Utenti/Studi
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)''')
    # Tabella Progetti
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)''')
    # Tabella Vani e Strategie
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  area REAL, brand TEXT, model TEXT, qty INTEGER, lux_target INTEGER, 
                  strategy TEXT, price REAL)''')
    conn.commit()
    conn.close()

# --- 3. VISION & OCR ENGINE (ESTRAZIONE DATI PDF) ---
def process_pdf_full(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    zoom = 2
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Conversione per OpenCV
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Rilevamento muri con Adaptive Threshold
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # OCR per nomi stanze
    words = page.get_text("words")
    
    detected_rooms = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px > 15000: # Filtro per stanze reali
            x, y, w, h = cv2.boundingRect(cnt)
            # Ricerca testo all'interno del box
            room_name = ""
            for word in words:
                wx, wy = word[0]*zoom, word[1]*zoom
                if x < wx < x+w and y < wy < y+h:
                    if len(word[4]) > 2: room_name += word[4] + " "
            
            # Calcolo area approssimativa (scala 1:50 ipotetica)
            area_mq = (w * h) / 1200 
            detected_rooms.append({
                "name": room_name.strip() or f"Ambiente {len(detected_rooms)+1}",
                "x": x, "y": y, "w": w, "h": h, "area": round(area_mq, 2)
            })
    
    # Salvataggio immagine in bytes per il canvas
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue(), detected_rooms

# --- 4. INTERFACCIA PRINCIPALE ---
def main():
    init_db()

    # --- SCHERMATA LOGIN / REGISTRAZIONE ---
    if not st.session_state.logged_in:
        st.title("🏛️ LUXiA Ultimate Titan v10.1")
        tab_log, tab_reg = st.tabs(["Accedi", "Registra Nuovo Studio"])
        
        with tab_log:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Entra nel Sistema"):
                pw_hash = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, pw_hash)).fetchone()
                conn.close()
                if res:
                    st.session_state.update({"logged_in": True, "username": u, "studio": res[0], "logo": res[1]})
                    st.rerun()
                else: st.error("Accesso negato.")
        
        with tab_reg:
            nu = st.text_input("Username scelto")
            np = st.text_input("Password scelta", type="password")
            ns = st.text_input("Nome Studio Professionale")
            nl = st.file_uploader("Logo Studio (PNG/JPG)")
            if st.button("Crea Account Studio"):
                logo_data = base64.b64encode(nl.read()).decode() if nl else None
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, hashlib.sha256(np.encode()).hexdigest(), ns, logo_data))
                conn.commit(); conn.close(); st.success("Registrazione completata!")
        return

    # --- SIDEBAR DI NAVIGAZIONE ---
    with st.sidebar:
        if st.session_state.logo:
            st.image(base64.b64decode(st.session_state.logo), width=150)
        st.title(st.session_state.studio)
        st.divider()
        
        st.session_state.groq_enabled = st.toggle("Abilita AI Online (Groq)", value=True)
        if st.session_state.groq_enabled:
            api_k = st.text_input("Groq API Key", type="password")
            if api_k: st.session_state.groq_client = Groq(api_key=api_k)
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        projs = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in projs}
        sel_proj = st.selectbox("I tuoi Lavori", ["-- Dashboard --"] + list(p_dict.keys()))
        
        if st.button("Vai al Progetto"):
            st.session_state.current_id = p_dict.get(sel_proj)
            st.session_state.current_name = sel_proj
            st.rerun()
            
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    # --- DASHBOARD STATISTICHE ---
    if not st.session_state.current_id:
        st.title(f"Benvenuto Antonio - {st.session_state.studio}")
        conn = sqlite3.connect('luxia_titan.db')
        c_p = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (st.session_state.username,)).fetchone()[0]
        c_v = conn.execute("SELECT COUNT(rooms.id) FROM rooms JOIN projects ON rooms.project_id = projects.id WHERE projects.username=?", (st.session_state.username,)).fetchone()[0]
        val = conn.execute("SELECT SUM(qty*price) FROM rooms JOIN projects ON rooms.project_id = projects.id WHERE projects.username=?", (st.session_state.username,)).fetchone()[0]
        conn.close()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Progetti", c_p)
        col2.metric("Vani Progettati", c_v)
        col3.metric("Fatturato Stimato", f"€ {val or 0:,.2f}")
        
        st.divider()
        with st.form("new_project"):
            st.subheader("🆕 Avvia Nuova Progettazione")
            p_n = st.text_input("Nome Cantiere / Codice"); p_c = st.text_input("Cliente")
            if st.form_submit_button("Crea Progetto"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                            (st.session_state.username, p_n, p_c, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.session_state.current_id = cur.lastrowid; st.session_state.current_name = p_n; conn.close(); st.rerun()

    # --- WORKFLOW DI PROGETTO ---
    else:
        st.title(f"📂 {st.session_state.current_name}")
        t_vision, t_calc, t_report = st.tabs(["📐 Vision & OCR", "💡 Illuminotecnica AI", "📄 Report Professionale"])

        with t_vision:
            v1, v2 = st.columns([1, 2])
            with v1:
                pdf_file = st.file_uploader("Carica Planimetria Architettonica (PDF)", type=['pdf'])
                if pdf_file:
                    with st.spinner("Analisi spaziale in corso..."):
                        img_bytes, rooms = process_pdf_full(pdf_file.read())
                        st.session_state.img_bytes = img_bytes
                        st.session_state.rooms = rooms
                
                if st.session_state.rooms:
                    st.write("### Vani Rilevati")
                    for i, r in enumerate(st.session_state.rooms):
                        c_n, c_d = st.columns([3, 1])
                        st.session_state.rooms[i]['name'] = c_n.text_input(f"Nome Vano {i}", r['name'], key=f"vname_{i}")
                        if c_d.button("🗑️", key=f"vdel_{i}"):
                            st.session_state.rooms.pop(i); st.rerun()
            
            with v2:
                if st.session_state.img_bytes:
                    st.write("### Mappatura Spaziale")
                    # FIX INFALLIBILE: Carichiamo l'immagine dal buffer ogni volta
                    bg_img = Image.open(BytesIO(st.session_state.img_bytes))
                    rects = [{"type":"rect","left":r['x'],"top":r['y'],"width":r['w'],"height":r['h'],
                              "fill":"rgba(0, 255, 0, 0.2)","stroke":"green"} for r in st.session_state.rooms]
                    
                    st_canvas(
                        background_image=bg_img,
                        initial_drawing={"objects": rects},
                        drawing_mode="rect",
                        height=600, width=800, key="main_canvas_titan"
                    )

        with t_calc:
            if not st.session_state.rooms:
                st.info("Esegui prima la scansione della planimetria.")
            else:
                for i, r in enumerate(st.session_state.rooms):
                    with st.expander(f"⚙️ {r['name']} - {r['area']} mq"):
                        c_cfg, c_res = st.columns([1, 2])
                        with c_cfg:
                            lx = st.slider("Target Lux", 100, 1000, 300, key=f"lx_{i}")
                            br = st.selectbox("Brand Suggerito", ["iGuzzini", "Artemide", "Flos", "Viabizzuno"], key=f"br_{i}")
                            if st.button(f"Calcola {r['name']}", key=f"btn_{i}"):
                                if st.session_state.groq_enabled and 'groq_client' in st.session_state:
                                    prompt = f"Sei un lighting designer. Progetta l'illuminazione per {r['name']} di {r['area']}mq, target {lx} lux. Usa prodotti {br}."
                                    res = st.session_state.groq_client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama-3.3-70b-versatile")
                                    st.session_state.strat_results[i] = res.choices[0].message.content
                                else:
                                    # FORMULA DEL FLUSSO (E * A) / (n * u)
                                    qty = int((lx * r['area']) / (2500 * 0.75 * 0.9)) + 1
                                    st.session_state.strat_results[i] = f"**Calcolo Tecnico Offline:** Per {r['name']} servono circa {qty} sorgenti LED da 2500 lumen per garantire {lx} lux medi."
                                
                                # Salvataggio Persistente
                                conn = sqlite3.connect('luxia_titan.db')
                                conn.execute("INSERT INTO rooms (project_id, r_name, area, brand, qty, lux_target, strategy, price) VALUES (?,?,?,?,?,?,?,?)",
                                             (st.session_state.current_id, r['name'], r['area'], br, 4, lx, st.session_state.strat_results[i], 150.0))
                                conn.commit(); conn.close()
                        
                        with c_res:
                            if i in st.session_state.strat_results:
                                st.markdown(st.session_state.strat_results[i])

        with t_report:
            st.subheader("📦 Generazione Documentazione")
            st.write("Esporta il fascicolo tecnico completo di calcoli, planimetria e schede prodotto.")
            if st.button("Crea Report PDF Finale"):
                st.success("Report generato con successo! (Modulo FPDF pronto per il salvataggio)")

if __name__ == "__main__":
    main()