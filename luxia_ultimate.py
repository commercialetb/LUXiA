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

# --- 1. CONFIGURAZIONE & SESSIONE ---
st.set_page_config(page_title="LUXiA Ultimate Titan v10.8", layout="wide")

def init_session():
    defaults = {
        'logged_in': False, 'username': None, 'studio': None, 'logo': None,
        'rooms': [], 'current_id': None, 'current_name': None, 
        'img_b64': None, 'strat_results': {}, 'zoom_level': 2.0,
        'pdf_cache': None, 'canv_w': 800, 'canv_h': 600
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session()

# --- 2. DATABASE ENGINE (AUTOCORRETTIVO) ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS rooms (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, area REAL)')
    
    # Migrazione automatica colonne mancanti
    cols = [('brand', 'TEXT'), ('model', 'TEXT'), ('qty', 'INTEGER'), ('lux_target', 'INTEGER'), ('strategy', 'TEXT'), ('price', 'REAL')]
    for col_n, col_t in cols:
        try: c.execute(f'ALTER TABLE rooms ADD COLUMN {col_n} {col_t}')
        except: pass
    conn.commit(); conn.close()

# --- 3. VISION & ZOOM ENGINE (FRIZZATO) ---
def process_pdf_full(pdf_bytes, zoom_factor):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = page.get_text("words")
    detected = []
    for cnt in contours:
        if cv2.contourArea(cnt) > (15000 * (zoom_factor/2)):
            x, y, w, h = cv2.boundingRect(cnt)
            label = ""
            for wd in words:
                if x < wd[0]*zoom_factor < x+w and y < wd[1]*zoom_factor < y+h:
                    if len(wd[4]) > 2: label += wd[4] + " "
            detected.append({"name": label.strip() or f"Vano {len(detected)+1}", "x": x, "y": y, "w": w, "h": h, "area": round((w*h)/(1200*zoom_factor), 2)})
    buf = BytesIO(); img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), detected, pix.width, pix.height

# --- 4. INTERFACCIA PRINCIPALE ---
def main():
    init_db()

    if not st.session_state.logged_in:
        st.title("🏛️ LUXiA Ultimate Titan v10.8")
        t1, t2 = st.tabs(["Accedi", "Registra Studio"])
        with t1:
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", key="btn_login"):
                conn = sqlite3.connect('luxia_titan.db')
                r = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, hashlib.sha256(p.encode()).hexdigest())).fetchone()
                if r: 
                    st.session_state.update({"logged_in": True, "username": u, "studio": r[0], "logo": r[1]})
                    st.rerun()
                else: st.error("Credenziali non valide.")
        with t2:
            nu = st.text_input("Nuovo Username", key="reg_user")
            np = st.text_input("Password di Registrazione", type="password", key="reg_pass")
            ns = st.text_input("Nome Studio Professionale", key="reg_studio")
            nl = st.file_uploader("Carica Logo Studio", type=['png', 'jpg'], key="reg_logo")
            if st.button("Crea Account", key="btn_reg"):
                l_b64 = base64.b64encode(nl.read()).decode() if nl else None
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, hashlib.sha256(np.encode()).hexdigest(), ns, l_b64))
                conn.commit(); st.success("Studio registrato! Ora puoi accedere.")
        return

    # SIDEBAR
    with st.sidebar:
        if st.session_state.logo: st.image(base64.b64decode(st.session_state.logo), width=150)
        st.header(st.session_state.studio)
        st.session_state.zoom_level = st.slider("🔍 Zoom HD Planimetria", 1.0, 4.0, st.session_state.zoom_level, 0.5)
        ai_on = st.toggle("Usa AI Groq", value=True, key="ai_toggle")
        if ai_on:
            gk = st.text_input("Chiave API Groq", type="password", key="groq_key")
            if gk: st.session_state.groq = Groq(api_key=gk)
        
        conn = sqlite3.connect('luxia_titan.db')
        projs = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        sel = st.selectbox("Seleziona Progetto", ["-- Dashboard --"] + [p[1] for p in projs], key="sel_proj")
        if st.button("Carica Progetto", key="btn_load"):
            p_id = next((p[0] for p in projs if p[1] == sel), None)
            st.session_state.update({"current_id": p_id, "current_name": sel if p_id else None})
            st.rerun()
        if st.button("Logout", key="btn_logout"): st.session_state.clear(); st.rerun()

    # LOGICA DASHBOARD / PROGETTO
    if not st.session_state.current_id:
        st.title("📊 Gestione Cantieri")
        with st.form("new_proj_form"):
            pn = st.text_input("Nome Cantiere / Riferimento")
            if st.form_submit_button("Crea Nuovo Progetto"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, date) VALUES (?,?,?)", (st.session_state.username, pn, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.rerun()
    else:
        st.title(f"📂 Lavoro: {st.session_state.current_name}")
        tab1, tab2, tab3 = st.tabs(["📐 Vision & Mappa", "💡 Calcoli & AI", "📄 Report Economico"])

        with tab1:
            col1, col2 = st.columns([1, 3])
            with col1:
                pdf = st.file_uploader("Carica PDF Planimetria", type=['pdf'], key="uploader_pdf")
                if pdf or st.button("🔄 Aggiorna Visualizzazione", key="btn_refresh"):
                    if pdf: st.session_state.pdf_cache = pdf.read()
                    if st.session_state.pdf_cache:
                        b64, rooms, w, h = process_pdf_full(st.session_state.pdf_cache, st.session_state.zoom_level)
                        st.session_state.update({"img_b64": b64, "rooms": rooms, "canv_w": w, "canv_h": h})
                for i, r in enumerate(st.session_state.rooms):
                    st.session_state.rooms[i]['name'] = st.text_input(f"Nome Vano {i}", r['name'], key=f"room_name_{i}")

            with col2:
                if st.session_state.img_b64:
                    st.markdown(f"<style>.stCanvasContainer {{ overflow: auto; max-height: 800px; }} div[data-testid='stCanvas'] {{ background-image: url('data:image/png;base64,{st.session_state.img_b64}'); background-size: contain; background-repeat: no-repeat; }}</style>", unsafe_allow_html=True)
                    rects = [{"type":"rect","left":r['x'],"top":r['y'],"width":r['w'],"height":r['h'], "fill":"rgba(0, 255, 0, 0.1)","stroke":"green"} for r in st.session_state.rooms]
                    st_canvas(initial_drawing={"objects": rects}, background_color="rgba(0,0,0,0)", height=st.session_state.canv_h, width=st.session_state.canv_w, key=f"canvas_{st.session_state.zoom_level}")

        with tab2:
            for i, r in enumerate(st.session_state.rooms):
                with st.expander(f"⚙️ Progettazione {r['name']} ({r['area']} mq)"):
                    c_i, c_o = st.columns([1, 2])
                    with c_i:
                        lx = st.slider("Target Illuminamento (Lux)", 100, 1000, 300, key=f"lux_val_{i}")
                        br = st.selectbox("Brand Suggerito", ["iGuzzini", "Artemide", "Flos", "Viabizzuno"], key=f"brand_sel_{i}")
                        pr = st.number_input("Costo Unitario (€)", 10.0, 5000.0, 150.0, key=f"price_val_{i}")
                        if st.button(f"Calcola Illuminazione", key=f"btn_calc_{i}"):
                            if ai_on and 'groq' in st.session_state:
                                res = st.session_state.groq.chat.completions.create(messages=[{"role":"user","content":f"Progetta illuminazione per {r['name']} di {r['area']}mq, target {lx} lux con {br}."}], model="llama-3.3-70b-versatile")
                                st.session_state.strat_results[i] = res.choices[0].message.content
                            else:
                                n_lamp = int((lx * r['area']) / (2500 * 0.7)) + 1
                                st.session_state.strat_results[i] = f"**Calcolo Tecnico Offline:** Suggeriti {n_lamp} apparecchi LED da 2500lm."
                            
                            conn = sqlite3.connect('luxia_titan.db')
                            conn.execute("INSERT INTO rooms (project_id, r_name, area, brand, lux_target, strategy, price, qty) VALUES (?,?,?,?,?,?,?,?)", (st.session_state.current_id, r['name'], r['area'], br, lx, st.session_state.strat_results[i], pr, 4))
                            conn.commit(); conn.close()
                    with c_o:
                        if i in st.session_state.strat_results: st.markdown(st.session_state.strat_results[i])

        with tab3:
            st.subheader("Computo Metrico Estimativo")
            conn = sqlite3.connect('luxia_titan.db')
            df = pd.read_sql_query(f"SELECT r_name as Vano, area as MQ, brand as Marca, lux_target as Lux, price as Prezzo_Cad FROM rooms WHERE project_id={st.session_state.current_id}", conn)
            conn.close()
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.metric("Budget Totale Stimato", f"€ {df['Prezzo_Cad'].sum():,.2f}")
                st.button("📄 Esporta Report PDF", key="btn_pdf_export")

if __name__ == "__main__":
    main()