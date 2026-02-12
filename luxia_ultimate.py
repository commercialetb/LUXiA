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
st.set_page_config(page_title="LUXiA Ultimate Titan v10.6", layout="wide")

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

# --- 2. DATABASE ENGINE (Full Schema) ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  area REAL, brand TEXT, model TEXT, qty INTEGER, lux_target INTEGER, 
                  strategy TEXT, price REAL)''')
    conn.commit(); conn.close()

# --- 3. VISION & OCR ENGINE ---
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
            detected.append({
                "name": label.strip() or f"Ambiente {len(detected)+1}",
                "x": x, "y": y, "w": w, "h": h, 
                "area": round((w * h) / (1200 * zoom_factor), 2)
            })
    
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), detected, pix.width, pix.height

# --- 4. LOGICA APPLICATIVA ---
def main():
    init_db()

    # --- LOGIN & REGISTRAZIONE ---
    if not st.session_state.logged_in:
        st.title("🏛️ LUXiA Ultimate Titan v10.6")
        tab1, tab2 = st.tabs(["Accedi", "Registra Studio"])
        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                conn = sqlite3.connect('luxia_titan.db')
                r = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, hashlib.sha256(p.encode()).hexdigest())).fetchone()
                if r: 
                    st.session_state.update({"logged_in": True, "username": u, "studio": r[0], "logo": r[1]})
                    st.rerun()
                else: st.error("Credenziali errate.")
        with tab2:
            nu = st.text_input("Nuovo User"); np = st.text_input("Nuova Pw", type="password")
            ns = st.text_input("Nome Studio Tecnico"); nl = st.file_uploader("Logo Studio", type=['png', 'jpg'])
            if st.button("Registra"):
                l_b64 = base64.b64encode(nl.read()).decode() if nl else None
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, hashlib.sha256(np.encode()).hexdigest(), ns, l_b64))
                conn.commit(); st.success("Studio registrato correttamente!")
        return

    # --- SIDEBAR ---
    with st.sidebar:
        if st.session_state.logo: st.image(base64.b64decode(st.session_state.logo), width=150)
        st.header(st.session_state.studio)
        st.divider()
        st.session_state.zoom_level = st.slider("🔍 Zoom HD", 1.0, 4.0, st.session_state.zoom_level, 0.5)
        ai_on = st.toggle("AI Online (Groq)", value=True)
        if ai_on:
            gk = st.text_input("Groq Key", type="password")
            if gk: st.session_state.groq = Groq(api_key=gk)
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        projs = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        sel = st.selectbox("I tuoi Progetti", ["-- Dashboard --"] + [p[1] for p in projs])
        if st.button("Apri Progetto"):
            p_id = next((p[0] for p in projs if p[1] == sel), None)
            st.session_state.update({"current_id": p_id, "current_name": sel if p_id else None})
            st.rerun()
        if st.button("Logout"): st.session_state.clear(); st.rerun()

    # --- DASHBOARD STATS ---
    if not st.session_state.current_id:
        st.title(f"Benvenuto Antonio")
        conn = sqlite3.connect('luxia_titan.db')
        stats = conn.execute("SELECT COUNT(*), (SELECT SUM(qty*price) FROM rooms) FROM projects WHERE username=?", (st.session_state.username,)).fetchone()
        conn.close()
        c1, c2 = st.columns(2)
        c1.metric("Progetti Attivi", stats[0])
        c2.metric("Valore Pipeline", f"€ {stats[1] or 0:,.2f}")
        
        with st.form("new_p"):
            pn = st.text_input("Nome Nuovo Cantiere"); pc = st.text_input("Cliente")
            if st.form_submit_button("Avvia Progetto"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                            (st.session_state.username, pn, pc, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.rerun()

    # --- WORKFLOW PROGETTO ---
    else:
        st.title(f"📂 {st.session_state.current_name}")
        t1, t2, t3 = st.tabs(["📐 Vision & Zoom", "💡 Progettazione & AI", "📄 Report & Economico"])

        with t1:
            col1, col2 = st.columns([1, 3])
            with col1:
                pdf = st.file_uploader("Carica Planimetria PDF", type=['pdf'])
                if pdf or st.button("Aggiorna Zoom"):
                    if pdf: st.session_state.pdf_cache = pdf.read()
                    if st.session_state.pdf_cache:
                        b64, rooms, w, h = process_pdf_full(st.session_state.pdf_cache, st.session_state.zoom_level)
                        st.session_state.update({"img_b64": b64, "rooms": rooms, "canv_w": w, "canv_h": h})
                
                for i, r in enumerate(st.session_state.rooms):
                    st.session_state.rooms[i]['name'] = st.text_input(f"Nome Vano {i}", r['name'], key=f"vn_{i}")
            
            with col2:
                if st.session_state.img_b64:
                    # FIX CSS PROXY INFALLIBILE
                    st.markdown(f"""
                        <style>
                        .stCanvasContainer {{ overflow: auto; max-height: 800px; border: 1px solid #555; }}
                        div[data-testid="stCanvas"] {{
                            background-image: url("data:image/png;base64,{st.session_state.img_b64}");
                            background-size: contain; background-repeat: no-repeat;
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                    rects = [{"type":"rect","left":r['x'],"top":r['y'],"width":r['w'],"height":r['h'],
                              "fill":"rgba(0, 255, 0, 0.15)","stroke":"green"} for r in st.session_state.rooms]
                    st_canvas(initial_drawing={"objects": rects}, drawing_mode="rect",
                              background_color="rgba(0,0,0,0)", height=st.session_state.canv_h, 
                              width=st.session_state.canv_w, key="titan_v10_canvas")

        with t2:
            for i, r in enumerate(st.session_state.rooms):
                with st.expander(f"⚙️ Configurazione: {r['name']} ({r['area']} mq)"):
                    c_in, c_out = st.columns([1, 2])
                    with c_in:
                        lx = st.slider("Target Lux", 100, 1000, 300, key=f"lx_{i}")
                        br = st.selectbox("Brand", ["iGuzzini", "Artemide", "Flos", "Viabizzuno"], key=f"br_{i}")
                        price = st.number_input("Prezzo Medio Apparecchio", 50, 5000, 150, key=f"pr_{i}")
                        if st.button(f"Genera Soluzione", key=f"gen_{i}"):
                            if ai_on and 'groq' in st.session_state:
                                res = st.session_state.groq.chat.completions.create(
                                    messages=[{"role":"user","content":f"Progetta {r['name']} per {lx} lux con {br}."}], model="llama-3.3-70b-versatile")
                                st.session_state.strat_results[i] = res.choices[0].message.content
                            else:
                                qty = int((lx * r['area']) / (2500 * 0.75)) + 1
                                st.session_state.strat_results[i] = f"**Calcolo Tecnico Offline:** Per {r['name']} servono {qty} sorgenti LED ({br})."
                            
                            # SALVATAGGIO DB
                            conn = sqlite3.connect('luxia_titan.db')
                            conn.execute("INSERT INTO rooms (project_id, r_name, area, brand, qty, lux_target, strategy, price) VALUES (?,?,?,?,?,?,?,?)",
                                         (st.session_state.current_id, r['name'], r['area'], br, 4, lx, st.session_state.strat_results[i], price))
                            conn.commit(); conn.close()
                    with c_out:
                        if i in st.session_state.strat_results: st.markdown(st.session_state.strat_results[i])

        with t3:
            st.subheader("Riepilogo Economico e Documentale")
            conn = sqlite3.connect('luxia_titan.db')
            df = pd.read_sql_query(f"SELECT r_name as Vano, area, brand, lux_target, price FROM rooms WHERE project_id={st.session_state.current_id}", conn)
            conn.close()
            if not df.empty:
                st.table(df)
                st.button("📦 Esporta Computo Metrico PDF")
                st.button("📉 Scarica Pacchetto IES/LDT")

if __name__ == "__main__":
    main()