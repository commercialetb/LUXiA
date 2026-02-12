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

# --- [BLOCCO 1: CONFIGURAZIONE & SESSIONE - FRIZZATO] ---
st.set_page_config(page_title="LUXiA Ultimate Titan v10.5", layout="wide")

def init_session():
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'rooms' not in st.session_state: st.session_state.rooms = []
    if 'img_b64' not in st.session_state: st.session_state.img_b64 = None
    if 'current_id' not in st.session_state: st.session_state.current_id = None
    if 'zoom_level' not in st.session_state: st.session_state.zoom_level = 2.0
    if 'strat_results' not in st.session_state: st.session_state.strat_results = {}

init_session()

# --- [BLOCCO 2: DATABASE ENGINE - FRIZZATO] ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  area REAL, brand TEXT, qty INTEGER, lux_target INTEGER, strategy TEXT, price REAL)''')
    conn.commit(); conn.close()

# --- [BLOCCO 3: VISION & OCR - FRIZZATO & ZOOM-FRIENDLY] ---
def process_pdf_core(pdf_bytes, zoom_factor):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    # Lo zoom influenza la risoluzione della pixmap
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
            detected.append({"name": label.strip() or f"Vano {len(detected)+1}", "x": x, "y": y, "w": w, "h": h, "area": round((w*h)/(800*zoom_factor), 2)})
    
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), detected, pix.width, pix.height

# --- [BLOCCO 4: LOGICA APPLICATIVA - STABILE] ---
def main():
    init_db()

    if not st.session_state.logged_in:
        st.title("🏛️ LUXiA Ultimate Titan v10.5")
        t1, t2 = st.tabs(["Accedi", "Registra Studio"])
        with t1:
            u = st.text_input("Username"); p = st.text_input("Password", type="password")
            if st.button("Login"):
                conn = sqlite3.connect('luxia_titan.db')
                r = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, hashlib.sha256(p.encode()).hexdigest())).fetchone()
                if r: 
                    st.session_state.update({"logged_in": True, "username": u, "studio": r[0], "logo": r[1]})
                    st.rerun()
        return

    # SIDEBAR
    with st.sidebar:
        if st.session_state.logo: st.image(base64.b64decode(st.session_state.logo), width=150)
        st.header(st.session_state.studio)
        st.divider()
        st.session_state.zoom_level = st.slider("🔍 Zoom Planimetria", 1.0, 4.0, st.session_state.zoom_level, 0.5)
        st.divider()
        ai_on = st.toggle("AI Online (Groq)", value=True)
        if ai_on:
            gk = st.text_input("Groq Key", type="password")
            if gk: st.session_state.groq = Groq(api_key=gk)
        if st.button("Logout"): st.session_state.clear(); st.rerun()

    # DASHBOARD O PROGETTO
    if not st.session_state.current_id:
        st.title("📊 Dashboard Studio")
        with st.form("new_p"):
            name = st.text_input("Nome Nuovo Progetto")
            if st.form_submit_button("Crea"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, date) VALUES (?,?,?)", (st.session_state.username, name, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.session_state.current_id = cur.lastrowid; st.session_state.current_name = name; conn.close(); st.rerun()
    else:
        st.title(f"📂 Progetto: {st.session_state.current_name}")
        tab1, tab2, tab3 = st.tabs(["📐 Vision & Zoom", "💡 Calcolo Lux", "📄 Report"])

        with tab1:
            col1, col2 = st.columns([1, 3])
            with col1:
                pdf = st.file_uploader("Carica Planimetria PDF", type=['pdf'])
                if pdf or st.button("Applica Zoom"):
                    if pdf: st.session_state.pdf_cache = pdf.read()
                    if 'pdf_cache' in st.session_state:
                        b64, rooms, w, h = process_pdf_core(st.session_state.pdf_cache, st.session_state.zoom_level)
                        st.session_state.update({"img_b64": b64, "rooms": rooms, "canv_w": w, "canv_h": h})
                
                for i, r in enumerate(st.session_state.rooms):
                    st.session_state.rooms[i]['name'] = st.text_input(f"Nome Vano {i}", r['name'], key=f"v_{i}")

            with col2:
                if st.session_state.img_b64:
                    # --- FIX INFALLIBILE: CSS PROXY ---
                    st.markdown(f"""
                        <style>
                        .stCanvasContainer {{ overflow: auto; max-height: 800px; border: 1px solid #444; }}
                        div[data-testid="stCanvas"] {{
                            background-image: url("data:image/png;base64,{st.session_state.img_b64}");
                            background-size: contain; background-repeat: no-repeat;
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                    
                    rects = [{"type":"rect","left":r['x'],"top":r['y'],"width":r['w'],"height":r['h'],
                              "fill":"rgba(0, 255, 0, 0.15)","stroke":"green"} for r in st.session_state.rooms]
                    
                    # Il Canvas ora è trasparente e non crasha mai
                    st_canvas(initial_drawing={"objects": rects}, drawing_mode="rect",
                              background_color="rgba(0,0,0,0)", height=st.session_state.canv_h, 
                              width=st.session_state.canv_w, key="titan_canvas_fixed")

        with tab2:
            # Calcolo Lux e Strategie (Blocco Frizzato)
            for i, r in enumerate(st.session_state.rooms):
                with st.expander(f"⚙️ Configurazione {r['name']} ({r['area']} mq)"):
                    lx = st.slider("Target Lux", 100, 1000, 300, key=f"lx_{i}")
                    if st.button(f"Genera per {r['name']}", key=f"g_{i}"):
                        if ai_on and 'groq' in st.session_state:
                            res = st.session_state.groq.chat.completions.create(messages=[{"role":"user","content":f"Progetta {r['name']} per {lx} lux."}], model="llama-3.3-70b-versatile")
                            st.session_state.strat_results[i] = res.choices[0].message.content
                        else:
                            qty = int((lx * r['area']) / (2500 * 0.7)) + 1
                            st.session_state.strat_results[i] = f"**Calcolo Tecnico:** Suggerite {qty} lampade da 2500lm."
                        
                        # Salvataggio DB
                        conn = sqlite3.connect('luxia_titan.db')
                        conn.execute("INSERT INTO rooms (project_id, r_name, area, lux_target, strategy, qty, price) VALUES (?,?,?,?,?,?,?)",
                                     (st.session_state.current_id, r['name'], r['area'], lx, st.session_state.strat_results[i], 4, 150.0))
                        conn.commit(); conn.close()
                    
                    if i in st.session_state.strat_results:
                        st.markdown(st.session_state.strat_results[i])

        with tab3:
            st.button("📦 Genera Computo e Report Finale")

if __name__ == "__main__":
    main()