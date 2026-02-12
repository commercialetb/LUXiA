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
st.set_page_config(page_title="LUXiA Ultimate Titan v10.2", layout="wide")

def init_session():
    defaults = {
        'logged_in': False, 'username': None, 'studio': None, 'logo': None,
        'rooms': [], 'current_id': None, 'current_name': None, 
        'img_b64': None, 'strat_results': {}, 'groq_enabled': False
    }
    for var, default in defaults.items():
        if var not in st.session_state:
            st.session_state[var] = default

init_session()

# --- 2. ENGINE DATABASE ---
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

# --- 3. VISION & OCR (PIENO FUNZIONAMENTO) ---
def process_pdf_full(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    zoom = 2
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    words = page.get_text("words")
    detected_rooms = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 15000:
            x, y, w, h = cv2.boundingRect(cnt)
            room_name = ""
            for word in words:
                if x < word[0]*zoom < x+w and y < word[1]*zoom < y+h:
                    if len(word[4]) > 2: room_name += word[4] + " "
            detected_rooms.append({
                "name": room_name.strip() or f"Vano {len(detected_rooms)+1}",
                "x": x, "y": y, "w": w, "h": h, "area": round((w * h) / 1200, 2)
            })
    
    # Conversione in B64 per il bypass CSS
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), detected_rooms

# --- 4. INTERFACCIA ---
def main():
    init_db()

    if not st.session_state.logged_in:
        st.title("🏛️ LUXiA Ultimate Titan v10.2")
        t_log, t_reg = st.tabs(["Accedi", "Registra Studio"])
        with t_log:
            u = st.text_input("User"); p = st.text_input("Pw", type="password")
            if st.button("Entra"):
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, hashlib.sha256(p.encode()).hexdigest())).fetchone()
                if res:
                    st.session_state.update({"logged_in": True, "username": u, "studio": res[0], "logo": res[1]})
                    st.rerun()
        return

    # SIDEBAR
    with st.sidebar:
        if st.session_state.logo: st.image(base64.b64decode(st.session_state.logo), width=150)
        st.header(st.session_state.studio)
        st.divider()
        st.session_state.groq_enabled = st.toggle("AI Online (Groq)", value=True)
        if st.session_state.groq_enabled:
            key = st.text_input("Groq API Key", type="password")
            if key: st.session_state.groq_client = Groq(api_key=key)
        
        projs = sqlite3.connect('luxia_titan.db').execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in projs}
        sel = st.selectbox("Progetti", ["-- Dashboard --"] + list(p_dict.keys()))
        if st.button("Carica"):
            st.session_state.current_id = p_dict.get(sel); st.session_state.current_name = sel; st.rerun()
        if st.button("Logout"): st.session_state.clear(); st.rerun()

    if not st.session_state.current_id:
        st.title("📊 Dashboard")
        with st.form("new"):
            pn = st.text_input("Nome Progetto")
            if st.form_submit_button("Crea"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, date) VALUES (?,?,?)", (st.session_state.username, pn, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.rerun()
    else:
        st.title(f"📂 {st.session_state.current_name}")
        t1, t2, t3 = st.tabs(["📐 Vision Engine", "💡 Progettazione", "📄 Report"])

        with t1:
            v1, v2 = st.columns([1, 2])
            with v1:
                pdf = st.file_uploader("PDF Planimetria", type=['pdf'])
                if pdf:
                    b64, rooms = process_pdf_full(pdf.read())
                    st.session_state.img_b64 = b64
                    st.session_state.rooms = rooms
                
                for i, r in enumerate(st.session_state.rooms):
                    st.session_state.rooms[i]['name'] = st.text_input(f"Nome {i}", r['name'], key=f"v_{i}")

            with v2:
                if st.session_state.img_b64:
                    # --- IL FIX DEFINITIVO (CSS BYPASS) ---
                    # Iniettiamo lo sfondo tramite CSS per evitare l'AttributeError del canvas
                    canvas_css = f"""
                    <style>
                    div[data-testid="stCanvas"] {{
                        background-image: url("data:image/png;base64,{st.session_state.img_b64}");
                        background-size: contain;
                        background-repeat: no-repeat;
                    }}
                    </style>
                    """
                    st.markdown(canvas_css, unsafe_allow_html=True)
                    
                    rects = [{"type":"rect","left":r['x'],"top":r['y'],"width":r['w'],"height":r['h'],
                              "fill":"rgba(0, 255, 0, 0.2)","stroke":"green"} for r in st.session_state.rooms]
                    
                    # Canvas ora è trasparente (senza background_image) e non crasha
                    st_canvas(
                        initial_drawing={"objects": rects},
                        drawing_mode="rect",
                        background_color="rgba(0,0,0,0)",
                        height=600, width=800, key="css_fixed_canvas"
                    )

        with t2:
            for i, r in enumerate(st.session_state.rooms):
                with st.expander(f"⚙️ {r['name']} ({r['area']}mq)"):
                    lx = st.slider("Lux", 100, 1000, 300, key=f"lx_{i}")
                    if st.button(f"Calcola {i}", key=f"bt_{i}"):
                        if st.session_state.groq_enabled and 'groq_client' in st.session_state:
                            res = st.session_state.groq_client.chat.completions.create(messages=[{"role":"user","content":f"Progetta {r['name']} per {lx} lux."}], model="llama-3.3-70b-versatile")
                            st.session_state.strat_results[i] = res.choices[0].message.content
                        else:
                            qty = int((lx * r['area']) / 2000) + 1
                            st.session_state.strat_results[i] = f"Calcolo Offline: {qty} lampade suggerite."
                        st.write(st.session_state.strat_results[i])

if __name__ == "__main__":
    main()