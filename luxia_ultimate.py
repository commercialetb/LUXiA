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

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="LUXiA Ultimate Titan v10.3", layout="wide")

def init_session():
    defaults = {
        'logged_in': False, 'username': None, 'studio': None, 'logo': None,
        'rooms': [], 'current_id': None, 'current_name': None, 
        'img_b64': None, 'strat_results': {}, 'groq_enabled': False,
        'zoom_level': 2.0  # Zoom di default
    }
    for var, default in defaults.items():
        if var not in st.session_state: st.session_state[var] = default

init_session()

# --- 2. VISION & ZOOM ENGINE ---
def process_pdf_with_zoom(pdf_bytes, zoom_factor):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    # Applichiamo lo zoom scelto dall'utente alla renderizzazione
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
    img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    words = page.get_text("words")
    detected_rooms = []
    for cnt in contours:
        if cv2.contourArea(cnt) > (15000 * (zoom_factor/2)): # Adattiamo la soglia area allo zoom
            x, y, w, h = cv2.boundingRect(cnt)
            room_name = ""
            for word in words:
                # Ricalibrazione coordinate testo in base allo zoom
                if x < word[0]*zoom_factor < x+w and y < word[1]*zoom_factor < y+h:
                    if len(word[4]) > 2: room_name += word[4] + " "
            
            detected_rooms.append({
                "name": room_name.strip() or f"Vano {len(detected_rooms)+1}",
                "x": x, "y": y, "w": w, "h": h, 
                "area": round((w * h) / (600 * zoom_factor), 2)
            })
    
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), detected_rooms, pix.width, pix.height

# --- 3. DATABASE ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  area REAL, brand TEXT, qty INTEGER, strategy TEXT)''')
    conn.commit(); conn.close()

# --- 4. MAIN INTERFACE ---
def main():
    init_db()

    if not st.session_state.logged_in:
        st.title("🏛️ LUXiA Ultimate Titan v10.3")
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
        # CONTROLLO ZOOM
        st.subheader("🔍 Controlli Zoom")
        st.session_state.zoom_level = st.slider("Livello Ingrandimento", 1.0, 5.0, st.session_state.zoom_level, 0.5)
        st.caption("Aumenta lo zoom per vedere dettagli piccoli")
        
        st.divider()
        if st.button("Logout"): st.session_state.clear(); st.rerun()

    # WORKFLOW
    if not st.session_state.current_id:
        st.title("Gestione Progetti")
        with st.form("new"):
            pn = st.text_input("Nome Progetto")
            if st.form_submit_button("Crea"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, date) VALUES (?,?,?)", (st.session_state.username, pn, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.rerun()
    else:
        st.title(f"📂 {st.session_state.current_name}")
        t1, t2 = st.tabs(["📐 Vision & Zoom", "💡 Progettazione"])

        with t1:
            v1, v2 = st.columns([1, 3])
            with v1:
                pdf = st.file_uploader("Carica PDF", type=['pdf'])
                if pdf or st.button("Aggiorna Zoom"):
                    if pdf: st.session_state.pdf_cache = pdf.read()
                    if 'pdf_cache' in st.session_state:
                        b64, rooms, w, h = process_pdf_with_zoom(st.session_state.pdf_cache, st.session_state.zoom_level)
                        st.session_state.img_b64 = b64
                        st.session_state.rooms = rooms
                        st.session_state.canvas_w = w
                        st.session_state.canvas_h = h
                
                for i, r in enumerate(st.session_state.rooms):
                    st.session_state.rooms[i]['name'] = st.text_input(f"Vano {i}", r['name'], key=f"vn_{i}")

            with v2:
                if st.session_state.img_b64:
                    # BYPASS CSS CON SUPPORTO SCROLLBAR
                    st.markdown(f"""
                        <style>
                        .canvas-container {{
                            overflow: auto;
                            max-height: 800px;
                            border: 2px solid #444;
                        }}
                        div[data-testid="stCanvas"] {{
                            background-image: url("data:image/png;base64,{st.session_state.img_b64}");
                            background-size: contain;
                            background-repeat: no-repeat;
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                    
                    rects = [{"type":"rect","left":r['x'],"top":r['y'],"width":r['w'],"height":r['h'],
                              "fill":"rgba(0, 255, 0, 0.1)","stroke":"green"} for r in st.session_state.rooms]
                    
                    # Il Canvas si adatta alle dimensioni dell'immagine zoomata
                    st_canvas(
                        initial_drawing={"objects": rects},
                        drawing_mode="rect",
                        background_color="rgba(0,0,0,0)",
                        height=st.session_state.canvas_h, 
                        width=st.session_state.canvas_w, 
                        key="zoom_canvas"
                    )

        with t2:
            for i, r in enumerate(st.session_state.rooms):
                with st.expander(f"⚙️ {r['name']}"):
                    st.write(f"Area stimata allo zoom attuale: {r['area']} mq")
                    if st.button(f"Calcola", key=f"c_{i}"):
                        st.success(f"Analisi per {r['name']} completata.")

if __name__ == "__main__":
    main()