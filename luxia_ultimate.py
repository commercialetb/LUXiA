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
from fpdf import FPDF
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# --- 1. INITIALIZATION ---
def init_session():
    """Previene AttributeError inizializzando le chiavi necessarie"""
    defaults = {
        'logged_in': False,
        'username': None,
        'studio': None,
        'logo': None,
        'rooms': [],
        'current_id': None,
        'current_name': None,
        'img': None,
        'groq_online': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

st.set_page_config(page_title="LUXiA Ultimate v9.7", layout="wide")
init_session()

# --- 2. DATABASE ENGINE ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  area REAL, brand TEXT, model TEXT, qty INTEGER, lux_target INTEGER, 
                  strategy TEXT, price REAL)''')
    conn.commit()
    conn.close()

# --- 3. VISION & OCR ENGINE ---
def process_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    zoom = 2
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    words = page.get_text("words")
    
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rooms = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 15000:
            x, y, w, h = cv2.boundingRect(cnt)
            label = ""
            for wd in words:
                tx, ty = wd[0]*zoom, wd[1]*zoom
                if x < tx < x+w and y < ty < y+h:
                    if len(wd[4]) > 2: label += wd[4] + " "
            rooms.append({"name": label.strip() or f"Vano {len(rooms)+1}", "x": x, "y": y, "w": w, "h": h, "area": (w*h)/1000})
    return img, rooms

# --- 4. MAIN INTERFACE ---
def main():
    init_db()

    # --- LOGIN PANEL ---
    if not st.session_state.logged_in:
        st.title("🏛️ LUXiA Ultimate v9.7")
        col_a, col_b = st.tabs(["Accedi", "Registra Studio"])
        
        with col_a:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Accedi"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u,h)).fetchone()
                conn.close()
                if res:
                    st.session_state.update({"logged_in": True, "username": u, "studio": res[0], "logo": res[1]})
                    st.rerun()
                else: st.error("Credenziali non valide.")
        
        with col_b:
            nu = st.text_input("Nuovo Utente")
            np = st.text_input("Nuova Password", type="password")
            ns = st.text_input("Nome Studio Tecnico")
            nl = st.file_uploader("Upload Logo Studio", type=['png', 'jpg'])
            if st.button("Registra"):
                l_b64 = base64.b64encode(nl.read()).decode() if nl else None
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, hashlib.sha256(np.encode()).hexdigest(), ns, l_b64))
                conn.commit(); conn.close(); st.success("Registrazione completata.")
        return

    # --- SIDEBAR ---
    with st.sidebar:
        if st.session_state.logo:
            st.image(base64.b64decode(st.session_state.logo), width=150)
        st.header(st.session_state.studio)
        st.divider()
        
        mode = st.radio("AI Engine", ["Online (Groq)", "Offline (Calcolo Tecnico)"])
        st.session_state.groq_online = (mode == "Online (Groq)")
        if st.session_state.groq_online:
            gk = st.text_input("Groq API Key", type="password")
            if gk: st.session_state.groq_client = Groq(api_key=gk)
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel = st.selectbox("I tuoi Progetti", ["-- Dashboard --"] + list(p_dict.keys()))
        if st.button("Apri Progetto"):
            st.session_state.current_id = p_dict.get(sel)
            st.session_state.current_name = sel
            st.rerun()

        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    # --- WORKFLOW ---
    if not st.session_state.current_id:
        st.title(f"Benvenuto, {st.session_state.username}")
        # Dashboard Stats
        conn = sqlite3.connect('luxia_titan.db')
        stats = conn.execute("SELECT COUNT(*), (SELECT SUM(qty*price) FROM rooms) FROM projects WHERE username=?", (st.session_state.username,)).fetchone()
        conn.close()
        c1, c2 = st.columns(2)
        c1.metric("Progetti Totali", stats[0]); c2.metric("Pipeline Valore", f"€ {stats[1] or 0:,.2f}")
        
        with st.form("new_p"):
            st.subheader("Crea Nuovo Progetto")
            pn = st.text_input("Nome Cantiere"); cl = st.text_input("Cliente")
            if st.form_submit_button("Crea"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", (st.session_state.username, pn, cl, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); conn.close(); st.rerun()
    else:
        st.title(f"📂 {st.session_state.current_name}")
        t1, t2, t3 = st.tabs(["📐 Vision & Mappa", "💡 Calcolo & Strategia", "📄 Report"])

        with t1:
            l, r = st.columns([1, 2])
            with l:
                f = st.file_uploader("Upload Planimetria PDF", type=['pdf'])
                if f:
                    img, rooms = process_pdf(f.read())
                    st.session_state.img = img
                    st.session_state.rooms = rooms
                
                if st.session_state.rooms:
                    for i, rm in enumerate(st.session_state.rooms):
                        st.session_state.rooms[i]['name'] = st.text_input(f"Vano {i}", rm['name'], key=f"n_{i}")
                        if st.button(f"🗑️ Elimina", key=f"d_{i}"):
                            st.session_state.rooms.pop(i); st.rerun()
            with r:
                if st.session_state.img:
                    rects = [{"type":"rect","left":rm['x'],"top":rm['y'],"width":rm['w'],"height":rm['h'],"fill":"rgba(0,255,0,0.2)","stroke":"green"} for rm in st.session_state.rooms]
                    st_canvas(background_image=st.session_state.img, initial_drawing={"objects":rects}, height=600, width=800, key="cvs")

        with t2:
            if not st.session_state.rooms: st.info("Esegui prima la scansione nella Tab Vision.")
            else:
                for i, rm in enumerate(st.session_state.rooms):
                    with st.expander(f"📍 {rm['name']}"):
                        target = st.slider("Target Lux", 100, 1000, 300, key=f"lx_{i}")
                        if st.button(f"Genera Soluzione", key=f"gen_{i}"):
                            if st.session_state.groq_online and 'groq_client' in st.session_state:
                                res = st.session_state.groq_client.chat.completions.create(
                                    messages=[{"role":"user","content":f"Progetta {rm['name']} ({rm['area']}mq) per {target} lux. Brand: iGuzzini."}],
                                    model="llama-3.3-70b-versatile")
                                strat = res.choices[0].message.content
                            else:
                                qty = int((target * rm['area']) / (2500 * 0.8 * 0.9)) + 1
                                strat = f"CALCOLO OFFLINE: Suggerite {qty} sorgenti LED da 2500lm per {rm['name']}."
                            
                            st.session_state[f"res_{i}"] = strat
                        if f"res_{i}" in st.session_state:
                            st.markdown(st.session_state[f"res_{i}"])

        with t3:
            if st.button("Esporta Report Professionale"):
                st.success("Logica FPDF pronta per la generazione del file.")

if __name__ == "__main__":
    main()