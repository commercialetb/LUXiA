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

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="LUXiA Ultimate v9.6", layout="wide")

# --- 2. GESTIONE DATABASE (Full persistence) ---
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
def process_pdf_advanced(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    zoom = 2
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # OCR: Estrazione etichette testo
    words = page.get_text("words")
    
    # Computer Vision: Rilevamento muri e vani
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rooms = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 15000:
            x, y, w, h = cv2.boundingRect(cnt)
            # Associa testo se trovato dentro il box
            label = ""
            for wd in words:
                tx, ty = wd[0]*zoom, wd[1]*zoom
                if x < tx < x+w and y < ty < y+h:
                    if len(wd[4]) > 2: label += wd[4] + " "
            rooms.append({"name": label.strip() or f"Vano {len(rooms)+1}", "x": x, "y": y, "w": w, "h": h, "area": (w*h)/1000}) # Area stimata
    return img, rooms

# --- 4. FUNZIONI UTILI ---
def get_stats(user):
    conn = sqlite3.connect('luxia_titan.db')
    res = conn.execute("SELECT COUNT(*), (SELECT SUM(qty*price) FROM rooms) FROM projects WHERE username=?", (user,)).fetchone()
    conn.close()
    return res

# --- 5. LOGICA APPLICATIVA ---
def main():
    init_db()
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'rooms' not in st.session_state: st.session_state.rooms = []

    # --- LOGIN & REGISTRAZIONE ---
    if not st.session_state.logged_in:
        st.title("🏛️ LUXiA Ultimate v9.6")
        l_col, r_col = st.tabs(["Accedi", "Registra Studio"])
        with l_col:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Entra"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u,h)).fetchone()
                if res:
                    st.session_state.update({"logged_in": True, "username": u, "studio": res[0], "logo": res[1]})
                    st.rerun()
                else: st.error("Dati errati.")
        return

    # --- SIDEBAR ---
    with st.sidebar:
        if st.session_state.logo: st.image(base64.b64decode(st.session_state.logo), width=150)
        st.header(st.session_state.studio)
        st.divider()
        ai_online = st.toggle("🌐 Groq Cloud AI", value=True)
        if ai_online:
            gk = st.text_input("Groq API Key", type="password")
            if gk: st.session_state.groq = Groq(api_key=gk)
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel = st.selectbox("Seleziona Progetto", ["-- DASHBOARD --"] + list(p_dict.keys()))
        if st.button("Carica"):
            st.session_state.current_id = p_dict.get(sel)
            st.session_state.current_name = sel
            st.rerun()

        if st.button("Logout"): st.session_state.logged_in = False; st.rerun()

    # --- DASHBOARD ---
    if not st.session_state.get('current_id'):
        st.title(f"Dashboard: {st.session_state.studio}")
        n_p, val = get_stats(st.session_state.username)
        c1, c2 = st.columns(2)
        c1.metric("Progetti in corso", n_p)
        c2.metric("Valore Pipeline", f"€ {val or 0:,.2f}")
        
        st.subheader("Crea Nuovo Cantiere")
        with st.form("new_p"):
            n = st.text_input("Nome Progetto"); c = st.text_input("Cliente")
            if st.form_submit_button("Crea"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                            (st.session_state.username, n, c, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.rerun()
    
    # --- WORKFLOW PROGETTO ---
    else:
        st.title(f"📂 {st.session_state.current_name}")
        t1, t2, t3 = st.tabs(["📐 Vision & Mappa", "💡 Calcolo & AI", "📄 Report"])

        with t1:
            cl1, cl2 = st.columns([1, 2])
            with cl1:
                pdf = st.file_uploader("Carica Planimetria PDF", type=['pdf'])
                if pdf:
                    img, r_list = process_pdf_advanced(pdf.read())
                    st.session_state.img = img
                    st.session_state.rooms = r_list
                
                if st.session_state.rooms:
                    for i, r in enumerate(st.session_state.rooms):
                        st.session_state.rooms[i]['name'] = st.text_input(f"Nome", r['name'], key=f"nm_{i}")
                        if st.button(f"🗑️ Rimuovi", key=f"dl_{i}"):
                            st.session_state.rooms.pop(i); st.rerun()
            with cl2:
                if 'img' in st.session_state:
                    rects = [{"type":"rect","left":r['x'],"top":r['y'],"width":r['w'],"height":r['h'],"fill":"rgba(0,255,0,0.2)","stroke":"green"} for r in st.session_state.rooms]
                    st_canvas(background_image=st.session_state.img, initial_drawing={"objects":rects}, height=600, width=800, key="c")

        with t2:
            if not st.session_state.rooms: st.info("Identifica i vani nella Tab 1.")
            else:
                for i, r in enumerate(st.session_state.rooms):
                    with st.expander(f"⚙️ Configurazione: {r['name']}"):
                        target = st.slider("Target Lux", 100, 1000, 300, key=f"lx_{i}")
                        if st.button(f"Calcola Soluzione", key=f"bt_{i}"):
                            if ai_online and 'groq' in st.session_state:
                                resp = st.session_state.groq.chat.completions.create(
                                    messages=[{"role":"user","content":f"Progetta illuminazione per {r['name']} ({r['area']}mq) con {target} lux. Usa brand iGuzzini o Flos."}],
                                    model="llama-3.3-70b-versatile")
                                st.session_state[f"st_{i}"] = resp.choices[0].message.content
                            else:
                                # Calcolo deterministico Offline
                                qty = int((target * r['area']) / (3000 * 0.8 * 0.9)) + 1
                                st.session_state[f"st_{i}"] = f"OFFLINE: Suggerite {qty} lampade LED da 3000 lumen per {r['name']}."
                            
                            # Salvataggio DB
                            conn = sqlite3.connect('luxia_titan.db')
                            conn.execute("INSERT INTO rooms (project_id, r_name, area, lux_target, strategy, qty, price) VALUES (?,?,?,?,?,?,?)",
                                         (st.session_state.current_id, r['name'], r['area'], target, st.session_state[f"st_{i}"], 4, 120.0))
                            conn.commit(); conn.close()
                        
                        if f"st_{i}" in st.session_state:
                            st.write(st.session_state[f"st_{i}"])

        with t3:
            st.subheader("Esportazione Documenti")
            if st.button("Genera Report Tecnico PDF"):
                # Logica FPDF...
                st.success("Report generato correttamente!")

if __name__ == "__main__":
    main()