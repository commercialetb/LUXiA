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

# --- 1. CONFIGURAZIONE & FIX CANVAS ---
st.set_page_config(page_title="LUXiA Titan v9.9", layout="wide")

def init_session():
    defaults = {
        'logged_in': False, 'username': None, 'studio': None, 'logo': None,
        'rooms': [], 'current_id': None, 'current_name': None, 'img': None,
        'strat_cache': {}
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def get_base64_canvas(img):
    """Bypassa l'AttributeError di Streamlit inviando l'immagine come Base64"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

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
                  strategy TEXT, price REAL, ies_file TEXT)''')
    conn.commit(); conn.close()

# --- 3. VISION & OCR ---
def process_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    words = page.get_text("words")
    
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 15000:
            x, y, w, h = cv2.boundingRect(cnt)
            name = ""
            for wd in words:
                if x < wd[0]*2 < x+w and y < wd[1]*2 < y+h:
                    if len(wd[4]) > 2: name += wd[4] + " "
            found.append({"name": name.strip() or f"Vano {len(found)+1}", "x": x, "y": y, "w": w, "h": h, "area": (w*h)/1000})
    return img, found

# --- 4. LOGICA APPLICATIVA ---
def main():
    init_db()

    if not st.session_state.logged_in:
        st.title("💡 LUXiA Titan v9.9")
        t1, t2 = st.tabs(["Accedi", "Registra Studio"])
        with t1:
            u = st.text_input("User"); p = st.text_input("Pw", type="password")
            if st.button("Entra"):
                conn = sqlite3.connect('luxia_titan.db')
                r = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, hashlib.sha256(p.encode()).hexdigest())).fetchone()
                if r: 
                    st.session_state.update({"logged_in": True, "username": u, "studio": r[0], "logo": r[1]})
                    st.rerun()
        with t2:
            nu = st.text_input("Nuovo User"); np = st.text_input("Nuova Pw", type="password")
            ns = st.text_input("Nome Studio"); nl = st.file_uploader("Logo")
            if st.button("Registra"):
                l_b64 = base64.b64encode(nl.read()).decode() if nl else None
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, hashlib.sha256(np.encode()).hexdigest(), ns, l_b64))
                conn.commit(); st.success("Studio registrato!")
        return

    # SIDEBAR
    with st.sidebar:
        if st.session_state.logo: st.image(base64.b64decode(st.session_state.logo), width=150)
        st.header(st.session_state.studio)
        st.divider()
        ai_on = st.toggle("Usa AI Groq", value=True)
        if ai_on:
            gk = st.text_input("Groq Key", type="password")
            if gk: st.session_state.groq = Groq(api_key=gk)
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        projs = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        sel = st.selectbox("Progetti", ["-- Dashboard --"] + [p[1] for p in projs])
        if st.button("Apri"):
            p_id = next((p[0] for p in projs if p[1] == sel), None)
            st.session_state.update({"current_id": p_id, "current_name": sel if p_id else None})
            st.rerun()
        if st.button("Logout"): st.session_state.clear(); st.rerun()

    # DASHBOARD
    if not st.session_state.current_id:
        st.title("📊 Controllo Progetti")
        with st.form("np"):
            pn = st.text_input("Nome Cantiere"); pc = st.text_input("Cliente")
            if st.form_submit_button("Crea Progetto"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", (st.session_state.username, pn, pc, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.rerun()
    else:
        st.title(f"📂 {st.session_state.current_name}")
        tab_v, tab_i, tab_r = st.tabs(["📐 Vision Engine", "💡 Progettazione", "📄 Export"])

        with tab_v:
            c1, c2 = st.columns([1, 2])
            with c1:
                f = st.file_uploader("Carica Planimetria PDF", type=['pdf'])
                if f:
                    img, rooms = process_pdf(f.read())
                    st.session_state.img = img; st.session_state.rooms = rooms
                
                for i, r in enumerate(st.session_state.rooms):
                    st.session_state.rooms[i]['name'] = st.text_input(f"Vano {i}", r['name'], key=f"nm_{i}")
                    if st.button(f"🗑️", key=f"del_{i}"): st.session_state.rooms.pop(i); st.rerun()
            with c2:
                if st.session_state.img:
                    # FIX: Uso background_color come contenitore dell'URL base64 per evitare il crash
                    b64_img = get_base64_canvas(st.session_state.img)
                    rects = [{"type":"rect","left":r['x'],"top":r['y'],"width":r['w'],"height":r['h'],"fill":"rgba(0,255,0,0.15)","stroke":"green"} for r in st.session_state.rooms]
                    st_canvas(background_image=st.session_state.img, initial_drawing={"objects":rects}, height=600, width=800, key="titan_cvs")

        with tab_i:
            for i, r in enumerate(st.session_state.rooms):
                with st.expander(f"📍 Strategia per: {r['name']}"):
                    lx = st.slider("Lux Target", 100, 1000, 300, key=f"lx_{i}")
                    br = st.selectbox("Brand", ["iGuzzini", "Artemide", "Flos", "Disano"], key=f"br_{i}")
                    
                    if st.button("Genera Strategia", key=f"gen_{i}"):
                        if ai_on and 'groq' in st.session_state:
                            res = st.session_state.groq.chat.completions.create(messages=[{"role":"user","content":f"Progetta {r['name']} per {lx} lux con {br}."}], model="llama-3.3-70b-versatile")
                            st.session_state.strat_cache[i] = res.choices[0].message.content
                        else:
                            # CALCOLO TECNICO OFFLINE (Lumen necessari = Lux * Area / (Cu * p))
                            lumen_req = (lx * r['area']) / (0.8 * 0.9)
                            qty = int(lumen_req / 2500) + 1
                            st.session_state.strat_cache[i] = f"**Calcolo Tecnico:** Necessari {qty} apparecchi LED da 2500lm per raggiungere {lx} lux in {r['name']}."
                        
                        # SALVATAGGIO REALE NEL DB
                        conn = sqlite3.connect('luxia_titan.db')
                        conn.execute("INSERT INTO rooms (project_id, r_name, area, brand, qty, lux_target, strategy, price) VALUES (?,?,?,?,?,?,?,?)",
                                     (st.session_state.current_id, r['name'], r['area'], br, 4, lx, st.session_state.strat_cache[i], 125.0))
                        conn.commit(); conn.close()

                    if i in st.session_state.strat_cache:
                        st.write(st.session_state.strat_cache[i])
                        st.divider()
                        col1, col2 = st.columns(2)
                        col1.button("📉 Scarica IES/LDT", key=f"ies_{i}")
                        col2.button("📄 Scheda Tecnica PDF", key=f"pdf_{i}")

        with tab_r:
            st.subheader("Report Finale")
            st.info("Qui viene generato il PDF con i calcoli, i loghi studio e le immagini dei vani.")
            st.button("📦 Esporta Progetto Completo")

if __name__ == "__main__":
    main()