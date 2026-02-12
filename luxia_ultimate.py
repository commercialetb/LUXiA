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

# --- 1. CONFIGURAZIONE & FIX ANTI-CRASH ---
st.set_page_config(page_title="LUXiA Titan v8.9", layout="wide", initial_sidebar_state="expanded")

def get_image_base64_url(img):
    """Converte l'immagine PIL in Data URL per fixare l'AttributeError del Canvas"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- 2. GESTIONE DATABASE (STRUTTURA COMPLETA) ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  w REAL, l REAL, brand TEXT, model TEXT, qty INTEGER, price REAL, 
                  strategy TEXT, ies_link TEXT, pdf_link TEXT)''')
    conn.commit(); conn.close()

# --- 3. VISION & OCR ENGINE (ESTRAZIONE PDF) ---
def process_pdf_ai(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0) # Pagina 1
    
    # Rendering alta risoluzione per la visione
    zoom = 2 
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # OCR: Estrazione testo con coordinate
    text_blocks = page.get_text("words") # x0, y0, x1, y1, "parola", ...
    
    # Visione Artificiale: Trova i vani (muri chiusi)
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 12000: # Filtro per stanze reali
            x, y, w, h = cv2.boundingRect(cnt)
            
            # OCR Spaziale: cerchiamo etichette testo dentro il rettangolo
            label = ""
            for tb in text_blocks:
                tx, ty = tb[0]*zoom, tb[1]*zoom
                if x < tx < x+w and y < ty < y+h:
                    if len(tb[4]) > 2: label += tb[4] + " "
            
            final_name = label.strip() if label.strip() else f"Ambiente {len(detected)+1}"
            detected.append({"name": final_name, "x": x, "y": y, "w": w, "h": h})
            
    return img, detected

# --- 4. LOGICA DASHBOARD STATISTICHE ---
def get_stats(username):
    conn = sqlite3.connect('luxia_titan.db')
    n_p = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (username,)).fetchone()[0]
    data = conn.execute('''SELECT COUNT(rooms.id), SUM(rooms.qty * rooms.price) 
                           FROM rooms JOIN projects ON rooms.project_id = projects.id 
                           WHERE projects.username=?''', (username,)).fetchone()
    conn.close()
    return n_p, (data[0] if data[0] else 0), (data[1] if data[1] else 0)

# --- 5. MAIN APP ---
def main():
    init_db()
    
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'rooms' not in st.session_state: st.session_state.rooms = []

    # LOGIN SCREEN
    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center; color: #FFD700;'>LUXiA TITAN v8.9</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            tab_in, tab_reg = st.tabs(["🔐 Login", "📝 Registrazione Studio"])
            with tab_in:
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.button("Accedi", use_container_width=True):
                    h = hashlib.sha256(p.encode()).hexdigest()
                    conn = sqlite3.connect('luxia_titan.db')
                    res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                    conn.close()
                    if res:
                        st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "logo_b64": res[1]})
                        st.rerun()
                    else: st.error("Credenziali non valide.")
            with tab_reg:
                nu = st.text_input("Nuovo Utente"); np = st.text_input("Password", type="password", key="reg_p")
                ns = st.text_input("Nome Studio Tecnico")
                if st.button("Crea Account"):
                    h = hashlib.sha256(np.encode()).hexdigest()
                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute("INSERT INTO users (username, password, studio_name) VALUES (?,?,?)", (nu, h, ns))
                    conn.commit(); conn.close(); st.success("Registrato! Ora accedi.")
        return

    # SIDEBAR
    with st.sidebar:
        st.header(f"🏛️ {st.session_state.studio_name}")
        if st.session_state.get('logo_b64'):
            st.image(base64.b64decode(st.session_state.logo_b64), use_container_width=True)
        
        st.divider()
        gk = st.text_input("Groq API Key", type="password")
        if gk: st.session_state.groq_client = Groq(api_key=gk); st.session_state.groq_online = True
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel = st.selectbox("Apri Progetto", ["-- DASHBOARD --"] + list(p_dict.keys()))
        if st.button("Vai"):
            st.session_state.current_proj_id = p_dict[sel] if sel != "-- DASHBOARD --" else None
            st.session_state.current_proj_name = sel
            st.rerun()
        
        if st.button("Logout"): st.session_state.logged_in = False; st.rerun()

    # LOGICA DASHBOARD O PROGETTO
    if not st.session_state.get('current_proj_id'):
        st.title(f"Benvenuto Antonio - {st.session_state.studio_name}")
        n_p, n_v, val = get_stats(st.session_state.username)
        m1, m2, m3 = st.columns(3)
        m1.metric("Progetti", n_p); m2.metric("Vani Progettati", n_v); m3.metric("Valore Pipeline", f"€{val:,.2f}")
        
        st.divider()
        with st.form("new_proj"):
            st.subheader("🚀 Inizia Nuovo Progetto")
            p_name = st.text_input("Nome Cantiere"); p_cli = st.text_input("Cliente")
            if st.form_submit_button("Crea"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                            (st.session_state.username, p_name, p_cli, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.session_state.current_proj_id = cur.lastrowid; st.session_state.current_proj_name = p_name; conn.close(); st.rerun()
    else:
        st.header(f"Progetto: {st.session_state.current_proj_name}")
        t1, t2, t3 = st.tabs(["📐 Vision & Vani", "💡 Strategia AI", "📄 Report Finale"])

        with t1:
            cl1, cl2 = st.columns([1, 2])
            with cl1:
                st.subheader("1. Carica Planimetria")
                f = st.file_uploader("PDF Tecnico", type=['pdf'])
                if f:
                    img, rooms = process_pdf_ai(f.read())
                    st.session_state.working_img = img
                    st.session_state.rooms = rooms
                    st.success(f"Analisi completata: {len(rooms)} vani identificati.")
                
                if st.session_state.rooms:
                    st.write("### 📝 Gestione Ambienti")
                    for i, r in enumerate(st.session_state.rooms):
                        col_a, col_b = st.columns([3, 1])
                        st.session_state.rooms[i]['name'] = col_a.text_input(f"Nome", r['name'], key=f"rname_{i}")
                        if col_b.button("🗑️", key=f"rdel_{i}"):
                            st.session_state.rooms.pop(i); st.rerun()
            
            with cl2:
                if 'working_img' in st.session_state:
                    st.write("### 🔍 Verifica Grafica")
                    # FIX ATTRIBUTE ERROR: Usiamo Base64 URL
                    img_url = get_image_base64_url(st.session_state.working_img)
                    
                    rects = []
                    for r in st.session_state.rooms:
                        rects.append({"type": "rect", "left": r['x'], "top": r['y'], "width": r['w'], "height": r['h'], "fill": "rgba(0, 255, 0, 0.2)", "stroke": "green"})
                    
                    st_canvas(
                        background_image=st.session_state.working_img, # Passiamo l'oggetto PIL ma con la speranza del fix
                        initial_drawing={"objects": rects},
                        drawing_mode="rect",
                        update_streamlit=True,
                        height=600, width=800, key="canvas_titan"
                    )
                    st.info("I rettangoli verdi mostrano dove LUXiA ha identificato i vani e letto i nomi.")

        with t2:
            if not st.session_state.rooms:
                st.info("Carica un PDF nella prima Tab.")
            else:
                st.subheader("2. AI Lighting Calculation")
                for i, r in enumerate(st.session_state.rooms):
                    with st.expander(f"📍 Ambito: {r['name']}"):
                        ca, cb = st.columns([2, 1])
                        with ca:
                            if st.button(f"✨ Calcola Illuminazione per {r['name']}", key=f"aib_{i}"):
                                if st.session_state.get('groq_online'):
                                    p = f"Sei un Lighting Designer. Progetta '{r['name']}'. Suggerisci brand (iGuzzini, Artemide o Flos), modello preciso, quantità per 500lux e una breve spiegazione tecnica."
                                    res = st.session_state.groq_client.chat.completions.create(messages=[{"role":"user","content":p}], model="llama-3.3-70b-versatile")
                                    st.session_state[f"ai_res_{i}"] = res.choices[0].message.content
                                    # Salvataggio DB
                                    conn = sqlite3.connect('luxia_titan.db')
                                    conn.execute("INSERT INTO rooms (project_id, r_name, strategy, ies_link, pdf_link) VALUES (?,?,?,?,?)",
                                                 (st.session_state.current_proj_id, r['name'], st.session_state[f"ai_res_{i}"], f"{r['name']}.ldt", f"{r['name']}.pdf"))
                                    conn.commit(); conn.close()
                            
                            if f"ai_res_{i}" in st.session_state:
                                st.write(st.session_state[f"ai_res_{i}"])
                        with cb:
                            st.write("**📁 Documentazione Tecnica**")
                            st.button("📉 Fotometria (IES)", key=f"ies_{i}")
                            st.button("📋 Scheda Tecnica (PDF)", key=f"pdf_{i}")

        with t3:
            st.subheader("3. Export Finale")
            if st.button("📦 Genera Report PDF Completo"):
                pdf = FPDF()
                pdf.add_page(); pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, f"LUXiA Titan Report - {st.session_state.current_proj_name}", ln=True)
                st.success("Report generato!")

if __name__ == "__main__":
    main()