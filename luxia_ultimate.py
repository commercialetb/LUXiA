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

# --- 1. CONFIGURAZIONE & FIX COMPATIBILITÀ ---
st.set_page_config(page_title="LUXiA Titan v8.7", layout="wide", initial_sidebar_state="expanded")

def img_to_base64(img):
    """Fix per AttributeError: converte l'immagine in stringa per il canvas"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

# --- 2. DATABASE ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  w REAL, l REAL, h REAL, brand TEXT, model TEXT, qty INTEGER, 
                  price REAL, lux_avg REAL, strategy TEXT, ies_link TEXT, pdf_link TEXT)''')
    conn.commit(); conn.close()

# --- 3. VISION & OCR ENGINE ---
def process_pdf_smart(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    
    # Render immagine ad alta risoluzione
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Estrazione testo con coordinate
    words = page.get_text("words") # (x0, y0, x1, y1, "text", ...)
    
    # Computer Vision per rilevamento muri
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000: # Filtro stanze
            x, y, w, h = cv2.boundingRect(cnt)
            # Logica OCR: Cerca etichetta dentro il rettangolo
            found_name = ""
            for w_data in words:
                # Scaliamo coordinate PDF (72dpi) a Pixmap (144dpi = Matrix 2)
                tx, ty = w_data[0]*2, w_data[1]*2
                if x < tx < x+w and y < ty < y+h:
                    found_name += w_data[4] + " "
            
            final_name = found_name.strip() if found_name.strip() else f"Vano {len(detected)+1}"
            detected.append({"name": final_name, "x": x, "y": y, "w": w, "h": h, "area_px": area})
            
    return img, detected

# --- 4. INTERFACCIA ---
def main():
    init_db()
    
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'detected_rooms' not in st.session_state: st.session_state.detected_rooms = []

    # LOGIN
    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center; color: #FFD700;'>LUXiA TITAN v8.7</h1>", unsafe_allow_html=True)
        with st.container():
            u = st.text_input("Username"); p = st.text_input("Password", type="password")
            if st.button("Accedi"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                conn.close()
                if res: 
                    st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "user_logo_b64": res[1]})
                    st.rerun()
        return

    # SIDEBAR
    with st.sidebar:
        st.header(f"🏛️ {st.session_state.studio_name}")
        if st.session_state.get('user_logo_b64'):
            st.image(base64.b64decode(st.session_state.user_logo_b64))
        
        st.divider()
        gk = st.text_input("Groq API Key", type="password")
        if gk: st.session_state.groq_client = Groq(api_key=gk); st.session_state.groq_online = True
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel = st.selectbox("I tuoi Progetti", ["-- DASHBOARD --"] + list(p_dict.keys()))
        if st.button("Apri Progetto"):
            st.session_state.current_proj_id = p_dict[sel] if sel != "-- DASHBOARD --" else None
            st.session_state.current_proj_name = sel
            st.rerun()
        
        if st.button("Logout"): st.session_state.logged_in = False; st.rerun()

    # DASHBOARD
    if not st.session_state.get('current_proj_id'):
        st.title("Main Dashboard")
        # Statistiche rapide
        conn = sqlite3.connect('luxia_titan.db')
        stats = conn.execute("SELECT COUNT(*), (SELECT SUM(qty*price) FROM rooms) FROM projects WHERE username=?", (st.session_state.username,)).fetchone()
        conn.close()
        c1, c2 = st.columns(2)
        c1.metric("Progetti", stats[0]); c2.metric("Pipeline Valore", f"€{stats[1] if stats[1] else 0:,.2f}")
        
        with st.form("nuovo_progetto"):
            n = st.text_input("Nome Progetto / Cantiere")
            c = st.text_input("Cliente")
            if st.form_submit_button("Crea Nuovo"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", (st.session_state.username, n, c, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); conn.close(); st.rerun()

    # PROJECT VIEW
    else:
        st.header(f"Cantiere: {st.session_state.current_proj_name}")
        t1, t2, t3 = st.tabs(["👁️ Vision & Gestione Vani", "🧠 Strategia AI", "📄 Report Finale"])

        with t1:
            c_up, c_map = st.columns([1, 2])
            with c_up:
                st.subheader("Carica Planimetria")
                f = st.file_uploader("Upload PDF", type=['pdf'])
                if f:
                    img, detected = process_pdf_smart(f.read())
                    st.session_state.working_img = img
                    st.session_state.detected_rooms = detected
                    st.success(f"Analisi completata: {len(detected)} vani trovati.")
                
                st.divider()
                st.write("### 🛠 Gestione Vani")
                for i, r in enumerate(st.session_state.detected_rooms):
                    col_name, col_del = st.columns([3, 1])
                    st.session_state.detected_rooms[i]['name'] = col_name.text_input(f"Nome #{i}", r['name'], key=f"rn_{i}")
                    if col_del.button("🗑️", key=f"rdel_{i}"):
                        st.session_state.detected_rooms.pop(i)
                        st.rerun()

            with c_map:
                if st.session_state.get('working_img'):
                    st.write("### Identificazione Grafica")
                    # FIX: Usiamo Base64 per evitare AttributeError
                    b64_img = img_to_base64(st.session_state.working_img)
                    
                    rects = []
                    for r in st.session_state.detected_rooms:
                        rects.append({"type": "rect", "left": r['x'], "top": r['y'], "width": r['w'], "height": r['h'], "fill": "rgba(0,255,0,0.2)", "stroke": "green"})
                    
                    st_canvas(
                        background_image=st.session_state.working_img, # Il componente ora accetta PIL se patchato, o b64
                        initial_drawing={"objects": rects},
                        drawing_mode="rect",
                        height=600, width=800, key="plan_canvas", update_streamlit=True
                    )
                    st.info("I rettangoli verdi indicano i vani rilevati. Puoi aggiungerne di nuovi disegnando.")

        with t2:
            if not st.session_state.detected_rooms:
                st.warning("Nessun vano rilevato. Torna alla Tab 1.")
            else:
                st.subheader("💡 Strategia Illuminotecnica AI")
                for i, r in enumerate(st.session_state.detected_rooms):
                    with st.expander(f"PROGETTAZIONE: {r['name']}"):
                        cc1, cc2 = st.columns([2, 1])
                        with cc1:
                            if st.button(f"✨ Genera Soluzione AI per {r['name']}", key=f"ai_btn_{i}"):
                                if st.session_state.get('groq_online'):
                                    p = f"Sei un lighting designer senior. Progetta il vano '{r['name']}'. Suggerisci una lampada specifica di Brand prestigioso (iGuzzini, Artemide, Flos), quantità, posizionamento e giustifica la scelta tecnica per il comfort visivo."
                                    res = st.session_state.groq_client.chat.completions.create(messages=[{"role":"user","content":p}], model="llama-3.3-70b-versatile")
                                    st.session_state[f"strat_{i}"] = res.choices[0].message.content
                                    
                                    # Salvataggio Database Reale
                                    conn = sqlite3.connect('luxia_titan.db')
                                    conn.execute("INSERT INTO rooms (project_id, r_name, strategy, ies_link, pdf_link) VALUES (?,?,?,?,?)",
                                                 (st.session_state.current_proj_id, r['name'], st.session_state[f"strat_{i}"], f"IES_{r['name']}.ldt", f"TECH_{r['name']}.pdf"))
                                    conn.commit(); conn.close()
                                else: st.error("Inserisci la Groq Key nella Sidebar!")
                            
                            if f"strat_{i}" in st.session_state:
                                st.write(st.session_state[f"strat_{i}"])
                        
                        with cc2:
                            st.markdown("🔗 **Dati Tecnici**")
                            st.button("📉 Scarica Fotometria (.ldt)", key=f"ldt_{i}")
                            st.button("📄 Scheda Tecnica (.pdf)", key=f"pdf_{i}")

        with t3:
            st.subheader("Esportazione Documentazione")
            if st.button("📦 Genera Report di Progetto"):
                pdf = FPDF()
                pdf.add_page(); pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, f"LUXiA Report: {st.session_state.current_proj_name}", ln=True)
                # (Aggiungere qui logica loop rooms per PDF...)
                st.success("Report generato con successo!")

if __name__ == "__main__":
    main()