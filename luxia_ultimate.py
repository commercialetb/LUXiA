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

# --- 1. CONFIGURAZIONE & STILE ---
st.set_page_config(page_title="LUXiA Titan v8.6", layout="wide", initial_sidebar_state="expanded")

# --- 2. ARCHITETTURA DATABASE (Self-Healing) ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    # Tabelle Core
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS project_docs (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, filename TEXT, file_blob BLOB, upload_date TEXT)')
    # Tabella Vani Potenziata
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  w REAL, l REAL, h REAL, brand TEXT, model TEXT, qty INTEGER, 
                  price REAL, lux_avg REAL, strategy TEXT, img_blob BLOB, 
                  ies_link TEXT, pdf_link TEXT)''')
    
    # Migrazione colonne (sicurezza anti-crash)
    try:
        cols = [i[1] for i in c.execute("PRAGMA table_info(rooms)").fetchall()]
        if 'ies_link' not in cols: c.execute("ALTER TABLE rooms ADD COLUMN ies_link TEXT")
        if 'pdf_link' not in cols: c.execute("ALTER TABLE rooms ADD COLUMN pdf_link TEXT")
    except: pass
    
    conn.commit()
    conn.close()

# --- 3. UTILITY FUNCTIONS ---
def get_dashboard_stats(username):
    conn = sqlite3.connect('luxia_titan.db')
    n_proj = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (username,)).fetchone()[0]
    stats = conn.execute('''SELECT COUNT(rooms.id), SUM(rooms.qty * rooms.price) 
                            FROM rooms JOIN projects ON rooms.project_id = projects.id 
                            WHERE projects.username=?''', (username,)).fetchone()
    conn.close()
    return n_proj, (stats[0] if stats[0] else 0), (stats[1] if stats[1] else 0)

def pdf_to_image(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # OCR Nativo: estraiamo parole e coordinate
    words = page.get_text("words") # x0, y0, x1, y1, text...
    return img, words

def detect_rooms_cv(img_pil, text_blocks):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            # Tentativo OCR: Cerca testo dentro le coordinate del vano
            name = "Vano Ignoto"
            for tw in text_blocks:
                tx, ty = tw[0]*2, tw[1]*2 # Scaling per Pixmap 2x
                if x < tx < x+w and y < ty < y+h:
                    if len(tw[4]) > 2: # Evita polvere OCR
                        name = tw[4]
                        break
            found.append({"name": name, "x": x, "y": y, "w": w, "h": h})
    return found

# --- 4. INTERFACCIA PRINCIPALE ---
def main():
    init_db()
    
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None
    if 'scale_px_m' not in st.session_state: st.session_state.scale_px_m = 100

    # --- LOGIN SYSTEM ---
    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center; color: #FFD700;'>LUXiA TITAN v8.6</h1>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            t_l, t_r = st.tabs(["🔒 Accesso", "📝 Registrazione"])
            with t_l:
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.button("Entra nello Studio", use_container_width=True):
                    h = hashlib.sha256(p.encode()).hexdigest()
                    conn = sqlite3.connect('luxia_titan.db')
                    res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                    conn.close()
                    if res:
                        st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "user_logo_b64": res[1]})
                        st.rerun()
                    else: st.error("Credenziali errate.")
            with t_r:
                nu = st.text_input("Nuovo Utente")
                np = st.text_input("Nuova Password", type="password")
                ns = st.text_input("Nome Studio")
                if st.button("Crea Account"):
                    h = hashlib.sha256(np.encode()).hexdigest()
                    conn = sqlite3.connect('luxia_titan.db')
                    try:
                        conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, h, ns, None))
                        conn.commit(); st.success("Registrato!")
                    except: st.error("Utente esistente.")
                    finally: conn.close()
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown(f"### 🏛️ {st.session_state.studio_name}")
        if st.session_state.get('user_logo_b64'):
            st.image(base64.b64decode(st.session_state.user_logo_b64), use_container_width=True)
        
        st.divider()
        st.markdown("### 🧠 AI ENGINE")
        gk = st.text_input("Groq API Key", type="password", placeholder="sk-...")
        if gk:
            st.session_state.groq_client = Groq(api_key=gk)
            st.session_state.groq_online = True
            st.success("AI ONLINE")
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel = st.selectbox("Naviga Progetti", ["-- DASHBOARD --"] + list(p_dict.keys()))
        if st.button("🚀 APRI / VAI", use_container_width=True):
            st.session_state.current_proj_id = p_dict[sel] if sel != "-- DASHBOARD --" else None
            if sel != "-- DASHBOARD --": st.session_state.current_proj_name = sel
            st.rerun()

        st.divider()
        if st.button("🚪 LOGOUT"): st.session_state.logged_in = False; st.rerun()

    # --- DASHBOARD ---
    if not st.session_state.current_proj_id:
        st.title(f"Studio {st.session_state.studio_name}")
        n_p, n_v, val = get_dashboard_stats(st.session_state.username)
        k1, k2, k3 = st.columns(3)
        k1.metric("Progetti Attivi", n_p)
        k2.metric("Vani Calcolati", n_v)
        k3.metric("Fatturato Pipeline", f"€ {val:,.2f}")
        
        st.divider()
        c_new, c_list = st.columns([1, 2])
        with c_new:
            st.subheader("🆕 Nuovo Cantiere")
            with st.form("new_p"):
                name = st.text_input("Nome Progetto")
                cli = st.text_input("Cliente")
                if st.form_submit_button("Crea Progetto"):
                    conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                    cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                                (st.session_state.username, name, cli, datetime.now().strftime("%d/%m/%Y")))
                    conn.commit(); st.session_state.current_proj_id = cur.lastrowid; st.session_state.current_proj_name = name; conn.close(); st.rerun()
        with c_list:
            st.subheader("📂 Archivio")
            if plist:
                df = pd.DataFrame(plist, columns=['ID', 'Progetto'])
                st.dataframe(df[['Progetto']], use_container_width=True, hide_index=True)

    # --- PROGETTO APERTO (WORKFLOW) ---
    else:
        st.header(f"🏢 Cantiere: {st.session_state.current_proj_name}")
        t1, t2, t3 = st.tabs(["📂 1. Vision & Plan", "💡 2. AI Strategy & Tech", "📄 3. Report"])

        # TAB 1: VISION
        with t1:
            col_up, col_map = st.columns([1, 2])
            with col_up:
                st.subheader("Upload PDF")
                f_pdf = st.file_uploader("Trascina la planimetria", type=['pdf'])
                if f_pdf:
                    img, words = pdf_to_image(f_pdf.read())
                    st.session_state.working_img = img
                    st.session_state.text_blocks = words
                    if st.button("🔍 Avvia Rilevamento Vision"):
                        st.session_state.detected_rooms = detect_rooms_cv(img, words)
                        st.success(f"Trovati {len(st.session_state.detected_rooms)} vani con OCR!")

                if 'detected_rooms' in st.session_state:
                    st.divider()
                    st.write("### 📏 Calibrazione")
                    m_nota = st.number_input("Misura nota (m)", 0.1, 10.0, 1.0)
                    if st.button("Imposta Scala"): st.session_state.scale_px_m = 150 # Mock scaling

            with col_map:
                if 'working_img' in st.session_state:
                    st.write("### Mappa Interattiva Vani")
                    rects = []
                    if 'detected_rooms' in st.session_state:
                        for r in st.session_state.detected_rooms:
                            rects.append({"type": "rect", "left": r['x'], "top": r['y'], "width": r['w'], "height": r['h'], "fill": "rgba(0,255,0,0.2)", "stroke": "green"})
                    
                    canvas = st_canvas(
                        background_image=st.session_state.working_img,
                        initial_drawing={"objects": rects},
                        drawing_mode="rect",
                        update_streamlit=True,
                        height=600, width=800, key="plan_canvas"
                    )
                    st.caption("Disegna nuovi rettangoli per aggiungere vani manualmente.")

        # TAB 2: AI & TECH
        with t2:
            if 'detected_rooms' not in st.session_state:
                st.info("Carica ed elabora un PDF nella Tab 1.")
            else:
                for i, r in enumerate(st.session_state.detected_rooms):
                    with st.expander(f"📍 {r['name']} ({r['w']//st.session_state.scale_px_m}x{r['h']//st.session_state.scale_px_m}m)"):
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            v_name = st.text_input("Modifica Nome", r['name'], key=f"name_{i}")
                            if st.button(f"✨ AI Strategy per {v_name}", key=f"ai_{i}"):
                                if st.session_state.get('groq_online'):
                                    prompt = f"Sei un Lighting Designer. Progetta il vano {v_name}. Suggerisci Brand (BEGA, iGuzzini, Flos), modello, quantità e lux medi."
                                    res = st.session_state.groq_client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama-3.3-70b-versatile")
                                    strat = res.choices[0].message.content
                                    st.session_state[f"strat_{i}"] = strat
                                    
                                    # Salvataggio DB
                                    conn = sqlite3.connect('luxia_titan.db')
                                    conn.execute("INSERT INTO rooms (project_id, r_name, strategy, ies_link, pdf_link) VALUES (?,?,?,?,?)",
                                                 (st.session_state.current_proj_id, v_name, strat, f"ies_{v_name}.ldt", f"tech_{v_name}.pdf"))
                                    conn.commit(); conn.close()
                            
                            if f"strat_{i}" in st.session_state:
                                st.info(st.session_state[f"strat_{i}"])
                        
                        with c2:
                            st.write("**Documentazione:**")
                            st.button("📉 IES File", key=f"ies_{i}")
                            st.button("📄 Datasheet", key=f"pdf_{i}")
                            if st.button("🗑️ Rimuovi Vano", key=f"del_{i}"):
                                st.session_state.detected_rooms.pop(i); st.rerun()

        # TAB 3: REPORT
        with t3:
            st.subheader("Final Report")
            if st.button("📄 Genera Report PDF Completo"):
                pdf = FPDF()
                pdf.add_page(); pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, f"PROGETTO: {st.session_state.current_proj_name}", ln=True)
                pdf.set_font("Arial", "", 12); pdf.cell(0, 10, f"Studio: {st.session_state.studio_name}", ln=True)
                
                conn = sqlite3.connect('luxia_titan.db')
                rooms = conn.execute("SELECT r_name, strategy FROM rooms WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
                conn.close()
                
                for r in rooms:
                    pdf.ln(10); pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, f"Vano: {r[0]}", ln=True)
                    pdf.set_font("Arial", "", 10); pdf.multi_cell(0, 5, r[1][:500])
                
                output = pdf.output(dest='S').encode('latin-1')
                st.download_button("📥 Scarica PDF", output, "Report_LUXiA.pdf")

if __name__ == "__main__":
    main()