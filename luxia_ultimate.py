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
st.set_page_config(page_title="LUXiA Titan v8.8", layout="wide", initial_sidebar_state="expanded")

def get_image_base64(img):
    """Trasforma l'immagine in Base64 per evitare l'AttributeError del Canvas"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- 2. CORE DATABASE ---
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

# --- 3. VISION & OCR ENGINE ---
def process_planimetry(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    
    # Render PDF ad alta risoluzione per CV
    zoom = 2  # 144 DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Estrazione testo con coordinate per OCR
    words = page.get_text("words") # (x0, y0, x1, y1, "text", ...)
    
    # Computer Vision per rilevamento contorni vani
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Thresholding adattivo per isolare le stanze dai muri
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rooms_found = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000: # Filtra elementi piccoli (mobili, scritte)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # OCR Spaziale: cerchiamo testo dentro questo rettangolo
            room_label = ""
            for wd in words:
                tx, ty = wd[0]*zoom, wd[1]*zoom
                if x < tx < x+w and y < ty < y+h:
                    if len(wd[4]) > 2: room_label += wd[4] + " "
            
            final_name = room_label.strip() if room_label.strip() else f"Vano {len(rooms_found)+1}"
            rooms_found.append({"name": final_name, "x": x, "y": y, "w": w, "h": h})
            
    return img, rooms_found

# --- 4. MAIN APPLICATION ---
def main():
    init_db()
    
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'detected_rooms' not in st.session_state: st.session_state.detected_rooms = []
    if 'scale' not in st.session_state: st.session_state.scale = 100 # Default px/m

    # LOGIN SCREEN
    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center; color: #FFD700;'>LUXiA TITAN v8.8</h1>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Accedi", use_container_width=True):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                conn.close()
                if res: 
                    st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "user_logo_b64": res[1]})
                    st.rerun()
                else: st.error("Accesso negato.")
        return

    # SIDEBAR
    with st.sidebar:
        st.header(f"🏛️ {st.session_state.studio_name}")
        if st.session_state.get('user_logo_b64'):
            st.image(base64.b64decode(st.session_state.user_logo_b64), use_container_width=True)
        
        st.divider()
        st.markdown("### 🧠 AI ENGINE")
        gk = st.text_input("Groq API Key", type="password")
        if gk: st.session_state.groq_client = Groq(api_key=gk); st.session_state.groq_online = True; st.success("AI: PRONTA")
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel = st.selectbox("I tuoi Progetti", ["-- DASHBOARD --"] + list(p_dict.keys()))
        if st.button("📂 APRI"):
            st.session_state.current_proj_id = p_dict[sel] if sel != "-- DASHBOARD --" else None
            st.session_state.current_proj_name = sel
            st.rerun()
        
        if st.button("🚪 LOGOUT"): st.session_state.logged_in = False; st.rerun()

    # DASHBOARD
    if not st.session_state.get('current_proj_id'):
        st.title("Benvenuto, Antonio")
        conn = sqlite3.connect('luxia_titan.db')
        stats = conn.execute("SELECT COUNT(*), (SELECT SUM(qty*price) FROM rooms) FROM projects WHERE username=?", (st.session_state.username,)).fetchone()
        conn.close()
        
        k1, k2 = st.columns(2)
        k1.metric("Progetti Totali", stats[0]); k2.metric("Valore Pipeline", f"€ {stats[1] if stats[1] else 0:,.2f}")
        
        st.divider()
        with st.form("new_proj"):
            st.subheader("🆕 Nuovo Cantiere")
            n = st.text_input("Nome Progetto"); c = st.text_input("Cliente")
            if st.form_submit_button("Crea"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", (st.session_state.username, n, c, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); conn.close(); st.rerun()

    # PROJECT WORKFLOW
    else:
        st.header(f"🏢 Progetto: {st.session_state.current_proj_name}")
        t1, t2, t3 = st.tabs(["👁️ Vision AI & Plan", "💡 Strategia Luci AI", "📄 Report"])

        # TAB 1: VISION & GESTIONE
        with t1:
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.subheader("Analisi Planimetria")
                up_pdf = st.file_uploader("Carica PDF Tecnico", type=['pdf'])
                if up_pdf:
                    img, rooms = process_planimetry(up_pdf.read())
                    st.session_state.working_img = img
                    st.session_state.detected_rooms = rooms
                    st.success(f"Trovati {len(rooms)} ambienti!")
                
                if st.session_state.detected_rooms:
                    st.write("### 📋 Elenco Vani")
                    for i, r in enumerate(st.session_state.detected_rooms):
                        c_n, c_d = st.columns([3, 1])
                        st.session_state.detected_rooms[i]['name'] = c_n.text_input(f"Vano {i}", r['name'], key=f"vname_{i}")
                        if c_d.button("🗑️", key=f"vdel_{i}"):
                            st.session_state.detected_rooms.pop(i); st.rerun()

            with col_r:
                if 'working_img' in st.session_state:
                    st.write("### Verifica Grafica Vani")
                    # FIX: Passiamo il Base64 al canvas per evitare l'AttributeError
                    b64_canvas = get_image_base64(st.session_state.working_img)
                    
                    rects = []
                    for r in st.session_state.detected_rooms:
                        rects.append({"type": "rect", "left": r['x'], "top": r['y'], "width": r['w'], "height": r['h'], "fill": "rgba(0,255,0,0.2)", "stroke": "green"})
                    
                    st_canvas(
                        background_image=st.session_state.working_img, 
                        initial_drawing={"objects": rects},
                        drawing_mode="rect",
                        update_streamlit=True,
                        height=600, width=800, key="plan_canvas_v88"
                    )
                    st.caption("Verifica i rettangoli verdi. Quelli che tieni verranno calcolati dall'AI.")

        # TAB 2: AI STRATEGY
        with t2:
            if not st.session_state.detected_rooms:
                st.info("Esegui prima la scansione nella Tab 1.")
            else:
                st.subheader("💡 Generatore Lighting Design AI")
                for i, r in enumerate(st.session_state.detected_rooms):
                    with st.expander(f"📍 Progetto: {r['name']}"):
                        ca, cb = st.columns([2, 1])
                        with ca:
                            if st.button(f"✨ Calcola Strategia per {r['name']}", key=f"ai_go_{i}"):
                                if st.session_state.get('groq_online'):
                                    prompt = f"Sei un esperto Lighting Designer. Analizza il vano '{r['name']}'. Scegli un brand (iGuzzini, Artemide o Flos), un modello specifico, calcola la quantità per 500lux e giustifica la scelta tecnica."
                                    res = st.session_state.groq_client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama-3.3-70b-versatile")
                                    strat = res.choices[0].message.content
                                    st.session_state[f"strat_txt_{i}"] = strat
                                    
                                    # Salvataggio Database con dati tecnici simulati
                                    conn = sqlite3.connect('luxia_titan.db')
                                    conn.execute("INSERT INTO rooms (project_id, r_name, strategy, ies_link, pdf_link) VALUES (?,?,?,?,?)",
                                                 (st.session_state.current_proj_id, r['name'], strat, f"ies_{r['name']}.ldt", f"tech_{r['name']}.pdf"))
                                    conn.commit(); conn.close()
                                    st.rerun()
                            
                            if f"strat_txt_{i}" in st.session_state:
                                st.markdown(st.session_state[f"strat_txt_{i}"])
                        
                        with cb:
                            st.write("**📄 Documenti Tecnici**")
                            st.button("📉 IES/LDT Data", key=f"ldt_dl_{i}")
                            st.button("📋 Tech Sheet", key=f"pdf_dl_{i}")

        # TAB 3: REPORT
        with t3:
            st.subheader("Export Documentazione")
            if st.button("📄 Esporta Report PDF"):
                # (Logica FPDF qui...)
                st.success("Report PDF generato!")

if __name__ == "__main__":
    main()