import streamlit as st
import sqlite3
import hashlib
import base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from datetime import datetime
from groq import Groq
from fpdf import FPDF
from streamlit_drawable_canvas import st_canvas

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="LUXiA Titan v7.0 Vision", layout="wide", initial_sidebar_state="expanded")

# --- 1. CORE DATABASE ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS project_docs (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, filename TEXT, file_type TEXT, file_blob BLOB, upload_date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, w REAL, l REAL, h REAL, brand TEXT, model TEXT, qty INTEGER, price REAL, lux_avg REAL, strategy TEXT, img_blob BLOB)''')
    
    # Check colonne
    cols = [i[1] for i in c.execute("PRAGMA table_info(rooms)").fetchall()]
    if 'price' not in cols: c.execute("ALTER TABLE rooms ADD COLUMN price REAL DEFAULT 0")
    if 'strategy' not in cols: c.execute("ALTER TABLE rooms ADD COLUMN strategy TEXT")
    
    conn.commit(); conn.close()

# --- 2. COMPUTER VISION ENGINE ---
def detect_rooms_cv(image_bytes):
    # Converte l'immagine per OpenCV
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Trova contorni (potenziali stanze)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_rooms = []
    img_h, img_w = img.shape[:2]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000: # Filtra rumore (piccoli oggetti)
            x, y, w, h = cv2.boundingRect(cnt)
            # Ignora contorni troppo grandi (bordo foglio)
            if w < img_w * 0.9 and h < img_h * 0.9:
                detected_rooms.append({'left': x, 'top': y, 'width': w, 'height': h})
    
    return detected_rooms, img_w, img_h

# --- 3. UTILITY ---
def get_dashboard_stats(username):
    conn = sqlite3.connect('luxia_titan.db')
    n_p = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (username,)).fetchone()[0]
    stats = conn.execute('''SELECT COUNT(rooms.id), SUM(rooms.qty * rooms.price) FROM rooms JOIN projects ON rooms.project_id = projects.id WHERE projects.username=?''', (username,)).fetchone()
    conn.close()
    return n_p, (stats[0] if stats[0] else 0), (stats[1] if stats[1] else 0)

# --- 4. MAIN APP ---
def main():
    init_db()
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'scale_factor' not in st.session_state: st.session_state.scale_factor = None # Pixels per Metro

    # LOGIN
    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center; color: #FFD700;'>LUXiA TITAN v7.0</h1>", unsafe_allow_html=True)
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.button("Login"):
            h = hashlib.sha256(p.encode()).hexdigest()
            conn = sqlite3.connect('luxia_titan.db')
            res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
            conn.close()
            if res: st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "user_logo_b64": res[1]}); st.rerun()
        return

    # SIDEBAR
    with st.sidebar:
        st.markdown("### 🏛️ LUXiA PANEL")
        if st.session_state.get('user_logo_b64'): st.image(base64.b64decode(st.session_state.user_logo_b64), use_container_width=True)
        
        st.divider()
        st.markdown("### 🧠 AI CORE")
        gk = st.text_input("Groq API Key", type="password")
        if gk: st.session_state.groq_client = Groq(api_key=gk); st.session_state.groq_online = True; st.success("AI VISION: ACTIVE")
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        sel = st.selectbox("Progetti", ["-- DASHBOARD --"] + list(p_dict.keys()))
        if st.button("📂 APRI"):
            st.session_state.current_proj_id = p_dict[sel] if sel != "-- DASHBOARD --" else None
            if sel != "-- DASHBOARD --": st.session_state.current_proj_name = sel
            st.rerun()
        
        st.markdown("---"); st.caption("LUXiA™ TRADEMARK - Patent Pending\nVersion 7.0 Visionary")

    # DASHBOARD
    if not st.session_state.current_proj_id:
        st.title(f"Benvenuto, {st.session_state.studio_name}")
        np, nv, val = get_dashboard_stats(st.session_state.username)
        c1, c2, c3 = st.columns(3)
        c1.metric("Progetti", np); c2.metric("Vani", nv); c3.metric("Pipeline (€)", f"{val:,.2f}")
        with st.form("new_p"):
            if st.form_submit_button("Crea Nuovo Progetto"):
                conn = sqlite3.connect('luxia_titan.db'); cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", (st.session_state.username, "Nuovo Progetto", "Cliente", datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); st.session_state.current_proj_id = cur.lastrowid; st.session_state.current_proj_name = "Nuovo Progetto"; conn.close(); st.rerun()

    # PROGETTO APERTO
    else:
        st.header(f"🏢 Progetto: {st.session_state.current_proj_name}")
        if st.button("⬅️ Back"): st.session_state.current_proj_id = None; st.rerun()

        t1, t2, t3 = st.tabs(["📂 1. Upload & Setup", "👁️ 2. Vision AI & Vani", "📄 3. Report"])

        # TAB 1: UPLOAD
        with t1:
            st.subheader("Caricamento Planimetria")
            st.info("💡 Per la funzione AI Vision, carica un file immagine (JPG/PNG) della pianta.")
            up_file = st.file_uploader("Carica Pianta", type=['png', 'jpg', 'jpeg'])
            if up_file:
                # Salva nel DB ma anche in sessione per elaborazione immediata
                bytes_data = up_file.getvalue()
                st.session_state.plan_img = bytes_data
                st.image(bytes_data, caption="Planimetria Caricata", width=500)
                
                if st.button("Usa questa pianta per l'analisi"):
                    st.success("Pianta pronta per la Tab 2!")

        # TAB 2: VISION & AI
        with t2:
            st.subheader("Analisi Spaziale e Lighting Design AI")
            
            if 'plan_img' not in st.session_state:
                st.warning("⚠️ Carica prima una planimetria nella Tab 1")
            else:
                # --- FASE 1: CALIBRAZIONE ---
                st.markdown("### 📏 Fase 1: Calibrazione Scala")
                st.write("Disegna una linea sulla pianta e indica quanto misura in metri (es. larghezza porta = 0.9).")
                
                # Canvas per calibrazione
                img_pil = Image.open(BytesIO(st.session_state.plan_img))
                canvas_calib = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)", stroke_width=3, stroke_color="#FF0000",
                    background_image=img_pil, update_streamlit=True, height=500, drawing_mode="line", key="calib_canvas"
                )
                
                if canvas_calib.json_data is not None:
                    objects = canvas_calib.json_data["objects"]
                    if len(objects) > 0:
                        line = objects[-1] # Prendi l'ultima linea
                        px_len = np.sqrt((line['x2'] - line['x1'])**2 + (line['y2'] - line['y1'])**2)
                        real_m = st.number_input("Lunghezza reale della linea (metri)", 0.1, 100.0, 1.0)
                        if st.button("Imposta Scala"):
                            st.session_state.scale_factor = px_len / real_m # Pixels per metro
                            st.success(f"Scala salvata: {st.session_state.scale_factor:.2f} px/m")

                # --- FASE 2: RILEVAMENTO E GESTIONE VANI ---
                if st.session_state.get('scale_factor'):
                    st.divider()
                    st.markdown("### 🧩 Fase 2: Rilevamento Vani")
                    
                    col_tools, col_canvas = st.columns([1, 3])
                    
                    with col_tools:
                        st.write("**Strumenti:**")
                        mode = st.radio("Modalità", ["Disegno Manuale Vano", "AI Auto-Detect"])
                        
                        initial_rects = []
                        if mode == "AI Auto-Detect" and st.button("🔍 Scansiona Planimetria"):
                            rects, w_img, h_img = detect_rooms_cv(st.session_state.plan_img)
                            initial_rects = rects
                            st.toast(f"Trovati {len(rects)} potenziali vani!", icon="🤖")

                    with col_canvas:
                        # Prepara dati JSON per il canvas se ci sono rettangoli rilevati
                        canvas_objects = []
                        for r in initial_rects:
                            canvas_objects.append({
                                "type": "rect", "left": r['left'], "top": r['top'], 
                                "width": r['width'], "height": r['height'], 
                                "fill": "rgba(0, 255, 0, 0.3)", "stroke": "#00FF00", "strokeWidth": 2
                            })
                        
                        canvas_rooms = st_canvas(
                            fill_color="rgba(0, 0, 255, 0.3)", stroke_width=2, stroke_color="#0000FF",
                            background_image=img_pil, update_streamlit=True, height=500, 
                            drawing_mode="rect", initial_drawing={"objects": canvas_objects} if initial_rects else None,
                            key="rooms_canvas"
                        )

                    # --- FASE 3: AI LIGHTING DESIGN ---
                    if canvas_rooms.json_data is not None:
                        objects = [obj for obj in canvas_rooms.json_data["objects"] if obj["type"] == "rect"]
                        
                        if objects:
                            st.divider()
                            st.markdown("### 💡 Fase 3: AI Lighting Strategy")
                            st.write(f"Vani identificati: {len(objects)}")
                            
                            for i, obj in enumerate(objects):
                                # Calcolo Dimensioni Reali
                                real_w = obj['width'] / st.session_state.scale_factor
                                real_l = obj['height'] / st.session_state.scale_factor
                                area = real_w * real_l
                                
                                with st.expander(f"📍 Vano #{i+1} - {area:.1f} mq ({real_w:.1f}x{real_l:.1f}m)"):
                                    c1, c2 = st.columns(2)
                                    v_name = c1.text_input(f"Nome Vano {i+1}", f"Ambiente {i+1}", key=f"n_{i}")
                                    v_type = c2.selectbox("Tipo Ambiente", ["Ufficio", "Residenziale", "Retail", "Industriale", "Esterno"], key=f"t_{i}")
                                    
                                    if st.button(f"✨ Genera Progetto Luce per {v_name}", key=f"btn_{i}"):
                                        if st.session_state.groq_online:
                                            # AI LOGIC
                                            prompt = f"""
                                            Sei LUXiA AI. Agisci come Senior Lighting Designer.
                                            Analizza un ambiente {v_type} di {real_w:.2f} x {real_l:.2f} metri (Area: {area:.2f} mq).
                                            1. Scegli il prodotto ideale (es. Downlight, Lineare, Track) specificando Brand (iGuzzini/Bega/Artemide).
                                            2. Calcola quantità per avere 500 lux (ufficio) o 300 (residenziale).
                                            3. Genera una breve descrizione tecnica e commerciale.
                                            4. Rispondi in formato testo strutturato.
                                            """
                                            try:
                                                chat = st.session_state.groq_client.chat.completions.create(
                                                    messages=[{"role":"user","content":prompt}], model="llama-3.3-70b-versatile"
                                                )
                                                strat = chat.choices[0].message.content
                                                
                                                # Mock dati tecnici (l'AI non ha accesso ai DB reali in tempo reale, li simuliamo o li estraiamo dal testo se formattato JSON)
                                                est_qty = int(area / 4) + 1
                                                est_price = 150.00
                                                est_lux = 500 if v_type == "Ufficio" else 300
                                                
                                                # Salvataggio
                                                conn = sqlite3.connect('luxia_titan.db')
                                                conn.execute('''INSERT INTO rooms (project_id, r_name, w, l, h, brand, model, qty, price, lux_avg, strategy) 
                                                                VALUES (?,?,?,?,?,?,?,?,?,?,?)''', 
                                                             (st.session_state.current_proj_id, v_name, real_w, real_l, 3.0, "AI-Selection", "Ref-Auto", est_qty, est_price, est_lux, strat))
                                                conn.commit(); conn.close()
                                                st.success("✅ Vano salvato nel database!")
                                                st.write(strat)
                                                
                                                # Simulazione Schede Tecniche
                                                st.markdown("**📂 Documentazione Tecnica Automatica:**")
                                                cc1, cc2 = st.columns(2)
                                                cc1.button("📉 Curva Fotometrica (LDT)", disabled=True)
                                                cc2.button("📄 Scheda Tecnica (PDF)", disabled=True)
                                                
                                            except Exception as e: st.error(f"Errore AI: {e}")
                                        else:
                                            st.error("AI Offline. Inserisci API Key nella sidebar.")

        # TAB 3: REPORT (Stessa logica di prima, solida)
        with t3:
            st.subheader("Report Finale")
            if st.button("Genera PDF"):
                pdf = FPDF()
                pdf.add_page(); pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, f"Progetto: {st.session_state.current_proj_name}", ln=True)
                
                conn = sqlite3.connect('luxia_titan.db')
                rooms = conn.execute("SELECT * FROM rooms WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
                conn.close()
                
                for r in rooms:
                    pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, f"Vano: {r[2]}", ln=True)
                    pdf.set_font("Arial", "", 10); pdf.multi_cell(0, 5, f"AI Strategy: {r[11][:300]}...")
                    pdf.ln(5)
                
                html = base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('latin-1')
                st.download_button("📥 Scarica Report", data=base64.b64decode(html), file_name="LUXiA_Report.pdf")

if __name__ == "__main__":
    main()