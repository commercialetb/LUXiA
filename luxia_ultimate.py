import streamlit as st
import numpy as np
import sqlite3
import hashlib
import base64
import requests
import io
from fpdf import FPDF
from datetime import datetime
from groq import Groq

# --- 1. DATABASE CONFIG (SUPPORTO PDF E MEDIA) ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)''')
    # Supporto per foto (img_blob) e documenti PDF (pdf_blob)
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  w REAL, l REAL, h REAL, brand TEXT, model TEXT, qty INTEGER, 
                  price REAL, lux_avg REAL, strategy TEXT, img_blob BLOB, pdf_blob BLOB, pdf_name TEXT)''')
    conn.commit()
    conn.close()

# --- 2. GENERATORE PDF RELAZIONE ---
class LuxiaPDF(FPDF):
    def header(self):
        if st.session_state.get('user_logo_b64'):
            with open("temp_logo.png", "wb") as f:
                f.write(base64.b64decode(st.session_state.user_logo_b64))
            self.image("temp_logo.png", 10, 8, 25)
        self.set_font('Arial', 'B', 12)
        self.set_xy(40, 10)
        self.cell(0, 5, st.session_state.get('studio_name', 'LUXiA').upper(), ln=True)
        self.ln(20)

# --- 3. CORE APP ---
def main():
    st.set_page_config(page_title="LUXiA Titan v5.3", layout="wide", initial_sidebar_state="expanded")
    init_db()

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None

    # --- LOGIN ---
    if not st.session_state.logged_in:
        st.title("💡 LUXiA | Professional Lighting Design")
        t_auth = st.tabs(["Accedi", "Registrati"])
        with t_auth[0]:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Entra"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                if res:
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.session_state.studio_name = res[0]
                    st.session_state.user_logo_b64 = res[1]
                    st.rerun()
                else: st.error("Accesso negato.")
        with t_auth[1]:
            nu, npw, ns = st.text_input("User"), st.text_input("Pass", type="password"), st.text_input("Nome Studio")
            if st.button("Registrati"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, hashlib.sha256(npw.encode()).hexdigest(), ns, None))
                conn.commit()
                st.success("Registrato!")
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("LUXiA Titan")
        if st.session_state.user_logo_b64:
            st.image(base64.b64decode(st.session_state.user_logo_b64), width=150)
        
        up_logo = st.file_uploader("Upload Logo Studio", type=['png','jpg'])
        if up_logo:
            b64 = base64.b64encode(up_logo.read()).decode()
            conn = sqlite3.connect('luxia_titan.db')
            conn.execute("UPDATE users SET logo_b64=? WHERE username=?", (b64, st.session_state.username))
            conn.commit()
            st.session_state.user_logo_b64 = b64
            st.rerun()

        st.divider()
        groq_key = st.text_input("Groq API Key", type="password")
        if groq_key:
            st.session_state.groq_client = Groq(api_key=groq_key)
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        projs = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        proj_dict = {p[1]: p[0] for p in projs}
        sel = st.selectbox("📂 Progetti", ["-- Crea Nuovo --"] + list(proj_dict.keys()))
        
        if sel != "-- Crea Nuovo --":
            st.session_state.current_proj_id = proj_dict[sel]
            st.session_state.current_proj_name = sel
        else:
            st.session_state.current_proj_id = None

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # --- CONTENT ---
    if not st.session_state.current_proj_id:
        st.header("Crea un nuovo Progetto")
        pn = st.text_input("Nome Progetto")
        cl = st.text_input("Cliente")
        if st.button("Crea e Apri Progetto"):
            if pn:
                conn = sqlite3.connect('luxia_titan.db')
                cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                            (st.session_state.username, pn, cl, datetime.now().strftime("%Y-%m-%d")))
                new_id = cur.lastrowid
                conn.commit()
                st.session_state.current_proj_id = new_id
                st.session_state.current_proj_name = pn
                st.rerun()
    else:
        st.title(f"🏢 {st.session_state.current_proj_name}")
        t1, t2, t3 = st.tabs(["📋 Elenco Vani", "📐 Progettazione & PDF", "📄 Report Finale"])

        with t1:
            conn = sqlite3.connect('luxia_titan.db')
            rooms = conn.execute("SELECT r_name, lux_avg, model, qty, img_blob, pdf_blob, pdf_name FROM rooms WHERE project_id=?", 
                                (st.session_state.current_proj_id,)).fetchall()
            if rooms:
                for r in rooms:
                    with st.expander(f"📍 {r[0]}"):
                        c_info, c_media = st.columns([2,1])
                        c_info.write(f"Risultato: **{r[1]:.0f} Lux** | {r[3]}x {r[2]}")
                        if r[4]: c_media.image(r[4], caption="Planimetria/Foto")
                        if r[5]:
                            # Download del PDF salvato nel DB
                            c_info.download_button(f"📄 Scarica {r[6]}", data=r[5], file_name=r[6])
            else: st.info("Nessun vano presente.")

        with t2:
            st.subheader("Nuovo Vano: Carica Planimetrie PDF e Dati")
            with st.form("add_room_v53", clear_on_submit=True):
                r_name = st.text_input("Nome Ambiente")
                c1, c2, c3 = st.columns(3)
                w = c1.number_input("Larghezza (m)", 1.0, 100.0, 6.0)
                l = c2.number_input("Lunghezza (m)", 1.0, 100.0, 8.0)
                h = c3.number_input("Altezza (m)", 2.0, 20.0, 3.0)
                
                st.write("📂 **Allegati Tecnici**")
                file_img = st.file_uploader("Carica Foto/Immagine (JPG/PNG)", type=['png','jpg','jpeg'])
                file_pdf = st.file_uploader("Carica Documento/Planimetria (PDF)", type=['pdf'])
                
                st.divider()
                brand = st.selectbox("Marca", ["BEGA", "iGuzzini", "Artemide", "Altro"])
                mod = st.text_input("Codice Modello")
                q = st.number_input("Q.tà", 1, 500, 6)
                pr = st.number_input("Prezzo Unit. (€)", 0.0, 5000.0, 220.0)
                
                if st.form_submit_button("Salva Ambiente nel Database"):
                    img_data = file_img.read() if file_img else None
                    pdf_data = file_pdf.read() if file_pdf else None
                    pdf_name = file_pdf.name if file_pdf else None
                    
                    lux = (q * 3200 * 0.6) / (w * l)
                    strat = "Calcolo in corso..."
                    
                    if 'groq_client' in st.session_state:
                        p = f"Ambiente: {r_name} ({w*l}mq). Apparecchi: {q}x {brand} {mod}. Spiega perché questa scelta è professionale."
                        chat = st.session_state.groq_client.chat.completions.create(
                            messages=[{"role":"user","content":p}], model="llama-3.3-70b-versatile"
                        )
                        strat = chat.choices[0].message.content

                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute('''INSERT INTO rooms (project_id, r_name, w, l, h, brand, model, qty, price, lux_avg, strategy, img_blob, pdf_blob, pdf_name) 
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', 
                                 (st.session_state.current_proj_id, r_name, rw, rl, rh, brand, mod, q, pr, lux, strat, img_data, pdf_data, pdf_name))
                    conn.commit()
                    st.success("Ambiente e PDF archiviati correttamente!")
                    st.rerun()

        with t3:
            st.subheader("Generazione Report e Computo Metrico")
            if st.button("Genera PDF Finale Progetto"):
                pdf = LuxiaPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, f"RELAZIONE COMPLETA: {st.session_state.current_proj_name}", ln=True)
                
                conn = sqlite3.connect('luxia_titan.db')
                data = conn.execute("SELECT * FROM rooms WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
                total_cost = 0
                for r in data:
                    pdf.ln(8)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, f"Vano: {r[2]}", fill=False, ln=True)
                    pdf.set_font("Arial", '', 10)
                    pdf.multi_cell(0, 6, f"- Dimensioni: {r[3]}x{r[4]}m (Area: {r[3]*r[4]}mq)\n- Illuminamento stimato: {r[11]:.0f} Lux\n- Fornitura: {r[9]}x {r[8]} ({r[7]})")
                    if r[12]: 
                        pdf.set_font("Arial", 'I', 9)
                        pdf.multi_cell(0, 5, f"AI Strategy: {r[12]}")
                    total_cost += (r[9]*r[10])
                
                pdf.ln(10)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"TOTALE INVESTIMENTO FORNITURA: {total_cost:,.2f} EUR", ln=True)
                
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button("📥 Scarica Report LUXiA", data=pdf_bytes, file_name=f"{st.session_state.current_proj_name}_LUXiA.pdf")

if __name__ == "__main__":
    main()