import streamlit as st
import numpy as np
import sqlite3
import hashlib
import base64
import io
from fpdf import FPDF
from datetime import datetime
from groq import Groq

# --- 1. ARCHITETTURA DATABASE ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  w REAL, l REAL, h REAL, brand TEXT, model TEXT, qty INTEGER, 
                  price REAL, lux_avg REAL, strategy TEXT, img_blob BLOB, pdf_name TEXT)''')
    conn.commit()
    conn.close()

# --- 2. CLASSE PDF CON LOGO ---
class LuxiaPDF(FPDF):
    def header(self):
        if st.session_state.get('user_logo_b64'):
            logo_data = base64.b64decode(st.session_state.user_logo_b64)
            with open("temp_logo_pdf.png", "wb") as f:
                f.write(logo_data)
            self.image("temp_logo_pdf.png", 10, 8, 25)
        self.set_font('Arial', 'B', 12)
        self.set_xy(40, 10)
        self.cell(0, 5, st.session_state.get('studio_name', 'STUDIO').upper(), ln=True)
        self.ln(20)

# --- 3. LOGICA PRINCIPALE ---
def main():
    st.set_page_config(page_title="LUXiA Titan", layout="wide", initial_sidebar_state="expanded")
    init_db()

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None
    if 'groq_online' not in st.session_state: st.session_state.groq_online = False

    # --- FASE LOGIN ---
    if not st.session_state.logged_in:
        st.title("💡 LUXiA | Accesso Studio")
        tabs = st.tabs(["Login", "Registra Studio"])
        with tabs[0]:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("Accedi"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                if res:
                    st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "user_logo_b64": res[1]})
                    st.rerun()
                else: st.error("Credenziali errate.")
        with tabs[1]:
            nu = st.text_input("Nuovo User")
            npw = st.text_input("Nuova Pass", type="password")
            ns = st.text_input("Nome Studio")
            if st.button("Crea Account"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("INSERT INTO users (username, password, studio_name) VALUES (?,?,?)", (nu, hashlib.sha256(npw.encode()).hexdigest(), ns))
                conn.commit()
                st.success("Registrato! Ora effettua il login.")
        return

    # --- SIDEBAR: LOGO, GROQ E PROGETTI ---
    with st.sidebar:
        st.title("LUXiA Titan")
        
        # Gestione Logo (Se presente si rimpicciolisce e appare tasto Reset)
        if st.session_state.user_logo_b64:
            st.image(base64.b64decode(st.session_state.user_logo_b64), width=120)
            if st.button("🔄 Cambia Logo Studio", help="Rimuovi il logo attuale per caricarne uno nuovo"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=NULL WHERE username=?", (st.session_state.username,))
                conn.commit()
                st.session_state.user_logo_b64 = None
                st.rerun()
        else:
            up = st.file_uploader("Carica Logo Studio (JPG/PNG)", type=['png','jpg'])
            if up:
                b64 = base64.b64encode(up.read()).decode()
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=? WHERE username=?", (b64, st.session_state.username))
                conn.commit()
                st.session_state.user_logo_b64 = b64
                st.rerun()

        st.divider()
        
        # Groq Config
        key = st.text_input("Groq API Key", type="password", placeholder="Incolla chiave...")
        if key:
            try:
                st.session_state.groq_client = Groq(api_key=key)
                st.session_state.groq_online = True
                st.caption("🟢 Groq AI Attiva")
            except: st.caption("🔴 Errore Chiave")

        st.divider()
        
        # Archivio Progetti
        st.subheader("📁 I Tuoi Progetti")
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        
        sel = st.selectbox("Seleziona Progetto", ["-- Seleziona --"] + list(p_dict.keys()))
        if st.button("🚀 Apri Progetto"):
            if sel != "-- Seleziona --":
                st.session_state.current_proj_id = p_dict[sel]
                st.session_state.current_proj_name = sel
                st.rerun()

        if st.button("🚪 Esci"):
            st.session_state.logged_in = False
            st.rerun()

    # --- CONTROLLO FLUSSO PROGETTI ---
    if not st.session_state.current_proj_id:
        st.header("✨ Crea un nuovo Progetto")
        with st.container(border=True):
            np_name = st.text_input("Nome Progetto")
            np_client = st.text_input("Cliente")
            if st.button("🆕 Crea e Apri Progetto"):
                if np_name:
                    conn = sqlite3.connect('luxia_titan.db')
                    cur = conn.cursor()
                    cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                                (st.session_state.username, np_name, np_client, datetime.now().strftime("%Y-%m-%d")))
                    new_id = cur.lastrowid
                    conn.commit()
                    st.session_state.current_proj_id = new_id
                    st.session_state.current_proj_name = np_name
                    st.rerun()
    else:
        # AREA LAVORO PROGETTO
        st.header(f"🏢 {st.session_state.current_proj_name}")
        if st.button("⬅️ Torna alla Lista"):
            st.session_state.current_proj_id = None
            st.rerun()

        t1, t2, t3 = st.tabs(["📋 Elenco Vani", "➕ Aggiungi Vano", "📄 Export Report"])

        with t1:
            conn = sqlite3.connect('luxia_titan.db')
            rooms = conn.execute("SELECT r_name, lux_avg, model, qty, img_blob, strategy FROM rooms WHERE project_id=?", 
                                (st.session_state.current_proj_id,)).fetchall()
            if rooms:
                for r in rooms:
                    with st.expander(f"📍 {r[0]}"):
                        c_text, c_img = st.columns([2,1])
                        c_text.write(f"**Risultato:** {r[1]:.0f} Lux medi")
                        c_text.write(f"**Configurazione:** {r[3]}x {r[2]}")
                        if r[5] != "N/A": c_text.info(f"AI: {r[5]}")
                        if r[4]: c_img.image(r[4], caption="Planimetria")
            else: st.info("Progetto vuoto. Aggiungi il primo vano.")

        with t2:
            with st.form("new_room", clear_on_submit=True):
                st.subheader("Dati Ambiente")
                n = st.text_input("Nome Vano")
                ca, cb, cc = st.columns(3)
                rw, rl, rh = ca.number_input("Largh. (m)", 1.0), cb.number_input("Lungh. (m)", 1.0), cc.number_input("Alt. (m)", 2.0)
                
                st.divider()
                st.write("🔦 Illuminazione")
                brand = st.selectbox("Marca", ["BEGA", "iGuzzini", "Altro"])
                mod = st.text_input("Modello/Codice")
                q = st.number_input("Quantità", 1, 500)
                
                f_img = st.file_uploader("Allega Immagine/PDF (PNG/JPG)", type=['png','jpg'])
                
                if st.form_submit_button("💾 Salva Vano"):
                    lux = (q * 3000 * 0.6) / (rw * rl)
                    strat = "N/A"
                    if st.session_state.groq_online:
                        p = f"Sei un Lighting Designer. Spiega brevemente perché {brand} {mod} è adatto a un vano di {rw*rl}mq."
                        chat = st.session_state.groq_client.chat.completions.create(messages=[{"role":"user","content":p}], model="llama-3.3-70b-versatile")
                        strat = chat.choices[0].message.content
                    
                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute('''INSERT INTO rooms (project_id, r_name, w, l, h, brand, model, qty, price, lux_avg, strategy, img_blob) 
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', 
                                 (st.session_state.current_proj_id, n, rw, rl, rh, brand, mod, q, 0, lux, strat, f_img.read() if f_img else None))
                    conn.commit()
                    st.success("Vano Salvato!")
                    st.rerun()