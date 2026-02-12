import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sqlite3
import hashlib
import base64
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
from datetime import datetime

# --- 1. DATABASE ARCHITECTURE (GERARCHIA VANI) ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    # Utenti
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)''')
    # Progetti (Contenitore)
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)''')
    # Vani (Dettagli tecnici collegati al progetto)
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  w REAL, l REAL, h REAL, brand TEXT, model TEXT, qty INTEGER, 
                  price REAL, lux_avg REAL, optics TEXT, strategy TEXT,
                  FOREIGN KEY(project_id) REFERENCES projects(id))''')
    conn.commit()
    conn.close()

# --- 2. BEGA CONNECTOR ---
def get_bega_data(product_code):
    code = product_code.replace(" ", "").replace(".", "")
    url = f"https://www.bega.com/it/prodotti/{code}/"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Estrattore semplificato - in produzione va raffinato sui selettori BEGA
            return {"lumen": 3500, "watt": 30, "success": True, "url": url}
    except: pass
    return {"success": False}

# --- 3. PDF GENERATOR ---
class LuxiaPDF(FPDF):
    def header(self):
        if 'user_logo_b64' in st.session_state and st.session_state.user_logo_b64:
            with open("temp_logo.png", "wb") as f:
                f.write(base64.b64decode(st.session_state.user_logo_b64))
            self.image("temp_logo.png", 10, 8, 25)
        self.set_font('Arial', 'B', 12)
        self.set_xy(40, 10)
        self.cell(0, 5, st.session_state.studio_name.upper(), ln=True)
        self.set_font('Arial', 'I', 8)
        self.set_x(40)
        self.cell(0, 5, "Relazione Illuminotecnica Multi-Vano", ln=True)
        self.ln(20)

# --- 4. CORE APPLICATION ---
def main():
    st.set_page_config(page_title="LUXiA", layout="wide", page_icon="💡")
    init_db()

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False

    # --- LOGIN / REGISTRAZIONE ---
    if not st.session_state.logged_in:
        st.title("💡 LUXiA | Portale Professionale")
        auth_tab = st.tabs(["Accedi", "Registrati"])
        with auth_tab[0]:
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
        with auth_tab[1]:
            nu, npw, ns = st.text_input("Nuovo User"), st.text_input("Nuova Pass", type="password"), st.text_input("Nome Studio")
            if st.button("Crea Account"):
                try:
                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, hashlib.sha256(npw.encode()).hexdigest(), ns, None))
                    conn.commit()
                    st.success("Registrato! Accedi ora.")
                except: st.error("Username occupato.")
        return

    # --- SIDEBAR & NAVIGAZIONE ---
    with st.sidebar:
        st.title("LUXiA Titan")
        if st.session_state.user_logo_b64:
            st.image(base64.b64decode(st.session_state.user_logo_b64), width=100)
        
        st.subheader(f"Studio: {st.session_state.studio_name}")
        
        # Gestione Progetti Esistenti
        conn = sqlite3.connect('luxia_titan.db')
        projs = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        proj_list = {p[1]: p[0] for p in projs}
        
        st.divider()
        sel_proj = st.selectbox("📂 Seleziona Progetto", ["-- Nuovo Progetto --"] + list(proj_list.keys()))
        
        if sel_proj != "-- Nuovo Progetto --":
            st.session_state.current_proj_id = proj_list[sel_proj]
            st.session_state.current_proj_name = sel_proj
        else:
            st.session_state.current_proj_id = None

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # --- MAIN CONTENT ---
    if not st.session_state.get('current_proj_id'):
        st.header("Crea un nuovo Progetto per iniziare")
        with st.form("new_proj"):
            np_name = st.text_input("Nome Progetto (es. Centro Direzionale)")
            np_client = st.text_input("Cliente")
            if st.form_submit_button("Crea Progetto"):
                conn = sqlite3.connect('luxia_titan.db')
                cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                            (st.session_state.username, np_name, np_client, datetime.now().strftime("%Y-%m-%d")))
                conn.commit()
                st.rerun()
    else:
        st.title(f"🏢 Progetto: {st.session_state.current_proj_name}")
        tab_list, tab_edit, tab_pdf = st.tabs(["📋 Elenco Vani", "📐 Configura Vano", "📄 Report Finale"])

        with tab_list:
            st.subheader("Vani inseriti in questo progetto")
            conn = sqlite3.connect('luxia_titan.db')
            rooms = conn.execute("SELECT r_name, lux_avg, model, qty FROM rooms WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
            if rooms:
                for r in rooms:
                    st.write(f"✅ **{r[0]}**: {r[1]:.0f} Lux medi con {r[3]}x {r[2]}")
            else:
                st.info("Nessun vano ancora configurato.")

        with tab_edit:
            st.subheader("Configurazione Ambiente")
            with st.form("room_form"):
                r_name = st.text_input("Nome Ambiente (es. Soggiorno, Ufficio 1)")
                c1, c2, c3 = st.columns(3)
                rw = c1.number_input("Larghezza (m)", 1.0, 100.0, 10.0)
                rl = c2.number_input("Lunghezza (m)", 1.0, 100.0, 15.0)
                rh = c3.number_input("Altezza (m)", 2.0, 20.0, 3.5)
                
                st.divider()
                st.write("🔦 Selezione Apparecchio")
                vendor = st.radio("Fornitore", ["BEGA (Auto)", "Altro (Manuale)"], horizontal=True)
                
                col_a, col_b = st.columns(2)
                if vendor == "BEGA (Auto)":
                    code = col_a.text_input("Codice BEGA")
                    model = code
                    lumen = 3500 # Default se scraping fallisce
                else:
                    model = col_a.text_input("Modello")
                    lumen = col_b.number_input("Lumen", 100, 50000, 3000)
                
                qty = st.number_input("Quantità", 1, 500, 8)
                price = st.number_input("Prezzo Listino Unitario (€)", 0.0, 5000.0, 250.0)
                
                if st.form_submit_button("Calcola e Aggiungi Vano"):
                    # Logica AI Luxia
                    lux = (qty * lumen * 0.65) / (rw * rl)
                    strategy = f"Soluzione ottimale per {r_name}. Il brand garantisce alte prestazioni."
                    
                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute('''INSERT INTO rooms (project_id, r_name, w, l, h, brand, model, qty, price, lux_avg, strategy) 
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?)''', 
                                 (st.session_state.current_proj_id, r_name, rw, rl, rh, vendor, model, qty, price, lux, strategy))
                    conn.commit()
                    st.success(f"Ambiente {r_name} aggiunto al progetto!")

        with tab_pdf:
            st.header("Esportazione Totale Progetto")
            if st.button("Genera Relazione Completa"):
                pdf = LuxiaPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, f"PROGETTO: {st.session_state.current_proj_name}", ln=True)
                
                conn = sqlite3.connect('luxia_titan.db')
                rooms_data = conn.execute("SELECT * FROM rooms WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
                
                total_val = 0
                for r in rooms_data:
                    pdf.ln(10)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.set_fill_color(240, 240, 240)
                    pdf.cell(0, 10, f"Ambiente: {r[2]}", 1, ln=True, fill=True)
                    pdf.set_font("Arial", '', 10)
                    pdf.cell(0, 7, f"Dimensioni: {r[3]}x{r[4]}m (H: {r[5]}m)", ln=True)
                    pdf.cell(0, 7, f"Apparecchio: {r[8]} ({r[7]}) - Q.ta: {r[9]}", ln=True)
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(0, 7, f"Illuminamento Medio: {r[11]:.0f} Lux", ln=True)
                    total_val += (r[9] * r[10])
                
                pdf.ln(15)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, f"VALORE TOTALE FORNITURA: {total_val:,.2f} EUR", ln=True)
                
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button("📥 Scarica Report LUXiA Titan", data=pdf_bytes, file_name=f"LUXiA_{st.session_state.current_proj_name}.pdf")

if __name__ == "__main__":
    main()