import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sqlite3
import hashlib
import base64
import io
from fpdf import FPDF
from datetime import datetime

# --- CONFIGURAZIONE DATABASE ---
def init_db():
    conn = sqlite3.connect('luxia_data.db')
    c = conn.cursor()
    # Tabella Utenti
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT)''')
    # Tabella Progetti
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, 
                  client TEXT, brand TEXT, area_w REAL, area_l REAL, area_h REAL, 
                  model TEXT, qty INTEGER, price REAL, total REAL, lux_avg REAL)''')
    # Utente di test (se non esiste)
    hashed_pw = hashlib.sha256("bega2024".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?)", 
              ('admin', hashed_pw, 'Arch. Antonio Affusto'))
    conn.commit()
    conn.close()

def save_project_to_db(data):
    conn = sqlite3.connect('luxia_data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO projects (username, p_name, client, brand, area_w, area_l, area_h, model, qty, price, total, lux_avg) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', data)
    conn.commit()
    conn.close()

def get_user_projects(username):
    conn = sqlite3.connect('luxia_data.db')
    c = conn.cursor()
    c.execute("SELECT p_name FROM projects WHERE username = ?", (username,))
    projects = c.fetchall()
    conn.close()
    return [p[0] for p in projects]

# --- PDF GENERATOR ---
class LuxiaPDF(FPDF):
    def header(self):
        if 'user_logo' in st.session_state and st.session_state.user_logo:
            # Salvataggio temporaneo del logo per FPDF
            with open("temp_logo.png", "wb") as f:
                f.write(st.session_state.user_logo.getbuffer())
            self.image("temp_logo.png", 10, 8, 25)
        
        self.set_font('Arial', 'B', 12)
        self.set_xy(40, 10)
        self.cell(0, 5, st.session_state.studio_name.upper(), ln=True)
        self.set_font('Arial', '', 9)
        self.set_x(40)
        self.cell(0, 5, f"Progetto realizzato con LUXiA", ln=True)
        self.ln(20)

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="LUXiA", layout="wide")
    init_db()

    # --- SESSION STATE ---
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'studio_name' not in st.session_state: st.session_state.studio_name = ""

    # --- LOGIN SCREEN ---
    if not st.session_state.logged_in:
        st.title("💡 LUXiA | Accesso")
        col1, _ = st.columns([1, 2])
        with col1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Entra"):
                hashed = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_data.db')
                c = conn.cursor()
                c.execute("SELECT studio_name FROM users WHERE username=? AND password=?", (u, hashed))
                res = c.fetchone()
                if res:
                    st.session_state.logged_in = True
                    st.session_state.user = u
                    st.session_state.studio_name = res[0]
                    st.rerun()
                else: st.error("Accesso negato")
        return

    # --- SIDEBAR (ARCHIVIO E LOGO) ---
    with st.sidebar:
        st.title("LUXiA")
        st.write(f"Utente: **{st.session_state.user}**")
        st.session_state.studio_name = st.text_input("Nome Studio", st.session_state.studio_name)
        logo = st.file_uploader("Carica Logo Studio (PNG)", type=['png'])
        if logo: st.session_state.user_logo = logo
        
        st.divider()
        st.subheader("📁 Archivio Progetti")
        saved_p = get_user_projects(st.session_state.user)
        st.selectbox("Richiama Progetto", ["-- Seleziona --"] + saved_p)
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📂 Input Progetto", "📐 Calcolo & Listino", "📄 Report"])

    with tab1:
        st.header("1. Definizione Progetto & Area")
        c1, c2 = st.columns(2)
        with c1:
            p_name = st.text_input("Nome Progetto")
            client = st.text_input("Cliente")
        with c2:
            brand = st.text_input("Azienda / Mix Brand")
            
        up_file = st.file_uploader("Carica Planimetria", type=['jpg', 'png', 'pdf'])
        if up_file:
            st.image(up_file, use_container_width=True)
            st.info("🎯 Definisci l'area di intervento (Coordinate in metri)")
            col_a, col_b, col_c = st.columns(3)
            aw = col_a.number_input("Larghezza Area (m)", 1.0, 500.0, 10.0)
            al = col_b.number_input("Lunghezza Area (m)", 1.0, 500.0, 20.0)
            ah = col_c.number_input("Altezza Soffitto (m)", 2.0, 20.0, 3.0)
            
            if st.button("Salva Area"):
                st.session_state.project = {"name": p_name, "client": client, "brand": brand, "w": aw, "l": al, "h": ah}
                st.success("Area salvata correttamente.")

    with tab2:
        if 'project' in st.session_state:
            st.header("2. Selezione Apparecchi & Calcolo")
            p = st.session_state.project
            
            with st.expander("Configurazione Apparecchio"):
                c_a, c_b, c_c = st.columns(3)
                model = c_a.text_input("Modello/Codice", "BEGA 50 998.1")
                qty = c_b.number_input("Quantità", 1, 1000, 1)
                price = c_c.number_input("Prezzo Listino Unitario (€)", 0.0, 10000.0, 100.0)
                totale = qty * price
                st.metric("Totale Fornitura", f"{totale} €")

            if st.button("Esegui Calcolo Illuminotecnico"):
                # Simulazione Lux
                lux_avg = 500 # Valore simulato
                st.session_state.calc_res = {"model": model, "qty": qty, "price": price, "total": totale, "lux": lux_avg}
                
                # Salvataggio nel DB SQLite
                db_data = (st.session_state.user, p['name'], p['client'], p['brand'], 
                           p['w'], p['l'], p['h'], model, qty, price, totale, lux_avg)
                save_project_to_db(db_data)
                st.success("Calcolo eseguito e progetto archiviato in SQLite!")
                
                fig = go.Figure(data=[go.Surface(z=np.random.rand(10,10)*600)])
                st.plotly_chart(fig)

    with tab3:
        if 'calc_res' in st.session_state:
            st.header("3. Esportazione Relazione")
            if st.button("Genera PDF Finale"):
                pdf = LuxiaPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, f"Progetto: {st.session_state.project['name']}", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.cell(0, 7, f"Cliente: {st.session_state.project['client']}", ln=True)
                pdf.cell(0, 7, f"Azienda/Brand: {st.session_state.project['brand']}", ln=True)
                pdf.ln(10)
                
                # Tabella Listino
                pdf.set_fill_color(240, 240, 240)
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(90, 10, "Modello Apparecchio", 1, 0, 'C', True)
                pdf.cell(30, 10, "Quantita", 1, 0, 'C', True)
                pdf.cell(30, 10, "Prezzo Unit.", 1, 0, 'C', True)
                pdf.cell(40, 10, "Totale", 1, 1, 'C', True)
                
                pdf.set_font("Arial", '', 11)
                res = st.session_state.calc_res
                pdf.cell(90, 10, res['model'], 1)
                pdf.cell(30, 10, str(res['qty']), 1, 0, 'C')
                pdf.cell(30, 10, f"{res['price']} EUR", 1, 0, 'C')
                pdf.cell(40, 10, f"{res['total']} EUR", 1, 1, 'C')
                
                pdf.ln(10)
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 10, f"Risultato Illuminotecnico Medio: {res['lux']} Lux", ln=True)
                
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button("📥 Scarica Relazione LUXiA", data=pdf_bytes, file_name=f"Relazione_{st.session_state.project['name']}.pdf")

if __name__ == "__main__":
    main()