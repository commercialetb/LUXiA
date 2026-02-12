import streamlit as st
import sqlite3
import hashlib
import base64
from fpdf import FPDF
from datetime import datetime
from groq import Groq

# --- 1. INIZIALIZZAZIONE DATABASE ---
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
                  price REAL, lux_avg REAL, strategy TEXT, img_blob BLOB)''')
    conn.commit()
    conn.close()

# --- 2. LOGICA PDF ---
class LuxiaPDF(FPDF):
    def header(self):
        if st.session_state.get('user_logo_b64'):
            try:
                logo_data = base64.b64decode(st.session_state.user_logo_b64)
                with open("temp_logo.png", "wb") as f:
                    f.write(logo_data)
                self.image("temp_logo.png", 10, 8, 25)
            except: pass
        self.set_font('Arial', 'B', 12)
        self.set_xy(40, 10)
        self.cell(0, 5, st.session_state.get('studio_name', 'STUDIO').upper(), ln=True)
        self.ln(20)

# --- 3. APP PRINCIPALE ---
def main():
    st.set_page_config(page_title="LUXiA Titan", layout="wide", initial_sidebar_state="expanded")
    init_db()

    # Inizializzazione sicura del Session State
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None
    if 'user_logo_b64' not in st.session_state: st.session_state.user_logo_b64 = None

    # --- SCHERMATA LOGIN ---
    if not st.session_state.logged_in:
        st.title("💡 LUXiA | Accesso")
        tab_log, tab_reg = st.tabs(["Login", "Registra Studio"])
        
        with tab_log:
            u = st.text_input("Username", key="l_user")
            p = st.text_input("Password", type="password", key="l_pass")
            if st.button("Accedi"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                conn.close()
                if res:
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.session_state.studio_name = res[0]
                    st.session_state.user_logo_b64 = res[1]
                    st.rerun()
                else: st.error("Credenziali non corrette")
        
        with tab_reg:
            nu = st.text_input("Nuovo Username", key="r_user")
            np = st.text_input("Nuova Password", type="password", key="r_pass")
            ns = st.text_input("Nome Studio", key="r_studio")
            if st.button("Registrati"):
                h = hashlib.sha256(np.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                try:
                    conn.execute("INSERT INTO users (username, password, studio_name) VALUES (?,?,?)", (nu, h, ns))
                    conn.commit()
                    st.success("Registrazione completata! Effettua il login.")
                except: st.error("Username già esistente.")
                conn.close()
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("LUXiA Titan")
        
        # Gestione Logo Smart
        if st.session_state.user_logo_b64:
            st.image(base64.b64decode(st.session_state.user_logo_b64), width=120)
            if st.button("🔄 Cambia Logo"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=NULL WHERE username=?", (st.session_state.username,))
                conn.commit()
                conn.close()
                st.session_state.user_logo_b64 = None
                st.rerun()
        else:
            up = st.file_uploader("Carica Logo Studio", type=['png','jpg','jpeg'])
            if up:
                b64 = base64.b64encode(up.read()).decode()
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=? WHERE username=?", (b64, st.session_state.username))
                conn.commit()
                conn.close()
                st.session_state.user_logo_b64 = b64
                st.rerun()

        st.divider()
        
        # Groq Setup
        groq_key = st.text_input("Groq API Key", type="password")
        if groq_key:
            st.session_state.groq_client = Groq(api_key=groq_key)
            st.caption("🟢 AI Attiva")

        st.divider()
        
        # Selezione Progetti
        conn = sqlite3.connect('luxia_titan.db')
        projs = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in projs}
        
        sel_p = st.selectbox("Apri Progetto", ["-- Seleziona --"] + list(p_dict.keys()))
        if st.button("🚀 Vai al Progetto"):
            if sel_p != "-- Seleziona --":
                st.session_state.current_proj_id = p_dict[sel_p]
                st.session_state.current_proj_name = sel_p
                st.rerun()

        if st.button("🚪 Esci"):
            st.session_state.logged_in = False
            st.rerun()

    # --- CONTENUTO PRINCIPALE ---
    if not st.session_state.current_proj_id:
        st.header("✨ Nuovo Progetto")
        with st.form("new_proj"):
            name = st.text_input("Nome Progetto")
            client = st.text_input("Cliente")
            if st.form_submit_button("🆕 Crea e Apri"):
                if name:
                    conn = sqlite3.connect('luxia_titan.db')
                    cur = conn.cursor()
                    cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                                (st.session_state.username, name, client, datetime.now().strftime("%d/%m/%Y")))
                    new_id = cur.lastrowid
                    conn.commit()
                    conn.close()
                    st.session_state.current_proj_id = new_id
                    st.session_state.current_proj_name = name
                    st.rerun()
    else:
        st.header(f"🏢 Progetto: {st.session_state.current_proj_name}")
        if st.button("⬅️ Lista Progetti"):
            st.session_state.current_proj_id = None
            st.rerun()

        t1, t2, t3 = st.tabs(["📋 Riepilogo", "➕ Aggiungi Vano", "📄 Report"])

        with t1:
            conn = sqlite3.connect('luxia_titan.db')
            rooms = conn.execute("SELECT r_name, lux_avg, model, qty, img_blob FROM rooms WHERE project_id=?", 
                                (st.session_state.current_proj_id,)).fetchall()
            conn.close()
            if rooms:
                for r in rooms:
                    with st.expander(f"Vano: {r[0]}"):
                        st.write(f"Illuminamento: {r[1]:.0f} Lux | Apparecchio: {r[3]}x {r[2]}")
                        if r[4]: st.image(r[4], width=250)
            else: st.info("Ancora nessun vano. Usa la tab 'Aggiungi Vano'.")

        with t2:
            with st.form("add_room_final"):
                r_n = st.text_input("Nome Vano")
                ca, cb, cc = st.columns(3)
                rw = ca.number_input("Larghezza (m)", 0.1)
                rl = cb.number_input("Lunghezza (m)", 0.1)
                rh = cc.number_input("Altezza (m)", 2.0)
                
                br = st.selectbox("Marca", ["BEGA", "iGuzzini", "Altro"])
                md = st.text_input("Codice")
                qt = st.number_input("Quantità", 1)
                
                up_img = st.file_uploader("Allega Immagine", type=['jpg','png'])
                
                if st.form_submit_button("💾 Salva"):
                    lux = (qt * 3000 * 0.6) / (rw * rl)
                    img_data = up_img.read() if up_img else None
                    
                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute('''INSERT INTO rooms (project_id, r_name, w, l, h, brand, model, qty, lux_avg, img_blob) 
                                    VALUES (?,?,?,?,?,?,?,?,?,?)''', 
                                 (st.session_state.current_proj_id, r_n, rw, rl, rh, br, md, qt, lux, img_data))
                    conn.commit()
                    conn.close()
                    st.success("Vano aggiunto!")
                    st.rerun()

if __name__ == "__main__":
    main()