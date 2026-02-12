import streamlit as st
import sqlite3
import hashlib
import base64
import pandas as pd
from fpdf import FPDF
from datetime import datetime
from groq import Groq

# --- 1. DATABASE & AUTO-MIGRATION ---
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
    
    # Migrazione colonne critiche
    columns = [col[1] for col in c.execute("PRAGMA table_info(rooms)").fetchall()]
    if 'price' not in columns: c.execute("ALTER TABLE rooms ADD COLUMN price REAL DEFAULT 0")
    if 'strategy' not in columns: c.execute("ALTER TABLE rooms ADD COLUMN strategy TEXT")
    
    conn.commit()
    conn.close()

# --- 2. LOGICA STATISTICHE ---
def get_stats(username):
    conn = sqlite3.connect('luxia_titan.db')
    n_proj = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (username,)).fetchone()[0]
    stats = conn.execute('''SELECT COUNT(rooms.id), SUM(rooms.qty * rooms.price) 
                            FROM rooms 
                            JOIN projects ON rooms.project_id = projects.id 
                            WHERE projects.username=?''', (username,)).fetchone()
    conn.close()
    return n_proj, (stats[0] if stats[0] else 0), (stats[1] if stats[1] else 0)

# --- 3. APP PRINCIPALE ---
def main():
    st.set_page_config(page_title="LUXiA Titan v5.8", layout="wide", initial_sidebar_state="expanded")
    init_db()

    # Inizializzazione Session State
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None
    if 'username' not in st.session_state: st.session_state.username = None

    # --- LOGIN / REGISTRAZIONE ---
    if not st.session_state.logged_in:
        st.title("💡 LUXiA | Lighting Design Portal")
        t_login, t_reg = st.tabs(["Accedi", "Registra Studio"])
        
        with t_login:
            u = st.text_input("Username", key="l_u")
            p = st.text_input("Password", type="password", key="l_p")
            if st.button("Entra"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                conn.close()
                if res:
                    st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "user_logo_b64": res[1]})
                    st.rerun()
                else: st.error("Credenziali errate.")
        
        with t_reg:
            nu = st.text_input("Nuovo User")
            np = st.text_input("Nuova Pass", type="password")
            ns = st.text_input("Nome Studio")
            if st.button("Crea Account"):
                h = hashlib.sha256(np.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                try:
                    conn.execute("INSERT INTO users (username, password, studio_name) VALUES (?,?,?)", (nu, h, ns))
                    conn.commit()
                    st.success("Studio registrato! Accedi ora.")
                except: st.error("Username occupato.")
                finally: conn.close()
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("LUXiA Titan")
        if st.session_state.get('user_logo_b64'):
            st.image(base64.b64decode(st.session_state.user_logo_b64), width=150)
            if st.button("🔄 Cambia Logo"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=NULL WHERE username=?", (st.session_state.username,))
                conn.commit(); conn.close()
                st.session_state.user_logo_b64 = None; st.rerun()
        else:
            up = st.file_uploader("Carica Logo", type=['png','jpg','jpeg'])
            if up:
                b64 = base64.b64encode(up.read()).decode()
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=? WHERE username=?", (b64, st.session_state.username))
                conn.commit(); conn.close()
                st.session_state.user_logo_b64 = b64; st.rerun()

        st.divider()
        g_key = st.text_input("Groq API Key", type="password")
        if g_key:
            st.session_state.groq_client = Groq(api_key=g_key)
            st.caption("🟢 AI Attiva")

        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel_p = st.selectbox("I Tuoi Lavori", ["-- Dashboard --"] + list(p_dict.keys()))
        if st.button("🚀 Vai"):
            if sel_p == "-- Dashboard --":
                st.session_state.current_proj_id = None
            else:
                st.session_state.current_proj_id = p_dict[sel_p]
                st.session_state.current_proj_name = sel_p
            st.rerun()

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False; st.rerun()

    # --- LOGICA DASHBOARD vs PROGETTO ---
    if not st.session_state.current_proj_id:
        # --- SCHERMATA DASHBOARD ---
        st.title(f"Benvenuto nel tuo Studio, {st.session_state.studio_name}")
        n_p, n_v, val = get_stats(st.session_state.username)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Progetti Attivi", n_p)
        c2.metric("Vani Illuminati", n_v)
        c3.metric("Valore Offerte (€)", f"{val:,.2f}")

        st.divider()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("🆕 Nuovo Cantiere")
            with st.form("new_p_form"):
                name = st.text_input("Nome Progetto")
                cli = st.text_input("Cliente")
                if st.form_submit_button("Crea e Apri"):
                    if name:
                        conn = sqlite3.connect('luxia_titan.db')
                        cur = conn.cursor()
                        cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                                    (st.session_state.username, name, cli, datetime.now().strftime("%d/%m/%Y")))
                        st.session_state.current_proj_id = cur.lastrowid
                        st.session_state.current_proj_name = name
                        conn.commit(); conn.close()
                        st.rerun()
        with col_b:
            st.subheader("📂 Progetti Recenti")
            if plist:
                for p_n in list(p_dict.keys())[-5:]:
                    st.text(f"• {p_n}")
            else: st.info("Nessun progetto trovato.")

    else:
        # --- SCHERMATA OPERATIVA PROGETTO ---
        st.header(f"🏢 Progetto: {st.session_state.current_proj_name}")
        if st.button("⬅️ Torna alla Dashboard"):
            st.session_state.current_proj_id = None; st.rerun()

        tab1, tab2 = st.tabs(["📋 Riepilogo Vani", "➕ Aggiungi Ambiente"])
        
        with tab1:
            conn = sqlite3.connect('luxia_titan.db')
            rooms = conn.execute("SELECT r_name, lux_avg, model, qty, img_blob, price FROM rooms WHERE project_id=?", 
                                (st.session_state.current_proj_id,)).fetchall()
            conn.close()
            if rooms:
                for r in rooms:
                    with st.expander(f"Ambiente: {r[0]}"):
                        col_tx, col_im = st.columns([2,1])
                        col_tx.write(f"**Illuminamento:** {r[1]:.0f} Lux")
                        col_tx.write(f"**Apparecchio:** {r[3]}x {r[2]}")
                        col_tx.write(f"**Valore:** € {r[3]*r[5] if r[5] else 0:,.2f}")
                        if r[4]: col_im.image(r[4], use_container_width=True)
            else: st.info("Nessun vano creato per questo progetto.")

        with tab2:
            st.subheader("Inserimento Dati Tecnici")
            with st.form("add_room_form"):
                r_n = st.text_input("Nome Vano (es. Hall)")
                c1, c2 = st.columns(2)
                rw = c1.number_input("Larghezza (m)", 0.1)
                rl = c2.number_input("Lunghezza (m)", 0.1)
                
                brand = st.selectbox("Fornitore", ["BEGA", "iGuzzini", "Artemide", "Altro"])
                model = st.text_input("Codice/Modello")
                qty = st.number_input("Q.tà", 1)
                price = st.number_input("Prezzo Unitario (€)", 0.0)
                
                f_img = st.file_uploader("Foto/Planimetria", type=['jpg','png'])
                
                if st.form_submit_button("💾 Salva e Calcola"):
                    # Calcolo semplificato (Lumen metodo dei lumen)
                    lux = (qty * 3200 * 0.6) / (rw * rl)
                    img_data = f_img.read() if f_img else None
                    
                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute('''INSERT INTO rooms (project_id, r_name, w, l, brand, model, qty, price, lux_avg, img_blob) 
                                    VALUES (?,?,?,?,?,?,?,?,?,?)''', 
                                 (st.session_state.current_proj_id, r_n, rw, rl, brand, model, qty, price, lux, img_data))
                    conn.commit(); conn.close()
                    st.success("Vano aggiunto con successo!")
                    st.rerun()

if __name__ == "__main__":
    main()