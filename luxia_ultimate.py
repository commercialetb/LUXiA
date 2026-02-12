import streamlit as st
import sqlite3
import hashlib
import base64
import pandas as pd
from fpdf import FPDF
from datetime import datetime
from groq import Groq

# --- 1. DATABASE & MIGRATION ---
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
    
    # Check migrazione per prezzi e strategie
    columns = [col[1] for col in c.execute("PRAGMA table_info(rooms)").fetchall()]
    if 'price' not in columns: c.execute("ALTER TABLE rooms ADD COLUMN price REAL DEFAULT 0")
    if 'strategy' not in columns: c.execute("ALTER TABLE rooms ADD COLUMN strategy TEXT")
    
    conn.commit()
    conn.close()

# --- 2. FUNZIONI DI CALCOLO DASHBOARD ---
def get_stats(username):
    conn = sqlite3.connect('luxia_titan.db')
    # Numero progetti
    n_proj = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (username,)).fetchone()[0]
    # Numero vani totali e valore totale
    stats = conn.execute('''SELECT COUNT(rooms.id), SUM(rooms.qty * rooms.price) 
                            FROM rooms 
                            JOIN projects ON rooms.project_id = projects.id 
                            WHERE projects.username=?''', (username,)).fetchone()
    conn.close()
    return n_proj, stats[0] or 0, stats[1] or 0

# --- 3. APP PRINCIPALE ---
def main():
    st.set_page_config(page_title="LUXiA Titan", layout="wide", initial_sidebar_state="expanded")
    init_db()

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None

    if not st.session_state.logged_in:
        # --- LOGIN PAGE (Invariata) ---
        st.title("💡 LUXiA | Accesso Studio")
        t1, t2 = st.tabs(["Login", "Registra"])
        with t1:
            u = st.text_input("User")
            p = st.text_input("Pass", type="password")
            if st.button("Accedi"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                conn.close()
                if res:
                    st.session_state.update({"logged_in":True, "username":u, "studio_name":res[0], "user_logo_b64":res[1]})
                    st.rerun()
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("LUXiA Titan")
        if st.session_state.user_logo_b64:
            st.image(base64.b64decode(st.session_state.user_logo_b64), width=120)
            if st.button("🔄 Cambia Logo"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=NULL WHERE username=?", (st.session_state.username,))
                conn.commit(); conn.close()
                st.session_state.user_logo_b64 = None; st.rerun()
        
        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        sel_p = st.selectbox("Apri Progetto", ["-- Dashboard --"] + list(p_dict.keys()))
        
        if st.button("🚀 Vai") or (sel_p == "-- Dashboard --" and st.session_state.current_proj_id is not None):
            if sel_p == "-- Dashboard --":
                st.session_state.current_proj_id = None
            else:
                st.session_state.current_proj_id = p_dict[sel_p]
                st.session_state.current_proj_name = sel_p
            st.rerun()

        if st.button("🚪 Esci"):
            st.session_state.logged_in = False; st.rerun()

    # --- MAIN CONTENT ---
    if not st.session_state.current_proj_id:
        # --- DASHBOARD INIZIALE ---
        st.title(f"👋 Benvenuto, {st.session_state.studio_name}")
        n_p, n_v, val = get_stats(st.session_state.username)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Progetti Totali", n_p)
        c2.metric("Vani Progettati", n_v)
        c3.metric("Valore Offerte", f"€ {val:,.2f}")

        st.divider()
        
        col_new, col_list = st.columns([1, 2])
        
        with col_new:
            st.subheader("🆕 Nuovo Progetto")
            with st.form("new_p"):
                name = st.text_input("Nome Cantiere")
                cli = st.text_input("Cliente")
                if st.form_submit_button("Crea Progetto"):
                    if name:
                        conn = sqlite3.connect('luxia_titan.db')
                        cur = conn.cursor()
                        cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                                    (st.session_state.username, name, cli, datetime.now().strftime("%d/%m/%Y")))
                        st.session_state.current_proj_id = cur.lastrowid
                        st.session_state.current_proj_name = name
                        conn.commit(); conn.close()
                        st.rerun()

        with col_list:
            st.subheader("📂 Ultimi Progetti")
            if plist:
                df = pd.DataFrame(plist, columns=['ID', 'Nome Progetto'])
                st.table(df[['Nome Progetto']])
            else:
                st.info("Nessun progetto in archivio.")

    else:
        # --- AREA LAVORO (Vani) ---
        st.header(f"🏢 {st.session_state.current_proj_name}")
        if st.button("⬅️ Torna alla Dashboard"):
            st.session_state.current_proj_id = None; st.rerun()
        
        # ... (Qui restano le Tab Riepilogo e Aggiungi Vano come prima) ...
        st.write("Area operativa del progetto attiva.")