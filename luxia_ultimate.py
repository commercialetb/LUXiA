import streamlit as st
import sqlite3
import hashlib
import base64
import pandas as pd
from fpdf import FPDF
from datetime import datetime
from groq import Groq

# --- 1. DATABASE & AUTO-MIGRATION (Risolve l'errore OperationalError) ---
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
    
    # Controllo e aggiunta dinamica colonne mancanti
    columns = [col[1] for col in c.execute("PRAGMA table_info(rooms)").fetchall()]
    needed = {
        'price': 'REAL DEFAULT 0',
        'strategy': 'TEXT',
        'img_blob': 'BLOB',
        'h': 'REAL DEFAULT 3.0'
    }
    for col_name, col_type in needed.items():
        if col_name not in columns:
            c.execute(f"ALTER TABLE rooms ADD COLUMN {col_name} {col_type}")
    
    conn.commit()
    conn.close()

# --- 2. STATISTICHE DASHBOARD ---
def get_stats(username):
    conn = sqlite3.connect('luxia_titan.db')
    n_proj = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (username,)).fetchone()[0]
    stats = conn.execute('''SELECT COUNT(rooms.id), SUM(rooms.qty * rooms.price) 
                            FROM rooms 
                            JOIN projects ON rooms.project_id = projects.id 
                            WHERE projects.username=?''', (username,)).fetchone()
    conn.close()
    return n_proj, (stats[0] if stats[0] else 0), (stats[1] if stats[1] else 0)

# --- 3. CORE APP ---
def main():
    st.set_page_config(page_title="LUXiA Titan v5.9", layout="wide", initial_sidebar_state="expanded")
    init_db()

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None
    if 'groq_online' not in st.session_state: st.session_state.groq_online = False

    # --- ACCESSO ---
    if not st.session_state.logged_in:
        st.title("💡 LUXiA | Lighting Design Portal")
        t_l, t_r = st.tabs(["Login", "Registra Studio"])
        with t_l:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Entra"):
                h = hashlib.sha256(p.encode()).hexdigest()
                conn = sqlite3.connect('luxia_titan.db')
                res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                conn.close()
                if res:
                    st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "user_logo_b64": res[1]})
                    st.rerun()
                else: st.error("Credenziali errate.")
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("LUXiA Titan")
        
        # Gestione Logo Professionale
        if st.session_state.get('user_logo_b64'):
            st.image(base64.b64decode(st.session_state.user_logo_b64), width=150)
            if st.button("🔄 Cambia Logo Studio"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=NULL WHERE username=?", (st.session_state.username,))
                conn.commit(); conn.close()
                st.session_state.user_logo_b64 = None; st.rerun()
        else:
            up_logo = st.file_uploader("Carica Logo Studio", type=['png','jpg','jpeg'])
            if up_logo:
                b64 = base64.b64encode(up_logo.read()).decode()
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=? WHERE username=?", (b64, st.session_state.username))
                conn.commit(); conn.close()
                st.session_state.user_logo_b64 = b64; st.rerun()

        st.divider()
        
        # --- FUNZIONALITÀ AI (GROQ) ---
        st.subheader("🤖 AI Strategy Engine")
        g_key = st.text_input("Groq API Key", type="password", help="Inserisci la chiave per attivare l'AI")
        if g_key:
            try:
                st.session_state.groq_client = Groq(api_key=g_key)
                st.session_state.groq_online = True
                st.success("AI Online 🟢")
            except:
                st.error("Chiave non valida")

        st.divider()
        
        # Selezione Progetti
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel_p = st.selectbox("Archivio Progetti", ["-- Dashboard --"] + list(p_dict.keys()))
        if st.button("🚀 Apri"):
            st.session_state.current_proj_id = p_dict[sel_p] if sel_p != "-- Dashboard --" else None
            if sel_p != "-- Dashboard --": st.session_state.current_proj_name = sel_p
            st.rerun()

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False; st.rerun()

    # --- LOGICA DASHBOARD vs PROGETTO ---
    if not st.session_state.current_proj_id:
        st.title(f"Benvenuto, {st.session_state.studio_name}")
        n_p, n_v, val = get_stats(st.session_state.username)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Progetti", n_p)
        c2.metric("Vani Totali", n_v)
        c3.metric("Valore Offerte", f"€ {val:,.2f}")

        st.divider()
        col_new, col_list = st.columns([1, 2])
        with col_new:
            st.subheader("🆕 Crea Progetto")
            with st.form("new_proj"):
                name = st.text_input("Nome Cantiere")
                cli = st.text_input("Cliente")
                if st.form_submit_button("Crea e Lavora"):
                    if name:
                        conn = sqlite3.connect('luxia_titan.db')
                        cur = conn.cursor()
                        cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                                    (st.session_state.username, name, cli, datetime.now().strftime("%d/%m/%Y")))
                        st.session_state.current_proj_id = cur.lastrowid
                        st.session_state.current_proj_name = name
                        conn.commit(); conn.close()
                        st.rerun()
    else:
        # --- AREA PROGETTO ATTIVO ---
        st.header(f"🏢 Progetto: {st.session_state.current_proj_name}")
        if st.button("⬅️ Torna alla Dashboard"):
            st.session_state.current_proj_id = None; st.rerun()

        tab1, tab2 = st.tabs(["📋 Riepilogo Vani", "➕ Aggiungi Ambiente"])
        
        with tab1:
            conn = sqlite3.connect('luxia_titan.db')
            # Query aggiornata con tutte le colonne necessarie
            rooms = conn.execute("SELECT r_name, lux_avg, model, qty, img_blob, price, strategy, brand FROM rooms WHERE project_id=?", 
                                (st.session_state.current_proj_id,)).fetchall()
            conn.close()
            if rooms:
                for r in rooms:
                    with st.expander(f"📍 Ambiente: {r[0]}"):
                        c_tx, c_im = st.columns([2,1])
                        c_tx.write(f"**Calcolo:** {r[1]:.0f} Lux | **Prodotto:** {r[3]}x {r[7]} {r[2]}")
                        c_tx.write(f"**Valore Vano:** € {r[3]*r[5]:,.2f}")
                        if r[6]: st.info(f"✨ **Strategia AI:** {r[6]}")
                        if r[4]: c_im.image(r[4], use_container_width=True)
            else: st.info("Nessun vano configurato.")

        with tab2:
            st.subheader("Configurazione Tecnica e AI")
            with st.form("form_vano"):
                r_n = st.text_input("Nome Ambiente")
                c1, c2, c3 = st.columns(3)
                rw = c1.number_input("Larghezza (m)", 0.1)
                rl = c2.number_input("Lunghezza (m)", 0.1)
                rh = c3.number_input("Altezza (m)", 2.0, 10.0, 3.0)
                
                br = st.selectbox("Marca", ["BEGA", "iGuzzini", "Artemide", "Altro"])
                md = st.text_input("Codice Modello")
                qt = st.number_input("Quantità", 1)
                pr = st.number_input("Prezzo Unitario (€)", 0.0)
                
                up_f = st.file_uploader("Allega Immagine/Pianta", type=['jpg','png'])
                
                if st.form_submit_button("💾 Salva e Genera Strategia AI"):
                    lux = (qt * 3000 * 0.6) / (rw * rl)
                    strat_ai = "AI non attivata (inserisci chiave Groq nella sidebar)"
                    
                    if st.session_state.groq_online:
                        try:
                            prompt = f"Sei un lighting designer esperto. Motiva brevemente perché usare {qt} apparecchi {br} {md} in un ambiente di {rw*rl}mq è una scelta di eccellenza tecnica e design."
                            chat = st.session_state.groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="llama-3.3-70b-versatile"
                            )
                            strat_ai = chat.choices[0].message.content
                        except: strat_ai = "Errore durante la generazione AI."

                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute('''INSERT INTO rooms (project_id, r_name, w, l, h, brand, model, qty, price, lux_avg, strategy, img_blob) 
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', 
                                 (st.session_state.current_proj_id, r_n, rw, rl, rh, br, md, qt, pr, lux, strat_ai, up_f.read() if up_f else None))
                    conn.commit(); conn.close()
                    st.success("Ambiente salvato con strategia AI!")
                    st.rerun()

if __name__ == "__main__":
    main()