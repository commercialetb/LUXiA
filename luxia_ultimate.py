import streamlit as st
import sqlite3
import hashlib
import base64
import pandas as pd
from datetime import datetime
from groq import Groq
from fpdf import FPDF

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="LUXiA Titan v6.3", layout="wide", initial_sidebar_state="expanded")

# --- 1. ARCHITETTURA DATABASE ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS project_docs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, filename TEXT, file_type TEXT, file_blob BLOB, upload_date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  w REAL, l REAL, h REAL, brand TEXT, model TEXT, qty INTEGER, 
                  price REAL, lux_avg REAL, strategy TEXT, img_blob BLOB)''')
    conn.commit()
    conn.close()

def get_dashboard_stats(username):
    conn = sqlite3.connect('luxia_titan.db')
    n_proj = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (username,)).fetchone()[0]
    stats = conn.execute('''SELECT COUNT(rooms.id), SUM(rooms.qty * rooms.price) 
                            FROM rooms JOIN projects ON rooms.project_id = projects.id 
                            WHERE projects.username=?''', (username,)).fetchone()
    conn.close()
    return n_proj, (stats[0] if stats[0] else 0), (stats[1] if stats[1] else 0)

def main():
    init_db()
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None
    if 'groq_online' not in st.session_state: st.session_state.groq_online = False

    # --- LOGIN SYSTEM ---
    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center; color: #FFD700;'>LUXiA TITAN</h1>", unsafe_allow_html=True)
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
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### 🏛️ STUDIO PANEL")
        if st.session_state.get('user_logo_b64'):
            st.image(base64.b64decode(st.session_state.user_logo_b64), use_container_width=True)
        
        st.divider()
        st.markdown("### 🧠 AI ENGINE")
        gk = st.text_input("Groq API Key", type="password")
        if gk:
            st.session_state.groq_client = Groq(api_key=gk)
            st.session_state.groq_online = True
            st.success("AI ONLINE")

        st.divider()
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        sel = st.selectbox("I Tuoi Progetti", ["-- DASHBOARD --"] + list(p_dict.keys()))
        if st.button("🚀 APRI"):
            st.session_state.current_proj_id = p_dict[sel] if sel != "-- DASHBOARD --" else None
            if sel != "-- DASHBOARD --": st.session_state.current_proj_name = sel
            st.rerun()

        st.divider()
        st.markdown("<div style='font-size: 10px; color: grey;'><b>LUXiA™ TRADEMARK</b><br>Patent Pending 2024-2026<br>All Rights Reserved.</div>", unsafe_allow_html=True)

    # --- LOGICA DASHBOARD / PROGETTO ---
    if not st.session_state.current_proj_id:
        st.title(f"Dashboard {st.session_state.studio_name}")
        np, nv, val = get_dashboard_stats(st.session_state.username)
        k1, k2, k3 = st.columns(3)
        k1.metric("Progetti", np)
        k2.metric("Vani", nv)
        k3.metric("Budget Totale", f"€ {val:,.2f}")
        
        with st.form("new_p"):
            st.subheader("🆕 Nuovo Cantiere")
            name = st.text_input("Nome Progetto")
            cli = st.text_input("Cliente")
            if st.form_submit_button("Crea"):
                conn = sqlite3.connect('luxia_titan.db')
                cur = conn.cursor()
                cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                            (st.session_state.username, name, cli, datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); conn.close()
                st.rerun()
    else:
        st.header(f"🏢 Cantiere: {st.session_state.current_proj_name}")
        if st.button("⬅️ Dashboard"):
            st.session_state.current_proj_id = None; st.rerun()

        # TABS (SISTEMATE)
        t_docs, t_ai, t_rep = st.tabs(["📂 1. Documenti & CAD", "💡 2. Vani & AI Strategy", "📄 3. Report Finale"])

        with t_docs:
            st.subheader("📂 Documenti Tecnici")
            f_doc = st.file_uploader("Carica DWG, PDF o Immagini", type=['pdf', 'dwg', 'dxf', 'jpg', 'png'])
            if f_doc and st.button("💾 Salva in Archivio"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("INSERT INTO project_docs (project_id, filename, file_blob, upload_date) VALUES (?,?,?,?)",
                             (st.session_state.current_proj_id, f_doc.name, f_doc.read(), datetime.now().strftime("%d/%m/%Y")))
                conn.commit(); conn.close(); st.success("File salvato!"); st.rerun()
            
            conn = sqlite3.connect('luxia_titan.db')
            docs = conn.execute("SELECT filename, file_blob, id FROM project_docs WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
            conn.close()
            for d in docs:
                c1, c2 = st.columns([4,1])
                c1.write(f"📄 {d[0]}")
                c2.download_button("Scarica", d[1], file_name=d[0], key=f"file_{d[2]}")

        with t_ai:
            st.subheader("💡 Gestione Vani & Strategia AI")
            with st.form("room_f"):
                rn = st.text_input("Nome Ambiente")
                c1, c2 = st.columns(2)
                w, l = c1.number_input("Larghezza", 0.1), c2.number_input("Lunghezza", 0.1)
                br = st.selectbox("Brand", ["BEGA", "iGuzzini", "Artemide", "Flos"])
                md = st.text_input("Modello")
                qt = st.number_input("Quantità", 1)
                pr = st.number_input("Prezzo Unit.", 0.0)
                if st.form_submit_button("Salva e Genera AI"):
                    lux = (qt * 3000 * 0.6) / (w * l)
                    strat = "AI Offline"
                    if st.session_state.groq_online:
                        p = f"Sei un lighting designer. Descrivi il valore tecnico di {qt} pezzi {br} {md} in un ambiente {rn}."
                        chat = st.session_state.groq_client.chat.completions.create(messages=[{"role":"user","content":p}], model="llama-3.3-70b-versatile")
                        strat = chat.choices[0].message.content
                    conn = sqlite3.connect('luxia_titan.db')
                    conn.execute("INSERT INTO rooms (project_id, r_name, w, l, brand, model, qty, price, lux_avg, strategy) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 (st.session_state.current_proj_id, rn, w, l, br, md, qt, pr, lux, strat))
                    conn.commit(); conn.close(); st.rerun()

            conn = sqlite3.connect('luxia_titan.db')
            rooms = conn.execute("SELECT r_name, lux_avg, strategy, model FROM rooms WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
            conn.close()
            for r in rooms:
                with st.expander(f"📍 {r[0]} ({r[1]:.0f} Lux)"):
                    st.write(f"Prodotto: {r[3]}")
                    st.info(f"AI Strategy: {r[2]}")

        with t_rep:
            st.subheader("📄 Generazione Report")
            if st.button("Scarica Report Progetto"):
                st.write("Generazione in corso...")

if __name__ == "__main__":
    main()