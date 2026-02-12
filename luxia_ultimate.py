import streamlit as st
import sqlite3
import hashlib
import base64
import pandas as pd
from datetime import datetime
from groq import Groq
from fpdf import FPDF

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="LUXiA Titan v6.0", layout="wide", initial_sidebar_state="expanded")

# --- 1. ARCHITETTURA DATABASE (Self-Healing) ---
def init_db():
    conn = sqlite3.connect('luxia_titan.db')
    c = conn.cursor()
    
    # Tabelle Core
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, studio_name TEXT, logo_b64 TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, p_name TEXT, client TEXT, date TEXT)''')
    
    # Tabella Documenti Generali Progetto (DWG, DXF, PDF)
    c.execute('''CREATE TABLE IF NOT EXISTS project_docs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, filename TEXT, file_type TEXT, file_blob BLOB, upload_date TEXT)''')
    
    # Tabella Vani (con AI e Dati Tecnici)
    c.execute('''CREATE TABLE IF NOT EXISTS rooms 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, r_name TEXT, 
                  w REAL, l REAL, h REAL, brand TEXT, model TEXT, qty INTEGER, 
                  price REAL, lux_avg REAL, strategy TEXT, img_blob BLOB)''')
    
    # Migrazione colonne (sicurezza anti-crash)
    try:
        cols = [i[1] for i in c.execute("PRAGMA table_info(rooms)").fetchall()]
        if 'price' not in cols: c.execute("ALTER TABLE rooms ADD COLUMN price REAL DEFAULT 0")
        if 'strategy' not in cols: c.execute("ALTER TABLE rooms ADD COLUMN strategy TEXT")
        if 'img_blob' not in cols: c.execute("ALTER TABLE rooms ADD COLUMN img_blob BLOB")
    except: pass
    
    conn.commit()
    conn.close()

# --- 2. FUNZIONI UTILITY ---
def get_dashboard_stats(username):
    conn = sqlite3.connect('luxia_titan.db')
    # Conta solo i progetti reali esistenti nella tabella
    n_proj = conn.execute("SELECT COUNT(*) FROM projects WHERE username=?", (username,)).fetchone()[0]
    # Statistiche vani e fatturato
    stats = conn.execute('''SELECT COUNT(rooms.id), SUM(rooms.qty * rooms.price) 
                            FROM rooms 
                            JOIN projects ON rooms.project_id = projects.id 
                            WHERE projects.username=?''', (username,)).fetchone()
    conn.close()
    return n_proj, (stats[0] if stats[0] else 0), (stats[1] if stats[1] else 0)

def delete_project(proj_id):
    conn = sqlite3.connect('luxia_titan.db')
    conn.execute("DELETE FROM rooms WHERE project_id=?", (proj_id,))
    conn.execute("DELETE FROM project_docs WHERE project_id=?", (proj_id,))
    conn.execute("DELETE FROM projects WHERE id=?", (proj_id,))
    conn.commit()
    conn.close()

# --- 3. INTERFACCIA PRINCIPALE ---
def main():
    init_db()

    # Session State Init
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'current_proj_id' not in st.session_state: st.session_state.current_proj_id = None
    if 'groq_online' not in st.session_state: st.session_state.groq_online = False

    # --- LOGIN SYSTEM ---
    if not st.session_state.logged_in:
        st.markdown("""<h1 style='text-align: center; color: #FFD700;'>LUXiA TITAN</h1>""", unsafe_allow_html=True)
        st.markdown("""<h3 style='text-align: center;'>Professional Lighting Design Suite</h3>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            tab_l, tab_r = st.tabs(["🔒 Accesso", "📝 Registrazione"])
            with tab_l:
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.button("Entra nello Studio", use_container_width=True):
                    h = hashlib.sha256(p.encode()).hexdigest()
                    conn = sqlite3.connect('luxia_titan.db')
                    res = conn.execute("SELECT studio_name, logo_b64 FROM users WHERE username=? AND password=?", (u, h)).fetchone()
                    conn.close()
                    if res:
                        st.session_state.update({"logged_in": True, "username": u, "studio_name": res[0], "user_logo_b64": res[1]})
                        st.rerun()
                    else: st.error("Accesso Negato.")
            with tab_r:
                nu = st.text_input("Nuovo Utente")
                np = st.text_input("Nuova Password", type="password")
                ns = st.text_input("Nome Studio")
                if st.button("Crea Account", use_container_width=True):
                    h = hashlib.sha256(np.encode()).hexdigest()
                    conn = sqlite3.connect('luxia_titan.db')
                    try:
                        conn.execute("INSERT INTO users VALUES (?,?,?,?)", (nu, h, ns, None))
                        conn.commit()
                        st.success("Registrato! Procedi al login.")
                    except: st.error("Utente già esistente.")
                    finally: conn.close()
        return

    # --- SIDEBAR PROFESSIONALE ---
    with st.sidebar:
        st.markdown("### 🏛️ STUDIO PANEL")
        
        # Logo Logic
        if st.session_state.get('user_logo_b64'):
            st.image(base64.b64decode(st.session_state.user_logo_b64), use_container_width=True)
            if st.button("🔄 Cambia Logo", help="Rimuovi logo attuale"):
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=NULL WHERE username=?", (st.session_state.username,))
                conn.commit(); conn.close()
                st.session_state.user_logo_b64 = None; st.rerun()
        else:
            up = st.file_uploader("Upload Logo", type=['png','jpg'])
            if up:
                b64 = base64.b64encode(up.read()).decode()
                conn = sqlite3.connect('luxia_titan.db')
                conn.execute("UPDATE users SET logo_b64=? WHERE username=?", (b64, st.session_state.username))
                conn.commit(); conn.close()
                st.session_state.user_logo_b64 = b64; st.rerun()

        st.divider()
        
        # AI Config
        st.markdown("### 🧠 AI ENGINE")
        gk = st.text_input("Groq API Key", type="password", placeholder="sk-...")
        if gk:
            st.session_state.groq_client = Groq(api_key=gk)
            st.session_state.groq_online = True
            st.success("AI SYSTEM: ONLINE")
        else:
            st.warning("AI SYSTEM: OFFLINE")

        st.divider()
        
        # Project Navigation
        conn = sqlite3.connect('luxia_titan.db')
        plist = conn.execute("SELECT id, p_name FROM projects WHERE username=?", (st.session_state.username,)).fetchall()
        p_dict = {p[1]: p[0] for p in plist}
        conn.close()
        
        st.markdown("### 📂 NAVIGATORE")
        sel = st.selectbox("Seleziona Progetto", ["-- DASHBOARD --"] + list(p_dict.keys()))
        
        if st.button("🚀 APRI / VAI", use_container_width=True):
            if sel == "-- DASHBOARD --":
                st.session_state.current_proj_id = None
            else:
                st.session_state.current_proj_id = p_dict[sel]
                st.session_state.current_proj_name = sel
            st.rerun()

        st.divider()
        if st.button("🚪 LOGOUT", use_container_width=True):
            st.session_state.logged_in = False; st.rerun()
            
        # LEGAL FOOTER
        st.markdown("---")
        st.markdown("""
        <div style='font-size: 10px; color: grey; text-align: center;'>
        <b>LUXiA™ TRADEMARK</b><br>
        Patent Pending 2024-2025<br>
        All Rights Reserved.<br>
        Version 6.0 Ultimate
        </div>
        """, unsafe_allow_html=True)

    # --- DASHBOARD (SE NESSUN PROGETTO APERTO) ---
    if not st.session_state.current_proj_id:
        st.title(f"Benvenuto, {st.session_state.studio_name}")
        
        # KPI
        n_p, n_v, val = get_dashboard_stats(st.session_state.username)
        k1, k2, k3 = st.columns(3)
        k1.metric("Progetti Attivi", n_p)
        k2.metric("Vani Calcolati", n_v)
        k3.metric("Pipeline (€)", f"{val:,.2f}")
        
        st.markdown("---")
        
        # Gestione Progetti
        c_new, c_list = st.columns([1, 2])
        
        with c_new:
            st.info("🆕 **Nuovo Cantiere**")
            with st.form("new_proj_f"):
                np_name = st.text_input("Nome Progetto")
                np_cli = st.text_input("Cliente")
                if st.form_submit_button("Crea Progetto"):
                    conn = sqlite3.connect('luxia_titan.db')
                    cur = conn.cursor()
                    cur.execute("INSERT INTO projects (username, p_name, client, date) VALUES (?,?,?,?)", 
                                (st.session_state.username, np_name, np_cli, datetime.now().strftime("%d/%m/%Y")))
                    conn.commit()
                    new_id = cur.lastrowid
                    conn.close()
                    st.session_state.current_proj_id = new_id
                    st.session_state.current_proj_name = np_name
                    st.rerun()

        with c_list:
            st.warning("🛠️ **Gestione Archivio**")
            if plist:
                df = pd.DataFrame(plist, columns=['ID', 'Nome Progetto'])
                st.dataframe(df[['Nome Progetto']], use_container_width=True, hide_index=True)
                
                # Funzione Cancellazione
                del_p = st.selectbox("Elimina Progetto", ["-- Nessuno --"] + list(p_dict.keys()))
                if del_p != "-- Nessuno --":
                    if st.button(f"🗑️ Conferma Eliminazione {del_p}"):
                        delete_project(p_dict[del_p])
                        st.success("Progetto eliminato.")
                        st.rerun()
            else: st.write("Nessun progetto in archivio.")

    # --- PROGETTO APERTO (WORKFLOW COMPLETO) ---
    else:
        st.header(f"🏢 Cantiere: {st.session_state.current_proj_name}")
        if st.button("⬅️ Torna alla Dashboard Generale"):
            st.session_state.current_proj_id = None; st.rerun()

        # TABS WORKFLOW
        t_docs, t_ai, t_rep = st.tabs(["📂 1. Documenti & CAD", "💡 2. Vani & AI Strategy", "📄 3. Report Finale"])

        # TAB 1: DOCUMENTI PROGETTO (DWG, DXF, PDF)
        with t_docs:
            st.subheader("Archivio Tecnico del Cantiere")
            col_up, col_view = st.columns([1, 2])
            
            with col_up:
                with st.form("upload_doc"):
                    st.write("Carica file generale (Planimetria, DWG, DXF)")
                    f_doc = st.file_uploader("Seleziona File", type=['pdf', 'dwg', 'dxf', 'jpg', 'png'])
                    if st.form_submit_button("⬆️ Carica in Archivio"):
                        if f_doc:
                            blob = f_doc.read()
                            conn = sqlite3.connect('luxia_titan.db')
                            conn.execute("INSERT INTO project_docs (project_id, filename, file_type, file_blob, upload_date) VALUES (?,?,?,?,?)",
                                         (st.session_state.current_proj_id, f_doc.name, f_doc.type, blob, datetime.now().strftime("%d/%m/%Y")))
                            conn.commit(); conn.close()
                            st.toast("File archiviato correttamente!", icon="✅")
                            st.rerun()

            with col_view:
                st.write("**File Disponibili:**")
                conn = sqlite3.connect('luxia_titan.db')
                docs = conn.execute("SELECT filename, upload_date, file_blob FROM project_docs WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
                conn.close()
                if docs:
                    for d in docs:
                        c1, c2 = st.columns([3, 1])
                        c1.text(f"📄 {d[0]} (del {d[1]})")
                        c2.download_button("Scarica", data=d[2], file_name=d[0])
                else: st.info("Nessun documento tecnico caricato.")

        # TAB 2: VANI & AI
        with t_ai:
            st.subheader("Progettazione Illuminotecnica & AI")
            
            c_input, c_summary = st.columns([1, 2])
            
            with c_input:
                st.markdown("#### ➕ Nuovo Ambiente")
                with st.form("room_ai_form"):
                    rn = st.text_input("Nome Vano (es. Hall)")
                    w, l, h = st.number_input("Larghezza", 1.0), st.number_input("Lunghezza", 1.0), st.number_input("Altezza", 3.0)
                    br = st.selectbox("Brand", ["BEGA", "iGuzzini", "Artemide", "Flos", "Custom"])
                    mod = st.text_input("Codice/Modello")
                    qt = st.number_input("Quantità", 1)
                    pr = st.number_input("Prezzo Unitario (€)", 0.0)
                    img = st.file_uploader("Foto Vano", type=['jpg','png'])
                    
                    st.markdown("**AI Settings:**")
                    ai_req = st.checkbox("Genera Strategia di Vendita con AI", value=True)

                    if st.form_submit_button("Calcola e Salva"):
                        lux = (qt * 3000 * 0.6) / (w * l)
                        strat = "AI non richiesta o Offline."
                        
                        if ai_req and st.session_state.groq_online:
                            try:
                                p = f"Scrivi una strategia di vendita convincente e tecnica per un {rn} di {w*l}mq usando {qt} apparecchi {br} {mod}. Focus su comfort visivo e design."
                                chat = st.session_state.groq_client.chat.completions.create(
                                    messages=[{"role":"user","content":p}], model="llama-3.3-70b-versatile"
                                )
                                strat = chat.choices[0].message.content
                            except Exception as e: strat = f"Errore AI: {e}"

                        conn = sqlite3.connect('luxia_titan.db')
                        conn.execute('''INSERT INTO rooms (project_id, r_name, w, l, h, brand, model, qty, price, lux_avg, strategy, img_blob) 
                                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', 
                                     (st.session_state.current_proj_id, rn, w, l, h, br, mod, qt, pr, lux, strat, img.read() if img else None))
                        conn.commit(); conn.close()
                        st.success("Vano aggiunto!")
                        st.rerun()

            with c_summary:
                st.markdown("#### 📋 Riepilogo Vani")
                conn = sqlite3.connect('luxia_titan.db')
                rooms = conn.execute("SELECT r_name, lux_avg, model, qty, price, strategy, img_blob FROM rooms WHERE project_id=?", 
                                    (st.session_state.current_proj_id,)).fetchall()
                conn.close()
                
                if rooms:
                    for r in rooms:
                        with st.container(border=True):
                            c_t, c_i = st.columns([3, 1])
                            with c_t:
                                st.markdown(f"**{r[0]}** | {r[1]:.0f} Lux | {r[3]}x {r[2]}")
                                st.caption(f"Totale: € {r[3]*r[4]:,.2f}")
                                if r[5] and "AI" not in r[5]:
                                    st.info(f"✨ **Strategia AI:** {r[5]}")
                                else:
                                    st.text(f"Note: {r[5]}")
                            if r[6]:
                                with c_i: st.image(r[6], use_container_width=True)
                else:
                    st.info("Nessun ambiente progettato. Usa il form a sinistra.")

        # TAB 3: REPORT
        with t_rep:
            st.subheader("Esportazione Documentale")
            if st.button("📄 Genera Report PDF Completo"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, f"PROGETTO: {st.session_state.current_proj_name}", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Studio: {st.session_state.studio_name}", ln=True)
                pdf.ln(10)
                
                conn = sqlite3.connect('luxia_titan.db')
                rooms_data = conn.execute("SELECT * FROM rooms WHERE project_id=?", (st.session_state.current_proj_id,)).fetchall()
                conn.close()
                
                tot = 0
                for rd in rooms_data: # rd index: 2=name, 8=qty, 9=price, 10=lux, 11=strat
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"Ambiente: {rd[2]} ({rd[10]:.0f} Lux)", ln=True)
                    pdf.set_font("Arial", "", 10)
                    pdf.multi_cell(0, 5, f"Fornitura: {rd[8]}x {rd[6]} {rd[7]}\nStrategia: {rd[11][:500]}...") # Limit text for PDF
                    tot += (rd[8] * rd[9])
                    pdf.ln(5)
                
                pdf.ln(10)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, f"TOTALE OFFERTA: {tot:,.2f} EUR", ln=True)
                
                # Copyright Footer in PDF
                pdf.set_y(-30)
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 10, "Generated by LUXiA Titan - Patent Pending", align="C")

                html = base64.b64encode(pdf.output(dest='S').encode('latin-1')).decode('latin-1')
                st.download_button("📥 Scarica PDF", data=base64.b64decode(html), file_name="Report_LUXiA.pdf")

if __name__ == "__main__":
    main()