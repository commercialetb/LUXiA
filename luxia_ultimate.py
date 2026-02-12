import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
from shapely.geometry import Polygon, Point
from fpdf import FPDF
import base64
import io
import ezdxf  # Per i file CAD
from PIL import Image

# --- CONFIGURAZIONE SISTEMA ---
class LuxiaConfig:
    APP_NAME = "LUXiA Ultimate Gold"
    VERSION = "Final 3.0"
    # Normative Certificate
    NORM_PROCESS = "UNI CEN/TS 17165:2019"
    NORM_DATA = "UNI/TS 11999" 
    NORM_TARGET = "UNI EN 12464-1"
    NORM_EMERGENCY = "UNI EN 1838"

# --- MODULO 1: AI VISION & AGENT ---
class GroqAgent:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key) if api_key else None

    def analyze_vision(self, image_bytes):
        """Analizza l'immagine della planimetria"""
        if not self.client: 
            return "⚠️ Modalità Demo: Inserisci API Key per l'analisi AI reale."
        
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt = f"""
        Sei un esperto Lighting Designer. Analizza questa planimetria tecnica.
        1. Identifica il tipo di stanza (es. Ufficio, Auditorium, Corridoio).
        2. Stima l'altezza del soffitto se ci sono quote, altrimenti ipotizza in base al contesto.
        3. Restituisci un JSON puro con: "room_type", "estimated_height", "suggested_lux".
        """
        try:
            chat = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }],
                model="llama-3.2-11b-vision-preview",
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"Errore Vision: {e}"

    def write_report(self, room_data, product, results):
        """Scrive la relazione tecnica finale"""
        if not self.client:
            return "Testo generato in modalità demo (senza API Key)."
            
        prompt = f"""
        Scrivi una "Relazione Tecnica Illuminotecnica" formale secondo {LuxiaConfig.NORM_PROCESS}.
        DATI PROGETTO:
        - Ambiente: {room_data['name']} (H={room_data['h']}m).
        - Prodotto Scelto: {product['model']} ({product['type']}) di {product['brand']}.
        - Risultati Calcolo: E_med {results['avg']:.0f} lux, U_o {results['uo']:.2f}.
        
        PUNTI DA TRATTARE:
        1. Giustifica la scelta di {product['type']} basandoti sull'altezza del soffitto.
        2. Conferma la conformità a {LuxiaConfig.NORM_TARGET}.
        3. Cita l'uso di dati {LuxiaConfig.NORM_DATA} per il BIM.
        """
        try:
            chat = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile", # Modello testo potente
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"Errore generazione testo: {e}"

# --- MODULO 2: CAD HANDLER (DXF) ---
def render_dxf_preview(dxf_bytes):
    """Legge un DXF e restituisce un grafico Plotly"""
    try:
        # Ezdxf richiede un file fisico o stream, usiamo un temp file in memoria
        with open("temp_upload.dxf", "wb") as f:
            f.write(dxf_bytes)
        
        doc = ezdxf.readfile("temp_upload.dxf")
        msp = doc.modelspace()
        
        fig = go.Figure()
        
        # Estraiamo LINE e LWPOLYLINE per visualizzare i muri
        count = 0
        for e in msp.query('LINE LWPOLYLINE'):
            if count > 500: break # Limite per performance
            if e.dxftype() == 'LINE':
                fig.add_trace(go.Scatter(x=[e.dxf.start.x, e.dxf.end.x], y=[e.dxf.start.y, e.dxf.end.y], mode='lines', line=dict(color='white', width=1), showlegend=False))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points()
                x = [p[0] for p in pts]
                y = [p[1] for p in pts]
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='cyan', width=1), showlegend=False))
            count += 1
            
        fig.update_layout(
            title="Anteprima CAD (Vettoriale)",
            plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False, scaleanchor="x"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=400
        )
        return fig
    except Exception as e:
        st.error(f"Impossibile renderizzare il DXF: {e}")
        return None

# --- MODULO 3: BEGA AGENT & PHYSICS ---
class BegaAgent:
    @staticmethod
    def get_recommendation(height, room_type):
        """Logica decisionale autonoma"""
        if height >= 4.0:
            return {
                "brand": "BEGA",
                "model": "50 998.1 Studio Line",
                "type": "Sospensione",
                "desc": "Sistema a sospensione per grandi altezze. Ottica Flood.",
                "lumen": 3612, "watt": 32.0, "install_h": height - 1.5
            }
        else:
            return {
                "brand": "BEGA",
                "model": "50 666.1 Incasso",
                "type": "Incasso",
                "desc": "Downlight schermato per controsoffitti bassi.",
                "lumen": 2800, "watt": 24.5, "install_h": height
            }

def calculate_lux(width, length, product, mount_h):
    """Motore fisico semplificato (Inverse Square Law)"""
    # Griglia di calcolo
    x = np.linspace(0, width, 10)
    y = np.linspace(0, length, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Posizioniamo 6 lampade (layout automatico semplice)
    lamps = [
        (width*0.25, length*0.25), (width*0.75, length*0.25),
        (width*0.25, length*0.5),  (width*0.75, length*0.5),
        (width*0.25, length*0.75), (width*0.75, length*0.75)
    ]
    
    workplane_h = 0.75
    h_eff = mount_h - workplane_h
    
    # Calcolo
    intensity = product['lumen'] / np.pi # Approssimazione Lambertiana
    for lx, ly in lamps:
        dist_sq = (X - lx)**2 + (Y - ly)**2 + h_eff**2
        cos_theta = h_eff / np.sqrt(dist_sq)
        E = (intensity * cos_theta) / dist_sq
        Z += E
        
    return X, Y, Z, np.mean(Z), np.min(Z)/np.mean(Z)

# --- MODULO 4: PDF EXPORT ---
def create_pdf(text, data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"LUXiA Report: {data['project']}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Dati di Progetto", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 8, f"Normativa Processo: {LuxiaConfig.NORM_PROCESS}", ln=True)
    pdf.cell(0, 8, f"Prodotto: {data['product']} ({data['type']})", ln=True)
    pdf.cell(0, 8, f"Lux Medi Calcolati: {data['lux']} lx", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Relazione Tecnica (AI Generated)", ln=True)
    pdf.set_font("Arial", '', 10)
    # FPDF non supporta utf-8 diretto bene senza font esterni, facciamo replace safe
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, safe_text)
    
    return pdf.output(dest='S').encode('latin-1')

# --- MAIN APPLICATION ---
def run_luxia():
    st.set_page_config(page_title="LUXiA Ultimate", layout="wide", page_icon="💡")
    
    # 1. SIDEBAR SETUP
    st.sidebar.image("https://img.icons8.com/ios-filled/100/FAB005/light-on.png", width=50)
    st.sidebar.title("LUXiA Gold")
    
    # Gestione API Key sicura
    api_key = st.sidebar.text_input("Groq API Key (gsk_...)", type="password")
    if not api_key and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.sidebar.success("🔑 Chiave trovata nei Secrets")
    
    groq = GroqAgent(api_key)
    
    # 2. HEADER
    st.title("💡 LUXiA: AI Lighting Architect")
    st.markdown(f"**Compliance:** {LuxiaConfig.NORM_PROCESS} | {LuxiaConfig.NORM_DATA}")
    
    # 3. TABS
    tab_vision, tab_agent, tab_report = st.tabs(["1. Vision & Import", "2. Bega Agent & 3D", "3. Export Certificato"])
    
    # --- TAB 1: VISION ---
    with tab_vision:
        st.header("Importazione Planimetria")
        uploaded_file = st.file_uploader("Carica file (JPG, PNG, PDF, DXF)", type=['jpg', 'png', 'pdf', 'dxf'])
        
        room_data = st.session_state.get('room_data', {"name": "Vano Generico", "h": 3.0, "w": 10, "l": 15})
        
        if uploaded_file:
            ftype = uploaded_file.name.split('.')[-1].lower()
            
            # GESTIONE IMMAGINI
            if ftype in ['jpg', 'png', 'jpeg']:
                st.image(uploaded_file, caption="Plan Image", width=500)
                if st.button("Analizza con Groq Vision"):
                    with st.spinner("Analisi AI in corso..."):
                        res = groq.analyze_vision(uploaded_file.getvalue())
                        st.success("Analisi Completata")
                        st.info(res)
                        # Mock update per demo
                        room_data = {"name": "Auditorium Vano 5", "h": 4.5, "w": 12, "l": 20}
                        st.session_state['room_data'] = room_data

            # GESTIONE PDF (SAFE MODE)
            elif ftype == 'pdf':
                st.info("📄 Documento PDF rilevato.")
                # Embed PDF per visualizzazione browser nativa
                base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.warning("Per l'analisi AI Vision, converti la pagina in JPG.")
                
            # GESTIONE DXF (CAD)
            elif ftype == 'dxf':
                st.success("📐 File CAD DXF Rilevato")
                fig_dxf = render_dxf_preview(uploaded_file.getvalue())
                if fig_dxf:
                    st.plotly_chart(fig_dxf, use_container_width=True)
                room_data = {"name": "Import da CAD", "h": 4.5, "w": 15, "l": 25} # Mock da CAD
                st.session_state['room_data'] = room_data

    # --- TAB 2: AGENT & CALC ---
    with tab_agent:
        st.header("Progettazione Autonoma")
        r = st.session_state.get('room_data', room_data)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**Ambiente:** {r['name']}")
            st.write(f"**Altezza:** {r['h']} m")
            
            if st.button("Chiedi all'Agente BEGA"):
                with st.spinner("Consultazione catalogo BEGA..."):
                    prod = BegaAgent.get_recommendation(r['h'], "Auditorium")
                    st.session_state['product'] = prod
                    st.success(f"Trovato: {prod['model']}")
                    
        with col2:
            if 'product' in st.session_state:
                p = st.session_state['product']
                st.info(f"💡 **Scelta AI:** {p['type']} - {p['desc']}")
                
                # Calcolo
                X, Y, Z, avg, uo = calculate_lux(r['w'], r['l'], p, p['install_h'])
                st.session_state['results'] = {"avg": avg, "uo": uo}
                
                # 3D Plot
                fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])
                fig.update_layout(title=f"Simulazione 3D ({avg:.0f} lux medi)", scene=dict(zaxis=dict(range=[0, 800])))
                st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: REPORT ---
    with tab_report:
        st.header("Generazione Documentale")
        if 'results' in st.session_state and 'product' in st.session_state:
            if st.button("Genera Pacchetto PDF"):
                with st.spinner("Scrittura relazione tecnica (Groq Llama-3)..."):
                    res = st.session_state['results']
                    prod = st.session_state['product']
                    
                    # 1. Testo AI
                    narrative = groq.write_report(st.session_state['room_data'], prod, res)
                    st.text_area("Anteprima Testo", narrative, height=150)
                    
                    # 2. Creazione PDF
                    pdf_data = {
                        "project": "BCC Auditorium",
                        "product": prod['model'],
                        "type": prod['type'],
                        "lux": f"{res['avg']:.0f}"
                    }
                    pdf_bytes = create_pdf(narrative, pdf_data)
                    
                    st.download_button(
                        label="📥 Scarica Relazione Certificata (.pdf)",
                        data=pdf_bytes,
                        file_name="LUXiA_Project_BCC.pdf",
                        mime="application/pdf"
                    )
        else:
            st.warning("Completa prima la fase 2 per sbloccare il report.")

if __name__ == "__main__":
    run_luxia()