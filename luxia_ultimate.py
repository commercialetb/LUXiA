import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from shapely.geometry import Polygon
from fpdf import FPDF
import base64
import io
import ezdxf
from PIL import Image

# --- CONFIGURAZIONE ---
class LuxiaConfig:
    APP_NAME = "LUXiA Ultimate Gold"
    VERSION = "Final 3.1"
    NORM_PROCESS = "UNI CEN/TS 17165:2019"
    NORM_DATA = "UNI/TS 11999" 
    NORM_TARGET = "UNI EN 12464-1"

# --- AGENTE VISION & TEXT ---
class GroqAgent:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key) if api_key else None

    def analyze_vision(self, image_bytes):
        if not self.client: return "⚠️ API Key mancante."
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        try:
            chat = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analizza questa planimetria. Identifica tipo stanza, h soffitto e lux target. Rispondi in JSON."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }],
                model="llama-3.2-11b-vision-preview",
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"Errore: {e}"

    def write_report(self, room_data, product, results):
        if not self.client: return "Testo demo."
        prompt = f"Scrivi relazione tecnica {LuxiaConfig.NORM_PROCESS} per {room_data['name']}. Prodotto: {product['model']}. Risultati: {results['avg']} lux."
        try:
            chat = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            return chat.choices[0].message.content
        except: return "Errore generazione testo."

# --- GESTORE CAD ---
def render_dxf_preview(dxf_bytes):
    try:
        with open("temp_upload.dxf", "wb") as f:
            f.write(dxf_bytes)
        doc = ezdxf.readfile("temp_upload.dxf")
        msp = doc.modelspace()
        fig = go.Figure()
        for e in msp.query('LINE LWPOLYLINE')[:500]:
            if e.dxftype() == 'LINE':
                fig.add_trace(go.Scatter(x=[e.dxf.start.x, e.dxf.end.x], y=[e.dxf.start.y, e.dxf.end.y], mode='lines', line=dict(color='white', width=1), showlegend=False))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points()
                fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode='lines', line=dict(color='cyan', width=1), showlegend=False))
        fig.update_layout(plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
        return fig
    except: return None

# --- ENGINE ---
def calculate_lux(w, l, product, mount_h):
    x, y = np.meshgrid(np.linspace(0, w, 15), np.linspace(0, l, 15))
    z = np.zeros_like(x)
    lamps = [(w*0.3, l*0.3), (w*0.7, l*0.3), (w*0.3, l*0.7), (w*0.7, l*0.7)]
    h_eff = mount_h - 0.75
    intensity = product['lumen'] / np.pi
    for lx, ly in lamps:
        dist_sq = (x - lx)**2 + (y - ly)**2 + h_eff**2
        z += (intensity * (h_eff / np.sqrt(dist_sq))) / dist_sq
    return x, y, z, np.mean(z), np.min(z)/np.mean(z)

# --- APP PRINCIPALE ---
def main():
    st.set_page_config(page_title="LUXiA Gold", layout="wide")
    st.sidebar.title("LUXiA Config")
    api_key = st.sidebar.text_input("Groq API Key", type="password") or st.secrets.get("GROQ_API_KEY")
    
    groq = GroqAgent(api_key)
    st.title("💡 LUXiA Ultimate Gold")

    tab1, tab2, tab3 = st.tabs(["📂 Vision & CAD", "📐 Progetto & 3D", "📥 Export"])

    with tab1:
        uploaded_file = st.file_uploader("Carica Planimetria", type=['jpg', 'png', 'pdf', 'dxf'])
        if uploaded_file:
            ftype = uploaded_file.name.split('.')[-1].lower()
            
            if ftype in ['jpg', 'png', 'jpeg']:
                st.image(uploaded_file, width=500)
                if st.button("Analizza AI Vision"):
                    res = groq.analyze_vision(uploaded_file.getvalue())
                    st.info(res)
                    st.session_state['room'] = {"name": "Auditorium", "h": 4.5, "w": 12, "l": 18}

            elif ftype == 'pdf':
                st.info("PDF Caricato. Se Chrome blocca l'anteprima, usa il tasto scarica.")
                pdf_bytes = uploaded_file.getvalue()
                st.download_button("Apri PDF", data=pdf_bytes, file_name="plan.pdf")
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" style="border:none;"></iframe>', unsafe_allow_html=True)

            elif ftype == 'dxf':
                st.success("File CAD rilevato.")
                fig_cad = render_dxf_preview(uploaded_file.getvalue())
                if fig_cad: st.plotly_chart(fig_cad, use_container_width=True)
                st.session_state['room'] = {"name": "Vano da CAD", "h": 4.5, "w": 15, "l": 20}

    with tab2:
        if 'room' in st.session_state:
            r = st.session_state['room']
            st.write(f"Ambiente: **{r['name']}** | Altezza: **{r['h']}m**")
            if st.button("Seleziona Prodotto BEGA"):
                tipo = "Sospensione" if r['h'] > 4.0 else "Incasso"
                p = {"model": "BEGA 50 998.1", "brand": "BEGA", "type": tipo, "lumen": 3612, "watt": 32}
                st.session_state['prod'] = p
                
            if 'prod' in st.session_state:
                p = st.session_state['prod']
                st.success(f"Scelto: {p['model']} ({p['type']})")
                X, Y, Z, avg, uo = calculate_lux(r['w'], r['l'], p, r['h']-1.0 if p['type']=="Sospensione" else r['h'])
                st.session_state['res'] = {"avg": avg, "uo": uo}
                fig3 = go.Figure(data=[go.Surface(z=Z)])
                st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        if 'res' in st.session_state:
            if st.button("Genera Report"):
                text = groq.write_report(st.session_state['room'], st.session_state['prod'], st.session_state['res'])
                st.text_area("Relazione", text, height=200)
                st.download_button("Scarica Progetto", data=b"PDF DATA", file_name="LUXiA_Report.pdf")

if __name__ == "__main__":
    main()
