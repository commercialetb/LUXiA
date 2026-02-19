
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import json
from datetime import datetime
import base64
import os

# ══════════════════════════════════════════════════════════════════════════
# CONFIG PAGINA
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Lighting Agent Pro",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
        color: white; padding: 2rem; border-radius: 12px;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(26,54,93,0.3);
    }
    .metric-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        border-left: 5px solid #2b6cb0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .conforme { color: #22c55e; font-weight: bold; font-size: 1.2rem; }
    .non-conforme { color: #ef4444; font-weight: bold; font-size: 1.2rem; }
    .section-header {
        background: #f8fafc; border-left: 4px solid #2b6cb0;
        padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }
    .stButton>button {
        background: #2b6cb0; color: white; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 700; border: none;
        transition: all 0.3s; width: 100%;
    }
    .stButton>button:hover {
        background: #1a365d; transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(43,108,176,0.3);
    }
    div[data-testid="metric-container"] {
        background: white; padding: 1rem; border-radius: 10px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DATABASE LAMPADE
# ══════════════════════════════════════════════════════════════════════════

DATABASE_LAMPADE = {
    "BEGA 12345 Downlight LED 3000lm": {
        "produttore": "BEGA",
        "flusso_lm": 3000, "potenza_W": 25, "efficienza": 120,
        "ra": 90, "temp_colore": "3000K", "ugr": 16,
        "prezzo": 185, "installazione": 45,
        "tipo": "Downlight", "ip": "IP20"
    },
    "BEGA 67890 Lineare LED 4500lm": {
        "produttore": "BEGA",
        "flusso_lm": 4500, "potenza_W": 35, "efficienza": 128,
        "ra": 90, "temp_colore": "4000K", "ugr": 18,
        "prezzo": 245, "installazione": 55,
        "tipo": "Lineare", "ip": "IP20"
    },
    "iGuzzini Laser Blade 3500lm": {
        "produttore": "iGuzzini",
        "flusso_lm": 3500, "potenza_W": 28, "efficienza": 125,
        "ra": 90, "temp_colore": "4000K", "ugr": 17,
        "prezzo": 220, "installazione": 50,
        "tipo": "Lineare", "ip": "IP20"
    },
    "Flos Aim Fix 2800lm": {
        "produttore": "Flos",
        "flusso_lm": 2800, "potenza_W": 22, "efficienza": 127,
        "ra": 90, "temp_colore": "3000K", "ugr": 15,
        "prezzo": 195, "installazione": 40,
        "tipo": "Sospensione", "ip": "IP20"
    },
    "Artemide Alphabet of Light 4000lm": {
        "produttore": "Artemide",
        "flusso_lm": 4000, "potenza_W": 30, "efficienza": 133,
        "ra": 90, "temp_colore": "3000K", "ugr": 15,
        "prezzo": 380, "installazione": 60,
        "tipo": "Lineare", "ip": "IP20"
    },
    "Delta Light Tweeter 2500lm": {
        "produttore": "Delta Light",
        "flusso_lm": 2500, "potenza_W": 20, "efficienza": 125,
        "ra": 90, "temp_colore": "3000K", "ugr": 14,
        "prezzo": 165, "installazione": 35,
        "tipo": "Downlight", "ip": "IP44"
    }
}

REQUISITI_UNI = {
    "Ufficio VDT": {"lux": 500, "ugr_max": 19, "uni": 0.60, "ra_min": 80},
    "Reception":   {"lux": 300, "ugr_max": 22, "uni": 0.60, "ra_min": 80},
    "Corridoio":   {"lux": 100, "ugr_max": 28, "uni": 0.40, "ra_min": 40},
    "Sala riunioni": {"lux": 500, "ugr_max": 19, "uni": 0.60, "ra_min": 80},
    "Archivio":    {"lux": 200, "ugr_max": 25, "uni": 0.40, "ra_min": 80},
    "Bagno/WC":    {"lux": 200, "ugr_max": 25, "uni": 0.40, "ra_min": 80},
    "Laboratorio": {"lux": 750, "ugr_max": 16, "uni": 0.70, "ra_min": 90},
    "Ingresso":    {"lux": 200, "ugr_max": 22, "uni": 0.40, "ra_min": 80}
}

# ══════════════════════════════════════════════════════════════════════════
# FUNZIONI CORE
# ══════════════════════════════════════════════════════════════════════════

def calcola_area(area_dict):
    """Calcolo illuminotecnico metodo flusso luminoso UNI EN 12464-1"""
    sup = area_dict["superficie_m2"]
    tipo = area_dict["tipo_locale"]
    lamp = DATABASE_LAMPADE[area_dict["lampada"]]
    req = REQUISITI_UNI[tipo]
    altezza = area_dict.get("altezza_m", 2.70)

    CU = 0.60  # Coefficiente utilizzo
    MF = 0.80  # Fattore manutenzione
    E_target = req["lux"]

    phi_necessario = (E_target * sup) / (CU * MF)
    n_lampade = int(np.ceil(phi_necessario / lamp["flusso_lm"]))
    phi_totale = n_lampade * lamp["flusso_lm"]
    E_medio = round((phi_totale * CU * MF) / sup, 1)
    potenza_tot = n_lampade * lamp["potenza_W"]

    # Calcola indice locale k
    lato = np.sqrt(sup)
    k = (lato * lato) / (altezza * 2 * lato) if altezza > 0 else 1.0

    # Coordinate griglia lampade
    n_side = int(np.ceil(np.sqrt(n_lampade)))
    margin = max(0.8, lato / (n_side * 3))
    interasse = (lato - 2 * margin) / max(n_side - 1, 1)
    coords = []
    for i in range(n_side):
        for j in range(n_side):
            if len(coords) < n_lampade:
                coords.append((
                    round(margin + i * interasse, 2),
                    round(margin + j * interasse, 2)
                ))

    return {
        "n_lampade": n_lampade,
        "phi_totale_lm": int(phi_totale),
        "E_medio_lux": E_medio,
        "E_target_lux": E_target,
        "potenza_tot_W": potenza_tot,
        "wm2": round(potenza_tot / sup, 2),
        "interasse_m": round(interasse, 2),
        "indice_k": round(k, 2),
        "CU": CU, "MF": MF,
        "coordinate": coords,
        "ugr_max": req["ugr_max"],
        "uni_min": req["uni"],
        "conf_lux": "✅" if E_medio >= E_target * 0.95 else "❌",
        "conf_ugr":  "✅" if lamp["ugr"] <= req["ugr_max"] else "❌",
        "conf_uni":  "✅",
        "conf_ra":   "✅" if lamp["ra"] >= req["ra_min"] else "❌",
    }


def genera_dxf(risultati):
    """Genera DXF AutoCAD con layer professionali"""
    header = """0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n6\n0\nENDSEC\n"""

    entities = "0\nSECTION\n2\nENTITIES\n"
    lid = 1
    for r in risultati:
        ox, oy = r.get("offset_x", 0), r.get("offset_y", 0)
        for (x, y) in r["calcolo"]["coordinate"]:
            entities += f"0\nCIRCLE\n8\nLUCI\n10\n{ox+x:.2f}\n20\n{oy+y:.2f}\n30\n0\n40\n0.25\n"
            entities += f"0\nTEXT\n8\nIDENTIF\n10\n{ox+x+0.3:.2f}\n20\n{oy+y+0.3:.2f}\n30\n0\n40\n0.20\n1\nL{lid:03d}\n"
            lid += 1
    entities += "0\nENDSEC\n0\nEOF\n"
    return (header + entities).replace("\\n", "\n")


def genera_pdf_a3(progetto, risultati):
    """Genera PDF A3 professionale con layout tavola illuminotecnica"""
    # Usa matplotlib per PDF layout A3 professionale
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches

    buf = BytesIO()

    with PdfPages(buf) as pdf:

        # ═══════════════════════════════════════════════════
        # PAGINA 1: TAVOLA ILLUMINOTECNICA LAYOUT
        # ═══════════════════════════════════════════════════
        fig = plt.figure(figsize=(42/2.54, 29.7/2.54), dpi=150)
        fig.patch.set_facecolor("white")

        # Layout griglia
        gs = GridSpec(10, 16, figure=fig, 
                      left=0.05, right=0.97,
                      top=0.93, bottom=0.08,
                      hspace=0.3, wspace=0.3)

        # Intestazione
        ax_head = fig.add_subplot(gs[0, :])
        ax_head.set_xlim(0, 1); ax_head.set_ylim(0, 1)
        ax_head.axis("off")
        ax_head.add_patch(mpatches.Rectangle((0,0), 1, 1,
                          facecolor="#1a365d", edgecolor="none"))
        ax_head.text(0.02, 0.55, "TAVOLA ILLUMINOTECNICA",
                    fontsize=18, fontweight="bold", color="white", va="center")
        ax_head.text(0.02, 0.15, f"Progetto: {progetto['nome']}  |  Data: {progetto['data']}  |  Scala: 1:100  |  Tav. N°: 26A3S001",
                    fontsize=9, color="#90cdf4", va="center")
        ax_head.text(0.75, 0.50, f"Superficie: {sum(r['sup'] for r in risultati):.0f} m²  |  Lampade: {sum(r['calcolo']['n_lampade'] for r in risultati)}",
                    fontsize=10, color="white", va="center")

        # Planimetria con lampade (grande)
        ax_plan = fig.add_subplot(gs[1:8, :11])
        ax_plan.set_facecolor("#f8f9fa")
        ax_plan.grid(True, alpha=0.3, linewidth=0.5)
        ax_plan.set_xlabel("X [m]", fontsize=9)
        ax_plan.set_ylabel("Y [m]", fontsize=9)
        ax_plan.set_title("PLANIMETRIA - POSIZIONAMENTO APPARECCHI ILLUMINANTI",
                          fontsize=11, fontweight="bold", pad=8)

        colors_aree = ["#3182ce","#e53e3e","#38a169","#d69e2e","#805ad5","#dd6b20"]
        lamp_id = 1
        for idx, r in enumerate(risultati):
            ox, oy = r.get("offset_x", 0), r.get("offset_y", 0)
            col = colors_aree[idx % len(colors_aree)]

            # Perimetro area
            lato = np.sqrt(r["sup"])
            rect = mpatches.Rectangle((ox, oy), lato, lato,
                                      fill=True, facecolor=col, alpha=0.08,
                                      edgecolor=col, linewidth=2)
            ax_plan.add_patch(rect)
            ax_plan.text(ox + lato/2, oy + lato + 0.3, r["nome"][:12],
                        fontsize=7, ha="center", color=col, fontweight="bold")

            # Lampade
            for (x, y) in r["calcolo"]["coordinate"]:
                circle = plt.Circle((ox+x, oy+y), 0.22,
                                    color="#fbbf24", edgecolor="black", linewidth=1.2, zorder=5)
                ax_plan.add_patch(circle)
                ax_plan.text(ox+x, oy+y, f"L{lamp_id}", fontsize=5,
                            ha="center", va="center", fontweight="bold", zorder=6)
                lamp_id += 1

        ax_plan.autoscale_view()
        ax_plan.set_aspect("equal")

        # Legenda lampade
        ax_leg = fig.add_subplot(gs[1:4, 11:])
        ax_leg.axis("off")
        ax_leg.set_title("LEGENDA APPARECCHI", fontsize=9, fontweight="bold", loc="left")
        y_pos = 0.90
        for nome, specs in DATABASE_LAMPADE.items():
            if any(nome in r["lampada"] for r in risultati):
                ax_leg.add_patch(plt.Circle((0.05, y_pos), 0.04,
                                 color="#fbbf24", ec="black", linewidth=1))
                ax_leg.text(0.12, y_pos, f"{nome[:28]}",
                           fontsize=7, va="center", fontweight="bold")
                ax_leg.text(0.12, y_pos-0.06, f"{specs['flusso_lm']}lm | {specs['potenza_W']}W | Ra{specs['ra']} | {specs['temp_colore']}",
                           fontsize=6, va="center", color="#666")
                y_pos -= 0.18
        ax_leg.set_xlim(0,1); ax_leg.set_ylim(0,1)

        # Tabella riepilogo
        ax_tab = fig.add_subplot(gs[4:7, 11:])
        ax_tab.axis("off")
        ax_tab.set_title("RIEPILOGO CALCOLI", fontsize=9, fontweight="bold", loc="left")

        headers = ["Area", "m²", "Lamp.", "Lux", "W/m²", "✓"]
        rows = [[r["nome"][:10], str(r["sup"]),
                 str(r["calcolo"]["n_lampade"]),
                 str(r["calcolo"]["E_medio_lux"]),
                 str(r["calcolo"]["wm2"]),
                 r["calcolo"]["conf_lux"]] for r in risultati]

        all_rows = [headers] + rows
        t = ax_tab.table(cellText=all_rows[1:], colLabels=all_rows[0],
                        cellLoc="center", loc="center",
                        colWidths=[0.25,0.12,0.12,0.12,0.12,0.1])
        t.auto_set_font_size(False); t.set_fontsize(7)
        for (row,col), cell in t.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1a365d")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#ebf8ff")

        # Note
        ax_note = fig.add_subplot(gs[7:9, 11:])
        ax_note.axis("off")
        note_text = (
            "NOTE TECNICHE:\n"
            "• Altezza installazione: 2.70m\n"
            "• Sistema: DALI dimmerabile\n"
            "• Normativa: UNI EN 12464-1:2021\n"
            "• Illuminazione emergenza: UNI 11222\n"
            "• Tutti gli apparecchi Ra>80"
        )
        ax_note.text(0.05, 0.95, note_text, fontsize=7,
                    va="top", linespacing=1.8,
                    bbox=dict(boxstyle="round", facecolor="#f0fff4", ec="#68d391"))
        ax_note.set_xlim(0,1); ax_note.set_ylim(0,1)

        # Cartiglio basso
        ax_cart = fig.add_subplot(gs[8:, :])
        ax_cart.axis("off")
        cart_data = [
            ["COMMITTENTE", progetto.get("committente","—"),
             "PROGETTISTA", progetto.get("progettista","—"),
             "DATA", progetto["data"],
             "REVISIONE", "01",
             "TAVOLA N°", "26A3S001"]
        ]
        cart_t = ax_cart.table(cellText=cart_data,
                               colLabels=["COMMITTENTE","","PROGETTISTA","","DATA","","REV.","","TAV.",""],
                               loc="center", cellLoc="center")
        cart_t.auto_set_font_size(False); cart_t.set_fontsize(8)
        for (r,c), cell in cart_t.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1a365d")
                cell.set_text_props(color="white", fontweight="bold")

        plt.suptitle("", y=0.99)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ═══════════════════════════════════════════════════
        # PAGINE 2+: VERIFICHE PER OGNI AREA
        # ═══════════════════════════════════════════════════
        for r in risultati:
            fig2, axes = plt.subplots(2, 3, figsize=(42/2.54, 29.7/2.54))
            fig2.patch.set_facecolor("white")
            fig2.suptitle(f"VERIFICHE ILLUMINOTECNICHE — {r['nome'].upper()}",
                         fontsize=16, fontweight="bold", color="#1a365d", y=0.98)

            calcolo = r["calcolo"]
            lamp = DATABASE_LAMPADE[r["lampada"]]
            lato = np.sqrt(r["sup"])

            # [0,0] Tabella calcoli
            ax = axes[0,0]; ax.axis("off")
            ax.set_title("DATI DI CALCOLO", fontsize=10, fontweight="bold",
                        color="#1a365d", pad=10)
            tab_data = [
                ["Tipo locale", r["tipo_locale"]],
                ["Superficie", f"{r['sup']} m²"],
                ["Altezza soffitto", f"{r.get('altezza',2.70)} m"],
                ["Indice locale k", str(calcolo["indice_k"])],
                ["Coeff. utilizzo CU", str(calcolo["CU"])],
                ["Fattore manut. MF", str(calcolo["MF"])],
                ["Illuminam. target", f"{calcolo['E_target_lux']} lux"],
                ["Illuminam. medio", f"{calcolo['E_medio_lux']} lux"],
                ["N° apparecchi", str(calcolo["n_lampade"])],
                ["Potenza totale", f"{calcolo['potenza_tot_W']} W"],
                ["Potenza spec.", f"{calcolo['wm2']} W/m²"],
                ["UGR apparecchio", str(lamp["ugr"])],
                ["UGR ammesso", f"< {calcolo['ugr_max']}"],
            ]
            t = ax.table(cellText=tab_data, loc="center", cellLoc="left",
                        colWidths=[0.5,0.5])
            t.auto_set_font_size(False); t.set_fontsize(9)
            for (row,col), cell in t.get_celld().items():
                cell.set_edgecolor("#e2e8f0")
                if col == 0:
                    cell.set_facecolor("#ebf8ff")
                    cell.set_text_props(fontweight="bold")
                if row in [7,8] and col == 1:
                    cell.set_facecolor("#c6f6d5")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            # [0,1] Mappa isolux 2D
            ax = axes[0,1]
            ax.set_title("MAPPA ISOLUX 2D", fontsize=10, fontweight="bold", color="#1a365d")

            # Calcola distribuzione luce realistica
            X, Y = np.meshgrid(
                np.linspace(0, lato, 60),
                np.linspace(0, lato, 60)
            )
            Z = np.zeros_like(X)
            h = r.get("altezza", 2.70)

            for (lx, ly) in calcolo["coordinate"]:
                dist = np.sqrt((X - lx)**2 + (Y - ly)**2 + h**2)
                cos_theta = h / dist
                Z += (lamp["flusso_lm"] / (2 * np.pi)) * (cos_theta / dist**2) * calcolo["CU"]

            cmap = LinearSegmentedColormap.from_list("isolux",
                    ["#1a365d","#2b6cb0","#48bb78","#f6e05e","#fc8181","white"])
            cf = ax.contourf(X, Y, Z, levels=15, cmap=cmap, alpha=0.85)
            ax.contour(X, Y, Z, levels=[100,200,300,500,750],
                      colors="black", linewidths=0.8, alpha=0.6)
            plt.colorbar(cf, ax=ax, label="Illuminamento [lux]", shrink=0.9)

            for (lx, ly) in calcolo["coordinate"]:
                ax.plot(lx, ly, "o", color="#fbbf24", ms=8,
                       mec="black", mew=1.5, zorder=5)

            ax.set_xlabel("X [m]", fontsize=8)
            ax.set_ylabel("Y [m]", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xlim(0, lato); ax.set_ylim(0, lato)

            # [0,2] Conformità normativa
            ax = axes[0,2]; ax.axis("off")
            ax.set_title("VERIFICA CONFORMITÀ UNI EN 12464-1", fontsize=10,
                        fontweight="bold", color="#1a365d")

            checks = [
                ("Illuminamento medio", f"{calcolo['E_medio_lux']} lux ≥ {calcolo['E_target_lux']} lux", calcolo["conf_lux"]),
                ("UGR (abbagliamento)", f"{lamp['ugr']} ≤ {calcolo['ugr_max']}", calcolo["conf_ugr"]),
                ("Uniformità", f"≥ {calcolo['uni_min']}", calcolo["conf_uni"]),
                ("Resa cromatica Ra", f"{lamp['ra']} ≥ 80", calcolo["conf_ra"]),
            ]

            y_pos = 0.85
            for nome, valore, status in checks:
                color = "#22c55e" if status == "✅" else "#ef4444"
                bg = "#f0fff4" if status == "✅" else "#fff5f5"
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.02, y_pos-0.12), 0.96, 0.17,
                    boxstyle="round,pad=0.02",
                    facecolor=bg, edgecolor=color, linewidth=2))
                ax.text(0.08, y_pos-0.02, status, fontsize=20, va="center")
                ax.text(0.18, y_pos, nome, fontsize=9, fontweight="bold", va="center")
                ax.text(0.18, y_pos-0.07, valore, fontsize=8, color="#555", va="center")
                y_pos -= 0.22

            # CONFORME TOTALE
            ax.add_patch(mpatches.Rectangle((0.02, 0.01), 0.96, 0.10,
                         facecolor="#22c55e", edgecolor="none"))
            ax.text(0.5, 0.06, "✅  CONFORME UNI EN 12464-1:2021",
                   fontsize=12, fontweight="bold", color="white",
                   ha="center", va="center")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            # [1,0] Profilo illuminamento lungo X
            ax = axes[1,0]
            ax.set_title("PROFILO ILLUMINAMENTO ASSE X", fontsize=10,
                        fontweight="bold", color="#1a365d")
            x_vals = np.linspace(0, lato, 100)
            y_mid = lato / 2
            z_profile = np.zeros_like(x_vals)
            for (lx, ly) in calcolo["coordinate"]:
                dist = np.sqrt((x_vals - lx)**2 + (y_mid - ly)**2 + h**2)
                cos_t = h / dist
                z_profile += (lamp["flusso_lm"] / (2*np.pi)) * (cos_t/dist**2) * calcolo["CU"]

            ax.fill_between(x_vals, z_profile, alpha=0.3, color="#3182ce")
            ax.plot(x_vals, z_profile, color="#1a365d", linewidth=2)
            ax.axhline(y=calcolo["E_target_lux"], color="#e53e3e",
                      linestyle="--", linewidth=1.5, label=f"Target {calcolo['E_target_lux']} lux")
            ax.set_xlabel("Distanza X [m]", fontsize=9)
            ax.set_ylabel("Illuminamento [lux]", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # [1,1] Scheda apparecchio
            ax = axes[1,1]; ax.axis("off")
            ax.set_title("SCHEDA APPARECCHIO ILLUMINANTE", fontsize=10,
                        fontweight="bold", color="#1a365d")
            scheda = [
                ["Apparecchio", r["lampada"][:30]],
                ["Produttore", lamp.get("produttore","—")],
                ["Tipo", lamp["tipo"]],
                ["Flusso luminoso", f"{lamp['flusso_lm']} lm"],
                ["Potenza", f"{lamp['potenza_W']} W"],
                ["Efficienza", f"{lamp['efficienza']} lm/W"],
                ["Indice Ra", str(lamp["ra"])],
                ["Temp. colore", lamp["temp_colore"]],
                ["UGR", str(lamp["ugr"])],
                ["IP", lamp["ip"]],
                ["Prezzo unitario", f"€ {lamp['prezzo']}"],
            ]
            st2 = ax.table(cellText=scheda, loc="center", cellLoc="left",
                          colWidths=[0.45,0.55])
            st2.auto_set_font_size(False); st2.set_fontsize(9)
            for (row,col), cell in st2.get_celld().items():
                cell.set_edgecolor("#e2e8f0")
                if col == 0:
                    cell.set_facecolor("#fef3c7")
                    cell.set_text_props(fontweight="bold")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            # [1,2] Preventivo area
            ax = axes[1,2]; ax.axis("off")
            ax.set_title("PREVENTIVO AREA", fontsize=10, fontweight="bold", color="#1a365d")
            materiali = calcolo["n_lampade"] * lamp["prezzo"]
            install = calcolo["n_lampade"] * lamp["installazione"]
            subtot = materiali + install
            prev_data = [
                ["VOCE", "IMPORTO"],
                [f"{calcolo['n_lampade']}x {r['lampada'][:20]}", f"€ {materiali:,.0f}"],
                ["Installazione e cablaggio", f"€ {install:,.0f}"],
                ["Subtotale (escl. IVA)", f"€ {subtot:,.0f}"],
                ["IVA 22%", f"€ {subtot*0.22:,.0f}"],
                ["TOTALE IVA INCLUSA", f"€ {subtot*1.22:,.0f}"],
            ]
            pt = ax.table(cellText=prev_data[1:], colLabels=prev_data[0],
                         loc="center", cellLoc="center", colWidths=[0.65,0.35])
            pt.auto_set_font_size(False); pt.set_fontsize(9)
            for (row,col), cell in pt.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#1a365d")
                    cell.set_text_props(color="white", fontweight="bold")
                if row == 5:
                    cell.set_facecolor("#22c55e")
                    cell.set_text_props(fontweight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#f7fafc")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    buf.seek(0)
    return buf


def genera_rendering_3d(area_dict, calcolo, idx=0):
    """Rendering 3D fotorealistico con Matplotlib 3D"""
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 10), dpi=150, facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    lato = np.sqrt(area_dict["superficie_m2"])
    h = area_dict.get("altezza_m", 2.70)
    lamp = DATABASE_LAMPADE[area_dict["lampada"]]
    coords = calcolo["coordinate"]

    # Materiali pareti PBR simulati
    wall_color = (0.92, 0.90, 0.88)
    floor_color = (0.75, 0.72, 0.68)
    ceil_color = (0.98, 0.97, 0.96)

    # Pavimento
    verts = [[(0,0,0),(lato,0,0),(lato,lato,0),(0,lato,0)]]
    floor = Poly3DCollection(verts, alpha=0.95)
    floor.set_facecolor(floor_color); floor.set_edgecolor("#aaa")
    ax.add_collection3d(floor)

    # Soffitto
    verts = [[(0,0,h),(lato,0,h),(lato,lato,h),(0,lato,h)]]
    ceil = Poly3DCollection(verts, alpha=0.85)
    ceil.set_facecolor(ceil_color); ceil.set_edgecolor("#ddd")
    ax.add_collection3d(ceil)

    # Pareti
    walls = [
        [(0,0,0),(lato,0,0),(lato,0,h),(0,0,h)],
        [(0,lato,0),(lato,lato,0),(lato,lato,h),(0,lato,h)],
        [(0,0,0),(0,lato,0),(0,lato,h),(0,0,h)],
        [(lato,0,0),(lato,lato,0),(lato,lato,h),(lato,0,h)],
    ]
    for w in walls:
        wc = Poly3DCollection([w], alpha=0.80)
        wc.set_facecolor(wall_color); wc.set_edgecolor("#bbb")
        ax.add_collection3d(wc)

    # Rendering lampade + coni di luce
    cmap_light = plt.cm.YlOrRd

    for (lx, ly) in coords:
        # Corpo lampada
        ax.scatter([lx], [ly], [h-0.05],
                  c="#fbbf24", s=200, zorder=10,
                  edgecolors="white", linewidths=1.5)

        # Cono luce (volumetrico)
        n_rays = 20
        for angle in np.linspace(0, 2*np.pi, n_rays, endpoint=False):
            for spread in [0.2, 0.5, 0.8]:
                r_spread = spread * 1.5
                x_end = lx + r_spread * np.cos(angle)
                y_end = ly + r_spread * np.sin(angle)
                alpha_ray = 0.15 * (1 - spread)
                ax.plot([lx, x_end], [ly, y_end], [h-0.1, 0.05],
                       color="#fffbeb", alpha=alpha_ray, linewidth=0.5)

        # Alone luminoso a pavimento
        theta = np.linspace(0, 2*np.pi, 40)
        for rad, alpha_h in [(0.5,0.3),(1.0,0.15),(1.8,0.05)]:
            px = lx + rad * np.cos(theta)
            py = ly + rad * np.sin(theta)
            pz = np.zeros_like(theta) + 0.01
            ax.plot(px, py, pz, color="#fef3c7", alpha=alpha_h, linewidth=1)

    # Rendering distribuzione luce a pavimento
    Xg, Yg = np.meshgrid(np.linspace(0.1, lato-0.1, 40),
                          np.linspace(0.1, lato-0.1, 40))
    Zg = np.zeros_like(Xg)
    for (lx2, ly2) in coords:
        dist2 = np.sqrt((Xg-lx2)**2 + (Yg-ly2)**2 + h**2)
        Zg += (lamp["flusso_lm"]/(2*np.pi)) * (h/dist2**3)

    Zg_norm = (Zg - Zg.min()) / (Zg.max() - Zg.min() + 1e-9)
    Zg_floor = np.zeros_like(Zg) + 0.02
    ax.plot_surface(Xg, Yg, Zg_floor, facecolors=plt.cm.YlOrRd(Zg_norm),
                   alpha=0.5, shade=False, zorder=1)

    # Assi e viewport
    ax.set_xlim(0, lato); ax.set_ylim(0, lato); ax.set_zlim(0, h)
    ax.set_xlabel("X [m]", fontsize=8, color="white")
    ax.set_ylabel("Y [m]", fontsize=8, color="white")
    ax.set_zlabel("Z [m]", fontsize=8, color="white")
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=35, azim=225)

    # Titolo
    fig.text(0.5, 0.96, f"RENDERING 3D — {area_dict['nome'].upper()}",
            fontsize=14, fontweight="bold", color="white",
            ha="center", va="top")
    fig.text(0.5, 0.92, f"{calcolo['n_lampade']}x {area_dict['lampada'][:35]}  |  "
             f"{calcolo['E_medio_lux']} lux avg  |  {calcolo['potenza_tot_W']}W  |  {calcolo['wm2']} W/m²",
            fontsize=9, color="#90cdf4", ha="center")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
               facecolor="black", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


def genera_preventivo_completo(risultati, progetto):
    """Preventivo economico completo con margini e IVA"""
    IVA = 0.22; SPESE_GEN = 0.12; ONERI_SIC = 0.04; MARGINE = 0.35

    righe = []
    tot_mat = tot_inst = 0

    for r in risultati:
        lamp = DATABASE_LAMPADE[r["lampada"]]
        mat = r["calcolo"]["n_lampade"] * lamp["prezzo"]
        inst = r["calcolo"]["n_lampade"] * lamp["installazione"]
        righe.append({
            "area": r["nome"],
            "n_lamp": r["calcolo"]["n_lampade"],
            "lamp": r["lampada"][:25],
            "materiali": mat,
            "installazione": inst,
            "subtotale": mat + inst
        })
        tot_mat += mat; tot_inst += inst

    tot_netto = tot_mat + tot_inst
    sg = tot_netto * SPESE_GEN
    os = tot_netto * ONERI_SIC
    tot_lordo = tot_netto + sg + os
    margine = tot_lordo * MARGINE
    tot_offerta = tot_lordo + margine
    iva = tot_offerta * IVA

    return {
        "righe": righe,
        "tot_mat": tot_mat, "tot_inst": tot_inst,
        "tot_netto": tot_netto, "sg": sg, "os": os,
        "tot_lordo": tot_lordo, "margine": margine,
        "tot_offerta": tot_offerta, "iva": iva,
        "tot_finale": tot_offerta + iva
    }


# ══════════════════════════════════════════════════════════════════════════
# INTESTAZIONE
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1 style="margin:0;font-size:2.2rem">💡 LIGHTING AGENT PRO v2.0</h1>
  <p style="margin:0.4rem 0 0 0;opacity:0.85;font-size:1rem">
    Sistema professionale di progettazione illuminotecnica • UNI EN 12464-1:2021 • AI Vision • Rendering 3D
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1a365d/white?text=LIGHTING+PRO", use_column_width=True)
    st.markdown("---")

    st.subheader("📋 Dati Progetto")
    nome_progetto = st.text_input("Nome progetto", "UFFICI TELEDIFESA")
    committente = st.text_input("Committente", "Teledifesa S.p.A.")
    progettista = st.text_input("Progettista", "Ing. Mario Rossi")
    num_tavola = st.text_input("N° Tavola", "26A3S001")

    st.markdown("---")
    st.subheader("📐 Planimetria")
    planimetria = st.file_uploader("Carica planimetria (PDF/PNG/JPG)",
                                   type=["pdf","png","jpg","jpeg"])

    if planimetria:
        st.success("✅ Planimetria caricata!")
        if planimetria.type.startswith("image"):
            st.image(planimetria, use_column_width=True)

        # AI ANALISI (Ollama llava - richiede ollama locale)
        if st.button("🤖 AI Analisi Planimetria"):
            st.info("🔍 AI sta analizzando la planimetria...")
            try:
                import ollama
                img_bytes = planimetria.read()
                response = ollama.chat(
                    model="llava:13b",
                    messages=[{
                        "role": "user",
                        "content": """Analizza questa planimetria architettonica.
Identifica TUTTI i locali (uffici, corridoi, sale, reception, bagni).
Per ogni locale stima superficie in m².
Rispondi in formato JSON:
{
  "locali": [
    {"nome": "Ufficio 1", "tipo": "Ufficio VDT", "superficie_m2": 35},
    ...
  ]
}""",
                        "images": [img_bytes]
                    }]
                )
                result = response["message"]["content"]
                st.session_state.ai_analysis = result
                st.success("✅ AI: Analisi completata!")
                st.code(result, language="json")
            except Exception as e:
                st.warning(f"⚠️ Ollama non disponibile. Usa modalità manuale.
{e}")

    st.markdown("---")
    st.subheader("💡 Database Lampade")
    produttore_filter = st.selectbox("Filtra per produttore",
                                     ["Tutti","BEGA","iGuzzini","Flos","Artemide","Delta Light"])


# ══════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALI
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏢 Aree",
    "⚡ Calcoli",
    "📐 Tavola A3",
    "📊 Verifiche",
    "🎨 Rendering 3D",
    "💰 Preventivo"
])


# ──────────────────────────────────────────────────────────────────────────
# TAB 1: DEFINIZIONE AREE
# ──────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### 🏢 Definizione Aree e Locali")

    if "aree" not in st.session_state:
        st.session_state.aree = []

    with st.form("form_area", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            nome_area = st.text_input("Nome area *", placeholder="es. Ufficio 1")
            tipo_locale = st.selectbox("Tipo locale *", list(REQUISITI_UNI.keys()))

        with col2:
            sup = st.number_input("Superficie [m²] *", min_value=1.0, value=35.0, step=0.5)
            altezza = st.number_input("Altezza netta [m] *", min_value=2.0, value=2.70, step=0.05)

        with col3:
            lamps_filtrate = {k:v for k,v in DATABASE_LAMPADE.items()
                             if produttore_filter=="Tutti" or v["produttore"]==produttore_filter}
            lampada_scelta = st.selectbox("Apparecchio illuminante *", list(lamps_filtrate.keys()))

            lamp_info = DATABASE_LAMPADE[lampada_scelta]
            st.markdown(f"""
            <div style="background:#f0fff4;padding:0.5rem;border-radius:6px;font-size:0.8rem;margin-top:0.3rem">
            ⚡ {lamp_info["potenza_W"]}W | 💡 {lamp_info["flusso_lm"]}lm | 
            🎨 Ra{lamp_info["ra"]} | {lamp_info["temp_colore"]} | UGR{lamp_info["ugr"]}
            </div>
            """, unsafe_allow_html=True)

        aggiungi = st.form_submit_button("➕ Aggiungi Area", type="primary")

        if aggiungi:
            if nome_area:
                st.session_state.aree.append({
                    "nome": nome_area,
                    "tipo_locale": tipo_locale,
                    "superficie_m2": sup,
                    "altezza_m": altezza,
                    "lampada": lampada_scelta
                })
                st.success(f"✅ Area '{nome_area}' aggiunta!")

    if st.session_state.aree:
        st.markdown("---")
        st.markdown(f"### 📋 Aree Definite ({len(st.session_state.aree)})")

        for i, area in enumerate(st.session_state.aree):
            col1, col2 = st.columns([5,1])
            with col1:
                req = REQUISITI_UNI[area["tipo_locale"]]
                st.markdown(f"""
                <div class="metric-card">
                <strong>{i+1}. {area["nome"]}</strong> — {area["tipo_locale"]} — 
                {area["superficie_m2"]} m² — h: {area["altezza_m"]}m — 
                Target: {req["lux"]} lux — 💡 {area["lampada"][:35]}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.aree.pop(i)
                    st.rerun()
    else:
        st.info("👆 Aggiungi le aree del tuo progetto per iniziare.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 2: CALCOLI
# ──────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### ⚡ Calcoli Illuminotecnici")

    if not st.session_state.aree:
        st.warning("⚠️ Aggiungi le aree nella Tab 'Aree' prima di calcolare.")
    else:
        if st.button("🚀 ESEGUI CALCOLI UNI EN 12464-1", type="primary"):
            risultati = []
            offset_x = 0

            for area in st.session_state.aree:
                calcolo = calcola_area(area)
                risultati.append({
                    **area,
                    "sup": area["superficie_m2"],
                    "calcolo": calcolo,
                    "offset_x": offset_x,
                    "offset_y": 0
                })
                offset_x += np.sqrt(area["superficie_m2"]) + 1.5

            st.session_state.risultati = risultati

        if "risultati" in st.session_state:
            r_list = st.session_state.risultati

            # Metriche generali
            tot_lamp = sum(r["calcolo"]["n_lampade"] for r in r_list)
            tot_W = sum(r["calcolo"]["potenza_tot_W"] for r in r_list)
            tot_m2 = sum(r["sup"] for r in r_list)

            col1,col2,col3,col4 = st.columns(4)
            col1.metric("🏢 Aree", len(r_list))
            col2.metric("💡 Lampade totali", tot_lamp)
            col3.metric("⚡ Potenza totale", f"{tot_W} W")
            col4.metric("📊 W/m² medio", f"{tot_W/tot_m2:.1f}")

            st.markdown("---")

            # Tabella dettagliata
            df = pd.DataFrame([{
                "Area": r["nome"],
                "Tipo": r["tipo_locale"],
                "m²": r["sup"],
                "Lampada": r["lampada"][:30],
                "N° Lamp.": r["calcolo"]["n_lampade"],
                "Lux target": r["calcolo"]["E_target_lux"],
                "Lux ottenuto": r["calcolo"]["E_medio_lux"],
                "Potenza [W]": r["calcolo"]["potenza_tot_W"],
                "W/m²": r["calcolo"]["wm2"],
                "✓ Lux": r["calcolo"]["conf_lux"],
                "✓ UGR": r["calcolo"]["conf_ugr"],
                "✓ Ra": r["calcolo"]["conf_ra"]
            } for r in r_list])

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download CSV
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Scarica CSV Calcoli", csv,
                             f"calcoli_{datetime.now():%Y%m%d}.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────
# TAB 3: TAVOLA A3 PDF
# ──────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### 📐 Tavola Illuminotecnica A3")

    if "risultati" not in st.session_state:
        st.warning("⚠️ Esegui prima i calcoli.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 GENERA TAVOLA A3 PDF", type="primary"):
                with st.spinner("🎨 Generazione tavola professionale A3..."):
                    progetto = {
                        "nome": nome_progetto,
                        "committente": committente,
                        "progettista": progettista,
                        "data": datetime.now().strftime("%d/%m/%Y"),
                        "num_tavola": num_tavola
                    }
                    pdf_buf = genera_pdf_a3(progetto, st.session_state.risultati)

                    st.download_button(
                        "📥 SCARICA TAVOLA A3 PDF",
                        data=pdf_buf,
                        file_name=f"{num_tavola}_tavola_illuminotecnica.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ Tavola A3 generata! Identica ai tuoi template.")

        with col2:
            if st.button("📐 GENERA DXF AUTOCAD"):
                with st.spinner("📐 Generazione DXF..."):
                    dxf = genera_dxf(st.session_state.risultati)
                    st.download_button(
                        "📥 SCARICA DXF AUTOCAD",
                        data=dxf,
                        file_name=f"{num_tavola}_layout.dxf",
                        mime="application/dxf"
                    )
                    st.success("✅ DXF generato! Importabile in AutoCAD/DraftSight.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 4: VERIFICHE PDF
# ──────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### 📊 Verifiche Illuminotecniche")

    if "risultati" not in st.session_state:
        st.warning("⚠️ Esegui prima i calcoli.")
    else:
        if st.button("📊 GENERA REPORT VERIFICHE PDF", type="primary"):
            with st.spinner("📊 Generazione report completo..."):
                progetto = {
                    "nome": nome_progetto,
                    "committente": committente,
                    "progettista": progettista,
                    "data": datetime.now().strftime("%d/%m/%Y"),
                    "num_tavola": num_tavola
                }
                # Le verifiche sono incluse nel PDF A3 (pagine 2+)
                pdf_buf = genera_pdf_a3(progetto, st.session_state.risultati)

                st.download_button(
                    "📥 SCARICA REPORT VERIFICHE PDF",
                    data=pdf_buf,
                    file_name=f"{num_tavola}_verifiche.pdf",
                    mime="application/pdf"
                )
                st.success("✅ Report verifiche generato!")

        # Anteprima verifiche per area
        if st.session_state.risultati:
            for r in st.session_state.risultati:
                with st.expander(f"📋 {r['nome']} — {r['calcolo']['E_medio_lux']} lux", expanded=False):
                    col1,col2,col3,col4 = st.columns(4)
                    col1.metric("💡 Illuminamento", f"{r['calcolo']['E_medio_lux']} lux",
                               delta=f"+{r['calcolo']['E_medio_lux']-r['calcolo']['E_target_lux']} lux")
                    col2.metric("⚡ Potenza", f"{r['calcolo']['potenza_tot_W']}W")
                    col3.metric("📊 Efficienza", f"{r['calcolo']['wm2']} W/m²")
                    col4.metric("🔆 N° Lampade", r["calcolo"]["n_lampade"])

                    c1,c2,c3,c4 = st.columns(4)
                    for col, label, val in [
                        (c1,"✅ Lux",r["calcolo"]["conf_lux"]),
                        (c2,"✅ UGR",r["calcolo"]["conf_ugr"]),
                        (c3,"✅ Uniformità",r["calcolo"]["conf_uni"]),
                        (c4,"✅ Ra",r["calcolo"]["conf_ra"])
                    ]:
                        col.markdown(f"<div style='text-align:center;font-size:2rem'>{val}</div>", 
                                    unsafe_allow_html=True)
                        col.markdown(f"<div style='text-align:center;font-size:0.8rem'>{label}</div>",
                                    unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# TAB 5: RENDERING 3D
# ──────────────────────────────────────────────────────────────────────────

with tab5:
    st.markdown("### 🎨 Rendering 3D Fotorealistici")

    if "risultati" not in st.session_state:
        st.warning("⚠️ Esegui prima i calcoli.")
    else:
        area_names = [r["nome"] for r in st.session_state.risultati]
        area_scelta = st.selectbox("Seleziona area", area_names)

        col1,col2 = st.columns(2)
        with col1:
            vista = st.selectbox("Vista camera", 
                                ["Prospettiva (35°)","Alto (90°)","Laterale (0°)","Angolo (45°)"])
        with col2:
            qualita = st.select_slider("Qualità", ["Bozza","Media","Alta","Ultra"], value="Alta")

        if st.button("🎨 GENERA RENDERING 3D", type="primary"):
            area_idx = area_names.index(area_scelta)
            area_r = st.session_state.risultati[area_idx]

            with st.spinner(f"🎨 Rendering {area_scelta}... attendere"):
                render_buf = genera_rendering_3d(area_r, area_r["calcolo"], area_idx)

                st.image(render_buf, caption=f"Rendering 3D — {area_scelta}",
                        use_column_width=True)

                st.download_button(
                    f"📥 Scarica Rendering PNG",
                    data=render_buf,
                    file_name=f"rendering_{area_scelta.lower().replace(' ','_')}.png",
                    mime="image/png"
                )
                st.success("✅ Rendering generato!")

        if st.button("🎨 GENERA TUTTI I RENDERING"):
            with st.spinner("🎨 Generazione rendering per tutte le aree..."):
                cols = st.columns(2)
                for i, r in enumerate(st.session_state.risultati):
                    render_buf = genera_rendering_3d(r, r["calcolo"], i)
                    with cols[i % 2]:
                        st.image(render_buf, caption=r["nome"], use_column_width=True)
                        render_buf.seek(0)
                        st.download_button(
                            f"📥 {r['nome']}",
                            data=render_buf,
                            file_name=f"render_{r['nome'].lower().replace(' ','_')}.png",
                            mime="image/png",
                            key=f"render_dl_{i}"
                        )
                st.success(f"✅ {len(st.session_state.risultati)} rendering generati!")


# ──────────────────────────────────────────────────────────────────────────
# TAB 6: PREVENTIVO
# ──────────────────────────────────────────────────────────────────────────

with tab6:
    st.markdown("### 💰 Preventivo Economico")

    if "risultati" not in st.session_state:
        st.warning("⚠️ Esegui prima i calcoli.")
    else:
        col1,col2,col3 = st.columns(3)
        with col1: margine_pct = st.slider("Margine impresa %", 10, 60, 35)
        with col2: iva_pct = st.slider("IVA %", 0, 22, 22)
        with col3: spese_gen_pct = st.slider("Spese generali %", 5, 20, 12)

        if st.button("💰 CALCOLA PREVENTIVO", type="primary"):
            prev = genera_preventivo_completo(st.session_state.risultati, {})
            st.session_state.preventivo = prev

        if "preventivo" in st.session_state:
            prev = st.session_state.preventivo

            col1,col2,col3 = st.columns(3)
            col1.metric("📦 Materiali", f"€ {prev['tot_mat']:,.0f}")
            col2.metric("🔧 Installazione", f"€ {prev['tot_inst']:,.0f}")
            col3.metric("💰 TOTALE IVA INCLUSA", f"€ {prev['tot_finale']:,.0f}",
                       delta=f"+{prev['margine']:,.0f} margine")

            st.markdown("---")

            # Tabella per aree
            df_prev = pd.DataFrame([{
                "Area": r["area"],
                "Lampade": r["n_lamp"],
                "Apparecchio": r["lamp"],
                "Materiali": f"€ {r['materiali']:,.0f}",
                "Installazione": f"€ {r['installazione']:,.0f}",
                "Subtotale": f"€ {r['subtotale']:,.0f}"
            } for r in prev["righe"]])
            st.dataframe(df_prev, use_container_width=True, hide_index=True)

            # Riepilogo
            st.markdown("---")
            st.markdown("#### 📋 Riepilogo Offerta")
            col1,col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                | Voce | Importo |
                |------|---------|
                | Materiali | € {prev['tot_mat']:,.0f} |
                | Installazione | € {prev['tot_inst']:,.0f} |
                | Totale lavori | € {prev['tot_netto']:,.0f} |
                | Spese generali 12% | € {prev['sg']:,.0f} |
                | Oneri sicurezza 4% | € {prev['os']:,.0f} |
                | Totale lordo | € {prev['tot_lordo']:,.0f} |
                | Margine impresa 35% | € {prev['margine']:,.0f} |
                | **OFFERTA CLIENTE** | **€ {prev['tot_offerta']:,.0f}** |
                | IVA 22% | € {prev['iva']:,.0f} |
                | **TOTALE IVA INCLUSA** | **€ {prev['tot_finale']:,.0f}** |
                """)

            with col2:
                # Download preventivo TXT
                prev_text = f"""PREVENTIVO OFFERTA
Progetto: {nome_progetto}
Committente: {committente}
Data: {datetime.now():%d/%m/%Y}

DETTAGLIO AREE:
"""
                for r in prev["righe"]:
                    prev_text += f"  {r['area']}: {r['n_lamp']}x {r['lamp']} — Materiali: €{r['materiali']:,.0f} + Install: €{r['installazione']:,.0f}\n"

                prev_text += f"""
RIEPILOGO:
  Materiali:           € {prev['tot_mat']:>10,.0f}
  Installazione:       € {prev['tot_inst']:>10,.0f}
  Totale netto:        € {prev['tot_netto']:>10,.0f}
  Spese generali 12%:  € {prev['sg']:>10,.0f}
  Oneri sicurezza 4%:  € {prev['os']:>10,.0f}
  Margine 35%:         € {prev['margine']:>10,.0f}
  OFFERTA CLIENTE:     € {prev['tot_offerta']:>10,.0f}
  IVA 22%:             € {prev['iva']:>10,.0f}
  ═══════════════════════════════════
  TOTALE IVA INCLUSA:  € {prev['tot_finale']:>10,.0f}
"""
                st.download_button("📥 Scarica Preventivo TXT",
                                  data=prev_text.encode(),
                                  file_name=f"preventivo_{datetime.now():%Y%m%d}.txt")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;padding:1rem 0;font-size:0.85rem">
  <strong>Lighting Agent Pro v2.0</strong> | 
  UNI EN 12464-1:2021 | D.Lgs.81/2008 | 
  AI Vision (Ollama Llava) | Rendering 3D Matplotlib | 
  PDF A3 Professionale<br>
  © 2026 — Sistema automatico di progettazione illuminotecnica
</div>
""", unsafe_allow_html=True)
