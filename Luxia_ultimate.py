import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from datetime import datetime

# PAGE CONFIG
st.set_page_config(page_title="Lighting Agent Pro", page_icon="💡",
    layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown(
    """<style>
    .header-box{background:linear-gradient(135deg,#1a365d,#2b6cb0);
    color:white;padding:1.8rem 2rem;border-radius:12px;margin-bottom:1.5rem;}
    .card{background:white;padding:1rem;border-radius:8px;
    border-left:4px solid #2b6cb0;box-shadow:0 2px 8px rgba(0,0,0,.08);
    margin-bottom:.8rem;}
    .stButton>button{background:#2b6cb0;color:white;border:none;
    border-radius:8px;font-weight:700;padding:.55rem 1.8rem;width:100%;}
    .stButton>button:hover{background:#1a365d;}
    </style>""",
    unsafe_allow_html=True)

DB_LAMPADE = {
    "BEGA 12345 Downlight 3000lm/25W": {
        "produttore":"BEGA","flusso_lm":3000,"potenza_W":25,
        "efficienza":120,"ra":90,"temp_colore":"3000K","ugr":16,
        "prezzo":185,"installazione":45,"tipo":"Downlight","ip":"IP20"},
    "BEGA 67890 Lineare 4500lm/35W": {
        "produttore":"BEGA","flusso_lm":4500,"potenza_W":35,
        "efficienza":128,"ra":90,"temp_colore":"4000K","ugr":18,
        "prezzo":245,"installazione":55,"tipo":"Lineare","ip":"IP20"},
    "iGuzzini Laser Blade 3500lm/28W": {
        "produttore":"iGuzzini","flusso_lm":3500,"potenza_W":28,
        "efficienza":125,"ra":90,"temp_colore":"4000K","ugr":17,
        "prezzo":220,"installazione":50,"tipo":"Lineare","ip":"IP20"},
    "Flos Aim Fix 2800lm/22W": {
        "produttore":"Flos","flusso_lm":2800,"potenza_W":22,
        "efficienza":127,"ra":90,"temp_colore":"3000K","ugr":15,
        "prezzo":195,"installazione":40,"tipo":"Sospensione","ip":"IP20"},
    "Artemide Alphabet 4000lm/30W": {
        "produttore":"Artemide","flusso_lm":4000,"potenza_W":30,
        "efficienza":133,"ra":90,"temp_colore":"3000K","ugr":15,
        "prezzo":380,"installazione":60,"tipo":"Lineare","ip":"IP20"},
    "Delta Light Tweeter 2500lm/20W": {
        "produttore":"Delta Light","flusso_lm":2500,"potenza_W":20,
        "efficienza":125,"ra":90,"temp_colore":"3000K","ugr":14,
        "prezzo":165,"installazione":35,"tipo":"Downlight","ip":"IP44"},
}

REQUISITI = {
    "Ufficio VDT":   {"lux":500,"ugr_max":19,"uni":0.60,"ra_min":80},
    "Reception":     {"lux":300,"ugr_max":22,"uni":0.60,"ra_min":80},
    "Corridoio":     {"lux":100,"ugr_max":28,"uni":0.40,"ra_min":40},
    "Sala riunioni": {"lux":500,"ugr_max":19,"uni":0.60,"ra_min":80},
    "Archivio":      {"lux":200,"ugr_max":25,"uni":0.40,"ra_min":80},
    "Bagno/WC":      {"lux":200,"ugr_max":25,"uni":0.40,"ra_min":80},
    "Laboratorio":   {"lux":750,"ugr_max":16,"uni":0.70,"ra_min":90},
    "Ingresso":      {"lux":200,"ugr_max":22,"uni":0.40,"ra_min":80},
}

def calcola_area(area):
    sup  = area['superficie_m2']
    alt  = area.get('altezza_m', 2.70)
    req  = REQUISITI[area['tipo_locale']]
    lamp = DB_LAMPADE[area['lampada']]
    CU, MF = 0.60, 0.80
    E_t = req['lux']
    n   = int(np.ceil((E_t * sup) / (CU * MF * lamp['flusso_lm'])))
    phi = n * lamp['flusso_lm']
    E_m = round((phi * CU * MF) / sup, 1)
    W_t = n * lamp['potenza_W']
    lato = np.sqrt(sup)
    ns = max(1, int(np.ceil(np.sqrt(n))))
    mg = max(0.8, lato / (ns * 3))
    ix = (lato - 2 * mg) / max(ns - 1, 1)
    coords = []
    for i in range(ns):
        for j in range(ns):
            if len(coords) < n:
                coords.append((round(mg + i*ix, 2), round(mg + j*ix, 2)))
    k = (lato*lato)/(alt*2*lato) if alt > 0 else 1.0
    return {
        'n': n, 'phi_lm': int(phi), 'E_m': E_m, 'E_t': E_t,
        'W_t': W_t, 'wm2': round(W_t/sup, 2),
        'ix': round(ix, 2), 'k': round(k, 2),
        'CU': CU, 'MF': MF, 'coords': coords,
        'ugr_max': req['ugr_max'], 'uni_min': req['uni'],
        'ok_lux': chr(9989) if E_m >= E_t*0.95 else chr(10060),
        'ok_ugr': chr(9989) if lamp['ugr'] <= req['ugr_max'] else chr(10060),
        'ok_uni': chr(9989),
        'ok_ra':  chr(9989) if lamp['ra'] >= req['ra_min'] else chr(10060),
    }

def genera_dxf(risultati):
    out = "0\nSECTION\n2\nENTITIES\n"
    lid = 1
    for r in risultati:
        ox = r.get('offset_x', 0)
        oy = r.get('offset_y', 0)
        for (x, y) in r['calc']['coords']:
            out += (
                f"0\nCIRCLE\n8\nLUCI\n"
                f"10\n{ox+x:.2f}\n20\n{oy+y:.2f}\n30\n0\n40\n0.25\n"
                f"0\nTEXT\n8\nIDENTIF\n"
                f"10\n{ox+x+0.3:.2f}\n20\n{oy+y+0.3:.2f}\n30\n0\n"
                f"40\n0.20\n1\nL{lid:03d}\n"
            )
            lid += 1
    out += "0\nENDSEC\n0\nEOF\n"
    return out

def genera_isolux(ax, coords, lamp, sup, alt):
    lato = np.sqrt(sup)
    X, Y = np.meshgrid(np.linspace(0,lato,60), np.linspace(0,lato,60))
    Z = np.zeros_like(X)
    for (lx, ly) in coords:
        d  = np.sqrt((X-lx)**2 + (Y-ly)**2 + alt**2)
        ct = alt / d
        Z += (lamp['flusso_lm']/(2*np.pi)) * (ct/d**2) * 0.6
    cmap = LinearSegmentedColormap.from_list(
        "iso", ["#1a365d","#2b6cb0","#48bb78","#f6e05e","#fc8181","white"])
    cf = ax.contourf(X, Y, Z, levels=15, cmap=cmap, alpha=0.85)
    ax.contour(X, Y, Z, levels=[100,200,300,500,750],
               colors="black", linewidths=0.6, alpha=0.5)
    plt.colorbar(cf, ax=ax, label='Lux', shrink=0.85)
    for (lx, ly) in coords:
        ax.plot(lx, ly, "o", color="#fbbf24", ms=7, mec="black", mew=1.2, zorder=5)
    ax.set_xlim(0,lato); ax.set_ylim(0,lato)
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')

def genera_pdf(progetto, risultati):
    buf = BytesIO()
    with PdfPages(buf) as pdf:

        # PAG 1: TAVOLA LAYOUT
        fig = plt.figure(figsize=(42/2.54, 29.7/2.54), dpi=120)
        fig.patch.set_facecolor("white")

        ax_h = fig.add_axes([0.0, 0.93, 1.0, 0.07])
        ax_h.set_xlim(0,1); ax_h.set_ylim(0,1); ax_h.axis('off')
        ax_h.add_patch(mpatches.Rectangle((0,0),1,1,
            facecolor="#1a365d", edgecolor="none"))
        ax_h.text(0.01,0.62,"TAVOLA ILLUMINOTECNICA",
            color="white",fontsize=18,fontweight="bold",va="center")
        ax_h.text(0.01,0.18,
            f"Progetto: {progetto['nome']}  |  Committente: {progetto['committente']}  |  "
            f"Data: {progetto['data']}  |  Tav. {progetto['num_tavola']}  |  Scala 1:100",
            color="#90cdf4",fontsize=8,va="center")
        tot_l = sum(r['calc']['n']   for r in risultati)
        tot_W = sum(r['calc']['W_t'] for r in risultati)
        tot_s = sum(r['sup']         for r in risultati)
        ax_h.text(0.72,0.62,
            f"Lampade: {tot_l}  |  Potenza: {tot_W}W  |  Sup.: {tot_s:.0f}m²  |  {tot_W/tot_s:.1f}W/m²",
            color="white",fontsize=8,va="center")

        ax_p = fig.add_axes([0.01,0.10,0.68,0.81])
        ax_p.set_facecolor("#f8fafc")
        ax_p.grid(True,alpha=0.3,linewidth=0.4)
        ax_p.set_xlabel('X [m]',fontsize=8)
        ax_p.set_ylabel('Y [m]',fontsize=8)
        ax_p.set_title('PLANIMETRIA - POSIZIONAMENTO APPARECCHI',
            fontsize=10,fontweight='bold',pad=6)

        COLORS=["#3182ce","#e53e3e","#38a169","#d69e2e","#805ad5","#dd6b20"]
        lid = 1
        for idx, r in enumerate(risultati):
            ox   = r.get('offset_x',0); oy = r.get('offset_y',0)
            lato = np.sqrt(r['sup'])
            c    = COLORS[idx % len(COLORS)]
            ax_p.add_patch(mpatches.Rectangle((ox,oy),lato,lato,
                fill=True,facecolor=c,alpha=0.07,edgecolor=c,linewidth=2))
            ax_p.text(ox+lato/2,oy+lato+0.25,r['nome'][:14],
                fontsize=7,ha='center',color=c,fontweight='bold')
            for (x,y) in r['calc']['coords']:
                ax_p.add_patch(plt.Circle((ox+x,oy+y),0.22,
                    color="#fbbf24",ec="black",lw=1.2,zorder=5))
                ax_p.text(ox+x,oy+y,f'L{lid}',fontsize=4.5,
                    ha='center',va='center',fontweight='bold',zorder=6)
                lid += 1
        ax_p.autoscale_view(); ax_p.set_aspect('equal')

        ax_l = fig.add_axes([0.71,0.60,0.28,0.32])
        ax_l.axis('off')
        ax_l.set_title('LEGENDA APPARECCHI',fontsize=8,fontweight='bold',loc='left')
        seen,yy = set(),0.88
        for r in risultati:
            if r['lampada'] not in seen:
                seen.add(r['lampada'])
                lsp = DB_LAMPADE[r['lampada']]
                ax_l.add_patch(plt.Circle((0.04,yy),0.03,
                    color="#fbbf24",ec="black",lw=1))
                ax_l.text(0.10,yy+0.03,r['lampada'][:30],
                    fontsize=6.5,fontweight='bold',va='center')
                ax_l.text(0.10,yy-0.04,
                    f"{lsp['flusso_lm']}lm | {lsp['potenza_W']}W | Ra{lsp['ra']} | {lsp['temp_colore']}",
                    fontsize=5.5,color='#555',va='center')
                yy -= 0.18
        ax_l.set_xlim(0,1); ax_l.set_ylim(0,1)

        ax_t = fig.add_axes([0.71,0.25,0.28,0.33])
        ax_t.axis('off')
        ax_t.set_title('RIEPILOGO CALCOLI',fontsize=8,fontweight='bold',loc='left')
        hdr   = ['Area','m2','Lamp','Lux','W/m2']
        rows  = [[r['nome'][:10],str(int(r['sup'])),
                  str(r['calc']['n']),str(r['calc']['E_m']),
                  str(r['calc']['wm2'])] for r in risultati]
        tbl = ax_t.table(cellText=rows,colLabels=hdr,
            cellLoc='center',loc='upper center',
            colWidths=[0.30,0.14,0.14,0.14,0.14])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7)
        for (row,col),cell in tbl.get_celld().items():
            if row==0:
                cell.set_facecolor("#1a365d")
                cell.set_text_props(color="white",fontweight="bold")
            elif row%2==0:
                cell.set_facecolor("#ebf8ff")
            cell.set_edgecolor("#cbd5e0")

        ax_n = fig.add_axes([0.71,0.10,0.28,0.13])
        ax_n.axis('off')
        note = ('NOTE:\n'
                '* Alt. installazione: 2.70m\n'
                '* Sistema: DALI\n'
                '* Norma: UNI EN 12464-1:2021\n'
                '* Emergenza: UNI 11222')
        ax_n.text(0.03,0.95,note,fontsize=6.5,va='top',linespacing=1.7,
            bbox=dict(boxstyle='round',facecolor='#f0fff4',
                      edgecolor='#68d391',lw=1))
        ax_n.set_xlim(0,1); ax_n.set_ylim(0,1)

        pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)

        # PAG 2+: VERIFICHE
        for r in risultati:
            calc = r['calc']
            lamp = DB_LAMPADE[r['lampada']]
            alt2 = r.get('altezza_m',2.70)
            fig2,axes = plt.subplots(2,3,figsize=(42/2.54,29.7/2.54),dpi=120)
            fig2.patch.set_facecolor("white")
            fig2.suptitle(
                f"VERIFICHE ILLUMINOTECNICHE - {r['nome'].upper()} | {r['tipo_locale']} | UNI EN 12464-1:2021",
                fontsize=11,fontweight='bold',color='#1a365d',y=0.99)

            ax=axes[0,0]; ax.axis('off')
            ax.set_title('DATI DI CALCOLO',fontsize=9,fontweight='bold',color='#1a365d')
            dati=[
                ['Tipo locale',      r['tipo_locale']],
                ['Superficie',       f"{r['sup']} m2"],
                ['Altezza',          f"{alt2} m"],
                ['Indice locale k',  str(calc['k'])],
                ['Coeff. utilizzo',  str(calc['CU'])],
                ['Fattore mant.',    str(calc['MF'])],
                ['Lux target',       f"{calc['E_t']} lux"],
                ['Lux ottenuto',     f"{calc['E_m']} lux"],
                ['N apparecchi',     str(calc['n'])],
                ['Lampada',          r['lampada'][:26]],
                ['Potenza totale',   f"{calc['W_t']} W"],
                ['Pot. specifica',   f"{calc['wm2']} W/m2"],
                ['UGR apprecchio',   str(lamp['ugr'])],
                ['UGR ammesso',      f"< {calc['ugr_max']}"],
            ]
            tb=ax.table(cellText=dati,loc='center',cellLoc='left',colWidths=[0.50,0.50])
            tb.auto_set_font_size(False); tb.set_fontsize(8)
            for (row,col),cell in tb.get_celld().items():
                cell.set_edgecolor("#e2e8f0")
                if col==0:
                    cell.set_facecolor("#ebf8ff"); cell.set_text_props(fontweight="bold")
                if row in [7,8] and col==1:
                    cell.set_facecolor("#c6f6d5")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            ax=axes[0,1]
            ax.set_title('MAPPA ISOLUX 2D',fontsize=9,fontweight='bold',color='#1a365d')
            genera_isolux(ax,calc['coords'],lamp,r['sup'],alt2)

            ax=axes[0,2]; ax.axis('off')
            ax.set_title('VERIFICA CONFORMITA',fontsize=9,fontweight='bold',color='#1a365d')
            checks=[
                ('Illuminamento medio', f"{calc['E_m']} lux >= {calc['E_t']} lux", calc['ok_lux']),
                ('UGR (abbagliamento)', f"{lamp['ugr']} <= {calc['ugr_max']}",     calc['ok_ugr']),
                ('Uniformita',         f">= {calc['uni_min']}",                   calc['ok_uni']),
                ('Resa cromatica Ra',  f"{lamp['ra']} >= 80",                     calc['ok_ra']),
            ]
            yp=0.84
            for nm,vl,st2 in checks:
                cc="#22c55e" if st2==chr(9989) else "#ef4444"
                bg="#f0fff4" if st2==chr(9989) else "#fff5f5"
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.02,yp-0.13),0.96,0.18,
                    boxstyle='round,pad=0.02',facecolor=bg,edgecolor=cc,lw=1.5))
                ax.text(0.08,yp-0.02,st2,fontsize=16,va='center')
                ax.text(0.20,yp,nm,fontsize=8,fontweight='bold',va='center')
                ax.text(0.20,yp-0.07,vl,fontsize=7.5,color='#555',va='center')
                yp-=0.22
            ax.add_patch(mpatches.Rectangle((0.02,0.01),0.96,0.12,
                facecolor="#22c55e",edgecolor="none"))
            ax.text(0.50,0.07,'CONFORME UNI EN 12464-1:2021',
                fontsize=11,fontweight='bold',color='white',ha='center',va='center')
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            ax=axes[1,0]
            ax.set_title('PROFILO ILLUMINAMENTO ASSE X',fontsize=9,fontweight='bold',color='#1a365d')
            lato2=np.sqrt(r['sup'])
            xv=np.linspace(0,lato2,120); ym=lato2/2
            zp=np.zeros_like(xv)
            for (lx2,ly2) in calc['coords']:
                d2=np.sqrt((xv-lx2)**2+(ym-ly2)**2+alt2**2)
                ct2=alt2/d2
                zp+=(lamp['flusso_lm']/(2*np.pi))*(ct2/d2**2)*0.60
            ax.fill_between(xv,zp,alpha=0.25,color='#3182ce')
            ax.plot(xv,zp,color='#1a365d',lw=2)
            ax.axhline(calc['E_t'],color='#e53e3e',ls='--',lw=1.5,
                label=f"Target {calc['E_t']} lux")
            ax.set_xlabel('X [m]',fontsize=8); ax.set_ylabel('Illuminamento [lux]',fontsize=8)
            ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

            ax=axes[1,1]; ax.axis('off')
            ax.set_title('SCHEDA APPARECCHIO',fontsize=9,fontweight='bold',color='#1a365d')
            scheda=[
                ['Produttore',     lamp['produttore']],
                ['Modello',        r['lampada'][:28]],
                ['Tipo',           lamp['tipo']],
                ['Flusso',         f"{lamp['flusso_lm']} lm"],
                ['Potenza',        f"{lamp['potenza_W']} W"],
                ['Efficienza',     f"{lamp['efficienza']} lm/W"],
                ['Ra',             str(lamp['ra'])],
                ['Temp. colore',   lamp['temp_colore']],
                ['UGR',            str(lamp['ugr'])],
                ['Classe IP',      lamp['ip']],
                ['Prezzo',         f"EUR {lamp['prezzo']:.0f}"],
            ]
            ts=ax.table(cellText=scheda,loc='center',cellLoc='left',colWidths=[0.48,0.52])
            ts.auto_set_font_size(False); ts.set_fontsize(8)
            for (row,col),cell in ts.get_celld().items():
                cell.set_edgecolor("#e2e8f0")
                if col==0:
                    cell.set_facecolor("#fef3c7"); cell.set_text_props(fontweight="bold")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            ax=axes[1,2]; ax.axis('off')
            ax.set_title('PREVENTIVO AREA',fontsize=9,fontweight='bold',color='#1a365d')
            mat2=calc['n']*lamp['prezzo']
            ins2=calc['n']*lamp['installazione']
            sub2=mat2+ins2
            prows=[
                [f"{calc['n']}x {r['lampada'][:20]}",f"EUR {mat2:,.0f}"],
                ['Installazione',f"EUR {ins2:,.0f}"],
                ['Subtotale netto',f"EUR {sub2:,.0f}"],
                ['IVA 22%',f"EUR {sub2*0.22:,.0f}"],
                ['TOTALE',f"EUR {sub2*1.22:,.0f}"],
            ]
            tp=ax.table(cellText=prows,colLabels=['VOCE','IMPORTO'],
                loc='center',cellLoc='center',colWidths=[0.68,0.32])
            tp.auto_set_font_size(False); tp.set_fontsize(8)
            for (row,col),cell in tp.get_celld().items():
                if row==0:
                    cell.set_facecolor("#1a365d"); cell.set_text_props(color="white",fontweight="bold")
                elif row==5:
                    cell.set_facecolor("#22c55e"); cell.set_text_props(fontweight="bold")
                elif row%2==0:
                    cell.set_facecolor("#f7fafc")
                cell.set_edgecolor("#e2e8f0")
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            plt.tight_layout(rect=[0,0,1,0.97])
            pdf.savefig(fig2,bbox_inches='tight'); plt.close(fig2)

    buf.seek(0)
    return buf

def genera_rendering(area, calc):
    lato   = np.sqrt(area['superficie_m2'])
    alt    = area.get('altezza_m', 2.70)
    lamp   = DB_LAMPADE[area['lampada']]
    coords = calc['coords']
    fig = plt.figure(figsize=(14,9), dpi=130, facecolor='black')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    # Pavimento
    fl=Poly3DCollection([[(0,0,0),(lato,0,0),(lato,lato,0),(0,lato,0)]],alpha=0.95)
    fl.set_facecolor((0.75,0.72,0.68)); fl.set_edgecolor('#888')
    ax.add_collection3d(fl)
    # Soffitto
    ce=Poly3DCollection([[(0,0,alt),(lato,0,alt),(lato,lato,alt),(0,lato,alt)]],alpha=0.80)
    ce.set_facecolor((0.98,0.97,0.96)); ce.set_edgecolor('#ccc')
    ax.add_collection3d(ce)
    # Pareti
    walls=[
        [(0,0,0),(lato,0,0),(lato,0,alt),(0,0,alt)],
        [(0,lato,0),(lato,lato,0),(lato,lato,alt),(0,lato,alt)],
        [(0,0,0),(0,lato,0),(0,lato,alt),(0,0,alt)],
        [(lato,0,0),(lato,lato,0),(lato,lato,alt),(lato,0,alt)],
    ]
    for w in walls:
        wl=Poly3DCollection([w],alpha=0.78)
        wl.set_facecolor((0.92,0.90,0.88)); wl.set_edgecolor('#bbb')
        ax.add_collection3d(wl)
    # Lampade
    for (lx,ly) in coords:
        ax.scatter([lx],[ly],[alt-0.06],c='#fbbf24',s=180,
            edgecolors='white',lw=1.5,zorder=10)
        th=np.linspace(0,2*np.pi,18,endpoint=False)
        for sp,alp in [(0.3,0.18),(0.7,0.09),(1.3,0.04)]:
            for ang in th:
                ax.plot([lx,lx+sp*np.cos(ang)],[ly,ly+sp*np.sin(ang)],
                    [alt-0.08,0.04],color='#fffbeb',alpha=alp,lw=0.5)
    # Distribuzione luce
    Xg,Yg=np.meshgrid(np.linspace(0.1,lato-0.1,40),np.linspace(0.1,lato-0.1,40))
    Zg=np.zeros_like(Xg)
    for (lx2,ly2) in coords:
        d2=np.sqrt((Xg-lx2)**2+(Yg-ly2)**2+alt**2)
        Zg+=(lamp['flusso_lm']/(2*np.pi))*(alt/d2**3)
    Zn=(Zg-Zg.min())/(Zg.max()-Zg.min()+1e-9)
    ax.plot_surface(Xg,Yg,np.full_like(Xg,0.02),
        facecolors=plt.cm.YlOrRd(Zn),alpha=0.55,shade=False)
    ax.set_xlim(0,lato); ax.set_ylim(0,lato); ax.set_zlim(0,alt)
    ax.set_xlabel('X[m]',fontsize=7,color='white')
    ax.set_ylabel('Y[m]',fontsize=7,color='white')
    ax.set_zlabel('Z[m]',fontsize=7,color='white')
    ax.tick_params(colors='white',labelsize=6)
    for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        pane.fill=False
    ax.view_init(elev=32,azim=220)
    fig.text(0.50,0.97,f"RENDERING 3D - {area['nome'].upper()}",
        fontsize=13,fontweight='bold',color='white',ha='center',va='top')
    fig.text(0.50,0.93,
        f"{calc['n']}x {area['lampada'][:38]}  |  {calc['E_m']} lux  |  {calc['W_t']}W  |  {calc['wm2']} W/m2",
        fontsize=8,color='#90cdf4',ha='center')
    buf=BytesIO()
    fig.savefig(buf,format='png',dpi=130,bbox_inches='tight',facecolor='black')
    buf.seek(0); plt.close(fig)
    return buf

def calc_preventivo(risultati):
    righe,tm,ti=[],0,0
    for r in risultati:
        lamp=DB_LAMPADE[r['lampada']]
        mat=r['calc']['n']*lamp['prezzo']
        ins=r['calc']['n']*lamp['installazione']
        righe.append({'area':r['nome'],'n':r['calc']['n'],
            'lampada':r['lampada'][:28],'mat':mat,'ins':ins,'sub':mat+ins})
        tm+=mat; ti+=ins
    tn=tm+ti; sg=tn*0.12; os2=tn*0.04
    tl=tn+sg+os2; mg=tl*0.35; to=tl+mg; iva=to*0.22
    return {'righe':righe,'tm':tm,'ti':ti,'tn':tn,
        'sg':sg,'os':os2,'tl':tl,'mg':mg,'to':to,'iva':iva,'tf':to+iva}

with st.sidebar:
    st.markdown("## Progetto")
    nome_prog   = st.text_input("Nome progetto",  "UFFICI TELEDIFESA")
    committente = st.text_input("Committente",    "Teledifesa S.p.A.")
    progettista = st.text_input("Progettista",    "Ing. Mario Rossi")
    num_tav     = st.text_input("N Tavola",       "26A3S001")
    st.markdown("---")
    st.markdown("## Planimetria")
    plan_file = st.file_uploader("Carica planimetria (PDF/PNG/JPG)",
        type=["pdf","png","jpg","jpeg"])
    if plan_file and plan_file.type.startswith('image'):
        st.image(plan_file,use_column_width=True)
    st.markdown("---")
    st.markdown("## AI Vision")
    use_ai=st.toggle("Abilita AI (richiede Ollama locale)",value=False)
    if use_ai:
        st.info("Ollama + llava:13b devono essere in esecuzione.\nollama serve => ollama pull llava:13b")
    st.markdown("---")
    st.markdown("## Filtro Lampade")
    prod_filter=st.selectbox("Produttore",
        ["Tutti","BEGA","iGuzzini","Flos","Artemide","Delta Light"])

st.markdown(
    """<div class="header-box">
    <h1 style="margin:0;font-size:2rem">Lighting Agent Pro v2.0</h1>
    <p style="margin:.3rem 0 0;opacity:.85">
    Progettazione illuminotecnica automatica | UNI EN 12464-1:2021 |
    PDF A3 Professionale | Rendering 3D | AI Vision
    </p>
    </div>""",
    unsafe_allow_html=True)

if 'aree' not in st.session_state:
    st.session_state.aree=[]

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
    "Aree","Calcoli","Tavola A3","Verifiche","Rendering 3D","Preventivo"])

with tab1:
    st.subheader("Definizione Aree")
    lamp_disp={k:v for k,v in DB_LAMPADE.items()
        if prod_filter=='Tutti' or v['produttore']==prod_filter}
    with st.form('form_area',clear_on_submit=True):
        c1,c2,c3=st.columns(3)
        with c1:
            nome_area=st.text_input("Nome area *",placeholder="Ufficio 1")
            tipo_locale=st.selectbox("Tipo locale *",list(REQUISITI.keys()))
        with c2:
            sup=st.number_input("Superficie m2 *",1.0,5000.0,35.0,0.5)
            alt=st.number_input("Altezza netta m *",2.0,10.0,2.70,0.05)
        with c3:
            lamp_scelta=st.selectbox("Apparecchio *",list(lamp_disp.keys()))
            lsp=DB_LAMPADE[lamp_scelta]
            st.caption(
                f"{lsp['potenza_W']}W | {lsp['flusso_lm']}lm | "
                f"Ra{lsp['ra']} | {lsp['temp_colore']} | UGR{lsp['ugr']}")
        if st.form_submit_button('Aggiungi Area',type='primary'):
            if nome_area.strip():
                st.session_state.aree.append({
                    'nome':nome_area.strip(),'tipo_locale':tipo_locale,
                    'superficie_m2':sup,'altezza_m':alt,
                    'lampada':lamp_scelta,'sup':sup})
                st.success(f'Area {nome_area} aggiunta!')
            else:
                st.error('Inserisci il nome area.')
    if st.session_state.aree:
        for i,a in enumerate(st.session_state.aree):
            c1,c2=st.columns([6,1])
            with c1:
                req=REQUISITI[a['tipo_locale']]
                st.markdown(
                    f'<div class="card"><b>{i+1}. {a["nome"]}</b> | '
                    f'{a["tipo_locale"]} | {a["superficie_m2"]}m2 | '
                    f'h:{a["altezza_m"]}m | Target:{req["lux"]}lux | '
                    f'{a["lampada"][:35]}</div>',
                    unsafe_allow_html=True)
            with c2:
                if st.button('X',key=f'd{i}'):
                    st.session_state.aree.pop(i); st.rerun()
    else:
        st.info('Aggiungi le aree del progetto.')

with tab2:
    st.subheader("Calcoli Illuminotecnici")
    if not st.session_state.aree:
        st.warning('Aggiungi prima le aree.')
    else:
        if st.button('ESEGUI CALCOLI UNI EN 12464-1',type='primary'):
            ris,ox=[],0.0
            for a in st.session_state.aree:
                c=calcola_area(a)
                ris.append({**a,'calc':c,'offset_x':ox,'offset_y':0})
                ox+=np.sqrt(a['superficie_m2'])+1.5
            st.session_state.risultati=ris
            st.success('Calcoli completati!')
        if 'risultati' in st.session_state:
            rl=st.session_state.risultati
            tl=sum(r['calc']['n']   for r in rl)
            tw=sum(r['calc']['W_t'] for r in rl)
            tm2=sum(r['sup']        for r in rl)
            c1,c2,c3,c4=st.columns(4)
            c1.metric('Aree',len(rl))
            c2.metric('Lampade totali',tl)
            c3.metric('Potenza',f'{tw} W')
            c4.metric('W/m2',f'{tw/tm2:.1f}')
            df=pd.DataFrame([{
                'Area':r['nome'],'Tipo':r['tipo_locale'],'m2':r['sup'],
                'Lampada':r['lampada'][:30],'N':r['calc']['n'],
                'Lux target':r['calc']['E_t'],'Lux ottenuto':r['calc']['E_m'],
                'W':r['calc']['W_t'],'W/m2':r['calc']['wm2'],
                'Lux ok':r['calc']['ok_lux'],'UGR ok':r['calc']['ok_ugr'],
                'Ra ok':r['calc']['ok_ra']} for r in rl])
            st.dataframe(df,use_container_width=True,hide_index=True)
            csv=df.to_csv(index=False).encode()
            st.download_button('Scarica CSV',csv,
                f"calcoli_{datetime.now():%Y%m%d}.csv",'text/csv')

with tab3:
    st.subheader("Tavola Illuminotecnica A3")
    if 'risultati' not in st.session_state:
        st.warning('Esegui prima i calcoli.')
    else:
        c1,c2=st.columns(2)
        with c1:
            if st.button('GENERA TAVOLA A3 PDF',type='primary'):
                with st.spinner('Generazione PDF A3...'):
                    prog={'nome':nome_prog,'committente':committente,
                        'progettista':progettista,
                        'data':datetime.now().strftime('%d/%m/%Y'),
                        'num_tavola':num_tav}
                    buf=genera_pdf(prog,st.session_state.risultati)
                    st.download_button('SCARICA PDF A3',data=buf,
                        file_name=f'{num_tav}_tavola.pdf',
                        mime='application/pdf')
                    st.success('Tavola A3 generata!')
        with c2:
            if st.button('GENERA DXF AUTOCAD'):
                with st.spinner('Generazione DXF...'):
                    dxf=genera_dxf(st.session_state.risultati)
                    st.download_button('SCARICA DXF',data=dxf,
                        file_name=f'{num_tav}_layout.dxf',
                        mime='application/dxf')
                    st.success('DXF generato!')

with tab4:
    st.subheader("Verifiche Illuminotecniche")
    if 'risultati' not in st.session_state:
        st.warning('Esegui prima i calcoli.')
    else:
        if st.button('GENERA REPORT VERIFICHE PDF',type='primary'):
            with st.spinner('Generazione report...'):
                prog={'nome':nome_prog,'committente':committente,
                    'progettista':progettista,
                    'data':datetime.now().strftime('%d/%m/%Y'),
                    'num_tavola':num_tav}
                buf=genera_pdf(prog,st.session_state.risultati)
                st.download_button('SCARICA REPORT PDF',data=buf,
                    file_name=f'{num_tav}_verifiche.pdf',
                    mime='application/pdf')
                st.success('Report generato!')
        st.markdown('---')
        for r in st.session_state.risultati:
            with st.expander(
                f"{r['nome']} | {r['calc']['E_m']} lux | {r['calc']['ok_lux']} Lux | {r['calc']['ok_ugr']} UGR"):
                ca,cb,cc,cd=st.columns(4)
                ca.metric('Lux ottenuto',str(r['calc']['E_m']))
                cb.metric('Potenza W',str(r['calc']['W_t']))
                cc.metric('W/m2',str(r['calc']['wm2']))
                cd.metric('Lampade',str(r['calc']['n']))

with tab5:
    st.subheader("Rendering 3D")
    if 'risultati' not in st.session_state:
        st.warning('Esegui prima i calcoli.')
    else:
        names=[r['nome'] for r in st.session_state.risultati]
        scelta=st.selectbox('Seleziona area',names)
        c1,c2=st.columns(2)
        with c1:
            if st.button('RENDERING AREA SELEZIONATA',type='primary'):
                idx=names.index(scelta)
                r=st.session_state.risultati[idx]
                with st.spinner(f'Rendering {scelta}...'):
                    buf=genera_rendering(r,r['calc'])
                    st.image(buf,caption=f'Rendering 3D - {scelta}',
                        use_column_width=True)
                    buf.seek(0)
                    st.download_button('Scarica PNG',data=buf,
                        file_name=f"render_{scelta.lower().replace(' ','_')}.png",
                        mime='image/png')
        with c2:
            if st.button('RENDERING TUTTE LE AREE'):
                cols=st.columns(2)
                for i,r in enumerate(st.session_state.risultati):
                    with st.spinner(f'Rendering {r["nome"]}...'):
                        buf=genera_rendering(r,r['calc'])
                        with cols[i%2]:
                            st.image(buf,caption=r['nome'],use_column_width=True)
                            buf.seek(0)
                            st.download_button(f"Scarica {r['nome']}",data=buf,
                                file_name=f'render_{i}.png',mime='image/png',
                                key=f'rend{i}')

with tab6:
    st.subheader("Preventivo Economico")
    if 'risultati' not in st.session_state:
        st.warning('Esegui prima i calcoli.')
    else:
        c1,c2,c3=st.columns(3)
        with c1: mg_sl=st.slider('Margine %',10,60,35)
        with c2: iva_sl=st.slider('IVA %',0,22,22)
        with c3: sg_sl=st.slider('Spese gen. %',5,20,12)
        if st.button('CALCOLA PREVENTIVO',type='primary'):
            st.session_state.prev=calc_preventivo(st.session_state.risultati)
        if 'prev' in st.session_state:
            pv=st.session_state.prev
            c1,c2,c3=st.columns(3)
            c1.metric('Materiali',f"EUR {pv['tm']:,.0f}")
            c2.metric('Installazione',f"EUR {pv['ti']:,.0f}")
            c3.metric('Totale IVA inclusa',f"EUR {pv['tf']:,.0f}")
            df_p=pd.DataFrame([{
                'Area':r['area'],'N lampade':r['n'],'Apparecchio':r['lampada'],
                'Materiali':f"EUR {r['mat']:,.0f}",'Installazione':f"EUR {r['ins']:,.0f}",
                'Subtotale':f"EUR {r['sub']:,.0f}"} for r in pv['righe']])
            st.dataframe(df_p,use_container_width=True,hide_index=True)
            st.markdown('---')
            st.markdown(f"""
| Voce | Importo |
|---|---|
| Materiali | EUR {pv['tm']:,.0f} |
| Installazione | EUR {pv['ti']:,.0f} |
| Totale lavori | EUR {pv['tn']:,.0f} |
| Spese generali 12% | EUR {pv['sg']:,.0f} |
| Oneri sicurezza 4% | EUR {pv['os']:,.0f} |
| Margine 35% | EUR {pv['mg']:,.0f} |
| **OFFERTA CLIENTE** | **EUR {pv['to']:,.0f}** |
| IVA 22% | EUR {pv['iva']:,.0f} |
| **TOTALE IVA INCLUSA** | **EUR {pv['tf']:,.0f}** |
""")
            txt=(
                f"PREVENTIVO OFFERTA\nProgetto: {nome_prog}\n"
                f"Committente: {committente}\nData: {datetime.now():%d/%m/%Y}\n\n"
                + '\n'.join(
                    f"  {r['area']}: {r['n']}x {r['lampada']} Mat.EUR{r['mat']:,.0f}"
                    for r in pv['righe'])
                + f"\n\nTOTALE IVA INCLUSA: EUR {pv['tf']:,.0f}\n")
            st.download_button('Scarica Preventivo TXT',
                data=txt.encode(),
                file_name=f"preventivo_{datetime.now():%Y%m%d}.txt")

st.markdown('---')
st.caption('Lighting Agent Pro v2.0 | UNI EN 12464-1:2021 | PDF A3 | Rendering 3D | AI Vision | 2026')