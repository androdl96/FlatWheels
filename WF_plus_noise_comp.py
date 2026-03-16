"""
compare_simulations.py
──────────────────────
Genera comparativas entre simulaciones WF+noise a partir de los pickles.

Tres tipos de comparación:
  1. Efecto del tamaño del defecto (L):   fijo defecto+vel, varía L
  2. Efecto de la velocidad (VEL):        fijo defecto+L,   varía vel
  3. Efecto del tipo de defecto:          fijo L+vel,        varía defecto

Salida: result_sim_2026/WF_plus_noise/figs/comparisions/
"""

import os
import glob
import pickle
import itertools
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm

# ── Rutas ─────────────────────────────────────────────────────────────────────
PKL_DIR    = "result_sim_2026/WF_plus_noise/pickle_data"
OUTPUT_DIR = "result_sim_2026/WF_plus_noise/figs/comparisions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Grupos de sensores ────────────────────────────────────────────────────────
GRUPOS = {
    "Cajas de grasa (Ejes 1-4)": ['az_AB_1R', 'az_AB_1L', 'az_AB_2R', 'az_AB_2L',
                                   'az_AB_3R', 'az_AB_3L', 'az_AB_4R', 'az_AB_4L'],
    "Bogie delantero":           ['az_BG_F_RF', 'az_BG_F_LF', 'az_BG_F_RB', 'az_BG_F_LB'],
    "Bogie trasero":             ['az_BG_R_RF', 'az_BG_R_LF', 'az_BG_R_RB', 'az_BG_R_LB'],
    "Plataforma":                ['az_Plat_F', 'az_Plat_R'],
}

# ── Paletas de color ──────────────────────────────────────────────────────────
COLORS_L       = {20: '#1f77b4', 30: '#ff7f0e', 45: '#2ca02c', 60: '#d62728'}
COLORS_VEL     = {20: '#1f77b4', 30: '#ff7f0e', 40: '#2ca02c', 50: '#9467bd', 60: '#d62728'}
COLORS_DEFECTO = {
    'SIN_DEFECTOS': '#2ca02c',
    'ERRI_Low':     '#ff7f0e',
    'ERRI_High':    '#d62728',
}

# ── Cargar todos los pickles ──────────────────────────────────────────────────
print("Cargando pickles...")
all_data = {}
for pkl_path in sorted(glob.glob(os.path.join(PKL_DIR, "*.pkl"))):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    key = d['meta']['filename']
    all_data[key] = d
print(f"  {len(all_data)} pickles cargados.\n")

# Construir índice por (L_mm, defecto, vel_kmh)
import re as _re

def _reparse(filename):
    """Re-extrae defecto y vel directamente del filename con regex más estrictas."""
    m_def = _re.search(r'_(ERRI_(?:High|Low)|SIN_DEFECTOS)_', filename)
    m_vel = _re.search(r'_VEL(\d+)_fs', filename)
    m_L   = _re.search(r'_L(\d+)_', filename)
    return (
        int(m_L.group(1))   if m_L   else None,
        m_def.group(1)      if m_def else None,
        int(m_vel.group(1)) if m_vel else None,
    )

index = {}   # (L_mm, defecto, vel_kmh) → data dict
for key, d in all_data.items():
    L, def_, vel = _reparse(d['meta']['filename'])
    if None not in (L, def_, vel):
        index[(L, def_, vel)] = d

# Valores únicos
Ls       = sorted({k[0] for k in index})
defectos = sorted({k[1] for k in index})
vels     = sorted({k[2] for k in index})

print(f"  L (mm):    {Ls}")
print(f"  Defectos:  {defectos}")
print(f"  Vel (km/h):{vels}\n")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: genera una página del PDF con comparativa de un grupo de sensores
# ══════════════════════════════════════════════════════════════════════════════
def plot_group_comparison(ax_list, grupo_cols, datasets, labels, colors,
                          lambda_axis, title_suffix="", show_integers=True):
    """
    ax_list    : lista de ejes (uno por sensor del grupo)
    datasets   : lista de dicts order_spectra {col: X_lambda}
    labels     : etiquetas de leyenda
    colors     : colores para cada dataset
    """
    λs = np.array(lambda_axis)
    for ax, col in zip(ax_list, grupo_cols):
        for data, label, color in zip(datasets, labels, colors):
            if col in data['order_spectra']:
                X = np.array(data['order_spectra'][col])
                ax.plot(λs, X, linewidth=0.9, label=label, color=color, alpha=0.85)
        if show_integers:
            for λ_int in range(1, int(λs[-1]) + 1):
                ax.axvline(x=λ_int, color='gray', linestyle='--',
                           linewidth=0.5, alpha=0.5)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.25)
    ax_list[0].legend(fontsize=8, loc='upper right')
    ax_list[-1].set_xlabel("λ (orden)")


def save_comparison_pdf(pdf_path, grupo_title, grupo_cols,
                        datasets, labels, colors, lambda_axis, suptitle):
    n = len(grupo_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{suptitle}\n{grupo_title}", fontsize=10, fontweight='bold')
    plot_group_comparison(axes, grupo_cols, datasets, labels, colors, lambda_axis)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# COMPARATIVA 1 – Efecto del tamaño L (fijo defecto + vel, varía L)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("COMPARATIVA 1: Efecto del tamaño del defecto (L)")
print("=" * 60)

count = 0
for defecto in defectos:
    for vel in vels:
        # Recopilar datos disponibles para cada L
        datasets = []
        labels   = []
        colors   = []
        for L in Ls:
            key = (L, defecto, vel)
            if key in index:
                datasets.append(index[key])
                labels.append(f"L={L} mm")
                colors.append(COLORS_L.get(L, 'black'))

        if len(datasets) < 2:
            continue

        lambda_axis = datasets[0]['lambda_axis']
        suptitle = (f"Efecto del tamaño del defecto (L)\n"
                    f"Defecto: {defecto}  |  Vel: {vel} km/h")

        safe_def = defecto.replace('_', '')
        pdf_path = os.path.join(
            OUTPUT_DIR, f"compare_L_{safe_def}_VEL{vel:03d}.pdf")

        with PdfPages(pdf_path) as pdf:
            for nombre_grupo, grupo_cols in GRUPOS.items():
                fig = save_comparison_pdf(
                    pdf_path, nombre_grupo, grupo_cols,
                    datasets, labels, colors, lambda_axis, suptitle)
                pdf.savefig(fig)
                plt.close(fig)

        print(f"  ✓  {os.path.basename(pdf_path)}")
        count += 1

print(f"\n  → {count} PDFs generados\n")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARATIVA 2 – Efecto de la velocidad (fijo defecto + L, varía vel)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("COMPARATIVA 2: Efecto de la velocidad (VEL)")
print("=" * 60)

count = 0
for defecto in defectos:
    for L in Ls:
        datasets = []
        labels   = []
        colors   = []
        for vel in vels:
            key = (L, defecto, vel)
            if key in index:
                datasets.append(index[key])
                labels.append(f"V={vel} km/h")
                colors.append(COLORS_VEL.get(vel, 'black'))

        if len(datasets) < 2:
            continue

        lambda_axis = datasets[0]['lambda_axis']
        suptitle = (f"Efecto de la velocidad\n"
                    f"Defecto: {defecto}  |  L: {L} mm")

        safe_def = defecto.replace('_', '')
        pdf_path = os.path.join(
            OUTPUT_DIR, f"compare_VEL_{safe_def}_L{L:03d}.pdf")

        with PdfPages(pdf_path) as pdf:
            for nombre_grupo, grupo_cols in GRUPOS.items():
                fig = save_comparison_pdf(
                    pdf_path, nombre_grupo, grupo_cols,
                    datasets, labels, colors, lambda_axis, suptitle)
                pdf.savefig(fig)
                plt.close(fig)

        print(f"  ✓  {os.path.basename(pdf_path)}")
        count += 1

print(f"\n  → {count} PDFs generados\n")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARATIVA 3 – Efecto del tipo de defecto (fijo L + vel, varía defecto)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("COMPARATIVA 3: Efecto del tipo de defecto")
print("=" * 60)

# Orden legible: primero sin defecto, luego severidad creciente
DEFECTO_ORDER = ['SIN_DEFECTOS', 'ERRI_Low', 'ERRI_High']

count = 0
for L in Ls:
    for vel in vels:
        datasets = []
        labels   = []
        colors   = []
        for defecto in DEFECTO_ORDER:
            key = (L, defecto, vel)
            if key in index:
                datasets.append(index[key])
                labels.append(defecto.replace('_', ' '))
                colors.append(COLORS_DEFECTO.get(defecto, 'black'))

        if len(datasets) < 2:
            continue

        lambda_axis = datasets[0]['lambda_axis']
        suptitle = (f"Efecto del tipo de defecto\n"
                    f"L: {L} mm  |  Vel: {vel} km/h")

        pdf_path = os.path.join(
            OUTPUT_DIR, f"compare_DEFECTO_L{L:03d}_VEL{vel:03d}.pdf")

        with PdfPages(pdf_path) as pdf:
            for nombre_grupo, grupo_cols in GRUPOS.items():
                fig = save_comparison_pdf(
                    pdf_path, nombre_grupo, grupo_cols,
                    datasets, labels, colors, lambda_axis, suptitle)
                pdf.savefig(fig)
                plt.close(fig)

        print(f"  ✓  {os.path.basename(pdf_path)}")
        count += 1

print(f"\n  → {count} PDFs generados\n")
print(f"✅  Todas las comparativas guardadas en: {OUTPUT_DIR}/")