"""
process_simulations.py
──────────────────────
Procesa todos los CSV de simulaciones WF+noise, genera PDFs con gráficos
temporales y order-like spectrum, y guarda los datos en pickle.

Estructura de salida:
  result_sim_2026/WF_plus_noise/figs/         → PDFs
  result_sim_2026/WF_plus_noise/pickle_data/  → .pkl
"""

import os
import re
import glob
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                 # sin GUI, necesario para guardar PDFs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal

# ── Rutas ─────────────────────────────────────────────────────────────────────
INPUT_DIR   = "simulaciones/fs=1000Hz/WF+noise"
OUTPUT_FIGS = "result_sim_2026/WF_plus_noise/figs"
OUTPUT_PKL  = "result_sim_2026/WF_plus_noise/pickle_data"

os.makedirs(OUTPUT_FIGS, exist_ok=True)
os.makedirs(OUTPUT_PKL,  exist_ok=True)

# ── Parámetros fijos ──────────────────────────────────────────────────────────
R  = 0.46       # radio rueda (m)
FS = 1000       # frecuencia de muestreo (Hz)

COL_NAMES = ['t',
             'az_AB_1R', 'az_AB_1L', 'az_AB_2R', 'az_AB_2L',
             'az_AB_3R', 'az_AB_3L', 'az_AB_4R', 'az_AB_4L',
             'az_BG_F_RF', 'az_BG_F_LF', 'az_BG_F_RB', 'az_BG_F_LB',
             'az_BG_R_RF', 'az_BG_R_LF', 'az_BG_R_RB', 'az_BG_R_LB',
             'az_Plat_F', 'az_Plat_R',
             'vel']

GRUPOS = {
    "Cajas de grasa (Ejes 1-4)": ['az_AB_1R', 'az_AB_1L', 'az_AB_2R', 'az_AB_2L',
                                   'az_AB_3R', 'az_AB_3L', 'az_AB_4R', 'az_AB_4L'],
    "Bogie delantero":           ['az_BG_F_RF', 'az_BG_F_LF', 'az_BG_F_RB', 'az_BG_F_LB'],
    "Bogie trasero":             ['az_BG_R_RF', 'az_BG_R_LF', 'az_BG_R_RB', 'az_BG_R_LB'],
    "Plataforma":                ['az_Plat_F', 'az_Plat_R'],
}

# ── Parámetros order spectrum ─────────────────────────────────────────────────
Λ        = 10
D_LAMBDA = 1 / 16
K        = int(Λ / D_LAMBDA)
λs       = np.arange(0, Λ, D_LAMBDA)

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_filename(fname):
    """Extrae metadatos del nombre del fichero CSV."""
    base = os.path.splitext(os.path.basename(fname))[0]
    meta = {"filename": base}
    m = re.search(r'_L(\d+)_', base)
    if m:
        meta["L_mm"] = int(m.group(1))
    m = re.search(r'_(ERRI_\w+|SIN_DEFECTOS)_', base)
    if m:
        meta["defecto"] = m.group(1)
    m = re.search(r'_VEL(\d+)_', base)
    if m:
        meta["vel_kmh"] = int(m.group(1))
    return meta


def leer_csv(path):
    df = pd.read_csv(path, sep=';', skiprows=[1], quotechar='"')
    df.columns = COL_NAMES
    return df


def calcular_G(f_fail, N, Ts):
    cumsum_f = np.cumsum(f_fail)
    G = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        G[k, :] = np.exp(-1j * 2 * np.pi * D_LAMBDA * k * Ts * cumsum_f)
    return G


def calcular_order_spectrum(señal_raw, G, N):
    s = señal_raw - señal_raw.mean()
    env = np.abs(signal.hilbert(s))
    env -= env.mean()
    env_analytic = signal.hilbert(env)
    return np.abs((1 / N) * np.dot(G, env_analytic))


# ── Plot temporal ─────────────────────────────────────────────────────────────
def plot_temporal(df, meta, pdf):
    t = df['t']
    fr_rot = df['vel'] / (2 * np.pi * R)

    all_groups = dict(GRUPOS)
    all_groups["Velocidad"]            = ['vel']
    all_groups["Frecuencia rotación"]  = ['_fr_rot_']   # señal calculada

    for nombre_grupo, columnas in all_groups.items():
        if columnas == ['_fr_rot_']:
            n = 1
            fig, axes = plt.subplots(n, 1, figsize=(12, 1.8), sharex=True)
            axes = [axes]
            fig.suptitle(
                f"Señales temporales – Frecuencia rotación\n{meta['filename']}",
                fontsize=10, fontweight='bold')
            axes[0].plot(t, fr_rot, linewidth=0.7, color='darkorange')
            axes[0].set_ylabel("fr (Hz)", fontsize=9)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlabel("Tiempo (s)")
        else:
            n = len(columnas)
            fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True)
            if n == 1:
                axes = [axes]
            fig.suptitle(
                f"Señales temporales – {nombre_grupo}\n{meta['filename']}",
                fontsize=10, fontweight='bold')
            for ax, col in zip(axes, columnas):
                ax.plot(t, df[col], linewidth=0.7)
                ax.set_ylabel(col, fontsize=9)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("Tiempo (s)")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# ── Plot order spectrum ───────────────────────────────────────────────────────
def plot_order_spectrum(df, G, N, meta, pdf):
    for nombre_grupo, columnas in GRUPOS.items():
        n = len(columnas)
        fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(
            f"Order-like spectrum – {nombre_grupo}\n{meta['filename']}",
            fontsize=10, fontweight='bold')

        for ax, col in zip(axes, columnas):
            X_lambda = calcular_order_spectrum(df[col].values, G, N)
            ax.plot(λs, X_lambda, linewidth=0.8, color='steelblue')
            for λ_int in range(1, int(λs[-1]) + 1):
                ax.axvline(x=λ_int, color='red', linestyle='--', linewidth=0.6,
                           label='λ enteros' if λ_int == 1 else None)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("λ")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# ── Procesado principal ───────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
print(f"Encontrados {len(csv_files)} archivos CSV\n")

all_results = {}   # dict global para el pickle

for i, csv_path in enumerate(csv_files):
    meta = parse_filename(csv_path)
    base = meta["filename"]
    print(f"\n[{i+1:02d}/{len(csv_files)}] {base}")

    # ── Leer datos ────────────────────────────────────────────────────────────
    print(f"  → Leyendo CSV...", end=' ', flush=True)
    df  = leer_csv(csv_path)
    N   = len(df)
    Ts  = 1 / FS
    fr_rot = (df['vel'] / (2 * np.pi * R)).values
    print(f"OK  ({N} muestras, fr_rot_mean={fr_rot.mean():.3f} Hz)")

    # ── Matriz G ──────────────────────────────────────────────────────────────
    print(f"  → Calculando matriz G ({K}x{N})...", end=' ', flush=True)
    G = calcular_G(fr_rot, N, Ts)
    print("OK")

    # ── PDF temporal ──────────────────────────────────────────────────────────
    print(f"  → Generando PDF temporal...", end=' ', flush=True)
    pdf_time_path = os.path.join(OUTPUT_FIGS, f"{base}_temporal.pdf")
    with PdfPages(pdf_time_path) as pdf:
        plot_temporal(df, meta, pdf)
    print("OK")

    # ── PDF order spectrum ────────────────────────────────────────────────────
    print(f"  → Calculando order spectra ({len(sum(GRUPOS.values(), []))} sensores)...", end=' ', flush=True)
    pdf_ord_path = os.path.join(OUTPUT_FIGS, f"{base}_order_spectrum.pdf")
    order_data   = {}
    with PdfPages(pdf_ord_path) as pdf:
        for nombre_grupo, columnas in GRUPOS.items():
            for col in columnas:
                X_lambda = calcular_order_spectrum(df[col].values, G, N)
                order_data[col] = X_lambda.tolist()
        plot_order_spectrum(df, G, N, meta, pdf)
    print("OK")

    # ── Guardar pickle ────────────────────────────────────────────────────────
    print(f"  → Guardando pickle...", end=' ', flush=True)
    result = {
        "meta":          meta,
        "lambda_axis":   λs.tolist(),
        "fr_rot_mean":   float(np.mean(fr_rot)),
        "order_spectra": order_data,
    }
    pkl_path = os.path.join(OUTPUT_PKL, f"{base}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f)
    all_results[base] = result
    print(f"OK  →  {pkl_path}")

# ── Índice JSON legible ───────────────────────────────────────────────────────
index = {
    k: {
        "meta":        v["meta"],
        "fr_rot_mean": v["fr_rot_mean"],
        "pkl_file":    f"{k}.pkl",
    }
    for k, v in all_results.items()
}
index_path = os.path.join(OUTPUT_PKL, "index.json")
with open(index_path, 'w') as f:
    json.dump(index, f, indent=2, ensure_ascii=False)

print(f"\n✅  Procesados {len(csv_files)} archivos")
print(f"   PDFs  → {OUTPUT_FIGS}/")
print(f"   PKL   → {OUTPUT_PKL}/")
print(f"   Índice→ {index_path}")