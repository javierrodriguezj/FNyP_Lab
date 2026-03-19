#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def ajuste2picos():
    """
    Versión sin ROOT para ajustar dos picos gaussianos con fondo (lineal o exponencial)
    usando NumPy, SciPy y Matplotlib.
    
    Se leen los datos (y, si existen, los fondos), se agrupan los canales en histogramas y
    se realizan ajustes preliminares para el fondo y cada pico, para finalmente ajustar globalmente
    la función total (fondo + 2 gaussianas) en la región de interés. Se muestran los resultados y
    se guardan dos figuras: una del espectro completo y otra con el ajuste en la región de picos.
    """
    # ------------------------------------------------------------
    # Configuración del sistema: Phywe o Spectech
    # ------------------------------------------------------------
    phywe = True  # True para Phywe, False para Spectech

    if phywe:
        filename        = "Co_Phywe.dat"
        filename_fondo1 = ""   # Ejemplo: "fondo1.dat" si hubiera fondo
        filename_fondo2 = ""   # Ejemplo: "fondo2.dat" si hubiera un segundo fondo
        nlines          = 3    # líneas de cabecera
        nchan           = 4000
        ngroup          = 10   # Agrupación para el espectro global
        ngroup_peak     = 5    # Agrupación para la región de picos
    else:
        filename        = "Co_Spectech.tsv"
        filename_fondo1 = ""
        filename_fondo2 = ""
        nlines          = 22
        nchan           = 1024  # o 2048, según corresponda
        ngroup          = 4
        ngroup_peak     = 2

    # Si se fueran a restar fondos, se definen los tiempos (en segundos)
    tiempo = 0.0
    tiempo_fondo1 = 0.0
    tiempo_fondo2 = 0.0
    if (filename_fondo1 != "") or (filename_fondo2 != ""):
        tiempo = 180.0
    if filename_fondo1 != "":
        tiempo_fondo1 = 600.0
    if filename_fondo2 != "":
        tiempo_fondo2 = 600.0

    # ------------------------------------------------------------
    # Definición de intervalos de interés
    # ------------------------------------------------------------
    # Región del histograma donde se encuentran los picos
    if phywe:
        hist_ini = 2800
        hist_fin = 3700
    else:
        hist_ini = 590
        hist_fin = 950

    # Intervalo para ajustar el fondo
    if phywe:
        bkg_ini = 2820
        bkg_fin = 2880
    else:
        bkg_ini = 590
        bkg_fin = 630

    # Tipo de fondo: True => lineal; False => exponencial
    bkg_lineal = False

    # Intervalos para cada gaussiana
    if phywe:
        g1_ini, g1_fin = 2900, 3200
        g2_ini, g2_fin = 3300, 3600
    else:
        g1_ini, g1_fin = 640, 730
        g2_ini, g2_fin = 770, 850

    # Intervalo total para el ajuste global (fondo + 2 gaussianas)
    if phywe:
        total_ini, total_fin = 2800, 3700
    else:
        total_ini, total_fin = 610, 950

    # Verificar que nchan es divisible por ngroup y ngroup_peak
    if (nchan % ngroup != 0) or (nchan % ngroup_peak != 0):
        print("El número de canales no es divisible por la agrupación. STOP.")
        return

    # ------------------------------------------------------------
    # Lectura de archivos y construcción de histogramas
    # ------------------------------------------------------------
    nbins_global = nchan // ngroup
    nbins_peak   = nchan // ngroup_peak
    global_hist = np.zeros(nbins_global)
    peak_hist   = np.zeros(nbins_peak)

    # Lectura del archivo principal (y de fondo, si existen)
    with open(filename, "r") as f:
        main_lines = f.readlines()
    if filename_fondo1:
        with open(filename_fondo1, "r") as f1:
            fondo1_lines = f1.readlines()
    else:
        fondo1_lines = None
    if filename_fondo2:
        with open(filename_fondo2, "r") as f2:
            fondo2_lines = f2.readlines()
    else:
        fondo2_lines = None

    row = 0
    for line in main_lines:
        if row < nlines:
            row += 1
            continue  # saltar cabecera
        cols = line.strip().split()
        if len(cols) < 2:
            row += 1
            continue
        x = float(cols[0])
        y = float(cols[1])
        # Procesar fondos (si existen)
        y1, y2 = 0.0, 0.0
        if fondo1_lines is not None and row < len(fondo1_lines):
            line1 = fondo1_lines[row]
            if line1.strip():
                cols1 = line1.strip().split()
                if len(cols1) >= 2:
                    y1 = float(cols1[1])
        if fondo2_lines is not None and row < len(fondo2_lines):
            line2 = fondo2_lines[row]
            if line2.strip():
                cols2 = line2.strip().split()
                if len(cols2) >= 2:
                    y2 = float(cols2[1])
        # Restar fondo ponderado si procede
        if (fondo1_lines is not None) and (fondo2_lines is not None):
            factor = tiempo / (tiempo_fondo1 + tiempo_fondo2) if (tiempo_fondo1+tiempo_fondo2)!=0 else 0
            y -= factor * (y1 + y2)
        elif fondo1_lines is not None:
            factor = tiempo / tiempo_fondo1 if tiempo_fondo1 != 0 else 0
            y -= factor * y1
        elif fondo2_lines is not None:
            factor = tiempo / tiempo_fondo2 if tiempo_fondo2 != 0 else 0
            y -= factor * y2

        # Agregar a los histogramas según la agrupación
        bin_index_global = int(x) // ngroup
        if 0 <= bin_index_global < nbins_global:
            global_hist[bin_index_global] += y
        bin_index_peak = int(x) // ngroup_peak
        if 0 <= bin_index_peak < nbins_peak:
            peak_hist[bin_index_peak] += y
        row += 1

    # Definir bordes y centros de los bins para graficar
    global_bin_edges = np.linspace(0, nchan, nbins_global+1)
    global_centers   = (global_bin_edges[:-1] + global_bin_edges[1:]) / 2
    peak_bin_edges   = np.linspace(0, nchan, nbins_peak+1)
    peak_centers     = (peak_bin_edges[:-1] + peak_bin_edges[1:]) / 2

    # ------------------------------------------------------------
    # Gráfico del espectro completo (estilo histograma "step")
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    hist_global_step = np.append(global_hist, global_hist[-1])
    ax1.step(global_bin_edges, hist_global_step, where='post', color='black')
    ax1.fill_between(global_bin_edges, hist_global_step, step='post', color='lightblue', alpha=0.5)
    ax1.set_xlabel("Canal")
    ax1.set_ylabel("# cuentas")
    ax1.set_title("Espectro completo")
    ax1.set_xlim(0, nchan)
    fig1.savefig("ajuste2picos_espectro.pdf", bbox_inches="tight")
    fig1.savefig("ajuste2picos_espectro.png", bbox_inches="tight")

    # ------------------------------------------------------------
    # Ajuste en la región de picos
    # ------------------------------------------------------------
    # Seleccionar la región del histograma para el ajuste de picos
    mask_peak = (peak_centers >= hist_ini) & (peak_centers <= hist_fin)
    
    x_peak = peak_centers[mask_peak]
    y_peak = peak_hist[mask_peak]
    
    #elimina valores con y = 0 (ey = 0)
    mask_valid = (y_peak > 0) & np.isfinite(y_peak)
    
    x_peak = x_peak[mask_valid]
    y_peak = y_peak[mask_valid]
    
    # Suponemos error estadístico: sqrt(cuentas)
    y_peak_err = np.sqrt(np.abs(y_peak))

    # --- Definir funciones de ajuste ---
    def linear_bkg(x, p0, p1):
        return p0 + p1 * x

    def expo_bkg(x, p0, p1):
        return np.exp(p0 + p1 * x)

    def gaus(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Función total: fondo + 2 gaussianas
    def total_func(x, p0, p1, p2, p3, p4, p5, p6, p7):
        bkg_val = linear_bkg(x, p0, p1) if bkg_lineal else expo_bkg(x, p0, p1)
        return bkg_val + gaus(x, p2, p3, p4) + gaus(x, p5, p6, p7)

    # --- Ajuste preliminar del fondo ---
    mask_bkg = (x_peak >= bkg_ini) & (x_peak <= bkg_fin)
    if np.sum(mask_bkg) < 2:
        print("No hay suficientes puntos para ajuste de fondo. Revisa bkg_ini y bkg_fin.")
        return
    if bkg_lineal:
        p0_bkg_guess = [np.mean(y_peak[mask_bkg]), 0.0]
        popt_bkg, _ = curve_fit(linear_bkg, x_peak[mask_bkg], y_peak[mask_bkg], p0=p0_bkg_guess)
    else:
        p0_bkg_guess = [np.log(np.mean(y_peak[mask_bkg]) + 1e-6), 0.0]
        popt_bkg, _ = curve_fit(expo_bkg, x_peak[mask_bkg], y_peak[mask_bkg], p0=p0_bkg_guess)

    # --- Ajuste preliminar de la gaussiana 1 ---
    mask_g1 = (x_peak >= g1_ini) & (x_peak <= g1_fin)
    if np.sum(mask_g1) < 3:
        print("No hay suficientes puntos para ajuste gaussiano 1.")
        return
    A1_guess = np.max(y_peak[mask_g1])
    mu1_guess = (g1_ini + g1_fin) / 2.0
    sigma1_guess = (g1_fin - g1_ini) / 6.0
    p0_g1 = [A1_guess, mu1_guess, sigma1_guess]
    popt_g1, _ = curve_fit(gaus, x_peak[mask_g1], y_peak[mask_g1], p0=p0_g1)

    # --- Ajuste preliminar de la gaussiana 2 ---
    mask_g2 = (x_peak >= g2_ini) & (x_peak <= g2_fin)
    if np.sum(mask_g2) < 3:
        print("No hay suficientes puntos para ajuste gaussiano 2.")
        return
    A2_guess = np.max(y_peak[mask_g2])
    mu2_guess = (g2_ini + g2_fin) / 2.0
    sigma2_guess = (g2_fin - g2_ini) / 6.0
    p0_g2 = [A2_guess, mu2_guess, sigma2_guess]
    popt_g2, _ = curve_fit(gaus, x_peak[mask_g2], y_peak[mask_g2], p0=p0_g2)

    # --- Parámetros iniciales para el ajuste total ---
    # Orden: [p0, p1] fondo, [p2, p3, p4] gaus1, [p5, p6, p7] gaus2
    p0_total_guess = [
        popt_bkg[0], popt_bkg[1],
        popt_g1[0], popt_g1[1], popt_g1[2],
        popt_g2[0], popt_g2[1], popt_g2[2]
    ]

    # Ajuste global en la región total
    mask_total = (x_peak >= total_ini) & (x_peak <= total_fin)
    if np.sum(mask_total) < 3:
        print("No hay suficientes puntos para ajuste total.")
        return
    popt_total, pcov_total = curve_fit(
        total_func,
        x_peak[mask_total],
        y_peak[mask_total],
        p0=p0_total_guess,
        sigma=y_peak_err[mask_total],
        absolute_sigma=True
    )
    perr = np.sqrt(np.diag(pcov_total))
    
    # Calcular chi2 y ndf
    fit_vals = total_func(x_peak[mask_total], *popt_total)
    chi2 = np.sum(((y_peak[mask_total] - fit_vals) / (y_peak_err[mask_total] + 1e-6))**2)
    ndf = np.sum(mask_total) - len(popt_total)

    # Extraer parámetros y errores
    p0_fit, p1_fit, p2_fit, p3_fit, p4_fit, p5_fit, p6_fit, p7_fit = popt_total
    p0_err, p1_err, p2_err, p3_err, p4_err, p5_err, p6_err, p7_err = perr

    # Imprimir resultados
    print("\nResultados del ajuste:\n")
    print("Fondo:")
    print(f"  Constante (p0): {p0_fit:.3f} +/- {p0_err:.3f}")
    print(f"  Pendiente (p1): {p1_fit:.3f} +/- {p1_err:.3f}")
    print("\nGaussiana 1:")
    print(f"  Amplitud (p2):  {p2_fit:.2f} +/- {p2_err:.2f}")
    print(f"  Centroide (p3): {p3_fit:.2f} +/- {p3_err:.2f}")
    print(f"  Sigma (p4):     {p4_fit:.2f} +/- {p4_err:.2f}")
    print("\nGaussiana 2:")
    print(f"  Amplitud (p5):  {p5_fit:.2f} +/- {p5_err:.2f}")
    print(f"  Centroide (p6): {p6_fit:.2f} +/- {p6_err:.2f}")
    print(f"  Sigma (p7):     {p7_fit:.2f} +/- {p7_err:.2f}")
    print("\nChi2 / ndf:")
    print(f"  {chi2:.2f} / {ndf}")

    # ------------------------------------------------------------
    # Gráfico de la región de picos y el ajuste total
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.errorbar(x_peak, y_peak, yerr=y_peak_err, fmt='o', label="Datos", capsize=3)
    x_fit = np.linspace(total_ini, total_fin, 200)
    y_fit_total = total_func(x_fit, *popt_total)
    if bkg_lineal:
        y_fit_bkg = linear_bkg(x_fit, p0_fit, p1_fit)
    else:
        y_fit_bkg = expo_bkg(x_fit, p0_fit, p1_fit)
    y_fit_g1 = gaus(x_fit, p2_fit, p3_fit, p4_fit)
    y_fit_g2 = gaus(x_fit, p5_fit, p6_fit, p7_fit)
    ax2.plot(x_fit, y_fit_total, 'r-', label="Ajuste total")
    ax2.plot(x_fit, y_fit_bkg, 'g--', label="Fondo")
    ax2.plot(x_fit, y_fit_g1, 'b--', label="Gaussiana 1")
    ax2.plot(x_fit, y_fit_g2, 'm--', label="Gaussiana 2")
    ax2.set_xlabel("Canal")
    ax2.set_ylabel("# cuentas")
    ax2.set_title("Ajuste de dos picos")
    ax2.legend()
    fig2.savefig("ajuste2picos.pdf", bbox_inches="tight")
    fig2.savefig("ajuste2picos.png", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    ajuste2picos()
