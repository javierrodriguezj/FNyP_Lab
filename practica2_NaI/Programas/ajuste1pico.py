import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def ajuste1pico():
    """
    Versión sin ROOT para ajustar un pico gaussiano con fondo (lineal o exponencial)
    usando NumPy, SciPy y Matplotlib.
    
    Se leen los datos de un archivo (y, si existen, de uno o dos fondos),
    se agrupan los canales (con dos factores de agrupación: uno para el espectro global
    y otro para la región del pico) y se realizan ajustes:
      - Ajuste de la gaussiana en la región del pico.
      - Ajuste del fondo en una región definida.
      - Ajuste total (fondo + gaussiana) en un intervalo de ajuste.
    
    Se imprimen los parámetros del ajuste y se guardan dos gráficos en PDF/PNG.
    """

    # ------------------------------------------------------------
    # Configuración (elige sistema: Phywe o Spectech)
    # ------------------------------------------------------------
    phywe = True  # True para Phywe, False para Spectech

    if phywe:
        filename        = "Na_Phywe.dat"   # Archivo con el espectro
        filename_fondo1 = ""               # Si hubiera fondo, ej. "fondo1.dat"
        filename_fondo2 = ""               # Si hubiera fondo, ej. "fondo2.dat"
        nlines          = 3                # Líneas de cabecera
        nchan           = 4000
        ngroup          = 10               # Agrupación para espectro global
        ngroup_peak     = 4                # Agrupación para la región del pico
    else:
        filename        = "Na_Spectech.tsv"
        filename_fondo1 = ""
        filename_fondo2 = ""
        nlines          = 22
        nchan           = 1024
        ngroup          = 4
        ngroup_peak     = 2

    # Si se usan fondos, definir tiempos de adquisición (segundos)
    tiempo         = 122.0
    tiempo_fondo1  = 300.0
    tiempo_fondo2  = 0.0
    if (filename_fondo1 != "") or (filename_fondo2 != ""):
        tiempo = 180.0
    if filename_fondo1 != "":
        tiempo_fondo1 = 600.0
    if filename_fondo2 != "":
        tiempo_fondo2 = 600.0

    # ------------------------------------------------------------
    # Definición de intervalos para los ajustes
    # ------------------------------------------------------------
    # 1) Región del histograma donde está el pico
    if phywe:
        hist_ini = 1200
        hist_fin = 1600
    else:
        hist_ini = 220
        hist_fin = 400

    # 2) Intervalo para ajustar el fondo
    if phywe:
        bkg_ini = 1100
        bkg_fin = 1200
    else:
        bkg_ini = 220
        bkg_fin = 250

    bkg_lineal = False  # True: fondo lineal; False: fondo exponencial

    # 3) Intervalo para ajustar la gaussiana
    if phywe:
        g1_ini = 1300
        g1_fin = 1500
    else:
        g1_ini = 260
        g1_fin = 360

    # 4) Intervalo total de ajuste (para la función total: fondo + gaussiana)
    if phywe:
        total_ini = 1200
        total_fin = 1600
    else:
        total_ini = 230
        total_fin = 400

    # Verificar que nchan es divisible por ngroup y ngroup_peak
    if (nchan % ngroup != 0) or (nchan % ngroup_peak != 0):
        print("El número de canales no es divisible por la agrupación. STOP.")
        return

    # ------------------------------------------------------------
    # Lectura de archivos y construcción de los histogramas
    # ------------------------------------------------------------
    nbins_global = nchan // ngroup
    nbins_peak   = nchan // ngroup_peak
    global_hist = np.zeros(nbins_global)
    peak_hist   = np.zeros(nbins_peak)

    # Leer archivos (saltando la cabecera)
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
            factor = tiempo / (tiempo_fondo1 + tiempo_fondo2) if (tiempo_fondo1+tiempo_fondo2) != 0 else 0
            y -= factor * (y1 + y2)
        elif fondo1_lines is not None:
            factor = tiempo / tiempo_fondo1 if tiempo_fondo1 != 0 else 0
            y -= factor * y1
        elif fondo2_lines is not None:
            factor = tiempo / tiempo_fondo2 if tiempo_fondo2 != 0 else 0
            y -= factor * y2

        # Agregar a los histogramas (la asignación se hace según la agrupación)
        bin_index_global = int(x) // ngroup
        if 0 <= bin_index_global < nbins_global:
            global_hist[bin_index_global] += y
        bin_index_peak = int(x) // ngroup_peak
        if 0 <= bin_index_peak < nbins_peak:
            peak_hist[bin_index_peak] += y

        row += 1

    # Definir los bordes de los bins (para graficar como histograma)
    global_bin_edges = np.linspace(0, nchan, nbins_global+1)
    peak_bin_edges   = np.linspace(0, nchan, nbins_peak+1)
    # También definimos los centros para algunos ajustes posteriores
    global_centers = (global_bin_edges[:-1] + global_bin_edges[1:]) / 2
    peak_centers   = (peak_bin_edges[:-1] + peak_bin_edges[1:]) / 2

    # ------------------------------------------------------------
    # Gráfico del espectro completo (dibujado como histograma estilo "step")
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    # Para usar step se requiere que el array tenga longitud = len(bordes)
    hist_global_step = np.append(global_hist, global_hist[-1])
    ax1.step(global_bin_edges, hist_global_step, where='post', color='black')
    ax1.fill_between(global_bin_edges, hist_global_step, step='post', color='lightblue', alpha=0.5)
    ax1.set_xlabel("Canal")
    ax1.set_ylabel("# cuentas")
    ax1.set_title("Espectro completo")
    ax1.set_xlim(0, nchan)
    fig1.savefig("ajuste1pico_espectro.pdf", bbox_inches="tight")
    fig1.savefig("ajuste1pico_espectro.png", bbox_inches="tight")
    
    # ------------------------------------------------------------
    # Ajuste en la región del pico (con fondo + gaussiana)
    # ------------------------------------------------------------
    # Seleccionar la región del pico (usamos los datos del histograma 'peak')
    mask_peak = (peak_centers >= hist_ini) & (peak_centers <= hist_fin)
    peak_x = peak_centers[mask_peak]
    peak_y = peak_hist[mask_peak]
    
    #elimina valores con y = 0 (ey = 0)
    mask_valid = (peak_y > 0) & np.isfinite(peak_y)
    
    peak_x = peak_x[mask_valid]
    peak_y = peak_y[mask_valid]
    
    # Usamos errores sqrt(n) (se usa valor absoluto para evitar NaN)
    peak_y_err = np.sqrt(np.abs(peak_y))

    # --- Funciones de ajuste ---
    def linear_bkg(x, p0, p1):
        return p0 + p1 * x

    def expo_bkg(x, p0, p1):
        return np.exp(p0 + p1 * x)

    def gaus(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Función total: fondo + gaussiana
    def total_func(x, p0, p1, p2, p3, p4, linear=True):
        if linear:
            return linear_bkg(x, p0, p1) + gaus(x, p2, p3, p4)
        else:
            return expo_bkg(x, p0, p1) + gaus(x, p2, p3, p4)

    # --- Ajuste de la gaussiana en la región [g1_ini, g1_fin] ---
    mask_g1 = (peak_x >= g1_ini) & (peak_x <= g1_fin)
    if np.sum(mask_g1) < 3:
        print("No hay suficientes puntos para ajuste gaussiano.")
        return
    A0 = np.max(peak_y[mask_g1])
    mu0 = (g1_ini + g1_fin) / 2.0
    sigma0 = (g1_fin - g1_ini) / 6.0
    p0_gaus = [A0, mu0, sigma0]
    try:
        popt_g1, pcov_g1 = curve_fit(gaus, peak_x[mask_g1], peak_y[mask_g1], p0=p0_gaus)
    except Exception as e:
        print("Error en ajuste gaussiano:", e)
        return

    # --- Ajuste del fondo en la región [bkg_ini, bkg_fin] ---
    mask_bkg = (peak_x >= bkg_ini) & (peak_x <= bkg_fin)
    num_puntos_bkg = np.sum(mask_bkg)
    print("Cantidad de puntos en la región de fondo:", num_puntos_bkg)
    if num_puntos_bkg < 2:
        print("No hay suficientes puntos para ajuste de fondo. Revisa bkg_ini y bkg_fin.")
        return
    if bkg_lineal:
        p0_bkg_guess = [np.mean(peak_y[mask_bkg]), 0.0]
        try:
            popt_bkg, pcov_bkg = curve_fit(linear_bkg, peak_x[mask_bkg], peak_y[mask_bkg], p0=p0_bkg_guess)
        except Exception as e:
            print("Error en ajuste de fondo lineal:", e)
            return
    else:
        p0_bkg_guess = [np.log(np.mean(peak_y[mask_bkg]) + 1e-6), 0.0]
        try:
            popt_bkg, pcov_bkg = curve_fit(expo_bkg, peak_x[mask_bkg], peak_y[mask_bkg], p0=p0_bkg_guess)
        except Exception as e:
            print("Error en ajuste de fondo exponencial:", e)
            return

    # --- Ajuste total en la región [total_ini, total_fin] ---
    mask_total = (peak_x >= total_ini) & (peak_x <= total_fin)
    if np.sum(mask_total) < 3:
        print("No hay suficientes puntos para ajuste total.")
        return

    # Parámetros iniciales para la función total:
    # Orden: [Constante, Pendiente, Amplitud, Centroide, Sigma]
    p0_total = popt_bkg[0]
    p1_total = popt_bkg[1]
    p2_total = popt_g1[0]
    p3_total = popt_g1[1]
    p4_total = popt_g1[2]
    p0_total_guess = [p0_total, p1_total, p2_total, p3_total, p4_total]

    try:
        if bkg_lineal:
            popt_total, pcov_total = curve_fit(
                lambda x, a, b, c, d, e: linear_bkg(x, a, b) + gaus(x, c, d, e),
                peak_x[mask_total], peak_y[mask_total],
                p0=p0_total_guess, sigma=peak_y_err[mask_total], absolute_sigma=True
            )
        else:
            popt_total, pcov_total = curve_fit(
                lambda x, a, b, c, d, e: expo_bkg(x, a, b) + gaus(x, c, d, e),
                peak_x[mask_total], peak_y[mask_total],
                p0=p0_total_guess, sigma=peak_y_err[mask_total], absolute_sigma=True
            )
    except Exception as e:
        print("Error en ajuste total:", e)
        return

    # Extraer parámetros y errores
    p0_fit, p1_fit, p2_fit, p3_fit, p4_fit = popt_total
    perr = np.sqrt(np.diag(pcov_total))
    p0_err, p1_err, p2_err, p3_err, p4_err = perr

    # Cálculos adicionales
    fwhm    = 2.35 * p4_fit
    fwhm_err = 2.35 * p4_err
    resol   = fwhm / p3_fit if p3_fit != 0 else 0
    if (fwhm != 0) and (p3_fit != 0):
        e_resol = resol * math.sqrt((fwhm_err / fwhm)**2 + (p3_err / p3_fit)**2)
    else:
        e_resol = 0
    area    = math.sqrt(2 * math.pi) * p2_fit * p4_fit / ngroup_peak
    if (p2_fit != 0) and (p4_fit != 0):
        rel_err_area = math.sqrt((p2_err / p2_fit)**2 + (p4_err / p4_fit)**2)
    else:
        rel_err_area = 0
    area_err = area * rel_err_area

    # Calcular chi2 (suma de ((data - ajuste)/error)² en la región total)
    if bkg_lineal:
        fit_vals = linear_bkg(peak_x[mask_total], p0_fit, p1_fit) + gaus(peak_x[mask_total], p2_fit, p3_fit, p4_fit)
    else:
        fit_vals = expo_bkg(peak_x[mask_total], p0_fit, p1_fit) + gaus(peak_x[mask_total], p2_fit, p3_fit, p4_fit)
    chi2 = np.sum(((peak_y[mask_total] - fit_vals) / (peak_y_err[mask_total] + 1e-6))**2)
    ndf  = np.sum(mask_total) - 5

    # Mostrar resultados en consola
    print("\nResultados del ajuste:\n")
    print("Parámetros del fondo:")
    print(f"  Constante (p0): {p0_fit:.3f} +/- {p0_err:.3f}")
    print(f"  Pendiente (p1): {p1_fit:.3f} +/- {p1_err:.3f}")
    print("\nParámetros de la gaussiana:")
    print(f"  Amplitud (p2):  {p2_fit:.2f} +/- {p2_err:.2f}")
    print(f"  Centroide (p3): {p3_fit:.2f} +/- {p3_err:.2f}")
    print(f"  Sigma (p4):     {p4_fit:.2f} +/- {p4_err:.2f}")
    print(f"  FWHM: {fwhm:.2f} +/- {fwhm_err:.2f}")
    print(f"  Resolución: {resol:.5f} +/- {e_resol:.5f}")
    print(f"  Área Gaussiana: {area:.0f} +/- {area_err:.0f}")
    print("\nChi-2:")
    print(f"  chi2 / ndf = {chi2:.1f} / {ndf}")

    # ------------------------------------------------------------
    # Gráfico de la región del pico y ajuste total
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.errorbar(peak_x, peak_y, yerr=peak_y_err, fmt='o', label="Datos", capsize=3)
    x_fit = np.linspace(total_ini, total_fin, 200)
    if bkg_lineal:
        y_fit = linear_bkg(x_fit, p0_fit, p1_fit) + gaus(x_fit, p2_fit, p3_fit, p4_fit)
        y_bkg = linear_bkg(x_fit, p0_fit, p1_fit)
    else:
        y_fit = expo_bkg(x_fit, p0_fit, p1_fit) + gaus(x_fit, p2_fit, p3_fit, p4_fit)
        y_bkg = expo_bkg(x_fit, p0_fit, p1_fit)
    y_gaus = gaus(x_fit, p2_fit, p3_fit, p4_fit)
    ax2.plot(x_fit, y_fit, 'r-', label="Ajuste total")
    ax2.plot(x_fit, y_bkg, 'g--', label="Fondo")
    ax2.plot(x_fit, y_gaus, 'b--', label="Gaussiana")
    ax2.set_xlabel("Canal")
    ax2.set_ylabel("# cuentas")
    ax2.set_title("Ajuste del pico")
    ax2.legend()
    fig2.savefig("ajuste1pico.pdf", bbox_inches="tight")
    fig2.savefig("ajuste1pico.png", bbox_inches="tight")
    
    # Mostrar ambas figuras
    plt.show()

if __name__ == "__main__":
    ajuste1pico()
