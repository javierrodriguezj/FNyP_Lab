
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def saxon_woods(x, A, mu, width, B):
    """
    Función tipo Saxon-Woods:
      f(x) = A/(1 + exp((x - mu)/width)) + B
    donde:
      A: Altura (pico)
      mu: Media (centro)
      width: Anchura (controla la pendiente)
      B: Fondo
    """
    return A / (1 + np.exp((x - mu) / width)) + B

def ajusteCompton():
    """
    Ajuste del borde Compton con función tipo Saxon-Woods,
    sin usar ROOT (utilizando NumPy, SciPy y Matplotlib).

    Se leen los datos (y los fondos, si existen) de archivos de texto,
    se agrupan los canales en histogramas y se realiza el ajuste en la
    región del borde Compton.
    """
    # ------------------------------------------------------------
    # Configuración del sistema: Phywe o Spectech
    # ------------------------------------------------------------
    phywe = True  # True para Phywe, False para Spectech

    if phywe:
        filename        = "Mn_Phywe_180s.dat"
        filename_fondo1 = "fondo1_Phywe_600s.dat"  # Dejar vacío ("") si no hay fondo
        filename_fondo2 = "fondo2_Phywe_600s.dat"  # Dejar vacío ("") si no hay fondo
        nlines          = 3         # Número de líneas de cabecera
        nchan           = 4000      # Número de canales
        ngroup          = 10        # Agrupación para el espectro global
        ngroup_peak     = 8         # Agrupación para la región del borde Compton
    else:
        filename        = "Na_Spectech.tsv"
        filename_fondo1 = ""
        filename_fondo2 = ""
        nlines          = 22
        nchan           = 1024      # (O 2048, según el MCA)
        ngroup          = 4
        ngroup_peak     = 2

    # ------------------------------------------------------------
    # Definir intervalos para la visualización y el ajuste
    # ------------------------------------------------------------
    # Intervalo del histograma para la zona del borde Compton (para dibujar)
    if phywe:
        hist_ini = 750
        hist_fin = 1450
    else:
        hist_ini = 500
        hist_fin = 700

    # Intervalo en el que se realizará el ajuste (borde Compton)
    if phywe:
        borde_ini = 800
        borde_fin = 1400
    else:
        borde_ini = 500
        borde_fin = 680

    # Valor inicial aproximado de la Altura (A) para el ajuste
    if phywe:
        altura = 110
    else:
        altura = 1100

    # ------------------------------------------------------------
    # Definir tiempos de medida (solo si se van a restar fondos)
    # ------------------------------------------------------------
    tiempo = 0.0
    tiempo_fondo1 = 0.0
    tiempo_fondo2 = 0.0
    if (filename_fondo1 != "") or (filename_fondo2 != ""):
        tiempo = 180.0    # Ej. 180 s para la medida principal
    if filename_fondo1 != "":
        tiempo_fondo1 = 600.0  # Ej. 600 s
    if filename_fondo2 != "":
        tiempo_fondo2 = 600.0  # Ej. 600 s

    # ------------------------------------------------------------
    # Comprobar que nchan es divisible por la agrupación
    # ------------------------------------------------------------
    if (nchan % ngroup != 0) or (nchan % ngroup_peak != 0):
        print("El número de canales no es divisible por la agrupación. STOP.")
        return

    # Parámetros iniciales para la función Saxon-Woods:
    anchura = borde_fin - borde_ini
    media = borde_ini + (anchura // 2)

    # ------------------------------------------------------------
    # Lectura de archivos y construcción de histogramas
    # ------------------------------------------------------------
    nbins_global = nchan // ngroup
    nbins_peak = nchan // ngroup_peak
    global_hist = np.zeros(nbins_global)
    peak_hist = np.zeros(nbins_peak)

    # Leer archivo principal
    with open(filename, "r") as f:
        main_lines = f.readlines()
    # Leer archivos de fondo (si se proporcionan)
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
            continue  # Saltar cabecera
        cols = line.strip().split()
        if len(cols) < 2:
            row += 1
            continue
        x = float(cols[0])
        y = float(cols[1])
        # Inicializar fondos
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
            y = y - factor * (y1 + y2)
        elif fondo1_lines is not None:
            factor = tiempo / tiempo_fondo1 if tiempo_fondo1 != 0 else 0
            y = y - factor * y1
        elif fondo2_lines is not None:
            factor = tiempo / tiempo_fondo2 if tiempo_fondo2 != 0 else 0
            y = y - factor * y2

        # Agregar a los histogramas según la agrupación
        bin_index_global = int(x) // ngroup
        if 0 <= bin_index_global < nbins_global:
            global_hist[bin_index_global] += y
        bin_index_peak = int(x) // ngroup_peak
        if 0 <= bin_index_peak < nbins_peak:
            peak_hist[bin_index_peak] += y
        row += 1

    # Definir bordes y centros de bins
    global_bin_edges = np.linspace(0, nchan, nbins_global+1)
    global_centers = (global_bin_edges[:-1] + global_bin_edges[1:]) / 2

    peak_bin_edges = np.linspace(0, nchan, nbins_peak+1)
    peak_centers = (peak_bin_edges[:-1] + peak_bin_edges[1:]) / 2

    # ------------------------------------------------------------
    # Gráfico del espectro global (histograma tipo "step")
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    hist_global_step = np.append(global_hist, global_hist[-1])
    ax1.step(global_bin_edges, hist_global_step, where='post', color='black')
    ax1.fill_between(global_bin_edges, hist_global_step, step='post', color='lightblue', alpha=0.5)
    ax1.set_xlabel("Canal")
    ax1.set_ylabel("# cuentas")
    ax1.set_title("Espectro global")
    ax1.set_xlim(0, nchan)
    fig1.savefig("ajusteCompton_espectro_noROOT.pdf", bbox_inches="tight")
    fig1.savefig("ajusteCompton_espectro_noROOT.png", bbox_inches="tight")

    # ------------------------------------------------------------
    # Ajuste en la región del borde Compton
    # Se utiliza el histograma 'peak'
    # Para la visualización se usan datos en [hist_ini, hist_fin],
    # pero el ajuste se realiza en el rango [borde_ini, borde_fin].
    mask_peak = (peak_centers >= hist_ini) & (peak_centers <= hist_fin)
    x_peak = peak_centers[mask_peak]
    y_peak = peak_hist[mask_peak]
    
    #elimina valores con y = 0 (ey = 0)
    mask_valid = (y_peak > 0) & np.isfinite(y_peak)
    
    x_peak = x_peak[mask_valid]
    y_peak = y_peak[mask_valid]
    
    # Error estadístico de los datos en la región de interés
    # (Para el ajuste usaremos el error de los datos seleccionados)
    # y_peak_err originalmente se calculaba sobre toda la región, pero aquí usamos:
    y_peak_err = np.sqrt(np.abs(y_peak))

    # Seleccionar los datos en el rango de ajuste [borde_ini, borde_fin]
    mask_fit = (x_peak >= borde_ini) & (x_peak <= borde_fin)
    if np.sum(mask_fit) < 3:
        print("No hay suficientes puntos para ajuste.")
        return
    x_fit = x_peak[mask_fit]
    y_fit = y_peak[mask_fit]
    y_fit_err = np.sqrt(np.abs(y_fit))  # Usamos los errores correspondientes a los puntos de ajuste

    # Valor inicial para el ajuste: [A, mu, width, B]
    p0_guess = [altura, media, anchura, 10.0]

    try:
        popt, pcov = curve_fit(saxon_woods, x_fit, y_fit, p0=p0_guess, method='trf', maxfev=10000)
    except Exception as e:
        print("Error en el ajuste:", e)
        return
    perr = np.sqrt(np.diag(pcov))

    # Calcular chi² usando los errores de los datos de ajuste
    fit_vals = saxon_woods(x_fit, *popt)
    chi2 = np.sum(((y_fit - fit_vals) / (y_fit_err + 1e-6))**2)
    ndf = np.sum(mask_fit) - len(popt)

    # Imprimir resultados
    print("\nResultados del ajuste (Saxon-Woods):\n")
    print(f"  Altura (A): {popt[0]:.3f} +/- {perr[0]:.3f}")
    print(f"  Media (mu): {popt[1]:.3f} +/- {perr[1]:.3f}")
    print(f"  Anchura:    {popt[2]:.3f} +/- {perr[2]:.3f}")
    print(f"  Fondo (B):  {popt[3]:.3f} +/- {perr[3]:.3f}")
    print(f"\n  Chi2/ndf: {chi2:.2f} / {ndf}")

    # ------------------------------------------------------------
    # Gráfico de la región del borde Compton y del ajuste
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.errorbar(x_peak, y_peak, yerr=y_peak_err, fmt='o', label="Datos", capsize=3)
    x_plot = np.linspace(borde_ini, borde_fin, 200)
    y_plot = saxon_woods(x_plot, *popt)
    ax2.plot(x_plot, y_plot, 'r-', label="Ajuste Saxon-Woods")
    ax2.set_xlabel("Canal")
    ax2.set_ylabel("# cuentas")
    ax2.set_title("Ajuste del borde Compton")
    ax2.legend()
    fig2.savefig("ajusteCompton.pdf", bbox_inches="tight")
    fig2.savefig("ajusteCompton.png", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    ajusteCompton()
