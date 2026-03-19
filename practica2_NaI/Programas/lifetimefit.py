import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

def lifetimefit():
   
    # ---------------------------------------------------------------------
    # Parámetros de configuración (elige True para Phywe, False para Spectech)
    # ---------------------------------------------------------------------
    phywe = False  # True para Phywe, False para Spectech

    if phywe:
        filename = "lifetime_Phywe.dat"
        nlines   = 3      # Líneas de cabecera
        nchan    = 60     # Número de medidas
        t_int    = 30.0   # Tiempo de intervalo en segundos
        l0_ini   = 50
        l0_fin   = nchan
        expo_ini = 1
        expo_fin = 50
        hist_ini = 1
        hist_fin = nchan
        ngroup   = 1
    else:
        filename = "lifetime_Spectech.tsv"
        nlines   = 22
        nchan    = 166
        t_int    = 8.0
        l0_ini   = 157
        l0_fin   = nchan
        expo_ini = 0
        expo_fin = 156
        hist_ini = 0
        hist_fin = nchan
        ngroup   = 1

    # Verificar que nchan sea divisible por ngroup
    if (nchan % ngroup) != 0:
        print("El número de canales no es divisible por el número de agrupación de canales. STOP")
        return

    # ---------------------------------------------------------------------
    # Lectura de datos y construcción de 'histograma' (array)
    # ---------------------------------------------------------------------
    n_bins = nchan // ngroup
    hist_data = np.zeros(n_bins)
    hist_x = np.arange(n_bins, dtype=float)  # eje X (0..n_bins-1)

    # Leer archivo
    with open(filename, "r") as f:
        row = 0
        for line in f:
            row += 1
            if row <= nlines:
                # Ignorar cabecera
                continue

            cols = line.strip().split()
            if len(cols) < 2:
                continue

            x_val = float(cols[0])
            y_val = float(cols[1])

            # Cálculo del bin
            bin_index = int(x_val // ngroup)
            if 0 <= bin_index < n_bins:
                hist_data[bin_index] += y_val

    # Error estadístico por conteo
    hist_err = np.sqrt(hist_data)

    # ---------------------------------------------------------------------
    # Definición de funciones de ajuste (vectorizadas)
    # ---------------------------------------------------------------------
    def f_bkg(x, p0):
        # Fondo constante
        return p0 * np.ones_like(x)

    def f_exp(x, p1, p2):
        # Exponencial
        return np.exp(p1 + p2*x)

    def f_tot(x, p0, p1, p2):
        # Suma de fondo y exponencial
        return p0 * np.ones_like(x) + np.exp(p1 + p2*x)

    # ---------------------------------------------------------------------
    # Ajuste preliminar: fondo en [l0_ini, l0_fin] (solo p0)
    #                    exponencial en [expo_ini, expo_fin] (p1, p2)
    # ---------------------------------------------------------------------
    l0_ini_bin   = max(0, min(n_bins-1, l0_ini))
    l0_fin_bin   = max(0, min(n_bins-1, l0_fin))
    expo_ini_bin = max(0, min(n_bins-1, expo_ini))
    expo_fin_bin = max(0, min(n_bins-1, expo_fin))

    # Fondo
    x_bkg   = hist_x[l0_ini_bin:l0_fin_bin+1]
    y_bkg   = hist_data[l0_ini_bin:l0_fin_bin+1]
    err_bkg = hist_err[l0_ini_bin:l0_fin_bin+1]

    #elimina valores con y = 0 (ey = 0)
    mask_bkg_valid = (y_bkg > 0) & np.isfinite(y_bkg) & (err_bkg > 0)
    x_bkg = x_bkg[mask_bkg_valid]
    y_bkg = y_bkg[mask_bkg_valid]
    err_bkg = err_bkg[mask_bkg_valid]
    
    # Ajuste del fondo: f_bkg(x, p0) = p0
    popt_bkg, pcov_bkg = curve_fit(
        f_bkg, x_bkg, y_bkg,
        p0=[np.mean(y_bkg)],  # valor inicial
        sigma=err_bkg,
        absolute_sigma=True
    )
    p0_bkg = popt_bkg[0]

    # Exponencial
    x_exp   = hist_x[expo_ini_bin:expo_fin_bin+1]
    y_exp   = hist_data[expo_ini_bin:expo_fin_bin+1]
    err_exp = hist_err[expo_ini_bin:expo_fin_bin+1]

   #elimina valores con y = 0 (ey = 0)
    mask_exp_valid = (y_exp > 0) & np.isfinite(y_exp) & (err_exp > 0)
    x_exp = x_exp[mask_exp_valid]
    y_exp = y_exp[mask_exp_valid]
    err_exp = err_exp[mask_exp_valid]
 
    # Ajuste exponencial inicial con log (si hay valores > 0)
    mask_pos = (y_exp > 0)
    if np.any(mask_pos):
        x_lin = x_exp[mask_pos]
        y_lin = np.log(y_exp[mask_pos])
        # Ajuste lineal ln(y) = p1 + p2*x  => polyfit da [pendiente, intercepto]
        b, a = np.polyfit(x_lin, y_lin, 1)
        p1_init, p2_init = a, b
    else:
        p1_init, p2_init = (1.0, -0.01)

    try:
        popt_exp, pcov_exp = curve_fit(
            f_exp, x_exp, y_exp,
            p0=[p1_init, p2_init],
            sigma=err_exp,
            absolute_sigma=True
        )
        p1_exp, p2_exp = popt_exp
    except RuntimeError:
        # Si falla el ajuste
        p1_exp, p2_exp = (p1_init, p2_init)

    # ---------------------------------------------------------------------
    # Ajuste total: f_tot(x, p0, p1, p2) = p0 + exp(p1 + p2*x)
    # Rango [hist_ini, hist_fin]
    # ---------------------------------------------------------------------
    hist_ini_bin = max(0, min(n_bins-1, hist_ini))
    hist_fin_bin = max(0, min(n_bins-1, hist_fin))

    x_tot   = hist_x[hist_ini_bin:hist_fin_bin+1]
    y_tot   = hist_data[hist_ini_bin:hist_fin_bin+1]
    err_tot = hist_err[hist_ini_bin:hist_fin_bin+1]

    #elimina valores con y = 0 (ey = 0)
    mask_tot_valid = (y_tot > 0) & np.isfinite(y_tot) & (err_tot > 0)
    x_tot = x_tot[mask_tot_valid]
    y_tot = y_tot[mask_tot_valid]
    err_tot = err_tot[mask_tot_valid]

    # Punto de partida: p0_bkg, p1_exp, p2_exp
    init_guess = [p0_bkg, p1_exp, p2_exp]

    popt_tot, pcov_tot = curve_fit(
        f_tot, x_tot, y_tot,
        p0=init_guess,
        sigma=err_tot,
        absolute_sigma=True
    )
    p0_fit, p1_fit, p2_fit = popt_tot
    perr_tot = np.sqrt(np.diag(pcov_tot))
    p0_err, p1_err, p2_err = perr_tot

    # ---------------------------------------------------------------------
    # Cálculo de chi^2 y ndf
    # ---------------------------------------------------------------------
    y_model = f_tot(x_tot, p0_fit, p1_fit, p2_fit)
    mask = (err_tot > 0)
    chi2 = np.sum(((y_tot[mask] - y_model[mask]) / err_tot[mask])**2)
    ndf  = np.sum(mask) - 3  # (nº de puntos) - (nº parámetros)

    # ---------------------------------------------------------------------
    # Cálculo de vida media y semivida
    # ---------------------------------------------------------------------
    life_ch = -1.0 / p2_fit
    er_life_ch = abs(p2_err * life_ch**2)

    # Vida media en minutos
    life_min    = (t_int * life_ch) / 60.0
    er_life_min = (t_int * er_life_ch) / 60.0

    # Semivida en minutos
    half_life_min    = life_min * math.log(2)
    er_half_life_min = er_life_min * math.log(2)

    # ---------------------------------------------------------------------
    # Mostrar resultados
    # ---------------------------------------------------------------------
    print("\nParámetros del ajuste total (p0 + exp(p1 + p2*x)):")
    print(f"   Fondo = {p0_fit: .5f}  +/- {p0_err: .5f}")
    print(f"   Constante exponential = {p1_fit: .7f}  +/- {p1_err: .7f}")
    print(f"   Pendiente exponential = {p2_fit: .7f}  +/- {p2_err: .7f}")

    print("\nChi2:")
    print(f"   chi2 / ndf = {chi2:.3f} / {ndf} = {chi2/ndf if ndf>0 else np.nan:.3f}")

    print("\nVida media:")
    print(f"  ( {life_min: .9f} +/- {er_life_min: .9f} ) min ")

    print("\nSemivida:")
    print(f"  ( {half_life_min: .9f} +/- {er_half_life_min: .9f} ) min\n")

    # ---------------------------------------------------------------------
    # Gráficas: escala lineal y logarítmica
    # ---------------------------------------------------------------------
    fig_lin, ax_lin = plt.subplots(figsize=(6,4))
    fig_log, ax_log = plt.subplots(figsize=(6,4))

    # Escala lineal
    ax_lin.errorbar(hist_x, hist_data, yerr=hist_err, fmt='o', ms=4, color='blue', label='Datos',zorder=1)
    ax_lin.plot(x_tot, f_bkg(x_tot, p0_fit), 'g--', label='Fondo',zorder=2)  # vectorizada
    ax_lin.plot(x_tot, f_tot(x_tot, p0_fit, p1_fit, p2_fit), 'r-', label='Ajuste total',zorder=3)
    ax_lin.set_xlabel("Canal")
    ax_lin.set_ylabel("Cuentas")
    ax_lin.set_title("Ajuste en escala lineal")
    ax_lin.legend()
    ax_lin.grid(True)

    # Escala logarítmica
    ax_log.errorbar(hist_x, hist_data, yerr=hist_err, fmt='o', ms=4, color='blue', label='Datos',zorder=1)
    ax_log.plot(x_tot, f_bkg(x_tot, p0_fit), 'g--', label='Fondo',zorder=2)
    ax_log.plot(x_tot, f_tot(x_tot, p0_fit, p1_fit, p2_fit), 'r-', label='Ajuste total',zorder=3)
    ax_log.set_yscale('log')
    ax_log.set_xlabel("Canal")
    ax_log.set_ylabel("Cuentas (escala log)")
    ax_log.set_title("Ajuste en escala logarítmica")
    ax_log.legend()
    ax_log.grid(True, which="both", ls=":")

    fig_lin.savefig("lifetimefit_lin.png", dpi=150, bbox_inches='tight')
    fig_lin.savefig("lifetimefit_lin.pdf", bbox_inches='tight')
    fig_log.savefig("lifetimefit_log.png", dpi=150, bbox_inches='tight')
    fig_log.savefig("lifetimefit_log.pdf", bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    lifetimefit()
