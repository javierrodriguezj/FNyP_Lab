
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def expofit():
    """
    Versión Python para ajustar una función exponencial más un fondo constante:
    f(x) = p0 + exp(p1 + p2*x).

    Lee un archivo (expo.dat) con líneas de cabecera y columnas: x, xerr, y, yerr.
    Hace el ajuste en el rango [graph_ini, graph_fin] y dibuja el resultado.
    """

    # ---------------------------------------------------------------------
    # Parámetros de lectura y configuración
    # ---------------------------------------------------------------------
    filename    = "expo.dat"  # Archivo donde están los datos
    nlines      = 2           # Número de líneas de cabecera (a ignorar al inicio)
    graph_ini   = -1.0        # Límite inferior del eje X para el ajuste
    graph_fin   = 18.0        # Límite superior del eje X para el ajuste
    graphy_ini  = 3.0         # Límite inferior del eje Y (para dibujar)
    graphy_fin  = 30.0        # Límite superior del eje Y (para dibujar)

    # Vectores para almacenar los datos
    x_vals     = []
    x_errs     = []
    y_vals     = []
    y_errs     = []

    # ---------------------------------------------------------------------
    # Lectura del archivo
    # ---------------------------------------------------------------------
    with open(filename, "r") as f:
        row = 0
        for line in f:
            row += 1
            # Ignorar las primeras 'nlines' líneas como cabecera
            if row <= nlines:
                continue

            cols = line.strip().split()
            if len(cols) >= 4:
                x   = float(cols[0])
                xe  = float(cols[1])
                y   = float(cols[2])
                ye  = float(cols[3])
                x_vals.append(x)
                x_errs.append(xe)
                y_vals.append(y)
                y_errs.append(ye)

    # Convertir a numpy arrays
    x_vals  = np.array(x_vals, dtype=float)
    x_errs  = np.array(x_errs, dtype=float)
    y_vals  = np.array(y_vals, dtype=float)
    y_errs  = np.array(y_errs, dtype=float)

    # ---------------------------------------------------------------------
    # Selección de datos en el rango [graph_ini, graph_fin]
    # (equivalente a 'Fit("R")' en ROOT, que ajusta solo en ese rango)
    # ---------------------------------------------------------------------
    mask_fit = (x_vals >= graph_ini) & (x_vals <= graph_fin)
    x_fit    = x_vals[mask_fit]
    y_fit    = y_vals[mask_fit]
    yerr_fit = y_errs[mask_fit]
    
    #elimina valores con y = 0 (ey = 0)
    mask_valid = (y_fit > 0) & np.isfinite(y_fit) & (yerr_fit > 0)
    
    x_fit = x_fit[mask_valid]
    y_fit = y_fit[mask_valid]
    yerr_fit = y_errs[mask_valid]

    # ---------------------------------------------------------------------
    # Definir la función de ajuste (vectorizada):
    #    f(x) = p0 + exp(p1 + p2*x)
    # ---------------------------------------------------------------------
    def f_tot(x, p0, p1, p2):
        return p0 + np.exp(p1 + p2*x)

    # ---------------------------------------------------------------------
    # Ajuste con scipy.optimize.curve_fit
    # Pesamos cada punto usando su error en y (si no es 0)
    # ---------------------------------------------------------------------
    # Valor inicial aproximado de los parámetros: p0=1, p1=1, p2=-0.1, por ejemplo.
    p0_init = [1.0, 1.0, -0.1]  

    popt, pcov = curve_fit(
        f_tot,
        x_fit, y_fit,
        p0=p0_init,
        sigma=yerr_fit,
        absolute_sigma=True  # interpretar 'sigma' como errores absolutos
    )

    # popt = [p0, p1, p2]
    p0_fit, p1_fit, p2_fit = popt
    # Errores estándar en los parámetros
    perr = np.sqrt(np.diag(pcov))
    p0_err, p1_err, p2_err = perr

    # ---------------------------------------------------------------------
    # Cálculo de chi^2 y ndf
    # ---------------------------------------------------------------------
    # Reproducimos la idea de ROOT: sum((y_i - f(x_i))^2 / err_i^2)
    # en el rango ajustado
    y_model = f_tot(x_fit, p0_fit, p1_fit, p2_fit)
    mask_err = (yerr_fit > 0)
    chi2 = np.sum(((y_fit[mask_err] - y_model[mask_err]) / yerr_fit[mask_err])**2)
    ndf  = np.sum(mask_err) - 3  # 3 parámetros

    # ---------------------------------------------------------------------
    # Mostrar resultados
    # ---------------------------------------------------------------------
    print("\nParámetros del ajuste (f(x) = p0 + exp(p1 + p2*x)):")
    print(f"   p0 = {p0_fit: .5f} +/- {p0_err: .5f}")
    print(f"   p1 = {p1_fit: .7f} +/- {p1_err: .7f}")
    print(f"   p2 = {p2_fit: .7f} +/- {p2_err: .7f}")

    print("\nChi-2:")
    print(f"   chi2 / ndf: {chi2:.3f} / {ndf} = {chi2/ndf if ndf>0 else np.nan:.3f}")

    # ---------------------------------------------------------------------
    # Gráficas: lineal y logarítmica
    # ---------------------------------------------------------------------
    # 1) Escala lineal
    fig_lin, ax_lin = plt.subplots(figsize=(6,4))
    ax_lin.set_title("Escala lineal")
    ax_lin.set_xlabel("Espesor [g/cm^2]")
    ax_lin.set_ylabel("# cuentas/s")

    # Rango X
    ax_lin.set_xlim(graph_ini, graph_fin)
    # Rango Y (si se desea)
    if graphy_ini < graphy_fin:
        ax_lin.set_ylim(graphy_ini, graphy_fin)

    # Dibujar datos con barras de error
    ax_lin.errorbar(
        x_vals, y_vals, yerr=y_errs, xerr=x_errs, fmt='o', color='blue',
        ecolor='blue', capsize=3, label='Datos'
    )

    # Para dibujar la función ajustada, creamos puntos de x en [graph_ini, graph_fin]
    x_plot = np.linspace(graph_ini, graph_fin, 200)
    y_plot = f_tot(x_plot, p0_fit, p1_fit, p2_fit)
    ax_lin.plot(x_plot, y_plot, 'r-', label='Ajuste')

    ax_lin.legend()
    ax_lin.grid(True)

    # 2) Escala logarítmica
    fig_log, ax_log = plt.subplots(figsize=(6,4))
    ax_log.set_title("Escala logarítmica")
    ax_log.set_xlabel("Espesor [g/cm^2]")
    ax_log.set_ylabel("# cuentas/s (log)")
    ax_log.set_xlim(graph_ini, graph_fin)
    if graphy_ini < graphy_fin:
        ax_log.set_ylim(graphy_ini, graphy_fin)
    ax_log.set_yscale('log')

    ax_log.errorbar(
        x_vals, y_vals, yerr=y_errs, xerr=x_errs, fmt='o', color='blue',
        ecolor='blue', capsize=3, label='Datos'
    )
    ax_log.plot(x_plot, y_plot, 'r-', label='Ajuste')
    ax_log.legend()
    ax_log.grid(True, which='both', ls=":")

    # Guardar imágenes (opcional)
    fig_lin.savefig("expofit_lin.png", dpi=150, bbox_inches='tight')
    fig_lin.savefig("expofit_lin.pdf", bbox_inches='tight')
    fig_log.savefig("expofit_log.png", dpi=150, bbox_inches='tight')
    fig_log.savefig("expofit_log.pdf", bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    expofit()
