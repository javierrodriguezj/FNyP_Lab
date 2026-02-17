#!/usr/bin/env python3
"""
Script para comparar un conjunto de datos con una distribución de Poisson.
- Lee datos desde un archivo de texto.
- Calcula la media y el error de la media.
- Construye histogramas con bins constantes y variables.
- Ajusta la distribución de Poisson.
- Evalúa la bondad de ajuste mediante un test Chi².
- Genera gráficos comparativos (observado vs. esperado).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, chi2
from scipy.optimize import curve_fit
import scipy.special as sp

# --------------------------
# EXTENSIÓN CONTINUA DE POISSON
# --------------------------
def poisson_continuous(x, lamb):
    """Extensión continua de la Poisson usando la función Gamma."""
    return np.exp(-lamb) * (lamb**x) / sp.gamma(x+1)

def main():
    # --------------------------
    # PARÁMETROS INICIALES
    # --------------------------
    filename = "fondo.dat"    # Archivo de datos
    nlines = 1                # Número de líneas de cabecera a saltar
    hist_ini, hist_fin = 0, 10

    # Bins de anchura constante (centrados en enteros)
    edges_const = np.arange(hist_ini - 0.5, hist_fin + 1.5, 1.0)

    # Bins de anchura variable (ejemplo: últimos valores agrupados)
    edges_var = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 10.5])

    # --------------------------
    # LECTURA DE DATOS
    # --------------------------
    data = np.loadtxt(filename, skiprows=nlines, usecols=2)
    nmeasu = len(data)

    # --------------------------
    # ESTADÍSTICA DESCRIPTIVA
    # --------------------------
    mean = np.mean(data)
    sample_variance = np.var(data, ddof=1)
    mean_error = np.sqrt(sample_variance / nmeasu)

    # --------------------------
    # HISTOGRAMAS
    # --------------------------
    hist_counts_const, _ = np.histogram(data, edges_const)
    hist_counts_var, _   = np.histogram(data, edges_var)

    bin_centers_const = 0.5 * (edges_const[:-1] + edges_const[1:])
    bin_centers_var   = 0.5 * (edges_var[:-1] + edges_var[1:])

    # --------------------------
    # AJUSTE DE POISSON
    # --------------------------
    def poisson_model(k, lamb):
        return nmeasu * poisson.pmf(k, lamb)

    mask = hist_counts_const > 0
    popt, pcov = curve_fit(poisson_model,
                           bin_centers_const[mask],
                           hist_counts_const[mask],
                           p0=[mean],
                           sigma=np.sqrt(hist_counts_const[mask]),
                           absolute_sigma=True)
    lamb_fit = popt[0]
    lamb_err = np.sqrt(np.diag(pcov))[0]

    # Expectativas con λ ajustada
    expected_const = poisson_model(bin_centers_const, lamb_fit)

    def expected_var_bins(lamb, N, edges):
        expected = []
        for i in range(len(edges) - 1):
            low, high = edges[i], edges[i+1]
            # Probabilidad de que k esté dentro del bin [low, high)
            prob = poisson.cdf(high, lamb) - poisson.cdf(low, lamb)
            expected.append(N * prob)
        return np.array(expected)

    expected_var = expected_var_bins(lamb_fit, nmeasu, edges_var)

    # --------------------------
    # TEST CHI²
    # --------------------------
    chi2_val = np.sum(((hist_counts_const[mask] - expected_const[mask]) /
                       np.sqrt(hist_counts_const[mask]))**2)
    ndof = np.sum(mask) - 1
    p_value = 1 - chi2.cdf(chi2_val, ndof)

    print(f"\nMedia de las {nmeasu} medidas: {mean:.7f} ± {mean_error:.7f}")
    print("Ajuste Poisson (bins constantes):")
    print(f"  λ = {lamb_fit:.4f} ± {lamb_err:.4f}")
    print(f"  Chi2 / ndf = {chi2_val:.4f} / {ndof}")
    print(f"  CL = {p_value:.5f}\n")

    mask_var = hist_counts_var > 0
    chi2_val_var = np.sum(((hist_counts_var - expected_var) ** 2) / expected_var)
    ndof_var = len(hist_counts_var) - 1
    p_value_var = 1 - chi2.cdf(chi2_val_var, ndof_var)

    print("Comparacion de histogramas de anchura bin variable:")
    print(f"  Chi2 / ndf = {chi2_val_var:.4f} / {ndof_var}")
    print(f"  CL = {p_value_var:.5f}\n")
    
    # --------------------------
    # GRÁFICOS
    # --------------------------
    # Histograma de anchura constante + curva suave
    plt.figure(figsize=(8,6))
    plt.errorbar(bin_centers_const, hist_counts_const,
                 yerr=np.sqrt(hist_counts_const), fmt='o', label='Observado')
    # Ajuste en versión histograma
    plt.step(edges_const, np.append(expected_const, expected_const[-1]),
             where="post", color="r", label='Ajuste Poisson (histograma)')
    # Ajuste en versión curva suave (extensión gamma)
    x_dense = np.linspace(hist_ini, hist_fin, 500)
    y_dense = nmeasu * poisson_continuous(x_dense, lamb_fit)
    plt.plot(x_dense, y_dense, "b--", lw=2, label='Ajuste Poisson (curva suave)')
    plt.xlabel("# cuentas")
    plt.ylabel("frecuencia")
    plt.title("Histograma de anchura constante con ajuste Poisson")
    plt.legend()
    plt.savefig("poisson_fit.png", dpi=150)
    plt.savefig("poisson_fit.pdf", dpi=150)

    # Histograma de anchura variable
    plt.figure(figsize=(8,6))
    plt.errorbar(bin_centers_var, hist_counts_var,
                 yerr=np.sqrt(hist_counts_var), fmt='o', label='Observado')
    plt.step(edges_var, np.append(expected_var, expected_var[-1]),
             where="post", color="r", label='Esperado (Poisson)')
    plt.xlabel("# cuentas")
    plt.ylabel("frecuencia")
    plt.title("Histograma de anchura variable")
    plt.legend()
    plt.savefig("poisson.png", dpi=150)
    plt.savefig("poisson.pdf", dpi=150)

    plt.show()

if __name__ == "__main__":
    main()
