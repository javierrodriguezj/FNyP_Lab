from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    # --------------------------
    # PARÁMETROS INICIALES
    # --------------------------
    ruta_datos = Path(__file__).resolve().parent / "data"
    ruta_resultados = Path(__file__).resolve().parent / "results"

    filename = ruta_datos / "parte1.tsv" # Archivo de datos 
    nlines = 11                        # Número de líneas de cabecera a saltar
    hist_ini, hist_fin = 0, 10

    # --------------------------
    # LECTURA DE DATOS
    # --------------------------
    data = np.loadtxt(filename, skiprows=nlines, usecols=2)
    nmeasu = len(data)

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
    plt.savefig(ruta_guardado / "poisson_fit.png", dpi=150)
    plt.savefig(ruta_guardado / "poisson_fit.pdf", dpi=150)

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
    plt.savefig(ruta_guardado / "poisson.png", dpi=150)
    plt.savefig(ruta_guardado / "poisson.pdf", dpi=150)

    plt.show()

if __name__ == "__main__":
    main()
