import numpy as np
import matplotlib.pyplot as plt

def espectro():
    """
    Versión Python (sin ROOT) para visualizar un espectro energético
    adquirido con un MCA, con posibilidad de restar fondos.
    - Lee archivo principal y, si existen, dos archivos de fondo.
    - Resta los fondos (ponderados por tiempos).
    - Agrupa canales en bins y crea un 'histograma' en un array.
    - Dibuja y guarda la figura en PDF/PNG.
    """

    # ---------------------------------------------------------------------
    # Configuración
    # ---------------------------------------------------------------------
    phywe = True  # True para Phywe, False para Spectech

    if phywe:
        filename        = "Co_Phywe.dat"    # Archivo con el espectro
        filename_fondo1 = ""               # "XXX_fondo1.dat" si existiera
        filename_fondo2 = ""               # "XXX_fondo2.dat" si existiera
        nlines          = 3                # Número de líneas de cabecera en archivo principal
        nchan           = 4000             # Número de canales totales del MCA
        ngroup          = 40               # Agrupación de canales
    else:
        filename        = "Co_Spectech.tsv"
        filename_fondo1 = ""
        filename_fondo2 = ""
        nlines          = 22
        nchan           = 1024             # O el que corresponda a tu MCA
        ngroup          = 4

    # Si quisiéramos restar fondo, definimos tiempos de adquisición (segundos).
    # Solo se usan si filename_fondo1/2 no están vacíos.
    tiempo        = 0.0  # Tiempo de la medida principal
    tiempo_fondo1 = 0.0
    tiempo_fondo2 = 0.0
    if filename_fondo1 != "" or filename_fondo2 != "":
        tiempo = 180.0  # p.ej. 3 minutos
    if filename_fondo1 != "":
        tiempo_fondo1 = 600.0
    if filename_fondo2 != "":
        tiempo_fondo2 = 600.0

    # Comprobar que nchan es divisible por ngroup
    if nchan % ngroup != 0:
        print("El número de canales no es divisible por la agrupación de canales. STOP.")
        return

    # ---------------------------------------------------------------------
    # Lectura de archivos
    # ---------------------------------------------------------------------
    try:
        f_main = open(filename, "r")
    except IOError:
        print(f"ERROR: No se puede abrir el archivo principal {filename}.")
        return

    if filename_fondo1:
        try:
            f1 = open(filename_fondo1, "r")
        except IOError:
            print(f"ERROR: No se puede abrir el archivo de fondo 1: {filename_fondo1}.")
            f1 = None
    else:
        f1 = None

    if filename_fondo2:
        try:
            f2 = open(filename_fondo2, "r")
        except IOError:
            print(f"ERROR: No se puede abrir el archivo de fondo 2: {filename_fondo2}.")
            f2 = None
    else:
        f2 = None

    # ---------------------------------------------------------------------
    # Construir 'histograma' en un array de tamaño n_bins = nchan//ngroup
    # ---------------------------------------------------------------------
    n_bins = nchan // ngroup
    hist_data = np.zeros(n_bins, dtype=float)

    # Para leer de forma simultánea el fondo, necesitamos iterar línea a línea
    row = 0
    # Generadores (iteradores) para cada fondo (para leer línea a línea en paralelo)
    fondo1_iter = iter(f1) if f1 else None
    fondo2_iter = iter(f2) if f2 else None

    # Lectura línea a línea del archivo principal
    for line in f_main:
        # Tomar la línea correspondiente de cada fondo (o cadena vacía si no hay más)
        line1 = next(fondo1_iter, "") if fondo1_iter else ""
        line2 = next(fondo2_iter, "") if fondo2_iter else ""

        if row >= nlines:
            # Procesar datos del archivo principal
            cols = line.strip().split()
            if len(cols) < 2:
                # Línea no válida
                row += 1
                continue

            x = float(cols[0])
            y = float(cols[1])

            # Inicializamos en 0 las cuentas de fondo
            y1 = 0.0
            y2 = 0.0

            # Fondo 1
            if line1.strip():
                cols1 = line1.strip().split()
                if len(cols1) >= 2:
                    # x1 = float(cols1[0])  # A veces se usa, si se quisiese comprobar
                    y1 = float(cols1[1])

            # Fondo 2
            if line2.strip():
                cols2 = line2.strip().split()
                if len(cols2) >= 2:
                    # x2 = float(cols2[0])
                    y2 = float(cols2[1])

            # Restar fondo si procede
            if f1 and f2:
                factor = tiempo / (tiempo_fondo1 + tiempo_fondo2)
                y -= factor * (y1 + y2)
            elif f1:
                factor = tiempo / tiempo_fondo1
                y -= factor * y1
            elif f2:
                factor = tiempo / tiempo_fondo2
                y -= factor * y2

            # Agregar al histograma (agrupación de canales)
            bin_index = int(x // ngroup)
            if 0 <= bin_index < n_bins:
                hist_data[bin_index] += y

        row += 1

    # Cerrar ficheros
    f_main.close()
    if f1: f1.close()
    if f2: f2.close()

    # ---------------------------------------------------------------------
    # Graficar el espectro como histograma (estilo step)
    # ---------------------------------------------------------------------
    # Se definen los bordes de los bins
    bin_edges = np.arange(n_bins + 1) * ngroup

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Espectro global")
    ax.set_xlabel("Canal")
    ax.set_ylabel("# cuentas")

    # Dibujar histograma con estilo "step"
    # Se añade el último valor de hist_data para que la longitud de y coincida con bin_edges
    ax.step(bin_edges, np.append(hist_data, hist_data[-1]), where='post', color='black')
    # Rellenar el área bajo el histograma
    #ax.fill_between(bin_edges, np.append(hist_data, hist_data[-1]), step='post', alpha=0.4, color='lightblue')

    # Ajustar límites de ejes
    ax.set_xlim(0, nchan)

    # Guardar la figura en PDF y PNG
    plt.savefig("espectro.pdf", bbox_inches='tight')
    plt.savefig("espectro.png", bbox_inches='tight')

    # Mostrar en pantalla
    plt.show()

if __name__ == "__main__":
    espectro()
