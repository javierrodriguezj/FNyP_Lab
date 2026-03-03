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

    # --------------------------
    # LECTURA DE DATOS
    # --------------------------
    data = np.loadtxt(filename, skiprows=nlines, usecols=2)
    voltage = np.loadtxt(filename, skiprows=nlines, usecols=1)
    nmeasu = len(data)

    # --------------------------
    # GRÁFICOS
    # --------------------------
    plt.figure(figsize=(8,6))
    plt.grid()
    plt.plot(voltage,data,'lightskyblue')
    plt.plot(voltage, data,'o')

  # 1. Flecha y texto para Vs (El punto de inicio)
    plt.annotate('$V_s$', xy=(voltage[4], data[4]), xytext=(450, 1000),
             arrowprops=dict(facecolor='k', edgecolor='k', arrowstyle='->', lw=2))

# 2. Flecha de doble sentido para la "Curva Plateau"
    plt.annotate('', xy=(voltage[5], data[-6]+500), xytext=(voltage[-6], data[-6]+500), 
             arrowprops=dict(arrowstyle='<->', color='limegreen', lw=2))
    plt.text(650, 8500, 'Geiger plateau', color='limegreen', ha='center', fontweight='bold')

# 3. Flecha para la región de descarga continua
    plt.annotate('', xy=(1050, 9000), xytext=(1000, 10000),  
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.text(1000, 10000, 'Región de descarga\ncontinua', color='k', ha='center', fontweight='bold')

    plt.annotate('', xy=(400, 9000), xytext=(400, 6000),  
            arrowprops=dict(linestyle='--', color='limegreen', lw=2,arrowstyle='-'))
    plt.text(410, 5600, '$V_1$', color='limegreen', ha='center', fontweight='bold')

    plt.annotate('', xy=(1000, 9000), xytext=(1000, 6000),  
            arrowprops=dict(linestyle='--', color='limegreen', lw=2,arrowstyle='-'))
    plt.text(1000, 5600, '$V_2$', color='limegreen', ha='center', fontweight='bold')
    
    plt.xlabel(f"Voltaje ($V$)")
    plt.ylabel(f"g ($cuentas/30 s$)")
    plt.title("Curva Plateau del GM")
    plt.savefig(ruta_resultados / "plateau.png", dpi=150)


    ###########
    # pendiente
    ###########
    g2=data[-6]
    g1=data[5]
    V2=voltage[-6]
    V1=voltage[5]

    m=(g2-g1)/(V2-V1)*100/g1
    print(f"Pendiente del plateau: {m:.8f} %/100V")


if __name__ == "__main__":
    main()
