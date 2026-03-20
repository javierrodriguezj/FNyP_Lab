import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Resultados de cuentas frente ángulos , ángulos primero, cuentas segunda entrada

data=[[40,42], [45, 48],[50,59],[55,99],[60,139],[65,256],[70,352],[75,454],[80,496],[85,550],[90,542],[95,562],[100,613],[105,574],[110,553],[115,563],[120,531],[125,481],[130,482],[135,477]
      ]

length=len(data)

sigma_theta=2.5
fondo=13.9
sigma_fondo=0.7

error_x=np.ones(length)*sigma_theta
error_y=np.ones(length)*sigma_fondo


# 1. Metemos tus datos manuales (ángulos en grados)
theta_deg = np.array(data)[:, 0]  # Extraemos los ángulos de la primera columna
cuentas = np.array(data)[:, 1]-fondo  # Extraemos las cuentas de la segunda columna



# 2. Constantes del montaje (Asegúrate de que B y R son los tuyos)
B = 0.3893  # Teslas
R = 1.493  # cm (16 mm)
me = 0.511 # Masa del electrón en MeV

# 3. Calculamos el Momento p [MeV/c] usando la fórmula del guion:
# p = 300 * B * (R/100) * cot(theta/2)  <-- Si R está en m
# En las unidades del código de kurie:
factor = 3.0 * B * R 
p = factor / np.tan(np.deg2rad(theta_deg) / 2.0)

# 4. Calculamos la Energía Cinética Te [MeV]
Te = np.sqrt(p**2 + me**2) - me

# --- VISUALIZACIÓN ---
plt.figure(figsize=(8, 5))
plt.plot(theta_deg, Te, color='red')
plt.plot(theta_deg, Te, 'o', color='blue')
plt.title('Calibración del Espectrómetro: Energía vs Ángulo')
plt.xlabel('Ángulo $\\theta$ [grados]')
plt.ylabel('Energía Cinética $T_e$ [MeV]')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.errorbar(theta_deg, cuentas, yerr=error_y,xerr=error_x, fmt='o', ecolor='grey',mfc='royalblue', label='Datos con error')
plt.title('Espectro neto de cuentas vs Ángulo')
plt.xlabel('Ángulo $\\theta$ [grados]')
plt.ylabel('Cuentas $g$ [cuentas/min]')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
