import numpy as np
from scipy.integrate import quad, fixed_quad
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#############################
# 1. Constantes y funciones auxiliares
#############################

me = 0.511       # Masa del e- (MeV)
alpha = 1/137.036

def computeTe(pe, me):
    """
    Convierte momento p_e [MeV/c] a energía cinética T_e [MeV].
    Acepta escalar o array.
    """
    return np.sqrt(pe*pe + me*me) - me

def computePe(Te, me):
    """
    Convierte energía cinética T_e [MeV] a momento p_e [MeV/c].
    Acepta escalar o array.
    """
    return np.sqrt(Te*(Te + 2.0*me))

def FermiFactor(Te, pe, Z, alpha):
    """
    Factor de Fermi aproximado (no relativista) + corrección relativista mínima.
    """
    a = Z * alpha * (Te + me) / pe
    # Factor no relativista
    F_NR = 2.0 * np.pi * a / (1.0 - np.exp(-2.0 * np.pi * a))
    # Corrección relativista simplificada
    S = np.sqrt(1.0 - alpha*alpha*Z*Z) - 1.0
    w = (Te / me) + 1.0
    F_Rcorr = (alpha*alpha*Z*Z*w*w + 0.25*(w*w - 1.0))**(S)
    return F_NR * F_Rcorr

def shape_factor(Te_or_pe, Q, Z, L, opt):
    """
    Calcula el factor de forma (incluye factor de densidad de estados)
    con o sin conversión T_e <-> p_e según opt.
     - opt > 0 => Te_or_pe = T_e
     - opt < 0 => Te_or_pe = p_e
    """
    if opt < 0:
        pe = Te_or_pe
        Te = computeTe(pe, me)
    else:
        Te = Te_or_pe
        pe = computePe(Te, me)

    # Factor de fase de espacio
    if opt < 0:
        ps_factor = pe*pe
    else:
        ps_factor = (Te + me) * np.sqrt(Te*(Te + 2.0*me))

    # Factor de Fermi
    fermi = FermiFactor(Te, pe, Z, alpha)

    # Spq factor: si L=0 => 1, si L!=0 => (p_e^2 + (Q - T_e)^2)
    q = (Q - Te) if (Q - Te) > 0 else 0.0
    Spq = 1.0 if L == 0 else (pe*pe + q*q)

    return ps_factor * fermi * Spq

def spectrum(Te_or_pe, Q, Z, L, opt):
    """
    Espectro ~ shape_factor * (Q - T_e)^2  (si Q> T_e).
    """
    if opt < 0:
        Te = computeTe(Te_or_pe, me)
    else:
        Te = Te_or_pe

    arg = (Q - Te)
    if arg > 0.0:
        return shape_factor(Te_or_pe, Q, Z, L, opt) * arg*arg
    else:
        return 0.0

#############################
# 2. Versión vectorizada de la función kurie
#############################
def kurie(Te_or_pe, Q, opt):
    """
    Maneja arrays o escalares:
      - Si opt < 0 => Te_or_pe = p_e, calculamos T_e
      - Si opt > 0 => Te_or_pe = T_e directamente

    Devuelve (Q - T_e) si (Q - T_e) > 0, o 0.0 en caso contrario.
    """
    Te_arr = np.atleast_1d(Te_or_pe)
    out = []
    for val in Te_arr:
        if opt < 0:
            Te_val = computeTe(val, me)
        else:
            Te_val = val

        diff = Q - Te_val
        out.append(diff if diff > 0.0 else 0.0)

    out = np.array(out)
    return out[0] if out.size == 1 else out

#############################
# 3. Funciones de convolución con la ResFcn (resolución angular)
#############################
def gaussian_pdf(x, mean, sigma):
    x_arr = np.asarray(x, dtype=float)     # por si x ya es un array
    norm = 1.0/(sigma*np.sqrt(2.0*np.pi))
    return norm * np.exp(-0.5*((x_arr - mean)/sigma)**2)

def spectrum_times_resFcn_integrand(thetai, Te_or_pe, Q, Z, L,
                                    B, R,
                                    mean_res_deg, sigma_res_deg,
                                    opt):
    """
    Integrando para la convolución: Gaus(theta_obs - thetai)*spectrum(...) dtheta
    """
    mean_res = mean_res_deg * np.pi/180.0
    sigma_res = sigma_res_deg * np.pi/180.0

    # Obtenemos pe y Te según opt
    if opt < 0:
        pe_obs = Te_or_pe
        Te_obs = computeTe(pe_obs, me)
    else:
        Te_obs = Te_or_pe
        pe_obs = computePe(Te_obs, me)

    factor = 3.0 * B * R
    # Ángulo observado en función de p_e obs
    theta_obs = 2.0 * np.arctan(np.abs(-factor)/pe_obs)
    thetai_delta = theta_obs - thetai
    # PDF gaussiana
    res_val = gaussian_pdf(thetai_delta, mean_res, sigma_res)

    # Momento real p_i
    p_i = np.abs(-factor)/np.tan(thetai/2.0)
    if opt < 0:
        return res_val * spectrum(p_i, Q, Z, L, opt)
    else:
        t_i = computeTe(p_i, me)
        return res_val * spectrum(t_i, Q, Z, L, +1)

def convolve_spectrum_with_resFcn(Te_or_pe, cte, Q, Z, L, B, R,
                                  mean_res_deg, sigma_res_deg, opt,
                                  integration_step_deg=1.0):
    """
    Integra entre 0 y pi la convolución del espectro con la gaussiana en theta.
    """
    def integrand_theta(thetai):
        return spectrum_times_resFcn_integrand(thetai, Te_or_pe, Q, Z, L,
                                               B, R,
                                               mean_res_deg, sigma_res_deg,
                                               opt)
    val, err = quad(integrand_theta, 0.0, np.pi, epsrel=1e-5, epsabs=1e-9, limit=50)
    return cte*val

def kurie_times_resFcn_integrand(thetai, Te_or_pe, Q, B, R,
                                 mean_res_deg, sigma_res_deg, opt):
    """
    Integrando para la convolución: Gaus(...) * kurie(...)
    """
    mean_res = mean_res_deg * np.pi/180.0
    sigma_res = sigma_res_deg * np.pi/180.0

    if opt < 0:
        pe_obs = Te_or_pe
        Te_obs = computeTe(pe_obs, me)
    else:
        Te_obs = Te_or_pe
        pe_obs = computePe(Te_obs, me)

    factor = 3.0*B*R
    theta_obs = 2.0*np.arctan(np.abs(-factor)/pe_obs)
    thetai_delta = theta_obs - thetai
    res_val = gaussian_pdf(thetai_delta, mean_res, sigma_res)

    p_i = np.abs(-factor)/np.tan(thetai/2.0)
    if opt < 0:
        return res_val * kurie(p_i, Q, opt)
    else:
        t_i = computeTe(p_i, me)
        return res_val * kurie(t_i, Q, +1)

def convolve_kurie_with_resFcn(Te_or_pe, cte, Q, B, R,
                               mean_res_deg, sigma_res_deg, opt):
    """
    Integración numérica de kurie * resFcn.
    """
    def integrand_theta(thetai):
        return kurie_times_resFcn_integrand(thetai, Te_or_pe, Q, B, R,
                                            mean_res_deg, sigma_res_deg, opt)
    val, err = quad(integrand_theta, 0.0, np.pi, epsrel=1e-5, epsabs=1e-9, limit=50)
    return cte*val

#############################
# 4. Modelos para el ajuste con curve_fit
#############################

def model_spectrum_noResFcn(Te_array, cte, Q, Z, L, opt):
    """
    Devuelve array de valores: cte * spectrum(Te[i], Q, Z, L, opt).
    """
    yvals = []
    for Te in Te_array:
        yvals.append(cte * spectrum(Te, Q, Z, L, opt))
    return np.array(yvals)

def model_spectrum_withResFcn(Te_array, cte, Q, Z, L, B, R,
                              mean_res_deg, sigma_res_deg, opt):
    """
    Evalúa para cada Te el espectro convolucionado con la función de resolución angular.
    """
    yvals = []
    for Te in Te_array:
        val = convolve_spectrum_with_resFcn(Te, cte, Q, Z, L, B, R,
                                            mean_res_deg, sigma_res_deg, opt)
        yvals.append(val)
    return np.array(yvals)

#############################
# 4b. Modelos para el Kurie con o sin resFcn
#############################
def kurie_model_noResFcn(Te_array, cte, Q, opt):
    """
    Evalúa cte * kurie(Te, Q, opt) para un array Te_array.
    """
    return cte * kurie(Te_array, Q, opt)

def kurie_model_withResFcn(Te_array, cte, Q, B, R,
                           mean_res_deg, sigma_res_deg, opt):
    """
    Evalúa la convolución (kurie * resFcn) para cada Te en Te_array,
    y lo multiplica por cte.
    """
    yvals = []
    for Te in Te_array:
        val = convolve_kurie_with_resFcn(Te, cte, Q, B, R,
                                         mean_res_deg, sigma_res_deg, opt)
        yvals.append(val)
    return np.array(yvals)

#############################
# 5. Funciones de ayuda: chi2, ndf y print
#############################


def compute_chi2_and_ndf(x, y, yerr, popt, model, method="least_squares"):
    """
    Calcula el Chi^2 y los grados de libertad (ndf).
    method puede ser:
      - "least_squares" → (y - model)^2 / σ^2
      - "baker_cousins" → 2*(μ - n + n*ln(n/μ))
    """
    y_model = model(x, *popt)
    ndf = len(x) - len(popt)

    if method == "baker_cousins":
        mask = (y > 0) & (y_model > 0)
        term = y_model[mask] - y[mask] + y[mask] * np.log(y[mask] / y_model[mask])
        chi2 = 2.0 * np.sum(term)
    else:
        residuals = (y - y_model) / yerr
        chi2 = np.sum(residuals**2)

    return chi2, ndf

def print_fit_results_scipy(label, popt, perr, chi2, ndf, fixed_params={}):
    """
    Imprime en pantalla resultados del ajuste, al estilo ROOT.
    - label: texto para indicar si es "Espectro" o "Kurie".
    - popt, perr: parámetros ajustados y errores.
    - chi2, ndf: chi^2 y grados de libertad.
    - fixed_params: diccionario con {nombre_param_fijo: valor}.
    """
    print(f"\n*** Resultados del ajuste {label} ***")
    print(f"Chi2                      = {chi2:.4f}")
    print(f"NDf                       = {ndf}")
    print(f"Constant (cte)           = {popt[0]:10.5f} +/- {perr[0]:10.5f}")
    print(f"Q                         = {popt[1]:10.5f} +/- {perr[1]:10.5f}")

    for k, v in fixed_params.items():
        print(f"{k:25s} = {v}   (fixed)")

#############################
# 6. Lectura de datos y flujo principal
#############################

def plotKuriefit_np(filename_root="Sr-90", save_figures=True):
    """
    Lee un archivo 'Sr-90.fit', extrae datos y
    ajusta usando las funciones definidas arriba.
    Luego dibuja y opcionalmente guarda las figuras (save_figures=True).
    """
    filename_in = filename_root + ".fit"
    try:
        fin = open(filename_in,"r")
    except:
        print("No se pudo abrir el archivo", filename_in)
        return

    B_T = 0.31
    R_cm = 1.6
    with_resFcn = 0
    mean_resFcn = 0.0
    sigma_resFcn = 10.0
    eps_resFcn = -1.0
    Z_ = 0.0
    Te_min = 0.0
    Te_max = 2.0
    L_ = 0

    theta_deg = []
    Te_arr = []
    sigma_Te = []
    nS_Te = []
    sigma_nS_Te = []

    row = 0
    for line in fin:
        line=line.strip()
        parts = line.split()
        if row == 2 and len(parts)>1:
            B_T = float(parts[1])
        elif row == 3 and len(parts)>1:
            R_cm = float(parts[1])
        elif row == 4 and len(parts)>1:
            with_resFcn = float(parts[1])
        elif row == 5 and len(parts)>1:
            mean_resFcn = float(parts[1])
        elif row == 6 and len(parts)>1:
            sigma_resFcn = float(parts[1])
        elif row == 7 and len(parts)>1:
            eps_resFcn = float(parts[1])
        elif row == 9 and len(parts)>1:
            Z_ = float(parts[1])
        elif row == 10 and len(parts)>1:
            Te_min = float(parts[1])
        elif row == 11 and len(parts)>1:
            Te_max = float(parts[1])
        elif row == 12 and len(parts)>1:
            L_ = float(parts[1])
        elif row > 14:
            vals = parts
            if len(vals) >=5:
                x = float(vals[0])  # theta
                y = float(vals[1])  # Te
                z = float(vals[2])  # sigma Te
                u = float(vals[3])  # nS_Te
                v = float(vals[4])  # sigma nS_Te
                if x >=30.0:
                    theta_deg.append(x)
                    Te_arr.append(y)
                    sigma_Te.append(z)
                    nS_Te.append(u)
                    sigma_nS_Te.append(v)
        row+=1
    fin.close()

    print("\n*** Input data ***\n")
    print("B[T] =", B_T)
    print("R[cm] =", R_cm)
    print("with_resFcn =", with_resFcn)
    print("mean_resFcn[deg] =", mean_resFcn)
    print("sigma_resFcn[deg] =", sigma_resFcn)
    print("eps_resFcn =", eps_resFcn)
    print("Z' =", Z_)
    print("Te_min[MeV] =", Te_min)
    print("Te_max[MeV] =", Te_max)
    print("L =", L_)
    print("\ntheta(o)  Te[MeV]  sig(Te)[MeV]  nS_Te[/min]  sig(nS_Te)[/min]")
    for i in range(len(theta_deg)):
        print(f"{theta_deg[i]:.1f}    {Te_arr[i]:.2f}    {sigma_Te[i]:.2f}    {nS_Te[i]:.2f}    {sigma_nS_Te[i]:.2f}")

    # Convertir a numpy arrays
    Te_data = np.array(Te_arr)
    nS_data = np.array(nS_Te)
    errTe_data = np.array(sigma_Te)
    errnS_data = np.array(sigma_nS_Te)

    maskX = (Te_data >= Te_min) & (Te_data <= Te_max) #& (nS_data > 0)
    Te_data_fit = Te_data[maskX]
    nS_data_fit = nS_data[maskX]
    errTe_data_fit = errTe_data[maskX]
    errnS_data_fit = errnS_data[maskX]
    
    #######################
    # 6.1 Ajuste del espectro
    #######################
    if with_resFcn == 0:
        def fitfunc_noRes(x, cte, Q):
            return model_spectrum_noResFcn(x, cte, Q, Z_, L_, +1)
        p0 = [1000.0, 2.3]
        popt, pcov = curve_fit(fitfunc_noRes,
                               Te_data_fit, nS_data_fit,
                               p0=p0,
                               sigma=errnS_data_fit, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        # Chi2 y ndf
        chi2_val, ndf_val = compute_chi2_and_ndf(Te_data_fit, nS_data_fit, errnS_data_fit,
                                                 popt, fitfunc_noRes, method="baker_cousins")

        cte_fit, Q_fit = popt
        # Imprimir resultados
        print_fit_results_scipy("Espectro (sin resFcn)", popt, perr,
                                chi2_val, ndf_val,
                                fixed_params={
                                    "Z":Z_, "L":L_, "B":B_T, "R":R_cm,
                                    "me":me, "alpha":alpha,
                                    "mean_resFcn":mean_resFcn,
                                    "sigma_resFcn":sigma_resFcn,
                                    "eps_resFcn":eps_resFcn,
                                    "opt":1
                                })

        Te_plot = np.linspace(Te_min, Te_max, 200)
        y_fit = fitfunc_noRes(Te_plot, cte_fit, Q_fit)

    else:
        def fitfunc_withRes(x, cte, Q):
            return model_spectrum_withResFcn(x, cte, Q, Z_, L_, B_T, R_cm,
                                             mean_resFcn, sigma_resFcn, +1)
        p0 = [1000.0, 2.3]
        # Evitar errores nulos o muy pequeños
        errnS_data_fit = np.where(errnS_data_fit <= 0, 1e-8, errnS_data_fit)
        popt, pcov = curve_fit(fitfunc_withRes,
                               Te_data_fit, nS_data_fit,
                               p0=p0,
                               sigma=errnS_data_fit, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        chi2_val, ndf_val = compute_chi2_and_ndf(Te_data_fit, nS_data_fit, errnS_data_fit,
                                                 popt, fitfunc_withRes, method="baker_cousins")

        cte_fit, Q_fit = popt
        print_fit_results_scipy("Espectro (con resFcn)", popt, perr,
                                chi2_val, ndf_val,
                                fixed_params={
                                    "Z":Z_, "L":L_, "B":B_T, "R":R_cm,
                                    "me":me, "alpha":alpha,
                                    "mean_resFcn":mean_resFcn,
                                    "sigma_resFcn":sigma_resFcn,
                                    "eps_resFcn":eps_resFcn,
                                    "opt":1
                                })

        Te_plot = np.linspace(Te_min, Te_max, 200)
        y_fit = fitfunc_withRes(Te_plot, cte_fit, Q_fit)

    #######################
    # 6.2 Dibujar el espectro
    #######################
    plt.figure(figsize=(7,5))
    plt.errorbar(Te_data, nS_data, yerr=errnS_data, xerr=errTe_data,
                 fmt='o', label="Datos", color='blue', ecolor='lightblue')
    plt.plot(Te_plot, y_fit, 'r-', label="Ajuste espectro")
    plt.title(f"Espectro beta {filename_root}")
    plt.xlabel("T_e [MeV]")
    plt.ylabel("Cuentas/min")
    plt.grid(True)
    plt.legend()

    if save_figures:
        plt.savefig(f"{filename_root}_espectro_Te_noROOT.pdf")
        plt.savefig(f"{filename_root}_espectro_Te_noROOT.png")
    plt.show()

    #######################
    # 6.3 Plot de Kurie
    #######################
    kurie_vals = []
    kurie_errs = []
    for i, Tei in enumerate(Te_data):
        # shape_factor en el punto i
        sf = shape_factor(Tei, Q_fit, Z_, L_, +1)  # simplificado

        if nS_data[i] <= 0 or sf <= 0:
            kurie_vals.append(0.0)
            kurie_errs.append(0.0)
        else:
            fe = np.sqrt(nS_data[i]/sf)
            sfe = 0.5*fe/nS_data[i]*errnS_data[i]
            kurie_vals.append(fe)
            kurie_errs.append(sfe)

    kurie_vals = np.array(kurie_vals)
    kurie_errs = np.array(kurie_errs)

    # Definimos la función de ajuste para el plot de Kurie
    if with_resFcn == 0:
        def kurie_model_fit(Te, cte, Q):
            return kurie_model_noResFcn(Te, cte, Q, +1)
    else:
        def kurie_model_fit(Te, cte, Q):
            return kurie_model_withResFcn(Te, cte, Q, B_T, R_cm,
                                          mean_resFcn, sigma_resFcn, +1)

    p0_kurie = [10.0, Q_fit]
    mask = (kurie_vals > 0)
    Te_masked = Te_data[mask]
    kurie_masked = kurie_vals[mask]
    err_kurie_masked = kurie_errs[mask]

    maskXX = (Te_masked >= Te_min) & (Te_masked <= Te_max)
    Te_masked_fit = Te_masked[maskXX]
    kurie_masked_fit = kurie_masked[maskXX]
    err_kurie_masked_fit = err_kurie_masked[maskXX]
    popt_kurie, pcov_kurie = curve_fit(kurie_model_fit,
                                       Te_masked_fit,
                                       kurie_masked_fit,
                                       p0=p0_kurie,
                                       sigma=err_kurie_masked_fit,
                                       absolute_sigma=True)
    perr_kurie = np.sqrt(np.diag(pcov_kurie))
    chi2_kurie, ndf_kurie = compute_chi2_and_ndf(Te_masked_fit, kurie_masked_fit,
                                                 err_kurie_masked_fit,
                                                 popt_kurie, kurie_model_fit, method="baker_cousins")

    cte_kurie_fit, Q_kurie_fit = popt_kurie
    # Imprimimos resultados
    print_fit_results_scipy("Kurie", popt_kurie, perr_kurie,
                            chi2_kurie, ndf_kurie,
                            fixed_params={
                                "Z":Z_, "L":L_, "B":B_T, "R":R_cm,
                                "me":me, "alpha":alpha,
                                "mean_resFcn":mean_resFcn,
                                "sigma_resFcn":sigma_resFcn,
                                "eps_resFcn":eps_resFcn,
                                "opt":1
                            })

    Te_plot2 = np.linspace(Te_min, Te_max, 200)
    kurie_plot = kurie_model_fit(Te_plot2, cte_kurie_fit, Q_kurie_fit)

    plt.figure(figsize=(7,5))
    plt.errorbar(Te_data, kurie_vals, xerr=errTe_data, yerr=kurie_errs,
                 fmt='o', color='blue', ecolor='lightblue', label="Kurie data")
    plt.plot(Te_plot2, kurie_plot, 'r-', label="Kurie fit")
    plt.title(f"Plot de Kurie {filename_root}")
    plt.xlabel("T_e [MeV]")
    plt.ylabel("Kurie-like function")
    plt.grid(True)
    plt.legend()

    if save_figures:
        plt.savefig(f"{filename_root}_plotKurie_Te_noROOT.pdf")
        plt.savefig(f"{filename_root}_plotKurie_Te_noROOT.png")
    plt.show()


#############################
# 7. Ejecutar (main)
#############################
if __name__ == "__main__":
    # Puedes elegir si se guardan figuras (True/False)
    plotKuriefit_np("Sr-90", save_figures=True)
