import numpy as np
from scipy.optimize import curve_fit

p0 = np.array([0.45, 0.455, 0.46, 0.465, 0.47, 0.475, 0.48, 0.485,
               0.49, 0.495, 0.5, 0.505, 0.51, 0.515, 0.52, 0.525,
               0.53, 0.535, 0.54, 0.545, 0.55, 0.555])

R  = np.array([19.92, 16.44, 15.34, 12.88, 10.76, 8.48, 7.02, 5.54,
               3.84,  2.82,  1.58,  0.90,  0.82,  0.38, 0.20, 0.06,
               0.02,  0.00,  0.00,  0.00,  0.00,  0.00])

def sigmoid(p0, A, k, pc):
    return A / (1 + np.exp(k * (p0 - pc)))

popt, pcov = curve_fit(sigmoid, p0, R,
                       p0=[20, 50, 0.5],
                       bounds=([0, 5, 0.45], [50, 500, 0.56]),
                       maxfev=10000)

A_fit, k_fit, pc_fit = popt
perr  = np.sqrt(np.diag(pcov))
width = 2 * np.log(9) / k_fit

print(f"Sigmoid threshold : pc = {pc_fit:.4f} ± {perr[2]:.4f}")
print(f"Sharpness k       : {k_fit:.2f}")
print(f"Transition width  : {width:.4f}")
print(f"Directed theory   : 0.5000")
print(f"Difference        : {abs(pc_fit - 0.5):.4f}")