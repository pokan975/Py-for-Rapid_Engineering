#!/usr/bin/env python

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


Is = 1e-9    # source current (Amp)
n = 1.5      # constant
R = 1e4      # resistor (Ohm)
T = 350.     # temperature (K)
q = 1.6e-19  # Coulomb constant (Q)
boltz = 1.38e-23  # Boltzmann constant

volt_s = np.linspace(0.1, 2.5, 250)

############################################################
# Problem 1                                                #
############################################################
# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def current_diode(volt, Vs):
    diode = Is * (np.exp((volt * q) / (n * boltz * T)) - 1.)
    return ((volt - Vs) / R) + diode


# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
# define range of source voltage
current1 = []

for v in np.nditer(volt_s):
    v_d = optimize.fsolve(current_diode, 1., (v,))[0]
    i_d = Is * (np.exp((v_d * q) / (n * boltz * T)) - 1.)
    current1.append(i_d)

print("Voltage      I (amp)")
for i in range(len(current1)):
    print("{0:7.4f}   {1:7.4e}".format(volt_s[i], current1[i]))


plt.plot(volt_s, np.log(current1))
plt.xlabel("Source Voltage (V)", fontsize = 16)
plt.ylabel("$\log$($I_{diode}$) (Amp)", fontsize = 16)
plt.title("Curve of source voltage vs. $I_{diode}$", fontsize = 16)
plt.grid()
plt.show()


############################################################
# Problem 2(a)                                             #
############################################################
# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def residual_diode(res_vd, res_A, res_phi, res_n, res_T, res_R, res_vs):
    Vt = (res_n * boltz * res_T) / q
    Is = res_A * res_T * res_T * np.exp(-res_phi * q / (boltz * res_T))
    i_diode = Is * (np.exp(res_vd / Vt) - 1.)
    return ((res_vd - res_vs) / res_R) + i_diode


# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def DiodeI(Vs, x):
    current2 = []
    # initial guesses for A, phi, n, T and R values
    vd = 1.    # voltage across the diode (Volt)
    DiodeI_A = x[0]
    DiodeI_phi = x[1]
    DiodeI_n = x[2]
    DiodeI_T = x[3]
    DiodeI_R = x[4]

    for v in np.nditer(Vs):
        DiodeI_vd = optimize.fsolve(residual_diode, vd, (DiodeI_A, DiodeI_phi, DiodeI_n, DiodeI_T, DiodeI_R, v))[0]
        DiodeI_i = Is * (np.exp((DiodeI_vd * q) / (DiodeI_n * boltz * DiodeI_T)) - 1.)
        current2.append(DiodeI_i)
        
    return current2


# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
A = 1e-8   # cross-sectional area of the diode
phi = 0.8  # constant Phi
n = 1.5    # constant n
T = 375    # temperature (K)
R = 1e4    # resistor (Ohm)

X = np.zeros([5], dtype = float)
X[0] = A
X[1] = phi
X[2] = n
X[3] = T
X[4] = R

plt.plot(volt_s, np.log(DiodeI(volt_s, X)))
plt.xlabel("Source Voltage (V)", fontsize = 16)
plt.ylabel("$\log$($I_{diode}$) (Amp)", fontsize = 16)
plt.title("Curve of source voltage vs. $I_{diode}$", fontsize = 16)
plt.grid()
plt.show()
