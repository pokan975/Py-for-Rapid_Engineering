#!/usr/bin/env python

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


Is = 1e-9    # source current (Amp)
n = 1.5      # 
R = 1e4      # resistor (Ohm)
T = 350.     # temperature (K)
q = 1.6e-19  # Coulomb constant (Q)
boltz = 1.381e-23  # Boltzmann constant


############################################################
# Problem 1                                                #
############################################################

# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def current_diode(volt, Vs):
    diode = Is * (np.exp((volt * q) / (n * boltz * T)) - 1.)
    node = ((volt - Vs) / R) + diode
    return node


# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
# define range of source voltage
volt_s = np.linspace(0.1, 2.5, 250)
current = []

for v in np.nditer(volt_s):
    v_d = optimize.fsolve(current_diode, 1., (v,))
    i_d = Is * (np.exp((v_d[0] * q) / (n * boltz * T)) - 1.)
    current.append(i_d)

print("Voltage      I (amp)")
for i in range(len(current)):
    print("{0:7.4f}   {1:7.4e}".format(volt_s[i], current[i]))


plt.plot(volt_s, np.log10(current))
plt.xlabel("Source Voltage (V)", fontsize = 16)
plt.ylabel("$\log$($I_{diode}$) (Amp)", fontsize = 16)
plt.title("Curve of source voltage vs. $I_{diode}$", fontsize = 16)
plt.grid()
plt.show()

############################################################
# Problem 2                                                #
############################################################


