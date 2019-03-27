#!/usr/bin/env python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

############################################################
# Problem 1                                                #
############################################################
Is = 1e-9    # source current (Amp)
n = 1.5      # ideality constant
R = 1e4      # resistor (Ohm)
T = 350.     # temperature (K)
q = 1.6e-19  # Coulomb constant (Q)
boltz = 1.38e-23  # Boltzmann constant

volt_s = np.linspace(0.1, 2.5, 250)

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


