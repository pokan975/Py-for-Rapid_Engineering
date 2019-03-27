#!/usr/bin/env python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


Is = 1e-9    # source current (Amp)
n = 1.5      # ideality constant
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
def residual_diode(vd, A, phi, n, T, R, vs):
    Vt = (n * boltz * T) / q
    Is = A * T * T * np.exp(-phi * q / (boltz * T))
    i_diode = Is * (np.exp(vd / Vt) - 1.)
    return ((vd - vs) / R) + i_diode


# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def DiodeI(Vs, x):
    current2 = []
    # initial guesses for A, phi, n, T and R values
    vd = 1.    # voltage across the diode (Volt)
    A = x[0]
    phi = x[1]
    n = x[2]
    T = x[3]
    R = x[4]

    for v in np.nditer(Vs):
        vd = optimize.fsolve(residual_diode, vd, (A, phi, n, T, R, v))[0]
        i_d = Is * (np.exp((vd * q) / (n * boltz * T)) - 1.)
        current2.append(i_d)
        
    return current2


# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
A = 1e-8   # cross-sectional area of the diode
phi = 0.8  # constant Phi
n = 1.5    # ideality constant
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


############################################################
# Problem 2(b)                                             #
############################################################
# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def residualphi(phi, A, n, T, R, Vs):
    

# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================    
def residualn(n, A, phi, T, R, Vs):
    

# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def residualR(R, A, phi, n, T, Vs):
    

# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
max_tol = 1e-3
max_iter = 100

filename = "DiodeIV.txt"
fh = open(filename, "r")

lines = fh.readlines()

A = []
T = []

for line in lines:
    line = line.strip()  # remove space at the start/end of line
    if line:
        parameter = line.split(" ")
        A.append(float(parameter[0]))
        T.append(float(parameter[1]))

# =============================================================================
# while (err > max_tol and iteration <= max_iter):
#     phi = optimize.leastsq(residualphi, phi, all the other parameters including n and R)
#     n = optimize.leastsq(residualn, n, all the other parameters including n and R)
#     R = optimize.leastsq(residualR, R, all the other parameters including n and R)
# =============================================================================
