#!/usr/bin/env python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

############################################################
# Problem 2(a)                                             #
############################################################
A = 1e-8   # cross-sectional area of the diode
phi = 0.8  # constant Phi
n = 1.5    # ideality constant
T = 375    # temperature (K)
R = 1e4    # resistor (Ohm)
q = 1.6e-19  # Coulomb constant (Q)
boltz = 1.38e-23  # Boltzmann constant
Vt = (n * boltz * T) / q

volt_s = np.linspace(0.1, 2.5, 250)

# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def residual_diode(vd, A, phi, n, T, R, vs):
    Is = A * T * T * np.exp(-phi * q / (boltz * T))
    i_diode = Is * (np.exp(vd / Vt) - 1.)
    return ((vd - vs) / R) + i_diode


# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def DiodeI(Vs, x):
    current = []
    # initial guesses for A, phi, n, T and R values
#    vd = 1.    # voltage across the diode (Volt)
    A = x[0]
    phi = x[1]
    n = x[2]
    T = x[3]
    R = x[4]

    for v in np.nditer(Vs):
        vd = optimize.fsolve(residual_diode, 1., (A, phi, n, T, R, v))[0]
        Is = A * T * T * np.exp(-phi * q / (boltz * T))
        i_d = Is * (np.exp(vd / Vt) - 1.)
        current.append(i_d)
        
    return current


# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
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
def residualphi(phi_guess, A, n, T, R, v_src):
    Vt = (n * boltz * T) / q

# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================    
def residualn(n_guess, A, phi, T, R, v_src):
    Vt = (n_guess * boltz * T) / q

# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def residualR(R_guess, A, phi, n, T, v_src):
    Vt = (n * boltz * T) / q

# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
max_tol = 1e-3  # maximum allowed tolerance in least square iteration
max_iter = 100  # maximum allowed iteration times

## initial guesses of unknown parameters
phi_init = 0.8
R_init = 1e4
n_init = 1.5

# read datasets into array from file
filename = "DiodeIV.txt"
fh = open(filename, "r")

lines = fh.readlines()

# arrays to store datasets from file
v_src = []    # source voltage
i_diode = []  # diode current

for line in lines:
    line = line.strip()  # remove space at the start/end of each line
    if line:
        parameter = line.split(" ")         # split datasets in each line
        v_src.append(float(parameter[0]))   # 1st data for source voltage
        i_diode.append(float(parameter[1])) # 2nd data for diode current

v_src = np.asarray(v_src)
i_diode = np.asarray(i_diode)
# =============================================================================
# while (err > max_tol and iteration <= max_iter):
#     phi = optimize.leastsq(residualphi, phi, all the other parameters including n and R)
#     n = optimize.leastsq(residualn, n, all the other parameters including n and R)
#     R = optimize.leastsq(residualR, R, all the other parameters including n and R)
# =============================================================================
