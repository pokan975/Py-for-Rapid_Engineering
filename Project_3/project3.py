#!/usr/bin/env python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

q = 1.6e-19       # Coulomb constant (Q)
boltz = 1.38e-23  # Boltzmann constant


############################################################
# Problem 1                                                #
############################################################
# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def current_diode(volt, Vs):
    diode = Is_prob1 * (np.exp((volt * q) / (n_prob1 * boltz * T_prob1)) - 1.)
    return ((volt - Vs) / R_prob1) + diode


# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
Is_prob1 = 1e-9    # source current (Amp)
n_prob1 = 1.5      # ideality
R_prob1 = 1e4      # resistor (Ohm)
T_prob1 = 350.     # temperature (K)
v_src_prob1 = np.linspace(0.1, 2.5, 250)  # range of source voltage
current1 = []

for v in np.nditer(v_src_prob1):
    v_d = optimize.fsolve(current_diode, 1., (v,))[0]
    i_d = Is_prob1 * (np.exp((v_d * q) / (n_prob1 * boltz * T_prob1)) - 1.)
    current1.append(i_d)

print("Voltage      I (amp)")
for i in range(len(current1)):
    print("{0:7.4f}   {1:7.4e}".format(v_src_prob1[i], current1[i]))


plt.plot(v_src_prob1, np.log10(current1))
plt.xlabel("Source Voltage (V)", fontsize = 16)
plt.ylabel("$\log$($I_{diode}$) (Amp)", fontsize = 16)
plt.title("Curve of source voltage vs. $I_{diode}$", fontsize = 16)
plt.grid()
plt.show()


############################################################
# Problem 2                                                #
############################################################
# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def calc_i_diode(vd, n, T, is_val):
    v = np.asarray(vd)
    vt = (n * boltz * T) / q
    i_diode = is_val * (np.exp(v / vt) - 1.)
    return i_diode


# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def solve_v_diode(vd, vs, R, n, T, is_val):
    vt = (n * boltz * T) / q
    diode = is_val * (np.exp(vd / vt) - 1.)
    return ((vd - vs) / R) + diode


def solve_i_diode(A, phi, R, n, T, v_src):
    vd_est = np.zeros_like(v_src)
    i_diode = np.zeros_like(v_src)
    v_guess = Vd_InitGuess
    
    is_val = A * T * T * np.exp(-phi * q / ( boltz * T ) )
    
    for index in range(len(v_src)):
        v_guess = optimize.fsolve(solve_v_diode, v_guess,
                                (v_src[index], R, n, T, is_val),
                                xtol = 1e-12)[0]
        
        vd_est[index] = v_guess
    
    # compute the diode current
    i_diode = calc_i_diode(vd_est, n, T, is_val)
    return i_diode


# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def optimize_R(R_guess, phi_guess, n_guess, A, T, v_src, i_meas):
    
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T, v_src)
    return (i_diode - i_meas) / (i_diode + i_meas + 1e-15)


# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def optimize_phi(phi_guess, R_guess, n_guess, A, T, v_src, i_meas):
    
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T_prob2, v_src)
    return (i_diode - i_meas) / (i_diode + i_meas + 1e-15)

# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================    
def optimize_n(n_guess, R_guess, phi_guess, A, T, v_src, i_meas):
    
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T_prob2, v_src)
    return (i_diode - i_meas) / (i_diode + i_meas + 1e-15)


# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
max_tol = 1e-3  # maximum allowed tolerance in least square iteration
max_iter = 100  # maximum allowed iteration times
T_prob2 = 375   # temperature (K)
A_prob2 = 1e-8  # sectional-area

## initial guesses for unknown parameters
Vd_InitGuess = 1.
phi_opt = 0.8
R_opt = 1e4
n_opt = 1.5


# arrays to store datasets from file
v_src_prob2 = []  # source voltage
i_meas = []       # measured diode current

# read datasets into array from file
filename = "DiodeIV_nonoise.txt"
fh = open(filename, "r")
lines = fh.readlines()

for line in lines:
    line = line.strip()  # remove space at the start/end of each line
    if line:
        parameter = line.split(" ")         # split datasets in each line
        v_src_prob2.append(float(parameter[0]))   # 1st data for source voltage
        i_meas.append(float(parameter[1])) # 2nd data for diode current

v_src_prob2 = np.asarray(v_src_prob2)
i_meas = np.asarray(i_meas)


iteration = 0

current2 = solve_i_diode(A_prob2, phi_opt, R_opt, n_opt, T_prob2, v_src_prob2)
err = abs(np.mean((current2 - i_meas) / (current2 + i_meas + 1e-15)))

while (err > max_tol and iteration < max_iter):
    iteration += 1
    
    R_opt = optimize.leastsq(optimize_R, R_opt, 
                             args = (phi_opt, n_opt, A_prob2, T_prob2, v_src_prob2, i_meas))[0]
    phi_opt = optimize.leastsq(optimize_phi, phi_opt, 
                               args = (R_opt, n_opt, A_prob2, T_prob2, v_src_prob2, i_meas))[0]
    n_opt = optimize.leastsq(optimize_n, n_opt, 
                             args = (R_opt, phi_opt, A_prob2, T_prob2, v_src_prob2, i_meas))[0]
    

    # compute the diode current
    current2 = solve_i_diode(A_prob2, phi_opt, R_opt, n_opt, T_prob2, v_src_prob2)
    err = abs(np.mean((current2 - i_meas) / (current2 + i_meas + 1e-15)))
    

plt.plot(v_src_prob2, np.log10(i_meas), "b")
plt.plot(v_src_prob2, np.log10(current2), "r")
plt.xlabel("Source Voltage (V)", fontsize = 16)
plt.ylabel("$\log$($I_{diode}$) (Amp)", fontsize = 16)
plt.title("Curve of source voltage vs. $I_{diode}$", fontsize = 16)
plt.grid()
plt.show()

plt.plot(v_src_prob2, current2 - i_meas, "b")
plt.xlabel("Source Voltage (V)", fontsize = 16)
plt.ylabel("Current diff. (Amp)", fontsize = 16)
plt.title("Diff. between measured and estimated $I_{diode}$", fontsize = 16)
plt.grid()
plt.show()
    