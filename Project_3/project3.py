#!/usr/bin/env python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# global constants for all problems
q = 1.6e-19       # Coulomb constant (Q)
boltz = 1.38e-23  # Boltzmann constant


############################################################
# Problem 1                                                #
############################################################
# =============================================================================
# Function:
# nodal analysis function for solving diode voltage
# =============================================================================
def current_diode(volt, Vs):
# type volt: array[float]
# type Vs: array[float]
# rtype: function

    # diode current equation
    diode = Is_prob1 * (np.exp((volt * q) / (n_prob1 * boltz * T_prob1)) - 1.)
    # return nodal function = 0
    return ((volt - Vs) / R_prob1) + diode


# =============================================================================
# Main Code:
# Determine the current I thorough the circuit by doing a nodal analysis
# for the circuit and using the current through the diode defined by
# the diode current equation.
# =============================================================================
Is_prob1 = 1e-9  # source current (Amp)
n_prob1 = 1.5    # ideality
R_prob1 = 1e4    # resistor (Ohm)
T_prob1 = 350.   # temperature (K)
v_src_prob1 = np.linspace(0.1, 2.5, 250)  # range of source voltage
current1 = []    # store diode current

for v in np.nditer(v_src_prob1):
    # get diode voltage by solving f_{node current}(Vd) = 0 
    v_d = optimize.fsolve(current_diode, 1., (v,))[0]
    # calc diode current using diode voltaage and diode current equation
    i_d = Is_prob1 * (np.exp((v_d * q) / (n_prob1 * boltz * T_prob1)) - 1.)
    
    current1.append(i_d)

# print source voltage and corresponding diode current
print("Problem 1:\n")
print("Voltage      I (amp)")
for i in range(len(current1)):
    print("{0:7.4f}   {1:7.4e}".format(v_src_prob1[i], current1[i]))

# plot the relationship of source voltage and log10(diode current) 
plt.plot(v_src_prob1, np.log10(current1))
plt.xlabel("Source Voltage (V)", fontsize = 16)
plt.ylabel("$\log_{10}$($I_{diode}$) (Amp)", fontsize = 16)
plt.title("Curve of source voltage vs. $I_{diode}$", fontsize = 16)
plt.grid()
plt.show()


############################################################
# Problem 2                                                #
############################################################
# =============================================================================
# Function:
# nodal analysis function for solving diode voltage
# =============================================================================
def solve_v_diode(vd, vs, R, n, T, is_val):
# type vd: float
# type vs: float
# type R: float
# type n: float
# type T: float
# type is_val: float
# rtype: function
    
    # calc constant for diode current equation
    vt = (n * boltz * T) / q
    # diode current equation
    diode = is_val * (np.exp(vd / vt) - 1.)
    # return nodal function = 0
    return ((vd - vs) / R) + diode


# =============================================================================
# Function:
# Calc current through diode using nodal analysis equation
# =============================================================================
def solve_i_diode(A, phi, R, n, T, v_src):
# type A: float
# type phi: float
# type R: float
# type n: float
# type T: float
# type v_src: array[float]
# rtype: array[float]

    # create zero array to store computed diode current/voltage
    vd_est = np.zeros_like(v_src)
    i_diode = np.zeros_like(v_src)
    # specify initial diode voltage for fsolve()
    v_guess = Vd_InitGuess
    
    is_val = A * T * T * np.exp(-phi * q / ( boltz * T ) )
    
    # for every given source voltage, calc diode voltage by solving nodal analysis
    for index in range(len(v_src)):
        v_guess = optimize.fsolve(solve_v_diode, v_guess,
                                (v_src[index], R, n, T, is_val),
                                xtol = 1e-12)[0]
        
        vd_est[index] = v_guess
    
    # compute the diode current
    vt = (n * boltz * T) / q  # calc constant for diode current equation
    i_diode = is_val * (np.exp(vd_est / vt) - 1.) # calc diode current by its definition
    return i_diode


# =============================================================================
# Function:
# Doing the optimization for the resistor
# =============================================================================
def optimize_R(R_guess, phi_guess, n_guess, A, T, v_src, i_meas):
# type R_guess: float
# type phi_guess: float
# type n_guess: float
# type A: float
# type T: float
# type v_src: array[float]
# type i_meas: array[float]
# rtype: array[float]
    
    # get diode current using optimized parameters
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T, v_src)
    # normalized error (add a constant in denominator for avoiding 0/0 case)
    return (i_diode - i_meas) / (i_diode + i_meas + 1e-15)


# =============================================================================
# Function:
# Doing the optimization for the barrier height phi
# =============================================================================
def optimize_phi(phi_guess, R_guess, n_guess, A, T, v_src, i_meas):
# type phi_guess: float
# type R_guess: float
# type n_guess: float
# type A: float
# type T: float
# type v_src: array[float]
# type i_meas: array[float]
# rtype: array[float]
    
    # get diode current using optimized parameters
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T_prob2, v_src)
    # normalized error (add a constant in denominator for avoiding 0/0 case)
    return (i_diode - i_meas) / (i_diode + i_meas + 1e-15)

# =============================================================================
# Function:
# Doing the optimization for the ideality n
# =============================================================================    
def optimize_n(n_guess, R_guess, phi_guess, A, T, v_src, i_meas):
# type n_guess: float
# type R_guess: float
# type phi_guess: float
# type A: float
# type T: float
# type v_src: array[float]
# type i_meas: array[float]
# rtype: array[float]
    
    # get diode current using optimized parameters
    i_diode = solve_i_diode(A, phi_guess, R_guess, n_guess, T_prob2, v_src)
    # normalized error (add a constant in denominator for avoiding 0/0 case)
    return (i_diode - i_meas) / (i_diode + i_meas + 1e-15)


# =============================================================================
# Main Code:
# Repeat the diode problem in Problem 1 for a diode where more parameters need
# to be optimized. Use fsolve() to solve nodal analysis by using optimized parameters
# Use leastsq() to find optimized values of unknown parameters by using error between
# measured and estimated diode currents computed by optimized paramters.
# Iterate the optimization process until get error converges or reach max iter times
# =============================================================================
max_tol = 1e-3  # maximum allowed tolerance in least square iteration
max_iter = 100  # maximum allowed iteration times
T_prob2 = 375   # temperature (K)
A_prob2 = 1e-8  # sectional-area

## initial guesses for unknown parameters
Vd_InitGuess = 1. # initial guess for diode voltage value
phi_opt = 0.8     # initial guess for optimal barrier height value
R_opt = 1e4       # initial guess for optimal resistor value
n_opt = 1.5       # initial guess for optimal ideality value

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
        parameter = line.split(" ")             # split datasets in each line
        v_src_prob2.append(float(parameter[0])) # 1st data for source voltage
        i_meas.append(float(parameter[1]))      # 2nd data for measured diode current

v_src_prob2 = np.asarray(v_src_prob2)
i_meas = np.asarray(i_meas)


# initialize iteration counter
iteration = 0

# before iteration, calc diode current using initial guesses to get initial error values array
current2 = solve_i_diode(A_prob2, phi_opt, R_opt, n_opt, T_prob2, v_src_prob2)
# error = L1 norm of the normalized vector (vector = estimated current - measured current)
err = np.linalg.norm((current2 - i_meas) / (current2 + i_meas + 1e-15), ord = 1)

print("\n\nProblem 2:\n")
print("Iteration    R      Phi      n")

# terate optimization process until error function is satisfied
while (err > max_tol and iteration < max_iter):
    # update iteration counter
    iteration += 1
    
    # optimize resistor values for error values array
    R_opt = optimize.leastsq(optimize_R, R_opt, 
                             args = (phi_opt, n_opt, A_prob2, T_prob2, v_src_prob2, i_meas))[0][0]
    # optimize barrier height values for error values array
    phi_opt = optimize.leastsq(optimize_phi, phi_opt, 
                               args = (R_opt, n_opt, A_prob2, T_prob2, v_src_prob2, i_meas))[0][0]
    # optimize ideality values for error values array
    n_opt = optimize.leastsq(optimize_n, n_opt, 
                             args = (R_opt, phi_opt, A_prob2, T_prob2, v_src_prob2, i_meas))[0][0]
    
    # calc the diode current
    current2 = solve_i_diode(A_prob2, phi_opt, R_opt, n_opt, T_prob2, v_src_prob2)
    # calc error values array for optimizing result check
    err = np.linalg.norm((current2 - i_meas) / (current2 + i_meas + 1e-15), ord = 1)
    
    # print the optimized resistor, phi, and ideality values with iteration counter.
    print("{0:9d} {1:7.2f} {2:7.4f} {3:7.4f}".format(iteration, R_opt, phi_opt, n_opt))
    
# plot the relationship of source voltage and log10(diode current) after optimization
plt.plot(v_src_prob2, np.log10(i_meas), "bs-", label = "measured I")
plt.plot(v_src_prob2, np.log10(current2), "r*-", label = "estimated I")
plt.xlabel("Source Voltage (V)", fontsize = 16)
plt.ylabel("$\log_{10}$($I_{diode}$) (Amp)", fontsize = 16)
plt.title("Curve of source voltage vs. $I_{diode}$", fontsize = 16)
plt.legend(loc = 'center right')
plt.grid()
plt.show()
