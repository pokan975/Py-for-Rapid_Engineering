#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =============================================================================
# Function:
# Calculate the factorial of input n:
# n! = 1,           if n = 1
# n! = n * (n-1)!,  if n > 1
# =============================================================================
def calc_derivative(xyz, time, sigma, r, b):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = (r * x) - y - (x * z)
    dzdt = (x * y) - (b * z)
    return [dxdt, dydt, dzdt]

# =============================================================================
# Main Code:
# Given a netlist of a circuit description, calculate the voltage/current on each node of the circuit
# solve the problem in the form of linear equation Ax = b
# where resistors are stamped on matrix A, current on vector b, then x is voltage (answer)
# =============================================================================
# define constants
Sigma = 10
R = 28
B = 8/3

xyz_init = (0., 1., 0.)
time_vec = np.linspace(0, 50, 501)

xyz_vec = odeint(calc_derivative, xyz_init, time_vec, args = (Sigma, R, B))

plt.plot(time_vec, xyz_vec[:, 0])
plt.xlabel('time')
plt.ylabel('x')
plt.show()

plt.plot(time_vec, xyz_vec[:, 1])
plt.xlabel('time')
plt.ylabel('y')
plt.show()

plt.plot(time_vec, xyz_vec[:, 2])
plt.xlabel('time')
plt.ylabel('z')
plt.show()

plt.plot(xyz_vec[:, 0], xyz_vec[:, 2])
plt.xlabel('x')
plt.ylabel('z')
plt.show()