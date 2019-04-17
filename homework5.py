#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =============================================================================
# Function:
# Define the sets of Lorenz differential equation, for solving
# dx/dt = sigma*(y-x); dy/dt = (r*x)-y-(x*z); dz/dt = (x*y)-(b*z)
# x, y, z are dependent, define their differential eq. together
# =============================================================================
def calc_derivative(xyz, time, sigma, r, b):
    # extract initial values of x, y, z
    x, y, z = xyz
    # define differential eq. of x, y, x versus time
    dxdt = sigma * (y - x)
    dydt = (r * x) - y - (x * z)
    dzdt = (x * y) - (b * z)
    
    return [dxdt, dydt, dzdt]

# =============================================================================
# Main Code:
# Use ODE function to solve Lorenz differential equations 
# define ODE equation, time steps, constants, initial values, then calc the 
# differential values of x, y, z for each time index
# =============================================================================
# define constants for differential eq.
Sigma = 10
R = 28
B = 8/3

# give initial condition of x, y, z
xyz_init = (0., 1., 0.)
# define time indices
time_vec = np.linspace(0, 50, 5001)

# solve differential equation using ODE function
xyz_vec = odeint(calc_derivative, xyz_init, time_vec, args = (Sigma, R, B))

# plot x, y, z versus time
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

# plot x versus z
plt.plot(xyz_vec[:, 0], xyz_vec[:, 2])
plt.xlabel('x')
plt.ylabel('z')
plt.show()