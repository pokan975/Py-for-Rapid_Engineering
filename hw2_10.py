#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate and plot the period-amplitude relation of an anharmonic oscillator
# start amplitude: 0m~2m, with zero initial velocity
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad

mass = 1  # mass of the object
N = 20    # Gaussian quadrature with degree = 20

# define integration part, potential is power 6 of the amplitude
def integral(x, a):
    return np.sqrt(1 / (a**6 - x**6))

T = []                                # list of period
amplitude = np.arange(0, 2, 0.1)  # list of initial amplitude as input

# for each initial amplitude, integrate the period (using Gaussian quadrature) 
for aa in np.nditer(amplitude):
    val, err = fixed_quad(integral, 0, aa, args = (aa,), n = N)
    T.append(val)

T = np.asanyarray(T)

# sqrt(8*mass) is the constant part of every T period
T_part1 = np.sqrt(8 * mass)
T = T_part1*T

# show result
plt.plot(amplitude, T)
plt.xlabel("amplitude (m)")
plt.ylabel("period T (sec)")
plt.title('Period of Oscillator for Given Amplitude')
plt.grid()

