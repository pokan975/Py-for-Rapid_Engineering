#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate Stefan-Boltzmann constant
# Return error msg if user's input is invalid.
# =============================================================================

# import required lib & functions
import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxw

# define constants
mass = 1
N = 20

def integral(x, a): 
    return ((a**6 - x**6)**0.5)**-1

T = []
amplitude = np.arange(0.01, 2, 0.01)

T_part1 = (8 * mass)**0.5

for a in np.nditer(amplitude):
    x, w = gaussxw(N)
    xp = (0.5 * (a - 0) * x) + (0.5 * (a + 0))
    wp = 0.5 * (a - 0) * w
    s = 0.0
    for k in range(N):
        s += wp[k] * integral(xp[k], a)
    
    T.append(s)

T = np.asanyarray(T)
np.multiply(T_part1, T)

plt.plot(amplitude, T)