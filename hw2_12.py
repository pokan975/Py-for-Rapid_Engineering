#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate Stefan-Boltzmann constant
# Return error msg if user's input is invalid.
# =============================================================================

# import required lib & functions
import numpy as np
import scipy.integrate as integrate
from math import exp

# define constants
hbar = 1.055e-34
boltz = 1.381e-23
c = 2.998e8
pi = np.pi

def integral(z):
    x = z / (1 - z)
    return (x**3) / (exp(x) - 1)

res, err = integrate.quad(integral, 0.01, 1)

boltz_const_up = boltz**4
boltz_const_down = 4 * (pi**2) * (c**2) * (hbar**3)

boltz_const = (boltz_const_up * res) / boltz_const_down