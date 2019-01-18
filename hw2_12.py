#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate Stefan-Boltzmann constant
# Return error msg if user's input is invalid.
# =============================================================================

# import required lib & functions
import numpy as np
from math import expm1
import scipy.integrate as integrate

# define constants
hbar = 1.055e-34
boltz = 1.381e-23
c = 2.998e8
pi = np.pi

integral = lambda x: (x**3) / expm1(x)

result, err = integrate.quad(integral, 0, np.inf)

boltz_const_up = boltz**4
boltz_const_down = 4 * (pi**2) * (c**2) * (hbar**3)

boltz_const = (boltz_const_up * result) / boltz_const_down