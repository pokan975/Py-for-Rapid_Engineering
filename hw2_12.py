#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate Stefan-Boltzmann constant:
#   sigma = (K**4)/(4*(pi**2)*(c**2)*(h**3))*(integrate (x**3)/(exp(x)-1) from 0 to inf)
#   sigma: Stefan-Boltzmann constant
#   K: Boltzmann constant
#   c: speed of light
#   h: Plank constant over 2pi
# =============================================================================

import numpy as np
from scipy.integrate import quad

# define constants
hbar = 1.055e-34
boltz = 1.381e-23
c = 2.998e8
pi = np.pi

# define function in integration
def integral(x):
    return (x**3) / (np.exp(x) - 1)

# compute integration part
res, err = quad(integral, 0, np.inf)

# compute non-integration part
boltz_const_up = boltz**4
boltz_const_down = 4 * (pi**2) * (c**2) * (hbar**3)

# combine integration part & non-integration part
stf_boltz_const = boltz_const_up * res / boltz_const_down

#output result
print("")
print("The Stefan-Boltzmann constant is {:.3e}".format(stf_boltz_const))

