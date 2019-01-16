#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate Stefan-Boltzmann constant
# Return error msg if user's input is invalid.
# =============================================================================

# import required lib & functions
import numpy as np
import scipy.integrate as integrate

# define constants
mass = 1
amp_lowbound = 0.0
amp_uppbound = 2.0
sample_pts = 20

amp_array = np.arange(amp_lowbound, amp_uppbound + 0.02, 0.02)

integral = lambda x, a: np.reciprocal(np.sqrt(a**6 - x**6))

T_part1 = np.sqrt(8 * mass)

integ_part = integrate.quadrature(integral, amp_lowbound, amp_array)