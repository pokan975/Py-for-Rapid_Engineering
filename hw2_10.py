#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate Stefan-Boltzmann constant
# Return error msg if user's input is invalid.
# =============================================================================

# =============================================================================
# 
# import numpy as np
# import matplotlib.pyplot as plt
# from gaussxw import gaussxw
# 
# 
# mass = 1
# N = 20
# 
# def integral(x, a): 
#     return ((a**6 - x**6)**0.5)**-1
# 
# T = []
# amplitude = np.arange(0.01, 2, 0.01)
# 
# T_part1 = (8 * mass)**0.5
# 
# for a in np.nditer(amplitude):
#     x, w = gaussxw(N)
#     xp = (0.5 * (a - 0) * x) + (0.5 * (a + 0))
#     wp = 0.5 * (a - 0) * w
#     s = 0.0
#     for k in range(N):
#         s += wp[k] * integral(xp[k], a)
#     
#     T.append(s)
# 
# T = np.asanyarray(T)
# np.multiply(T_part1, T)
# 
# plt.plot(amplitude, T)
# =============================================================================

# import required lib & functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad

# define constants
mass = 1
N = 20

# define integration part
def integral(x, a):
    return ((a**6 - x**6)**0.5)**-1

T = []
amplitude = np.arange(0.01, 2, 0.01)

for aa in np.nditer(amplitude):
    val, err = fixed_quad(integral, 0, aa, args = (aa,), n = N)
     
    T.append(val)

T = np.asanyarray(T)
T_part1 = (8 * mass)**0.5
np.multiply(T_part1, T)

plt.plot(amplitude, T)
plt.xlabel("amplitude (m)")
plt.ylabel("period T (sec)")
plt.title('Period of Oscillator for Given Amplitude')
plt.grid()



