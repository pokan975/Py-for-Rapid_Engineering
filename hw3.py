#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate and plot the period-amplitude relation of an anharmonic oscillator
# start amplitude: 0m~2m, with zero initial velocity
# =============================================================================
import numpy as np
from read_netlist import read_netlist

netlist = read_netlist()

N = len(netlist)
A = np.zeros([N, N], float)

