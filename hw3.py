#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate and plot the period-amplitude relation of an anharmonic oscillator
# start amplitude: 0m~2m, with zero initial velocity
# =============================================================================
import numpy as np
import comp_constants as COMP
from numpy.linalg import solve
from read_netlist import read_netlist

# read netlist from file
netlist = read_netlist()

# get the total nodes of resistence = N
N = 0
for comp in netlist:
    if (comp[COMP.TYPE] == COMP.R):
        N = max(N, comp[COMP.I], comp[COMP.J])

# initialize admittance matrix, voltage vector, and current vector
A = np.zeros([N, N], float)
current_vector = np.zeros([N, 1], float)

# stamp the resistence, voltage, and current value on the admittance matrix
for comp in netlist:
    if (comp[COMP.TYPE] == COMP.R):
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1
        if (i >= 0):
            A[i, i] += 1.0/comp[COMP.VAL]
        
            if (j >= 0):
                A[j, j] += 1.0/comp[COMP.VAL]
                A[i, j] += -1.0/comp[COMP.VAL]
                A[j, i] += -1.0/comp[COMP.VAL]
        
    elif (comp[COMP.TYPE] == COMP.VS):
        j = comp[COMP.J] - 1
        if (j >= 0):
            volt = np.zeros([1, A.shape[1]], float)
            volt[0, j] = 1
            A = np.concatenate((A, volt), axis = 0)
            
            volt_add = comp[COMP.VAL] * np.ones([1, 1])
            current_vector = np.concatenate((current_vector, volt_add), axis = 0)
                
            current = np.zeros([A.shape[0], 1], float)
            current[j, 0] = 1
            A = np.concatenate((A, current), axis = 1)
            
# compute the voltage on each node
voltage_vector = solve(A, current_vector)

#output result
print(voltage_vector)