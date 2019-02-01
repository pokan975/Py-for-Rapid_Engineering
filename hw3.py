#!/usr/bin/env python
# =============================================================================
# Main Code:
# Given a netlist of a circuit description, calculate the voltage/current on each node of the circuit
# solve the problem in the form of linear equation Ax = b
# where resistors are stamped on matrix A, current on vector b, then x is voltage (answer)
# =============================================================================
import numpy as np
import comp_constants as COMP
from numpy.linalg import solve
from read_netlist import read_netlist

# read netlist from file
netlist = read_netlist()

# get the total nodes of resistor = N
N = 0
for comp in netlist:
    if (comp[COMP.TYPE] == COMP.R):
        N = max(N, comp[COMP.I], comp[COMP.J])

# initialize admittance matrix, voltage vector, and current vector
A = np.zeros([N, N], float)
current_vector = np.zeros([N, 1], float)

# stamp each resistor, voltage, and current value on the admittance matrix
for comp in netlist:
    # the component between node (from)i and (to)j
    i = comp[COMP.I] - 1
    j = comp[COMP.J] - 1
    
    # stamp the resistor on matrix A
    # rule: [i,i]&[j,j]: stamp 1/R; [i,j]&[j,i]: stamp -1/R
    if (comp[COMP.TYPE] == COMP.R):
        # if node I or J is 0(ground), do not stamp
        if (i >= 0):
            A[i, i] += 1.0/comp[COMP.VAL]
            
            if (j >= 0):
                A[i, j] += -1.0/comp[COMP.VAL]
                A[j, i] += -1.0/comp[COMP.VAL]
        
        if (j >= 0):
            A[j, j] += 1.0/comp[COMP.VAL]
        
    # stamp independent voltage source on matrix A
    # rule: for voltage source, append a new row & column to A, 
    #       then, [0, j](row) & [j, 0](column) = 1, ; [0, i](row) & [i, 0](column) = -1
    elif (comp[COMP.TYPE] == COMP.VS):
        # create new row & column for independent voltage source
        volt = np.zeros([1, A.shape[1]], float)
        current = np.zeros([A.shape[0] + 1 , 1], float)
        
        # if node I or J is 0(ground), do not stamp
        if (i >= 0):
            volt[0, i] = -1
            current[i, 0] = -1
        
        if (j >= 0):
            volt[0, j] = 1
            current[j, 0] = 1
            
        # append row & column onto A
        A = np.concatenate((A, volt), axis = 0)
        A = np.concatenate((A, current), axis = 1)
        
        # for independent voltage source, append its value V to vector b
        # (so its corresponding entry in vector x is current)
        volt_add = comp[COMP.VAL] * np.ones([1, 1])
        current_vector = np.concatenate((current_vector, volt_add), axis = 0)
        
# derive vector b (the voltage on each node)
voltage_vector = solve(A, current_vector)

# output result
print(voltage_vector)
