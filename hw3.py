#!/usr/bin/env python
# =============================================================================
# Main Code:
# Calculate and plot the period-amplitude relation of an anharmonic oscillator
# start amplitude: 0m~2m, with zero initial velocity
# =============================================================================
import numpy as np
import comp_constants as COMP
from read_netlist import read_netlist


def stamper(y_add, netlist, currents, voltage, num_nodes):
    for comp in netlist:
        if (comp[COMP.TYPE] == COMP.R):
            i = comp[COMP.I] - 1
            j = comp[COMP.j] - 1
            if (i >= 0):
                y_add[i, i] += 1.0 / comp[COMP.VAL]
                y_add[j, j] += 1.0 / comp[COMP.VAL]
                y_add[i, j] += -1.0 / comp[COMP.VAL]
                y_add[j, i] += -1.0 / comp[COMP.VAL]
        
        elif (comp[COMP.TYPE] == COMP.VS):
            
    return num_nodes


netlist = read_netlist()

N = len(netlist)
A = np.zeros([N, N], float)
