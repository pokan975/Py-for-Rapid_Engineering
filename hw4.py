#!/usr/bin/env python
# =============================================================================
# Main Code:
# Given a netlist of a circuit description, calculate the voltage/current on each node of the circuit
# solve the problem in the form of linear equation Ax = b
# where resistors are stamped on matrix A, current on vector b, then x is voltage (answer)
# =============================================================================

import tkinter as tk
import numpy as np
import matplotlib as plot

# root widget
root = tk.Tk()