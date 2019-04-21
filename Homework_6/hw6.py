#!/usr/bin/env python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

###############################################################################
# Function:
# Solve for pseudo 2D self heating based on stack. Start at the base layer
# which is at a constant temperature. Then work our way up.
# Inputs:
#    current - current through the structure
#    voltage - voltage across the structure
#    tbase   - temperatue of the bottom layer
# Outputs:  
#    temp_top- temperature of the top layer
###############################################################################
def calc_temperature(current, voltage, tbase):
    #initialize temper. at top the same as bottom layer
    temp_top = tbase
    # calc the heat flow across the whole structure
    heat_flow = current * voltage
    
    # calc temp. at each layer and add them up to get the temp. at top layer
    for layer in range(LAYERS):
        # calc thermal resistance at each layer
        thermal_rest = THICK[layer] / (K[layer] * AREA)
        # add up temp. at each layer to get the temp. at top layer
        temp_top = temp_top + heat_flow * thermal_rest

    return temp_top


###############################################################################
# Function: 
# Return diode current and temperature 
# Inputs:   
#    volt    - voltage across the device  
#    tguess  - guess at device temperature  
#    tbase   - temperature of the bottom layer 
# Outputs: 
#    diode_curr - current through the diode 
#    t_err   - difference between the guess and the calculated value 
###############################################################################
def diode_i_temp_err(volt, t_guess, tbase):
    # calc diode current based on given parameters
    diode_curr = I0 * (np.exp((volt * Q) / (IDE * KB * t_guess)) - 1.)
    # get the calculated temperature based on calculated diode current & parameters
    t_calc = calc_temperature(diode_curr, volt, tbase)

    return [diode_curr, (t_calc - t_guess)]


###############################################################################
# Function:
# Bridge with a diode with node and temperature error. Based on the guess of
# voltages, compute diode current and temp error. Then compute the current at
# each node, which should sum to 0. Return these sums and the temp error.
# Inputs:
#    variables - an array containing the following values fsolve will optimize:
#              index 0 - voltage at node 1
#              index 1 - voltage at node 2
#              index 2 - voltage at node 3
#              index 3 - temperature of the diode
#    source_v - source voltage
#    tbase    - temperate of the base layer
# Outputs:
#    An array of variables that need to be optimized to 0
###############################################################################
def bridge_i_diode_temp(variables, source_v, tbase):
    # extract the parameters
    n1_v   = variables[0]
    n2_v   = variables[1]
    n3_v   = variables[2]
    t_guess = variables[3]

    # compute the diode current and temperature error
    # (temperature error should be optimized to 0!)
    v_diode = n1_v - n2_v
    [i_diode, diode_temp_err] = diode_i_temp_err(v_diode, t_guess, tbase)

    # based on the calculated current and voltages, sum up the currents
    # at each node (they should be optimized to 0!)
    n1_i = ((n1_v - n3_v) / R1) + (n1_v / R2) + i_diode
    n2_i = ((n2_v - n3_v) / R3) + (n2_v / R4) - i_diode
    n3_i = ((n3_v - source_v) / R5) + ((n3_v - n1_v) / R1) + ((n3_v - n2_v) / R3)

    # goal is to let all values be 0!
    return [n1_i, n2_i, n3_i, diode_temp_err]


###############################################################################
# Main Code:
# Solve the nodal voltages and diode temperature & currents in a Wheatstone
# bridge for different source voltages, the diode is customized so its behaviors 
# of temperature & current need to be derived from its stack
###############################################################################
# First of all, defnie constants for this bridge circuit
MIL = 25      # conversion factor
UM2CM = 1e-4  # square centimeters in meters

# Diode Constants
I0 = 3e-9     # reverse bias saturation current
Q = 1.6e-19   # charge on the electron
KB = 1.38e-23 # Boltzmann constant
IDE = 2.0     # Ideality factor

MAX_V_SOURCE = 10  # Max source voltage to analyze
MIN_V_SOURCE =  1  # Min source voltage to analyze

# values of the resistors in bridge (Ohm)
R1 = 1e3 
R2 = 4e3
R3 = 3e3
R4 = 2e3
R5 = 1e3

# Constant temperature of the base
TBASE = 325
# Area of the diode (transform its unit to cm**2)
AREA = 10*UM2CM*10*UM2CM

# Thermal coefficients (W/cmC) of each layer in the diode structure
#   0  Si active       10 mil  1.3
#   1  Si bulk         15 mil  1.3
#   2  Cu back metal   5 mil   3.86
#   3  paste           25 mil  0.5
#   4  Metal flag      100 mil 5
K = np.array([1.3, 1.3, 3.86, 0.5, 5.0], float)
# thickness of each layer (transform their units to cm)
THICK = np.array([10, 15, 5, 25, 100], float) * MIL*UM2CM
# number of total layers of the diode
LAYERS = 5




# define source voltage grid
v_source = np.linspace(MIN_V_SOURCE, MAX_V_SOURCE, 100)

# create empty arrays to store optimized nodal voltages, diode temp. & current
n1_volt = np.zeros_like(v_source)              # node 1 voltages
n2_volt = np.zeros_like(v_source)              # node 2 voltages
n3_volt = np.zeros_like(v_source)              # node 3 voltages
diode_temp  = np.zeros_like(v_source)              # temperature values
diode_i = np.zeros_like(v_source)              # diode current values

# initial guesses for variables that need to be optimized
temp = TBASE       # diode temperature at top
v1 = MIN_V_SOURCE  # voltage at node 1
v2 = MIN_V_SOURCE  # voltage at node 2
v3 = MAX_V_SOURCE  # voltage at node 3  

# calc the optimized nodal voltages, diode temp. & currents for different source voltages
for index in range(len(v_source)):
    # get optimized values
    [v1, v2, v3, temp] = optimize.fsolve(bridge_i_diode_temp, [v1, v2, v3, temp], (v_source[index], TBASE))
    # store to corresponding arrays
    n1_volt[index] = v1
    n2_volt[index] = v2
    n3_volt[index] = v3
    diode_temp[index] = temp
    
    # use optimized diode temp. to calc diode currents & store
    diode_volt = v1 - v2
    current = I0 * (np.exp((diode_volt * Q) / (IDE * KB * temp)) - 1.)
    diode_i[index] = current
    
# plot nodal voltages versus source voltages
plt.plot(v_source, n1_volt, 'r', label = "volt at node 1")
plt.plot(v_source, n2_volt, 'g', label = "volt at node 2")
plt.plot(v_source, n3_volt, 'b', label = "volt at node 3")
plt.legend(loc = "upper left")
plt.xlabel("Source voltage (Volt)")
plt.ylabel("Nodal voltage (Volt)")
plt.title("Nodal voltages versus source voltage")
plt.grid()
plt.show()

# plot diode temperatures versus source voltages
plt.plot(v_source, diode_temp)
plt.xlabel("Source voltage (Volt)")
plt.ylabel("Diode temperature (K)")
plt.title("Diode temperature versus source voltage")
plt.grid()
plt.show()

# plot logarithm of diode currents versus source voltages
plt.plot(v_source, np.log10(diode_i))
plt.xlabel("Source voltage (Volt)")
plt.ylabel("Log diode current (Amp)")
plt.title("Diode current versus source voltage")
plt.grid()
plt.show()