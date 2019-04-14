#!/usr/bin/env python
import numpy as np
import subprocess
import shutil
import re

# =============================================================================
# Main Code:
# Use python script to automate the procedure of finding the optimum fanout
# and number of inverters config with minimum delay between the transient source
# and the load capacitor in an inverter chain.
# =============================================================================
# list all fanout values for test
fan = [1, 2, 3, 4, 5, 6, 7]
# list all num of inverters for test
inv_stage = [3, 5, 7, 9, 11, 13]

# list all possible (fanout & stage) configs (each pair as a tuple)
configs = []
for f in fan:
    for inv in inv_stage:
        comb = tuple([f, inv]) # each config = tuple(fanout, stage)
        configs.append(comb)

# a list to store all results of delay time
result = []

# store the content of the original hspice script as a list of strings for later use
file = open("InvChain.sp", "r")
# a list to store all script commands
script = []
for line in file:
    # store every line in script as an single element in list, 
    # except inverter nodes & fanout parameter
    # (because they are parameters needed to be modified in each test run)
    match = re.search(r"^Xinv", line)
    if not match:
        script.append(line)

file.close()


node = 98  # ASCii code of char "b", used as node name

# use for loop to run every config
for counter in range(len(configs)):
    # extract fanout & inverter nums of current test run
    fan_new = configs[counter][0]
    stage_new = configs[counter][1]

    # create a temp script, edit it, then overwrite the content of the original
    # script file using this temp file
    # (because always run hspice with the original script file name)
    file_temp = open("InvChain_temp.sp", "w")

    # write script command line by line from script string list
    for line in script:
        # modify the fanout and inverter name when writing the corresponding
        # lines
        match1 = re.search(r"(param\sfan\s=\s)(\d)", line)
        match2 = re.search(r"Xinv[\d]+\.z", line)
        
        # replace fanout parameter
        if match1:
            num = match1.group(2)
            line = re.sub(r"\d", str(fan_new), line)
        
        # replace the name of the last stage of inverters
        elif match2:
            line = re.sub(r"[\d]+\.z", str(stage_new) + ".z", line)

        # write script command to temp file
        file_temp.write(line)
    
        # insert stages of inverter after capacitor setting
        match = re.search(r"Cload", line)
        if match:
            # write the 1st stage, always starts at node "a", fanout always be 1
            line = "Xinv1 a " + chr(node) + " inv M = 1\n"
            file_temp.write(line)
        
            # write 2nd ~ last-1 stages of inverter
            for j in np.arange(1, stage_new - 1, 1):
                # set fanout multiplies for each stage
                fans = "*fan" * (j - 1)
                line = "Xinv" + str(j + 1) + " " + chr(j + node - 1) + ' ' + \
                chr(j + node) + " inv M = fan" + fans + "\n"
            
                file_temp.write(line)
        
            # write the last stage, always ends at node "z",
            # and set the fanout multiply
            fans = "*fan" * j
            line = "Xinv"+ str(stage_new) + " " + chr(j + node) + \
            " z inv M = fan" + fans + "\n"
        
            file_temp.write(line)
            
    
    file_temp.close()
    # overwrite the original script file using temp file
    shutil.copyfile('InvChain_temp.sp', 'InvChain.sp')


    # run hspice (always use the original script file name)
    proc = subprocess.Popen(["hspice", "InvChain.sp"], stdout = subprocess.PIPE)
    output, err = proc.communicate()
    print("*** Running hspice InvChain.sp command ***\n", output)

    # extract delay time from output csv
    Data = np.recfromcsv("InvChain.mt0.csv", comments = "$", skip_header = 3)
    print(Data["tphl_inv"])
    # store delay time for each config, use [()] to extract the float num from
    # an np.array with zero dimension
    result.append(Data["tphl_inv"][()])

# find the minimum delay time from result list
opt_val = min(result)
# use minimum delay time to find the corresponding config
opt_config = configs[result.index(opt_val)]

# print all configs of (fanout & num of inverters) and corresponding delay time 
print("fan inverter  delay(sec)")
for i in range(len(configs)):
    print("{0:3d} {1:8d}  {2:6.4e}".format(configs[i][0], configs[i][1], result[i]))

# print minumum delay time & its corresponding config
print("\nminimum delay (sec): {0:.4e}".format(opt_val))
print("optimum config => fan = {0:d}; inverter = {1:d}".format(opt_config[0], opt_config[1]))
