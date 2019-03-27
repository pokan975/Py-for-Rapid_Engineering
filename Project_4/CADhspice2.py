
###############################
import subprocess
import numpy as np
import shutil
import re

'''
#subprocess.call("date")

p=subprocess.Popen(["hspice","InvChain.sp"], stdout=subprocess.PIPE)
output,err = p.communicate()
print(" *** Running hspice InvChain.sp command ***\n", output)


#help(np.recfromcsv)
#help(np.genfromtxt)

Data = np.recfromcsv("InvChain.mt0.csv",comments="$",skip_header=3)
print(Data["tphl_inv"])
tphl_prev = Data["tphl_inv"]

f = open('InvChain.sp', 'r')
f1 = open('InvChain1.sp','w')

#with open('InvChain.sp') as f:
#    read_data = f.read()

for line in f:
    if line == '.param fan = 1\n':
        line = '.param fan = 2\n'
#    print(line,end='')
    f1.write(line)


f.close()
f1.close()
'''
#shutil.copyfile('InvChain1.sp', 'InvChain.sp') 
'''
p=subprocess.Popen(["hspice","InvChain.sp"], stdout=subprocess.PIPE)
output,err = p.communicate()
print(" *** Running hspice InvChain.sp command ***\n", output)

Data = np.recfromcsv("InvChain.mt0.csv",comments="$",skip_header=3)
print(Data["tphl_inv"])
tphl_next = Data["tphl_inv"]

while tphl_next < tphl_prev:
    '''
f = open('InvChain2.sp', 'r')
f1 = open('InvChain3.sp','w')

for line in f:
    match = re.search(r'(param\sfan\s=\s)(\d)',line)
    if match: 
        numbertxt = match.group(2)
        newnumber = int(numbertxt) + 1
        newnumbertxt = str(newnumber)
        line = re.sub(r'\d',newnumbertxt,line)
        f1.write(line)
    else:
        f1.write(line)


f.close()
f1.close()

shutil.copyfile('InvChain3.sp', 'InvChain2.sp') 
'''
    p=subprocess.Popen(["hspice","InvChain.sp"], stdout=subprocess.PIPE)
    output,err = p.communicate()
    print(" *** Running hspice InvChain.sp command ***\n", output)

    Data = np.recfromcsv("InvChain.mt0.csv",comments="$",skip_header=3)
    print(Data["tphl_inv"])
    tphl_prev = tphl_next
    tphl_next = Data["tphl_inv"]
'''   