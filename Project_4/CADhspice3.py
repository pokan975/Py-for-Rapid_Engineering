
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
index = 4
beta = 2
if index == 4 :
    f = open('InvChain4.sp', 'r')
    f1 = open('InvChain5.sp','w')

    for line in f:
        match = re.search(r'^X',line)
        if match: 
            f1.write('')
        else:
            f1.write(line)
        match = re.search(r'Cload', line)
        if match:
            line = 'Xinv a 2 inv M='+str(beta)+'\n'
            f1.write(line)
            for j in np.arange(1,index-1,1):
                line = 'Xinv '+str(j+1)+' '+str(j+2)+' inv M='+str(beta**(j+1))+'\n'
                f1.write(line)
            line = 'Xinv '+str(index)+' z inv M='+str(beta**index)+'\n'
            f1.write(line)
    f.close()
    f1.close()

    shutil.copyfile('InvChain5.sp', 'InvChain4.sp') 
'''
    p=subprocess.Popen(["hspice","InvChain.sp"], stdout=subprocess.PIPE)
    output,err = p.communicate()
    print(" *** Running hspice InvChain.sp command ***\n", output)

    Data = np.recfromcsv("InvChain.mt0.csv",comments="$",skip_header=3)
    print(Data["tphl_inv"])
    tphl_prev = tphl_next
    tphl_next = Data["tphl_inv"]
'''   