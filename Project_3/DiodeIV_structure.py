# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:43:31 2017

@author: olhartin@asu.edu
"""

from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np


## for problem 2 use a diode where we calculate the saturation
## current from physical parameters
print('Problem 2: Find n, phi and R to match diode data')

 
#actual values used to create data file

## Diode equation
def IdiodeA(vd,A,phi,n,T):

    return
##
##  nodal analysis solver gets vd
def IdiodeRA(vd,Vd,A,phi,n,T,R):

    return(err)
    

def EvalDiodeR(X,Vs):
##    print('residuals 1')
    A = X[0]; phi = X[1]; n = X[2];
    T = X[3]; R = X[4];
##      solve for diode voltage

    return(Vd,ID)
   
##
##  find these parameter values
##  create a residual equation for optimization
##  These are the residual differences between the diode data in the file and attempt to match that data.
##
## M = 1 exp, not normalized
## M = 2 exp, normalized
## M = 3 log, not normalized
## M = 4 log, normalized
def residuals(x,X,Vs,M,j,I2bpred):

    X[j] = x        ##  parameter to be optimized, replace with x
    A = X[0]; phi = X[1]; n = X[2];
    T = X[3]; R = X[4];
##
##      solve for diode voltage

    for vs in Vs:
        vd = optimize.fsolve(IdiodeRA,last_vd,(vs,A,phi,n,T,R),xtol=1e-12)[0]



##      normalization makes errors in low magnitude
##      regions of the probem as important as ones
##      in high magnitude regions

        return (Ierr)
##

#Read diode data file
filename = 'DiodeIV.txt'

I2bpred = np.asarray(Itemp)
Vs = np.asarray(Vstemp)

       

##  initial guesses for A,phi,n,T and R values

X = np.zeros([5],float)
X[0] = 1e-8
X[1] = 0.8
X[2] = 1.5
X[3] = 375
X[4] = 10000



##
##  find parameters to match currents in file
##


while (err > 1e-3 and i <= 30):
## good optimization of n, phi, and R, but not A and T

    for j in :    ## optmizes R then n the phi

        xd,cov,infodict,mesg,ier = optimize.leastsq(residuals,x,args=(X,Vv[beg:end],M,j,I2bpred[beg:end]),full_output=True)

##    print(' xd ', xd, ' max error ', np.max(resid),' avg error ',np.average(np.abs(resid)))


