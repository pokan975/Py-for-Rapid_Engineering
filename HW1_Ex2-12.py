#!/usr/bin/env python
"""
Created on Tue Jan  8 21:43:57 2019

@author: William
"""

## Exercise 2.12
from numpy import sqrt

upperbound = int(1e4)
prime_list = [2]

for num in range(3, upperbound+1):
    
    prime_base = [i for i in prime_list if i <= sqrt(num)]
    
    if prime_base is []:
        prime_list.append(num)
        continue
    
    fPrime_num = 1
    
    for prime_num in prime_base:
        if not ( num % prime_num ):
            fPrime_num = 0
            break
    
    if fPrime_num:
        prime_list.append(num)


# =============================================================================
# input("") without a variable means only a blank line in output
#        for and while loop can have else
#        do not compare 2 floating num (because of the precision issue)
#        "global" declaration is dangerous, use global variable name is dangerous
#        input of a function is passed by "value", but array and list passed by init addr of them, so they passed by reference
# =============================================================================
