# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:43:57 2019

@author: William
"""

## Exercise 2.12
from numpy import sqrt

upperbound = int(1e4)
prime_list = [2]

for num in range(3,upperbound+1):
    
    prime_base = [i for i in prime_list if i <= sqrt(num)]
    
    if prime_base is []:
        prime_list.append(num)
        continue
    
    prime_num_flag = 1
    
    for prime_num in prime_base:
        if not num % prime_num:
            prime_num_flag = 0
            break
    
    if prime_num_flag:
        prime_list.append(num)
