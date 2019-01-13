#!/usr/bin/env python
# =============================================================================
# Main Code:
# Find all prime numbers less or equal than 10^4.
# Prime factor examination:
#     an non-prime integer must have prime factors less or equal than its square root, or it's a prime.
# 
# Pseudo code:
# for all integers n:
#     prime_factor_list = elements in prime_num_list less or equal than sqrt(n)
#     if prime_factor_list empty:
#         n is prime, skip
#     
#     set prime_flag True
#     for factors in prime_factor_list:
#         if (n mod factors) == 0:
#             reset prime flag, interrupt loop
#     if prime flag True:
#         add n to prime_num_list
# =============================================================================


from numpy import sqrt

Max_num = int(1e4)
# initiate prime list from 2 (minimum prime)
Prime_list = [2]

for num in range(3, Max_num + 1):
    
    # list all primes less or equal than n as factors
    PrimeFactor_list = [i for i in Prime_list if i <= sqrt(num)]
    
    # if factors is empty, n is a prime, skip residual codes
    if PrimeFactor_list == []:
        Prime_list.append(num)
        continue
    
    # if factors not empty, default regard n as prime, set flag True
    Prime_flag = True
    
    # check if n has factor in factors, once find, stop check & set flag False
    for factor in PrimeFactor_list:
        if num % factor == 0:
            Prime_flag = False
            break
    
    # n is prime if flag remains True, add to list
    if Prime_flag:
        Prime_list.append(num)
