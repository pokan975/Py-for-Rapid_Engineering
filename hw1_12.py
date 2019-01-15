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
#         n is prime
# =============================================================================


from numpy import sqrt

max_num = int(1e4)
# initiate prime list from 2 (minimum prime)
prime_list = [2]

for num in range(3, max_num + 1):
    
    # default take n as prime
    prime_flag = True
    
    # find n's factor in factors, once find out, reset flag & break loop
    for factor in prime_list:
        # only check prime factors less or equal than sqrt(n)
        if factor > sqrt(num):
            break
        elif num % factor == 0:
            prime_flag = False
            break
    
    # add n to prime list if no factor is found
    if prime_flag:
        prime_list.append(num)

# print out final prime list
print(prime_list)