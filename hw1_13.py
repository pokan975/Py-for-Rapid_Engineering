#!/usr/bin/env python
# =============================================================================
# Function:
# Calculate & return Catalan number C_n of input n:
# C_n = 1,                             if n = 0
# C_n = (C_{n-1})*((4*n - 2) / (n + 1)), if n > 0
# =============================================================================
def Catalan(num):
# type num: int
# rtype: int
    
    # recursively call self with input n, n-1, n-2,...,until 0
    if num == 0:
        return 1
    else:
        return ( ((4 * num) - 2) / (num + 1) ) * Catalan(num - 1)


# =============================================================================
# Function:
# Calculate the Greatest Common Factor (GCD) of 2 positive integers:
# GCD(m, n) = m,               if n = 0
# GCD(m, n) = GCD(n, m mod n), if n > 0
# =============================================================================
def calcGCD(num1, num2):
# type num1: int
# type num2: int
# rtype: int

    # if (m ond n) = 0, n is GCD
    # recursively take (m, n), (n, m mod n), ((m mod n), n mod (m mod n))... as input until the 2nd argument is 0 
    if num2 == 0:
        return num1
    else:
        return calcGCD(num2, num1 % num2)


# =============================================================================
# Main Code:
# =============================================================================

# get Catalan number of 100 then print out
num_cata = 100

print("\nCatalan number of {} is".format(num_cata), Catalan(num_cata))


# get the GCD of 108 & 192 then print out
GCD_num1, GCD_num2 = 108, 192

print("\nThe GCD of {0} and {1} is".format(GCD_num1, GCD_num2), calcGCD(GCD_num1, GCD_num2))


