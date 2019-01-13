#!/usr/bin/env python

# =============================================================================
# Function:
# Calculate & return Catalan number C_n of input n:
# C_n = 1,                             if n = 0
# C_n = C_(n-1)*((4*n - 2) / (n + 1)), if n > 0
# =============================================================================
def Catalan(num):
    """
    type num: int
    rtype: int
    """
    # recursively call self with input n, n-1, n-2,...,until 0 then return 1
    if num == 0:
        return 1
    else:
        return ( (4*num - 2) / (num + 1) ) * Catalan(num - 1)


# =============================================================================
# Function:
# Calculate the Greatest Common Factor (GCD) of 2 positive integers:
# GCD(m, n) = m,               if n = 0
# GCD(m, n) = GCD(n, m mod n), if n > 0
# =============================================================================
def calcGCD(num1, num2):
    """
    type num1: int
    type num2: int
    rtype: int
    """
    # recursively call self with input (m, n), (n, m mod n),...,until (m ond n) is 0
    if num2 == 0:
        return num1
    else:
        return calcGCD(num2, num1 % num2)


# =============================================================================
# Main Code:
# =============================================================================

# call function Catalan to get Catalan number of 100
Num_cata = 100

print("\nCatalan number of {} is".format(Num_cata), Catalan(Num_cata))


# call function getGCD to get the GCD of 108 & 192
GCD_num1, GCD_num2 = 108, 192

print("\nThe GCD of {0} and {1} is".format(GCD_num1, GCD_num2), calcGCD(GCD_num1, GCD_num2))


