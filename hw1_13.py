#!/usr/bin/env python
# =============================================================================
# Function:
# Calculate the factorial of input n:
# n! = 1,           if n = 1
# n! = n * (n-1)!,  if n > 1
# =============================================================================
def factorial(num):
# type num: int
# rtype: int
    
    # recursively call self with input n, n-1, n-2,...,until 1 to calc n!
    if num == 1:
        return 1
    else:
        return num * factorial(num - 1)

# =============================================================================
# Function:
# Calculate & return Catalan number C_n of input n:
# C_n = 1,                               if n = 0
# C_n = (C_{n-1})*((4*n - 2) / (n + 1)), if n > 0
# =============================================================================
def Catalan(num):
# type num: int
# rtype: int
    
    # recursively call self with input n, n-1, n-2,...,until 0 to calc Catalan num
    if num == 0:
        return 1
    else:
        return ( ((4 * num) - 2) / (num + 1) ) * Catalan(num - 1)


# =============================================================================
# Function:
# Calculate the Greatest Common Divisor (GCD) of 2 positive integers:
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

# get factorial num and print out
num_factorial = int(input("Enter an integer for a factorial computation: "))

print("")
print("factorial of {} is".format(num_factorial), factorial(num_factorial))


# get Catalan number then print out
num_cata = int(input("Enter an integer for a Catalan number computation: "))

print("")
print("Catalan value of {} is".format(num_cata), Catalan(num_cata))


# get the GCD of 2 integers and print out
GCD_num1 = int(input("Enter the first of two integers for a GCD calculation: "))
GCD_num2 = int(input("Enter the second of two integers for a GCD calculation: "))

print("")
print("greatest common divisor of {0} and {1} is".format(GCD_num1, GCD_num2), calcGCD(GCD_num1, GCD_num2))


