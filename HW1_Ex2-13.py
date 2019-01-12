#!/usr/bin/env python
"""
Created on Tue Jan  8 23:28:37 2019

@author: William
"""

def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)
    
def Catalan(n):
    if n is 0:
        return 1
    else:
        return ( (4*n - 2) / (n + 1) ) * Catalan(n-1)

def GCD(m, n):
    if n is 0:
        return m
    else:
        return GCD(n, m % n)



Num_fact = int(input("Enter a positive integer to get factorial: "))

print("{}! = ".format(Num_fact), factorial(Num_fact))


Num_cata = int(input("Enter an positive integer for Catalan: "))

print("Catalan factorial of {} is".format(Num_cata), Catalan(Num_cata))


GCD_num1, GCD_num2 = input("Enter 2 positive integers to get their GCD: ").split()

print("The GCD of {0} and {1} is".format(GCD_num1, GCD_num2), GCD(int(GCD_num1), int(GCD_num2)))