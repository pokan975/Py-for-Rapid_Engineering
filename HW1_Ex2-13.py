# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:28:37 2019

@author: William
"""

## Exercise 2.13
def factorial(n):
    if n == 1:
        return 1
    else:
        return n*factorial(n-1)
    
def Catalan(n):
    if n is 0:
        return 1
    else:
        return ((4*n-2)/(n+1))*Catalan(n-1)

def GCD(m, n):
    if n is 0:
        return m
    else:
        return GCD(n, m%n)


print(factorial(5))
print(Catalan(100))
print(GCD(108, 192))