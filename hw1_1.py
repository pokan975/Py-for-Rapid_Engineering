#!/usr/bin/env python
"""
Created on Tue Jan  8 23:28:37 2019

@author: William
"""

from numpy import sqrt

error = True

while error:
    try:
        height = int(input("Enter the height of the tower in meter: "))
        error = False
    except:
        print("\nYour input is invalid!")


time = sqrt(2 * height / 9.81)

print("\nThe time the ball takes from {}m to hit the ground is".format(height), round(time, 2), "sec.")

