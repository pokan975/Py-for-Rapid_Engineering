#!/usr/bin/env python
# =============================================================================
# Main Code:
# Let user input a integer (height), then calculate the time a ball takes from this height to the ground.
# Return error msg if user enters invalid data format.
# =============================================================================


from numpy import sqrt

try:
    # catch user-defined height (integer, in meter)
    Height = int(input("Enter the height in meter: "))
    
    # compute time: height = 0.5*g*(t^2), takes g = 9.81 m/(s^2) as gravitational acceleration
    Time = sqrt(2 * Height / 9.81) 
    
    # output result
    print("\nThe time the ball takes from {}m to hit the ground is".format(Height), round(Time, 2), "sec.")

# error handling: catch all kinds of error, output error msg
except:
    print("\nYour input is invalid!")


