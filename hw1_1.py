#!/usr/bin/env python
# =============================================================================
# Main Code:
# Let user input the height, then output the time a ball takes from that altitude to the ground.
# Return error msg if user's input is invalid.
# =============================================================================


from numpy import sqrt

# takes g = 9.81 m/(s^2) as gravitational acceleration
GRAVITY_CONST = 9.81

try:
    # ask for user-defined starting altitude
    height = float(input("enter height: "))
    
    # because height = 0.5*g*(t^2), derive t
    time = sqrt(2 * height / GRAVITY_CONST)
    
    # add this to separate in and out
    print("")
    # output result
    print("time to hit the ground from {}m:".format(height), round(time, 2), "seconds.")

# error handling: catch all kinds of error, print error msg
except:
    print("\nYour input is invalid!")


