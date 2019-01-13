#!/usr/bin/env python
# =============================================================================
# Main Code:
# Let user input the height, then output the time a ball takes from that altitude to the ground.
# Return error msg if user's input is invalid.
# =============================================================================


from numpy import sqrt

try:
    # ask for user-defined starting altitude
    Height = float(input("enter height: "))
    
    # height = 0.5*g*(t^2), takes g = 9.81 m/(s^2) as gravitational acceleration
    Time = sqrt(2 * Height / 9.81) 
    
    # add this to separate in and out
    print("")
    # output result
    print("time to hit the ground from {}m:".format(Height), round(Time, 2), "seconds.")

# error handling: catch all kinds of error, output error msg
except:
    print("\nYour input is invalid!")


