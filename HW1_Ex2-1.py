## EEE591: Python for Rapid Engineering Solution
## Author: Po-Kan Shih


## Exercise 2.1
from numpy import sqrt

height = int(input("Enter the height of the tower in meter: "))

time = sqrt(2*height/9.81)

print("The time the ball takes to hit the ground is",round(time, 2),"sec.")

