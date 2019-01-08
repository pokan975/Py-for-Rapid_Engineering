## EEE591: Python for Rapid Engineering Solution
## Author: Po-Kan Shih


## (1)
from numpy import sqrt
from numpy import around

height = input("Enter the height of the tower in meter: ")
time = sqrt(2*float(height)/9.81)

print("The time the ball takes to hit the ground is",around(time,decimals=2),"sec")


## (2)
