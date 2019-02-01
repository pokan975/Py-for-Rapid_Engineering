#!/usr/bin/env python

from tkinter import *
import numpy as np
import matplotlib.pyplot as plt

def calc_wealth(msg):
    # get value of each entries
    mean_value = float(entry_1.get())
    STD = float(entry_2.get())
    yearly_income = int(entry_3.get())
    contribution_years = int(entry_4.get())
    retirement = int(entry_5.get())
    
    wealth = [yearly_income]
    
    for year in range(2, retirement + 1):
        risk = (STD / 100) * np.random.randn()
        
        if year <= contribution_years:
            wealth_next = wealth[-1] * (1 + (mean_value / 100) + risk) + yearly_income
        else:
            wealth_next = wealth[-1] * (1 + (mean_value / 100) + risk)
        
        wealth.append(wealth_next)
    
    
    # show the final amount
    string = "the final wealth is " + str(wealth[-1])
    msg.config(text = string)
    
    # clear all entries
# =============================================================================
#     entry_1.delete(0)
#     entry_2.delete(0)
#     entry_3.delete(0)
#     entry_4.delete(0)
#     entry_5.delete(0)
# =============================================================================
    print(wealth)
    
    years = range(1, retirement + 1)
    plt.plot(years, wealth)
    plt.xlabel("year")
    plt.ylabel("total wealth")
    plt.show()
    

# =============================================================================
# Main Code:
# Given a netlist of a circuit description, calculate the voltage/current on each node of the circuit
# solve the problem in the form of linear equation Ax = b
# where resistors are stamped on matrix A, current on vector b, then x is voltage (answer)
# =============================================================================
# create root widget
root = Tk()
root.title("Wealth Calculator")

# create each label for input arguments
Label(root, text = "Mean Return (%)").grid(row = 0, column = 0, sticky = W)
Label(root, text = "Std Dev Return (%)").grid(row = 1, column = 0, sticky = W)
Label(root, text = "Yearly Contribution ($)").grid(row = 2, column = 0, sticky = W)
Label(root, text = "No. of Years of Contribution").grid(row = 3, column = 0, sticky = W)
Label(root, text = "No. of Years to Retirement").grid(row = 4, column = 0, sticky = W)

# create entries for input arguments
entry_1 = Entry(root)
entry_2 = Entry(root)
entry_3 = Entry(root)
entry_4 = Entry(root)
entry_5 = Entry(root)

# align each entry
entry_1.grid(row = 0, column = 1)
entry_2.grid(row = 1, column = 1)
entry_3.grid(row = 2, column = 1)
entry_4.grid(row = 3, column = 1)
entry_5.grid(row = 4, column = 1)

msg = Label(root, fg = "dark green")
msg.grid(row = 5, column = 0, sticky = W, pady = 4)


Button(root, text = 'Calculate', command = lambda: calc_wealth(msg)).grid(row = 6, column = 1, sticky = W, pady = 4)
Button(root, text = 'Quit', command = root.destroy).grid(row = 6, column = 0, sticky = W, pady = 4)

mainloop()