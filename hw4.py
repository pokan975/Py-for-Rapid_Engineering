#!/usr/bin/env python
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Function:
# Calculate the wealth at the end of every year according to user's input
# wealth of year N is the cumulative number at the end of the year N
# during contribution years, a fixed investment will be added at the end of the year
# from 2nd year, all wealth will be taken into market
# market risk is Gaussian distribution with specified mean and standard deviation
# =============================================================================
def calc_wealth(entries, display_frame):
# type entries: dict
# type display_frame: Frame object
    
    # clear all widgets in the display frame
    for widget in display_frame.winfo_children():
        widget.destroy()
    
    # get values from each entries
    mean_value = (float(entries['Mean Return (%)'].get()))
    STD = float(entries['Std Dev Return (%)'].get())
    yearly_income =  float(entries['Yearly Contribution ($)'].get()) 
    contribution_years = int(entries['No. of Years of Contribution'].get())
    retirement = int(entries["No. of Years to Retirement"].get())
    
    # initialize the list of yearly wealth, set the investment of 1st year as 1st element
    wealth = [yearly_income]
    
    # wealth at year's end = (wealth at year's begining)*(1 + risk_mean + risk_STD) + investment of that year
    for year in range(2, retirement + 1):
        # generate Gaussian random num as risk by using specified STD
        risk = (STD / 100) * np.random.randn()
        
        # if contribution years < retirement year: no fixed investment after contribution years
        if year <= contribution_years:
            wealth_next = wealth[-1] * (1 + (mean_value / 100) + risk) + yearly_income
        else:
            wealth_next = wealth[-1] * (1 + (mean_value / 100) + risk)
        
        # append every year's final wealth to the list
        wealth.append(wealth_next)
        
    # show and align the final wealth at retirement (just the final element of wealth list)
    string = "wealth at retirement: " + str(wealth[-1])
    msg = Label(display_frame, text = string, anchor = W)
    msg.pack(side = LEFT)
    
    # plot the wealth variation curve (Y axis) as function of year (X axis)
    years = range(1, retirement + 1)
    plt.plot(years, wealth)
    plt.xlabel("year")
    plt.ylabel("total wealth")
    plt.show()

# =============================================================================
# Function:
# config each input entries, use dict to bond each entry (value) and its corresponding label (key)
# locate each entry and its label in an individual frame, and them in the frame
# align frame in the GUI window
# =============================================================================
def makeform(root, fields):
# type root: GUI instance
# type fields: list of field labels
# rtype: dict

    # initialize an empty dict
    entries = {}
   
    # orderly put each input entry and its corresponding label in its own frame
    for field in fields:
        # create and align frame in GUI window
        row = Frame(root) 
        row.pack(side = TOP, fill = X, padx = 5, pady = 5)
        
        # assign and align field label for input each entry in frame
        lab = Label(row, width = 30, text = field + ": ", anchor = W)
        lab.pack(side = LEFT)
        
        # create and align input entry in frame
        ent = Entry(row)
        # set 0 as default value of each entry
        ent.insert(0,"0")
        ent.pack(side = RIGHT, expand = YES, fill = X)
        
        # for each object of dict:
        # key: field label
        # value: corresponding entry object
        entries[field] = ent
    
    return entries


# =============================================================================
# Main Code:
# Create a GUI that allows user to input some arguments, then calculates his wealth at retirement
# GUI will display the final wealth, and plot the wealth variation as a function of year
# =============================================================================
if __name__ == "__main__":
    # designate the meaning of each input entries
    fields = ('Mean Return (%)',               # the mean value of risk distribution in market
              'Std Dev Return (%)',            # the standard deviation of risk in market
              'Yearly Contribution ($)',       # how much money will be invested per year?
              'No. of Years of Contribution',  # how many years will user take for investment?
              'No. of Years to Retirement')    # how many years will it take before retirement? (must >= above)
    
    # create GUI window instance
    root = Tk()
    root.title("Wealth Calculator")
    
    # for each input entry, create a frame to accommodate it and its corresponding labels,
    # allow user to input values
    entries = makeform(root, fields)
    
    # create and align a frame for displaying the calculation result 
    # (no input entry, just showing message)
    display_frame = Frame(root)
    display_frame.pack(side = TOP, fill = X, padx = 5, pady = 5)

    # create and align the "Quit" button: 
    # terminate the GUI application
    b1 = Button(root, text = "Quit", command = root.destroy)
    b1.pack(side = LEFT, padx = 5, pady = 5)
    
    # create and align the "Calculate" button: 
    # once pressed, compute the wealth for each year, display the remaining
    # wealth at retirement on GUI, and plot the wealth fluctuation as a function of year
    b2 = Button(root, text = "Calculate", command = (lambda: calc_wealth(entries, display_frame)))
    b2.pack(side = LEFT, padx = 5, pady = 5)
    
    # run GUI window
    root.mainloop()
