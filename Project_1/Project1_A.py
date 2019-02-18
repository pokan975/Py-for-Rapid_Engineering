#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

############################################################
# Problem 1                                                #
############################################################

# =============================================================================
# Function:
# create covariance for dataframes
# =============================================================================
def mosthighlycorrelated(mydataframe, numtoreport): 
    
    # compute the correlations 
    cormatrix = mydataframe.corr() 
    
    # set the correlations on the diagonal or lower triangle to zero, 
    # so they will not be reported as the highest ones: 
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T 
    
    # find the top n correlations 
    cormatrix = cormatrix.stack() 
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending = False).index).reset_index() 
    
    # assign human-friendly names 
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] 
    return cormatrix.head(numtoreport)


# =============================================================================
# Function:
# Compute and plot the covariance matrix
# =============================================================================
def corr_matrix(X,cols):
    # make the color maps for the covariance matrix
    fig = plt.figure(figsize = (8,8), dpi = 100)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap("jet", 30)
    cax = ax1.imshow(np.abs(X.corr()), interpolation = "nearest", cmap = cmap)
    
    # config colormap axis and grid
    major_ticks = np.arange(0, len(cols), 1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True, which = "both", axis = "both")

    # config axis labels and colorbar
    plt.title("Correlation Matrix")
    labels = cols
    ax1.set_xticklabels(labels, fontsize = 9)
    ax1.set_yticklabels(labels, fontsize = 9)
    fig.colorbar(cax, ticks = [-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()


# =============================================================================
# Function:
# Plot pairwise relationships in heart disease dataset
# =============================================================================
def pairplotting(df):
    # set pairplot parameters and plot the pairwise dot distribution
    sns.set(style = "whitegrid", context = "notebook")
    cols = df.columns
    sns.pairplot(df[cols], height = 2.5)
    plt.show()

# =============================================================================
# Main Code:
# Read the database in from heart1.csv file and analyze the data
# Calculate correlation between every pair of the variables, show the variables
# which are most highly correlated with each other, and pick those which are the most 
# highly correlated with result for training classifiers
# =============================================================================
# read data from csv file as dataframe
heart = pd.read_csv("heart1.csv")

#extract variable names
cols = heart.columns


# drop the observations that contain null value
for i in range(heart.values.shape[0]):
    null_value = sum(heart.iloc[i,:].isna())
    
    # drop the row that contains null section
    if null_value != 0:
        heart.drop(heart.index[i])
        
# show statistics of the data
print("\nDescriptive Statistics:")
print(heart.describe())

# show the first n highly correlated variables
print("\nMost Highly Correlated:")
print(mosthighlycorrelated(heart, 10))

# show the matrix of correlation between each pair of variables
print("\nCovariance Matrix:")
corr_matrix(heart.iloc[:, :], cols[:])

var_highlycorr = ["thal", "nmvcf", "eia", "mhr", "opst", "cpt", "a1p2"]
heart_highlycorr = heart.loc[:, [*var_highlycorr]]

# plot the pairwise distributions of all variables
print("\nPair plotting:")
pairplotting(heart_highlycorr)






