#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt

# =============================================================================
# Problem 1
# =============================================================================
def mosthighlycorrelated(mydataframe, numtoreport): 
    
    # find the correlations 
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


def corr_matrix(X,cols):
    fig = plt.figure(figsize = (7,7), dpi = 100)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(np.abs(X.corr()), interpolation = 'nearest', cmap = cmap)
    
    major_ticks = np.arange(0, len(cols), 1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True, which = 'both', axis = 'both')
##    plt.aspect('equal')
    plt.title('Correlation Matrix')
    labels = cols
    ax1.set_xticklabels(labels, fontsize = 9)
    ax1.set_yticklabels(labels, fontsize = 9)
    fig.colorbar(cax, ticks = [-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()
    return(1)


def pairplotting(df):
    sns.set(style = 'whitegrid', context = 'notebook')
    cols = df.columns
    sns.pairplot(df[cols],size=2.5)
    plt.show()


iris = pd.read_csv('heart1.csv')

cols = iris.columns


# get rid of observations containing null value
for i in range(iris.values.shape[0]):
    null_value = sum(iris.iloc[i,:].isnull())
    
    if null_value != 0:
        iris.drop(iris.index[i])
        
# =============================================================================
# print(' Descriptive Statistics ')
# print(iris.describe())
# =============================================================================

print("Most Highly Correlated")
print(mosthighlycorrelated(iris,5))

print(' Covariance Matrix ')
corr_matrix(iris.iloc[:,0:-1], cols[0:-1])

print(' Pair plotting ')
pairplotting(iris)