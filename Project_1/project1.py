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

# =============================================================================
# Main Code:


# =============================================================================
heart = pd.read_csv('heart1.csv')

cols = heart.columns


# get rid of observations containing null value
for i in range(heart.values.shape[0]):
    null_value = sum(heart.iloc[i,:].isnull())
    
    if null_value != 0:
        heart.drop(heart.index[i])
        
# =============================================================================
# print(' Descriptive Statistics ')
# print(iris.describe())
# =============================================================================

print("Most Highly Correlated")
print(mosthighlycorrelated(heart,10))

# =============================================================================
# print(' Covariance Matrix ')
# corr_matrix(iris.iloc[:,0:-1], cols[0:-1])
# 
# print(' Pair plotting ')
# pairplotting(iris)
# =============================================================================

# observation data
X = heart.iloc[:,[7,8,11,12]].values
# target
Y = heart.iloc[:,13].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 50)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  # do standard normalization on data: z = (x - u) / std
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

# =============================================================================
# Perceptron
# =============================================================================
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter = 50, tol = 1e-3, eta0 = 0.001, fit_intercept = True)
ppn.fit(x_train_std, y_train)

print('Number in test ', len(y_test))
y_pred = ppn.predict(x_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# =============================================================================
# Logistic Regression
# =============================================================================
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 50.0, solver = "liblinear")
lr.fit(x_train_std, y_train)

print('Number in test ', len(y_test))
y_pred = lr.predict(x_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# =============================================================================
# Support Vector Machine
# =============================================================================
from sklearn.svm import SVC

svm = SVC(kernel = 'rbf', gamma = 0.2, C = 2.0)
svm.fit(x_train_std, y_train)

print('Number in test ', len(y_test))
y_pred = svm.predict(x_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))




