#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

############################################################
# Problem 2                                                #
############################################################

# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
# read data from csv file as dataframe
heart = pd.read_csv("heart1.csv")
#extract variable names
cols = heart.columns

# drop observation data that contains null value
for i in range(heart.values.shape[0]):
    null_value = sum(heart.iloc[i,:].isnull())
    
    # drop the row that contains null section
    if null_value != 0:
        heart.drop(heart.index[i])
        

# based on the statistical result from Project1_A.py,
# pick variables that are the most highly correlated with the result for detection
var_highlycorr = ["thal", "nmvcf", "eia", "mhr", "opst", "cpt"]

X = heart.loc[:,[*var_highlycorr]].values
# desired result
Y = heart.loc[:,"a1p2"].values

# split data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 50)

# Preprocessing: standardly normalize data: x_std = (x - x_mean) / std
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

# recombine data for later test
x_combined = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

total_inst = len(y_combined)

print("Number in test: ", total_inst)

# =============================================================================
# Perceptron classifier
# =============================================================================
from sklearn.linear_model import Perceptron

# create classifier object and set parameters
ppn = Perceptron(max_iter = 100, tol = 1e-3, eta0 = 0.01, fit_intercept = True, random_state = 42)
# train classifier using training set
ppn.fit(x_train_std, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = ppn.predict(x_combined)
# show classification accuracy
print("\nPerceptron ==>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))


# =============================================================================
# Logistic Regression classifier
# =============================================================================
from sklearn.linear_model import LogisticRegression

# create classifier object and set parameters
lr = LogisticRegression(C = 1.0, solver = "liblinear", penalty = "l2", random_state = 42)
# train classifier using training set
lr.fit(x_train_std, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = lr.predict(x_combined)
# show classification accuracy
print("\nLogistic Regression ==>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))


# =============================================================================
# Support Vector Machine classifier
# use SVM with with linear and nonlinear kernel function, respectively
# =============================================================================
from sklearn.svm import SVC

# create linear kernel SVM classifier object and set parameters
svm = SVC(kernel = "linear", class_weight = "balanced", C = 2.0, random_state = 42)
# train classifier using training set
svm.fit(x_train_std, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = svm.predict(x_combined)
# show classification accuracy
print("\nSupport Vector Machine (Linear kernel) ==>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))


# create nonlinear kernel SVM classifier object and set parameters
lsvm = SVC(kernel = "rbf", gamma = 2.0, class_weight = "balanced", C = 2.0, random_state = 42)
# train classifier using training set
lsvm.fit(x_train_std, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = lsvm.predict(x_combined)
# show classification accuracy
print("\nSupport Vector Machine (Nonlinear kernel) ==>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))


# =============================================================================
# Decision Tree classifier
# use normal decision tree and random forest classifier, respectively
# =============================================================================
from sklearn.tree import DecisionTreeClassifier

# create normal decision tree classifier object and set parameters
tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 4, random_state = 42)
# train classifier using training set
tree.fit(x_train_std, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = tree.predict(x_combined)
# show classification accuracy
print("\nDecision Tree ==>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))


from sklearn.ensemble import RandomForestClassifier

# create random forest classifier object and set parameters
forest = RandomForestClassifier(criterion = "gini", n_estimators = 100, class_weight = "balanced", n_jobs = -1, random_state = 42)
# train classifier using training set
forest.fit(x_train_std, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = forest.predict(x_combined)
# show classification accuracy
print("\nRandom Forest ==>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))


# =============================================================================
# K Nearest Neighbor classifier
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

# create classifier object and set parameters
knn = KNeighborsClassifier(n_neighbors = 20, weights = "distance", algorithm = "auto", p = 4, metric = "minkowski")
# train classifier using training set
knn.fit(x_train_std,y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = knn.predict(x_combined)
# show classification accuracy
print("\nK Nearest Neighbor ==>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))






