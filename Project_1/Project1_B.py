#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# =============================================================================
# Main Code:


# =============================================================================
heart = pd.read_csv('heart1.csv')

cols = heart.columns

# wipe out observation data containing null value
for i in range(heart.values.shape[0]):
    null_value = sum(heart.iloc[i,:].isnull())
    
    if null_value != 0:
        heart.drop(heart.index[i])
        

# observation data
X = heart.loc[:,["thal", "nmvcf", "eia", "mhr"]].values
# target
Y = heart.loc[:,"a1p2"].values

# split data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 50)

# apply standard normalization to data: x_std = (x - x_mean) / std
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

x_combined = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

total_inst = len(y_combined)

print('Number in test ', total_inst)

# =============================================================================
# Perceptron
# =============================================================================
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter = 100, tol = 1e-3, eta0 = 0.001, fit_intercept = True)
ppn.fit(x_train_std, y_train)

y_pred = ppn.predict(x_combined)
print("\nPerceptron ==>")
print('Misclassified samples: %d/%d' % ((y_combined != y_pred).sum(), total_inst))
print('Accuracy: %.2f' % accuracy_score(y_combined, y_pred))


# =============================================================================
# Logistic Regression
# =============================================================================
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 50.0, solver = "liblinear")
lr.fit(x_train_std, y_train)

y_pred = lr.predict(x_combined)
print("\nLogistic Regression ==>")
print('Misclassified samples: %d/%d' % ((y_combined != y_pred).sum(), total_inst))
print('Accuracy: %.2f' % accuracy_score(y_combined, y_pred))


# =============================================================================
# Support Vector Machine
# =============================================================================
from sklearn.svm import SVC

svm = SVC(kernel = 'rbf', gamma = 0.2, C = 2.0)
svm.fit(x_train_std, y_train)

y_pred = svm.predict(x_combined)
print("\nSupport Vector Machine (Linear) ==>")
print('Misclassified samples: %d/%d' % ((y_combined != y_pred).sum(), total_inst))
print('Accuracy: %.2f' % accuracy_score(y_combined, y_pred))


from sklearn.svm import LinearSVC

lsvm = LinearSVC(penalty = "l2", loss = "hinge", tol = 1e-3, C = 2.0, max_iter = 1000)
lsvm.fit(x_train_std, y_train)

y_pred = lsvm.predict(x_combined)
print("\nSupport Vector Machine (Nonlinear) ==>")
print('Misclassified samples: %d/%d' % ((y_combined != y_pred).sum(), total_inst))
print('Accuracy: %.2f' % accuracy_score(y_combined, y_pred))


# =============================================================================
# Decision Tree
# =============================================================================
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 4, random_state = 0)
tree.fit(x_train_std, y_train)

y_pred = tree.predict(x_combined)
print("\nDecision Tree ==>")
print('Misclassified samples: %d/%d' % ((y_combined != y_pred).sum(), total_inst))
print('Accuracy: %.2f' % accuracy_score(y_combined, y_pred))


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion = 'entropy', n_estimators = 10, random_state = 0, n_jobs = 2)
forest.fit(x_train_std, y_train)

y_pred = forest.predict(x_combined)
print("\nRandom Forest ==>")
print('Misclassified samples: %d/%d' % ((y_combined != y_pred).sum(), total_inst))
print('Accuracy: %.2f' % accuracy_score(y_combined, y_pred))


# =============================================================================
# K Nearest Neighbor
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10, p = 2, metric = 'minkowski')
knn.fit(x_train_std,y_train)

y_pred = knn.predict(x_combined)
print("\nK Nearest Neighbor ==>")
print('Misclassified samples: %d/%d' % ((y_combined != y_pred).sum(), total_inst))
print('Accuracy: %.2f' % accuracy_score(y_combined, y_pred))






