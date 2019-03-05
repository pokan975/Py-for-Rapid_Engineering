#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# =============================================================================
# Function:
# Plot the confusion matrix.
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, classes, title):
# type y_true: array[int]
# type y_pred: array[int]
# type classes: array[str]
# type title: str

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # create plot instance
    fig, ax = plt.subplots(figsize = (5, 5))
    im = ax.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    ax.figure.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
    # show all ticks...
    ax.set(xticks = np.arange(cm.shape[1]),
           yticks = np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels = classes, yticklabels = classes,
           title = title,
           # set x/y axes labels
           ylabel = 'True label',
           xlabel = 'Predicted label')
    # remove grid
    ax.grid(b = False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")

    # Loop over data dimensions and create text annotations of TP/TN/FP/FN results at the center.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]),
                    ha = "center", va = "center",
                    fontsize = 14,
                    color = "white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()



# =============================================================================
# Main Code:
# Apply different classifiers on data to determine whether heart disease is present
# =============================================================================
# read data from csv file as dataframe
sonar = pd.read_csv("sonar_all_data_2.csv")
# create tick labels
axes_label = np.array(["rock", "mine"])

# drop observation data that contains null value
for i in range(sonar.values.shape[0]):
    null_value = sum(sonar.iloc[i,:].isnull())
    
    # drop the row that contains null section
    if null_value != 0:
        sonar.drop(sonar.index[i])


x = sonar.iloc[:,:-2].values
# desired result
y = sonar.iloc[:,-2].values

# split data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Preprocessing: standardly normalize data: x_std = (x - x_mean) / std
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)
x_std = sc.fit_transform(x)

# use PCA (singular value decomposition) to analyze the influence of components
from sklearn.decomposition import PCA
# only show the top 30 components that have the highest influence on predicting result
pca = PCA(n_components = 30)

# calc the PCA of test data & show result
pca.fit(x_std)
print("Top 30 Singular values:\n", pca.singular_values_)

# based on the PCA result, only extract the first n highest influential components as variables
x_train_pca = pca.transform(x_train_std)[:, 0:7]
x_test_pca = pca.transform(x_test_std)[:, 0:7]

# recombine data for later test
x_combined = np.vstack((x_train_pca, x_test_pca))
y_combined = np.hstack((y_train, y_test))

total_inst = len(y_combined)
print("\nNumber in test: ", total_inst)


# =============================================================================
# Perceptron classifier
# =============================================================================
from sklearn.linear_model import Perceptron

# create classifier object and set parameters
ppn = Perceptron(max_iter = 100, tol = 1e-3, eta0 = 0.01, fit_intercept = True, random_state = 0)
# train classifier using training set
ppn.fit(x_train_pca, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = ppn.predict(x_combined)
# show classification accuracy
print("\nPerceptron =======>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_combined, y_pred, axes_label, "Confusion matrix")


# =============================================================================
# Logistic Regression classifier
# =============================================================================
from sklearn.linear_model import LogisticRegression

# create classifier object and set parameters
lr = LogisticRegression(C = 1.0, solver = "liblinear", penalty = "l2", random_state = 0)
# train classifier using training set
lr.fit(x_train_pca, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = lr.predict(x_combined)
# show classification accuracy
print("\nLogistic Regression =======>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_combined, y_pred, axes_label, "Confusion matrix")

# =============================================================================
# Support Vector Machine classifier
# use SVM with with linear and nonlinear kernel function, respectively
# =============================================================================
from sklearn.svm import SVC

# create linear kernel SVM classifier object and set parameters
svm = SVC(kernel = "linear", class_weight = "balanced", C = 2.0, random_state = 0)
# train classifier using training set
svm.fit(x_train_pca, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = svm.predict(x_combined)
# show classification accuracy
print("\nSupport Vector Machine (Linear kernel) =======>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_combined, y_pred, axes_label, "Confusion matrix")


# create nonlinear kernel SVM classifier object and set parameters
lsvm = SVC(kernel = "rbf", gamma = 2.0, class_weight = "balanced", C = 2.0, random_state = 0)
# train classifier using training set
lsvm.fit(x_train_pca, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = lsvm.predict(x_combined)
# show classification accuracy
print("\nSupport Vector Machine (Nonlinear kernel) =======>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_combined, y_pred, axes_label, "Confusion matrix")

# =============================================================================
# Decision Tree classifier
# use normal decision tree and random forest classifier, respectively
# =============================================================================
from sklearn.tree import DecisionTreeClassifier

# create normal decision tree classifier object and set parameters
tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 4, random_state = 0)
# train classifier using training set
tree.fit(x_train_pca, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = tree.predict(x_combined)
# show classification accuracy
print("\nDecision Tree =======>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_combined, y_pred, axes_label, "Confusion matrix")


from sklearn.ensemble import RandomForestClassifier

# create random forest classifier object and set parameters
forest = RandomForestClassifier(criterion = "gini", n_estimators = 100, class_weight = "balanced", n_jobs = -1, random_state = 0)
# train classifier using training set
forest.fit(x_train_pca, y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = forest.predict(x_combined)
# show classification accuracy
print("\nRandom Forest =======>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_combined, y_pred, axes_label, "Confusion matrix")

# =============================================================================
# K Nearest Neighbor classifier
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

# create classifier object and set parameters
knn = KNeighborsClassifier(n_neighbors = 20, weights = "distance", algorithm = "auto", p = 4, metric = "minkowski")
# train classifier using training set
knn.fit(x_train_pca,y_train)

# testing classifier using the whole dataset (training + testing set)
y_pred = knn.predict(x_combined)
# show classification accuracy
print("\nK Nearest Neighbor =======>")
print("Misclassified/total samples: %d/%d" % ((y_combined != y_pred).sum(), total_inst))
print("Accuracy: %.2f" % accuracy_score(y_combined, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y_combined, y_pred, axes_label, "Confusion matrix")