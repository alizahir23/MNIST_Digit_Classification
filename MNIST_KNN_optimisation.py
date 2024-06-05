#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.interpolation import shift
import numpy as np
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = X[:30000], X[30000:40000], y[:30000], y[30000:40000]

# Define the range of k values to evaluate
k_range = list(range(1, 16))
k_weights = ['uniform', 'distance']
k_scores = []

# Evaluate the accuracy of KNN for different values of k using cross-validation
for k in k_range:
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_clf, X_train, y_train, cv=3, scoring='accuracy')
    k_scores.append(scores.mean())

# Plot the cross-validated accuracy for different values of k
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# Perform grid search to find the best parameters for KNN
param_grid = dict(n_neighbors=k_range, weights=k_weights)
knn_clf = KNeighborsClassifier()
knn_grid = GridSearchCV(knn_clf, param_grid, cv=3, scoring='accuracy', return_train_score=False)

# Fit the grid search model on the training data
knn_grid.fit(X_train, y_train)

# Display the results of the grid search
df = pd.DataFrame(knn_grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(df)
print(f"Best cross-validated score: {knn_grid.best_score_}")
print(f"Best parameters: {knn_grid.best_params_}")

# Function to shift images for data augmentation
def shift_image(image, dy, dx):
    image = image.reshape(28, 28)
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape(-1)

# Create augmented training data by shifting images
X_train_augmented = list(X_train)
y_train_augmented = list(y_train)

# Augment the training data by shifting images in four directions
for dy, dx in ((0, -1), (0, 1), (1, 0), (-1, 0)):
    for image, label in zip(X_train, y_train):
        img = shift_image(image, dy, dx)
        X_train_augmented.append(img)
        y_train_augmented.append(label)

# Shuffle the augmented dataset
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = np.array(X_train_augmented)[shuffle_idx]
y_train_augmented = np.array(y_train_augmented)[shuffle_idx]

# Perform grid search on the augmented training data
knn_clf_aug = KNeighborsClassifier()
knn_grid_aug = GridSearchCV(knn_clf_aug, param_grid, cv=3, scoring='accuracy', return_train_score=False)

# Fit the grid search model on the augmented training data
knn_grid_aug.fit(X_train_augmented, y_train_augmented)

# Display the results of the grid search on the augmented data
df_aug = pd.DataFrame(knn_grid_aug.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(df_aug)
print(f"Best cross-validated score (augmented data): {knn_grid_aug.best_score_}")
print(f"Best parameters (augmented data): {knn_grid_aug.best_params_}")

# Evaluate the accuracy of the best model on the test data
y_test_pred = knn_grid.predict(X_test)
score = accuracy_score(y_test, y_test_pred)
print(f"Accuracy score after training on original dataset: {score}")

# Evaluate the accuracy of the best augmented model on the test data
y_test_pred_aug = knn_grid_aug.predict(X_test)
score_aug = accuracy_score(y_test, y_test_pred_aug)
print(f"Accuracy score after training on augmented dataset: {score_aug}")
