# MNIST Digit Classification

This repository contains a Jupyter Notebook that demonstrates the process of training, evaluating, and comparing different classifiers on the MNIST dataset.

## Overview

The MNIST dataset is a well-known dataset of handwritten digits. This notebook includes steps to:

- Load and explore the dataset.
- Train various classifiers.
- Evaluate and compare their performance.

## Feature Exploration

The MNIST dataset contains 70,000 images of handwritten digits, each 28x28 pixels. The images are grayscale, and each pixel is represented by an integer from 0 to 255.

## Feature Engineering

No specific feature engineering was performed for this project since the raw pixel values are used directly to train the models.

## Preprocessing

- Splitting the dataset into training and test sets.
- Normalizing the pixel values by scaling them to the range [0, 1].

## Models and Evaluation

### Binary Classification: Identifying Digit '5'

We trained the following models for identifying whether a digit is a '5' or not.

1. **SGD Classifier**

   - **Cross-validation (cv=3) Accuracy**:
     ```
     array([0.95035, 0.96035, 0.9604 ])
     ```
   - The SGD Classifier performs well with high accuracy due to its ability to handle large datasets efficiently.

   - **Precision/Recall vs Threshold for SGD Classifier**:

   ![Precision/Recall vs Threshold](https://i.ibb.co/hWvQBNG/Screenshot-2024-06-04-at-1-41-13-PM.png)

   - **ROC Curve for SGD Classifier**:

![ROC Curve for SGD Classifier](https://i.ibb.co/bJnYpPT/Screenshot-2024-06-04-at-1-43-51-PM.png) - **AUC Score**: 0.9604938554008616 - The ROC curve shows a high true positive rate with a low false positive rate, indicating that the classifier is performing well in distinguishing between digit '5' and other digits.

2. **Dummy Classifier**

   - **Cross-validation (cv=3) Accuracy**:
     ```
     array([0.90965, 0.90965, 0.90965])
     ```
   - **Explanation**: The number of '5's in the dataset is very small compared to the rest of the dataset. Due to this imbalance, even a dummy classifier that always predicts the majority class (not '5') achieves around 90% accuracy.

3. **Random Forest Classifier**

   - **Metrics**:

     - F1 Score: 0.9274509803921569
     - ROC-AUC: 0.9983436731328145
     - Precision: 0.9897468089558485
     - Recall: 0.8725327430363402

   - **ROC Curve Comparison: Random Forest**:

     - The Random Forest Classifier performs exceptionally well with a high ROC-AUC score, indicating excellent performance in distinguishing between digit '5' and other digits.

   - **Precision/Recall Curve for SGD vs Random Forest**:

![Precision/Recall Curve for SGD vs Random Forest](https://i.ibb.co/2gJGY8x/Screenshot-2024-06-04-at-1-56-00-PM.png)

### Multiclass Classification: Identifying Digits 0-9

1. **SVM Classifier**
   - **Confusion Matrix**:

![enter image description here](https://i.ibb.co/r4f50jg/Screenshot-2024-06-04-at-2-08-06-PM.png)

- **Cross-validation (cv=3) Accuracy**:
  ```
  array([0.977 , 0.9738, 0.9739])
  ```

2. **SGD Classifier**
   - **Confusion Matrix**(Standardized Dataset):

![Confusion Matrix for SGD Classifier](https://i.ibb.co/VCy9n7k/Screenshot-2024-06-04-at-3-04-26-PM.png)

- **Cross-validation (cv=3) Accuracy**:
  - Default dataset:
    ```
    array([0.87365, 0.85835, 0.8689 ])
    ```
  - Standardized dataset:
    ```
    array([0.8983, 0.891 , 0.9018])
    ```

### Analysis of Results

The **SGD Classifier** performs well with high accuracy due to its ability to handle large datasets efficiently. The **Dummy Classifier** achieves around 90% accuracy because it always predicts the majority class (not '5'), which highlights the class imbalance in the dataset. The **Random Forest Classifier** performs exceptionally well with high precision, recall, and ROC-AUC scores, indicating its effectiveness in distinguishing between digit '5' and other digits. The confusion matrices for the **SVM Classifier** and **SGD Classifier** provide insights into the performance of these classifiers in the multiclass classification task.

### KNN Classification with Data Augmentation

In addition to the models mentioned above, we conducted an experiment with the KNN classifier.

1. **Initial KNN Experiment**:

   - We applied GridSearchCV to find the best hyperparameters for the KNN classifier. The best parameters found were:
     ```
     {'n_neighbors': 4, 'weights': 'distance'}
     ```
   - This resulted in an improved accuracy of **97%** on the MNIST dataset.

2. **Data Augmentation**:
   - To further enhance the performance, we applied data augmentation by shifting each image in all four directions (up, down, left, right) by one pixel. This increased the number of training images from 30,000 to 150,000 (30000 original + 4\*30000 augmented).
   - Due to computational limitations, we used only 40,000 images from the MNIST dataset (30,000 for training and 10,000 for testing).
   - After applying data augmentation, the accuracy of the KNN classifier improved to **98%**.

## Acknowledgments

- The MNIST dataset: [Yann LeCun, Corinna Cortes, and Chris Burges](http://yann.lecun.com/exdb/mnist/)
