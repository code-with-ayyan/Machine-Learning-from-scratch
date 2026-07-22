# K-Nearest Neighbors (KNN) Classification From Scratch

A complete implementation of the **K-Nearest Neighbors (KNN) Classification** algorithm from scratch using **NumPy**, without relying on scikit-learn's `KNeighborsClassifier`. This project demonstrates how KNN classifies unseen samples by computing **Euclidean distances**, selecting the **K nearest neighbors**, and predicting the majority class through **majority voting**.

---

## Project Overview

This project implements the KNN Classification algorithm completely from scratch and evaluates it on the **Titanic Survival Prediction** dataset.

Unlike many machine learning algorithms, KNN is an **instance-based (lazy learning)** algorithm. It does not learn explicit model parameters during training. Instead, it stores the training data and predicts the class of a new sample by examining its nearest neighbors.

The primary objective of this project is to understand the internal working of KNN rather than simply using a machine learning library.

---

## Features

* KNN Classification implemented completely from scratch
* Euclidean Distance calculation using NumPy
* K Nearest Neighbor search
* Majority Voting classification
* Data preprocessing and feature engineering
* Feature scaling using StandardScaler
* Model evaluation using classification metrics
* Comparison with scikit-learn implementation

---

## Dataset

**Titanic Survival Prediction Dataset**

### Target Variable

* `Survived`

### Features Used

* Pclass
* Sex
* Age
* SibSp
* Parch
* Fare
* Embarked

---

## Project Structure

```text
KNN-Classification/

├── KNN_Classifier.py
├── KNN_classification.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Algorithm Workflow

1. Load the Titanic dataset
2. Handle missing values
3. Encode categorical variables
4. Scale numerical features using StandardScaler
5. Store the training dataset
6. Compute Euclidean distance between the query point and every training sample
7. Select the **K nearest neighbors**
8. Predict the majority class using voting
9. Evaluate the model performance

---

## Model Performance

The model was evaluated using different values of **K**. The best overall performance was achieved with **K = 7**.

| Metric    |      Value |
| --------- | ---------: |
| Accuracy  | **82.68%** |
| Precision | **72.97%** |
| Recall    | **83.08%** |
| F1 Score  | **77.70%** |

The scratch implementation achieved performance comparable to the scikit-learn implementation, confirming the correctness of the algorithm.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn (only for preprocessing, evaluation, and comparison)

---

## Model Comparison

To verify the correctness of the scratch implementation, the model was compared with **scikit-learn's `KNeighborsClassifier`** using the same preprocessed dataset, train-test split, and **K = 7**.

| Metric        | Custom Model | Scikit-learn Model |
| ------------- | -----------: | -----------------: |
| **Accuracy**  |   **82.68%** |         **82.68%** |
| **Precision** |   **72.97%** |         **72.97%** |
| **Recall**    |   **83.08%** |         **83.08%** |
| **F1 Score**  |   **77.70%** |         **77.70%** |

The custom implementation achieved **identical performance** to scikit-learn's implementation across all evaluation metrics. This confirms that the scratch implementation correctly reproduces the behavior of the standard `KNeighborsClassifier` and validates the correctness of the algorithm implementation.

---


## Learning Outcomes

Through this project, I learned:

* How KNN Classification works internally
* Euclidean Distance calculation
* Nearest Neighbor search
* Majority Voting
* Classification model evaluation
* Feature scaling for distance-based algorithms
* Comparing a scratch implementation with a production-ready library


---

## Author

**Ayyan Ahmed**

Aspiring AI/ML Engineer | Building Machine Learning algorithms from scratch to develop a strong mathematical and practical understanding of AI and Machine Learning.
