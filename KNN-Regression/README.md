# K-Nearest Neighbors (KNN) Regression From Scratch

A complete implementation of the **K-Nearest Neighbors (KNN) Regression** algorithm from scratch using **NumPy**, without relying on scikit-learn's `KNeighborsRegressor`. This project demonstrates how KNN performs regression by computing Euclidean distances, selecting the nearest neighbors, and predicting continuous values using the average of their target values.

---

## Project Overview

This project implements the KNN Regression algorithm from scratch and evaluates it on the **Medical Insurance Charges** dataset. Every step of the machine learning workflow is included, from data preprocessing to model evaluation.

Unlike parametric models such as Linear Regression, KNN is an **instance-based (lazy learning)** algorithm that does not learn explicit model parameters. Instead, it stores the training data and makes predictions by finding the **K nearest training samples**.

---

## Features

* KNN Regression implemented completely from scratch
* Euclidean Distance calculation using NumPy
* K Nearest Neighbor search
* Prediction using the mean of nearest target values
* Data preprocessing and feature engineering
* Feature scaling using StandardScaler
* Model evaluation using regression metrics
* Comparison with scikit-learn implementation

---

## Dataset

**Medical Insurance Charges Dataset**

**Target Variable**

* `charges`

**Features Used**

* age
* bmi
* children
* is_smoker
* is_female
* bmi_category_Obese

---

## Project Structure

```text
KNN-Regression/

├── KNN_Regression.py
├── insurance.ipynb
├── insurance.csv
├── images/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
├── README.md
└── requirements.txt
```

---

## Algorithm Workflow

1. Load and preprocess the dataset
2. Scale numerical features using StandardScaler
3. Store the training dataset
4. Compute Euclidean distance between the query point and all training samples
5. Select the **K nearest neighbors**
6. Predict the target value by averaging their target values
7. Evaluate the model performance

---

## Model Performance

| Metric      |      Value |
| ----------- | ---------: |
| R² Score    | **77.75%** |
| Adjusted R² | **77.25%** |

The scratch implementation achieves performance very close to the scikit-learn implementation, confirming the correctness of the algorithm.

---

## Visualizations

### Correlation Heatmap

Shows the relationships between features in the dataset.

### Actual vs Predicted

Compares the model predictions with the true insurance charges.

### Feature Importance

Displays the selected features used during training.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn (only for preprocessing and comparison)

---

## Model Comparison

To validate the correctness of the scratch implementation, the model was compared with **scikit-learn's `KNeighborsRegressor`** using the same preprocessed dataset, train-test split, and **K = 5**.

| Metric          | Custom Model | Scikit-learn Model |
| --------------- | -----------: | -----------------: |
| **R² Score**    |   **77.75%** |         **77.76%** |
| **Adjusted R²** |   **77.24%** |         **77.25%** |
| **RMSE**        |  **6393.56** |        **6392.69** |
| **MAE**         |  **3256.05** |        **3251.63** |

The custom implementation achieved performance **almost identical** to scikit-learn's implementation across all evaluation metrics. The minimal differences are due to numerical precision and implementation-level optimizations. These results validate the correctness of the scratch implementation while demonstrating that it closely reproduces the behavior of the standard `KNeighborsRegressor`.

---

## Learning Outcomes

Through this project, I learned:

* How KNN Regression works internally
* Euclidean Distance calculation
* Neighbor selection using distance sorting
* Instance-based learning
* Regression model evaluation
* Data preprocessing and feature scaling
* Comparing a scratch implementation with a production-ready library

---

## Author

**Ayyan Ahmed**

Aspiring AI/ML Engineer | Building Machine Learning algorithms from scratch to understand their mathematical foundations.