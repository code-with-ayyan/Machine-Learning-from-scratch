# Gaussian Naive Bayes From Scratch – Titanic Survival Prediction

## Overview

This project demonstrates the implementation of **Gaussian Naive Bayes from Scratch** using **NumPy** without relying on scikit-learn's `GaussianNB` class. The algorithm is implemented by manually computing **prior probabilities**, **feature-wise mean and variance**, and applying the **Gaussian Probability Density Function (PDF)** together with **Bayes' Theorem** for binary classification.

The project also compares the scratch implementation with scikit-learn's implementation to validate the correctness of the custom-built model.

---

# Dataset

The project uses the **Titanic Survival** dataset.

**Target Variable**

- Survived

**Features**

- Passenger Class
- Sex
- Age
- Fare
- SibSp
- Parch
- Embarked

---

# Project Workflow

- Data Loading
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Categorical Encoding
- Feature Scaling
- Train-Test Split
- Gaussian Naive Bayes Implementation from Scratch
- Model Evaluation
- Comparison with Scikit-learn

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# Project Structure

```text
Gaussian_Naive_Bayes/
│
├── Gaussian_NB.py
├── KNN_classification.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Model Evaluation

| Metric | Custom Model | Scikit-learn Model |
|---------|-------------:|-------------------:|
| Accuracy | **78.77%** | **78.77%** |
| Precision | **81.08%** | **81.08%** |
| Recall | **71.43%** | **71.43%** |
| F1 Score | **75.95%** | **75.95%** |

---

# Model Comparison

The custom Gaussian Naive Bayes implementation produced **identical performance** to scikit-learn's implementation across all evaluation metrics.

This confirms that the scratch-built algorithm correctly reproduces the behavior of the standard `GaussianNB` model while providing a deeper understanding of the mathematics behind Naive Bayes classification.

---

# Features of this Project

- Gaussian Naive Bayes implemented completely from scratch
- Manual calculation of Prior Probabilities
- Manual calculation of Mean
- Manual calculation of Variance
- Gaussian Probability Density Function (PDF)
- Bayes' Theorem based prediction
- Log Probability implementation for numerical stability
- Comparison with Scikit-learn implementation

---

# Learning Outcomes

Through this project, I learned:

- Bayes' Theorem
- Gaussian Distribution
- Prior Probability
- Posterior Probability
- Likelihood Estimation
- Classification Model Evaluation
- Building a Machine Learning algorithm completely from scratch

---

# Future Improvements

- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Complement Naive Bayes
- Categorical Naive Bayes

---

# Author

**Ayyan Ahmed**