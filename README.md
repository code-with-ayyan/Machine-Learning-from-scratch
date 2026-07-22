# Machine Learning From Scratch

## Overview

**Machine Learning From Scratch** is a collection of machine learning algorithms implemented manually using **Python and NumPy**. The primary goal of this repository is to understand the mathematical foundations of machine learning by implementing algorithms from first principles instead of relying solely on high-level libraries.

Each algorithm is developed step by step, including data preprocessing, mathematical derivations, model training, evaluation, visualization, and comparison with the equivalent **scikit-learn** implementation.

---

## Goals

* Learn Machine Learning fundamentals
* Understand the mathematics behind algorithms
* Implement machine learning algorithms from scratch
* Compare custom implementations with scikit-learn models
* Build practical machine learning projects
* Strengthen problem-solving and interview skills

---

## Repository Structure

```text
MachineLearningFromScratch/

├── Linear-Regression/
├── Logistic-Regression/
├── KNN-Classifier/
├── KNN-Regression/
└── README.md
```

---

## Current Progress

| Algorithm           | Scratch Implementation | Scikit-learn Comparison |    Status   |
| ------------------- | :--------------------: | :---------------------: | :---------: |
| Linear Regression   |            ✅           |            ✅            | ✅ Completed |
| Logistic Regression |            ✅           |            ✅            | ✅ Completed |
| KNN Classification  |            ✅           |            ✅            | ✅ Completed |
| KNN Regression      |            ✅           |            ✅            | ✅ Completed |
| Decision Tree       |            ⏳           |            ⏳            | Coming Soon |
| Random Forest       |            ⏳           |            ⏳            | Coming Soon |
| SVM                 |            ⏳           |            ⏳            | Coming Soon |
| Naive Bayes         |            ⏳           |            ⏳            | Coming Soon |
| K-Means             |            ⏳           |            ⏳            | Coming Soon |
| PCA                 |            ⏳           |            ⏳            | Coming Soon |

---

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn (for validation and performance comparison)
* Jupyter Notebook

---

## Learning Focus

Every algorithm in this repository includes:

* Mathematical intuition
* Step-by-step derivation
* From-scratch implementation
* Gradient-based optimization (where applicable)
* Data preprocessing
* Model evaluation
* Visualization
* Performance comparison with the equivalent scikit-learn implementation
* Well-documented Jupyter notebooks

---

## Validation Strategy

Every custom implementation is evaluated and compared against the corresponding **scikit-learn** model using the same dataset, preprocessing pipeline, and train-test split.

The comparison includes standard evaluation metrics such as:

* Accuracy (Classification)
* Precision
* Recall
* F1 Score
* R² Score (Regression)
* Adjusted R²
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

This comparison helps verify the correctness of each scratch implementation while demonstrating the performance differences between educational implementations and highly optimized production-grade machine learning libraries.

---

## Purpose

This repository is designed for students, beginners, and aspiring Machine Learning Engineers who want to build a strong understanding of how machine learning algorithms work internally.

Rather than only learning how to use machine learning libraries, the objective is to understand **why** the algorithms work by implementing them from scratch, validating them against industry-standard implementations, and applying them to real-world datasets.

---

## Author

**Ayyan Ahmed**

Machine Learning & AI Enthusiast
