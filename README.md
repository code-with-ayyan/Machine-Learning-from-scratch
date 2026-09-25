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
├── Naive-bayes/
├── Decision-Tree-Classifier/
├── Decision-Tree-Regressor/
├── SVM/
├── Practice/
├── ensemble-learning-scratch/
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
| Decision Tree Classifier|            ✅       |            ✅            | ✅ Completed |
| Decision Tree Regressor |            ✅       |            ✅            | ✅ Completed |
| ensemble learning techniques|       ⏳        |            ⏳          | In Progress 📈|
| SVM (Classifier)       |            ✅        |            ✅            | ✅ Completed |
| Naive Bayes (Gaussian) |            ✅        |            ✅            | ✅ Completed |

---

## Ensemble Learning Progress

```text
├── ensemble-learning-scratch/
```

| Technique          | Scratch Implementation | Scikit-learn Comparison |    Status   |
| ------------------ | :--------------------: | :---------------------: | :---------: |
| Bagging Classifier |            ✅           |            ✅            | ✅ Completed |
| Bagging Regressor  |            ✅           |            ✅            | ✅ Completed |
| Random Forest Classifier|            ✅       |            ✅            | ✅ Completed |
| Random Forest Regressor |            ✅       |            ✅            | ✅ Completed |
| AdaBoost Classifier          |            ✅       |            ✅            | ✅ Completed |
| AdaBoost Regressor          |            ✅       |            ✅            | ✅ Completed |
| Gradient Boosting  |            ⏳           |            ⏳            |  ⏳ Pending  |
| Stacking           |            ⏳           |            ⏳            |  ⏳ Pending  |

---

# 🧪 Practice — Model Tuning & Unsupervised Learning

The **`Practice/`** directory contains practical experiments covering **model evaluation, preprocessing, hyperparameter tuning, ensemble learning, and unsupervised learning** using real-world and built-in datasets.

## 📌 Topics Covered

### 🔧 Model Tuning & Ensemble Learning

`model_tuning_techniques_ensemble_learning.ipynb` covers:

- Scikit-learn Pipelines
- K-Fold Cross-Validation
- GridSearchCV & RandomizedSearchCV
- Baseline vs tuned models
- Stacking
- Bagging with Random Forest
- Boosting with AdaBoost, Gradient Boosting & XGBoost

The experiments focus on understanding practical model optimization and ensemble learning workflows.


### 🔍 Unsupervised Learning

The **`unsupervised_practice/`** directory contains practical experiments with unsupervised learning algorithms.

#### K-Means Clustering

Implemented **K-Means clustering** on the Iris dataset, covering:

- Feature scaling using `StandardScaler`
- Selecting the number of clusters using the **Elbow Method**
- Evaluating cluster quality using **Silhouette Score**
- Training K-Means with `K=3`
- Visualizing clusters and centroids
- Comparing discovered clusters with the actual Iris class labels
- Evaluating clustering using **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**
- Testing the trained model on new unseen samples and mapping clusters to Iris species

Results from the experiment:

- **ARI:** 0.7302
- **NMI:** 0.7582

## 📂 Structure

```text
Practice/
│
├── model_tuning_techniques_ensemble_learning.ipynb
│
└── unsupervised_practice/
    ├── KMeans.ipynb
    ├── PCA.ipynb                 # Coming soon
    └── DBSCAN.ipynb              # Coming soon

```
This directory is mainly for hands-on experimentation and strengthening practical understanding of machine learning techniques before applying them to larger projects.


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
