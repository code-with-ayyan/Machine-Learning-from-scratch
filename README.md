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
| Random Forest Classifier|            ⏳       |            ⏳            | Coming Soon |
| Random Forest Regressor |            ⏳       |            ⏳            | Coming Soon |
| SVM                    |            ✅        |            ✅            | ✅ Completed |
| Naive Bayes (Gaussian) |            ✅        |            ✅            | ✅ Completed |

---

## Practice, Experimental Modules & Ensemble Learning

```text
├── Practice/
```

The **`Practice/`** directory contains practical experiments and end-to-end workflows focused on **model evaluation, preprocessing pipelines, hyperparameter tuning, cross-validation, and ensemble learning** using real-world datasets such as the **Mobile Price Classification** dataset.

Key concepts covered in this directory:

### **Pipeline Basics**

Worked with the basics of **Scikit-learn Pipelines** to combine preprocessing steps and machine learning models into a single workflow, making the training and prediction process more organized and helping avoid data leakage during cross-validation and hyperparameter tuning.

### **Cross-Validation**

**K-Fold** cross-validation technique is used to evaluate model performance across different data splits and assess model stability and generalization.

### **Hyperparameter Tuning**

Systematically searching for optimal model configurations using **GridSearchCV** and **RandomizedSearchCV**.

### **Model Optimization**

Evaluating performance improvements by comparing **baseline models** with **hyperparameter-tuned estimators**, including models such as **Support Vector Machines** and **tree-based models**.

### **Ensemble Learning**

#### **1. Stacking**

Implemented using **Scikit-learn's `StackingClassifier`** with three base estimators:

* **LogisticRegression**
* **DecisionTreeClassifier**
* **SVC**

The final estimator is **LogisticRegression**.

Although Stacking did not improve the model's performance on the **Mobile Price Classification** dataset, implementing it was important for understanding how multiple different models can work together through a **meta-model** to make the final prediction. This is a valuable technique in real-world machine learning workflows.

#### **2. Bagging — Random Forest**

Implemented using **Scikit-learn's `RandomForestClassifier`** to understand the concept of **Bagging (Bootstrap Aggregating)**.

Bagging trains multiple models on different bootstrap samples of the training data and combines their predictions. In classification, the final prediction is generally determined through **majority voting**.

**Random Forest** extends this idea by combining multiple decision trees while also introducing randomness in feature selection, which helps improve model diversity, robustness, and generalization.
 


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
