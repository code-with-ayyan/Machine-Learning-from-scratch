# Support Vector Machine (SVM) — From Scratch

A Support Vector Machine (SVM) classifier implemented **from scratch using NumPy** and compared with the standard **Scikit-learn SVM** implementation.

The main goal of this project is to understand how SVM works internally instead of directly relying on a machine-learning library.

---

## 📌 Project Overview

In this project, a binary SVM classifier was implemented from scratch.

The custom implementation includes:

* Support Vector Machine classification
* SMO (Sequential Minimal Optimization) training
* Kernel functions
* RBF kernel
* Linear kernel
* Polynomial kernel
* Sigmoid kernel
* C parameter
* Gamma parameter
* Class weighting
* Decision function
* Prediction
* Support vector selection

The custom model was then tested on the same dataset used with Scikit-learn's SVM model.

---

## 🧠 How SVM Works

SVM tries to find the best decision boundary that separates two classes.

The main idea is to find a hyperplane that maximizes the margin between the classes.

The data points closest to the decision boundary are called **Support Vectors**.

These support vectors are extremely important because they determine the position of the decision boundary.

For non-linear data, SVM can use **kernel functions** to transform the data into a space where it can be separated more effectively.

---

## ⚙️ Implementation

The custom SVM was implemented using:

```text
Python
NumPy
```

The training process uses **Sequential Minimal Optimization (SMO)** to optimize the SVM's alpha values.

The implementation also calculates the kernel matrix and uses support vectors during prediction.

---

## 🔑 Supported Kernels

### Linear Kernel

Used when the data can be separated using a straight decision boundary.

### Polynomial Kernel

Allows more complex decision boundaries using polynomial transformations.

### RBF Kernel

The RBF kernel can handle non-linear relationships between features.

### Sigmoid Kernel

Uses a sigmoid function to calculate similarity between samples.

---

## 🧪 Model Comparison

The custom SVM was compared with the Scikit-learn SVM using the same:

* Training data
* Testing data
* Feature scaling
* Kernel
* Model parameters
* Evaluation metrics

### Results

| Model                |   Accuracy |  Precision |     Recall |   F1 Score |
| -------------------- | ---------: | ---------: | ---------: | ---------: |
| **Custom SVM**       | **81.56%** | **72.97%** | **80.60%** | **76.60%** |
| **Scikit-learn SVM** | **81.56%** | **72.97%** | **80.60%** | **76.60%** |

---

## 📊 Evaluation Metrics

### Accuracy

Measures the percentage of total predictions that were correct.

### Precision

Measures how many predicted positive samples were actually positive.

### Recall

Measures how many actual positive samples were correctly identified.

### F1 Score

Combines precision and recall into a single metric.

---

## ✅ Conclusion

The custom SVM implementation produced **exactly the same evaluation results** as the Scikit-learn SVM on the test dataset.

Both models achieved:

* **81.56% Accuracy**
* **72.97% Precision**
* **80.60% Recall**
* **76.60% F1 Score**

This confirms that the custom implementation is successfully working and producing the same predictions as the Scikit-learn implementation on this dataset.

The project helped demonstrate how an SVM classifier can be implemented from scratch using NumPy and how its results can be validated against a standard machine-learning library.

---

## 📁 Project Structure

```
SVM/
└── classification/
    ├── SVC.py
    ├── SVM_Classification.ipynb
    ├── README.md
    ├── .gitignore
    └── requirements.txt

```

---

## 🎯 Learning Goal

This project is part of a **Machine Learning From Scratch** journey focused on understanding the internal working of machine-learning algorithms before using high-level implementations.


