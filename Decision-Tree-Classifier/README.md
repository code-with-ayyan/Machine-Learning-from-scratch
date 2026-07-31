

# 🌳 Decision Tree Classifier from Scratch

A complete implementation of the **Decision Tree Classification Algorithm** from scratch using only **Python** and **NumPy**, followed by a comparison with **Scikit-learn's DecisionTreeClassifier** on the **Titanic Dataset**.

---

## 📌 Project Overview

This project demonstrates how a Decision Tree Classifier works internally without relying on machine learning libraries.

The implementation includes:

* Building a Decision Tree from scratch
* Entropy calculation
* Information Gain calculation
* Best feature and threshold selection
* Recursive tree construction
* Prediction using tree traversal
* Performance comparison with Scikit-learn

---

## 📂 Project Structure

```text
Decision-Tree-Classifier/
│
├── DecisionTreeClassifier.py      
├── DecisionTreeClassifier.ipynb           
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Features

* Decision Tree built completely from scratch
* Recursive Tree Construction
* Entropy-based splitting
* Information Gain calculation
* Binary Tree Traversal for prediction
* Supports One-Hot Encoded datasets
* Comparison with Scikit-learn implementation

---

## 📊 Dataset

**Dataset:** Titanic Survival Dataset

Target Variable:

* Survived

  * 0 → Did Not Survive
  * 1 → Survived

The dataset was preprocessed before training:

* Missing values handled
* Categorical features encoded
* Numerical features cleaned

---

## 🧠 Algorithm

The scratch implementation follows these steps:

1. Calculate Entropy of the current dataset.
2. Compute Information Gain for every feature and threshold.
3. Select the split with the highest Information Gain.
4. Create left and right child nodes.
5. Repeat recursively until a stopping condition is reached.
6. Predict by traversing the generated tree.

---

## 📈 Model Performance

| Model                      | Accuracy | Precision | Recall | F1 Score |
| -------------------------- | -------: | --------: | -----: | -------: |
| Scratch Decision Tree      |   80.45% |    71.62% | 79.10% |   75.18% |
| Scikit-learn Decision Tree |   81.01% |    71.62% | 80.30% |   75.71% |

---

## 🚀 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn (for model evaluation comparison)

---

## 📚 Learning Outcomes

This project helped in understanding:

* Binary Tree data structure
* Recursive algorithms
* Entropy
* Information Gain
* Decision Tree learning process
* Feature selection
* Tree traversal for prediction

---


## 👨‍💻 Author

**Ayyan Ahmed**

Machine Learning & AI Student

Building Machine Learning algorithms completely from scratch to understand their mathematical foundations.
