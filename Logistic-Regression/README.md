# 🚢 Logistic Regression from Scratch

## 📖 Overview

This project implements **Logistic Regression from scratch using NumPy** without relying on scikit-learn's `LogisticRegression` class. The primary objective is to understand the mathematical intuition and implementation of Logistic Regression, including the Sigmoid Function, Log Loss, Gradient Descent, parameter updates, and binary classification.

The Titanic dataset is used only to train and evaluate the custom implementation.

---

# 🎯 Objective

The main focus of this project is to:

* Implement Logistic Regression from scratch
* Understand the mathematics behind binary classification
* Learn how Gradient Descent updates model parameters
* Build a reusable Logistic Regression class using NumPy
* Evaluate the custom model on a real-world dataset

---

# 📂 Dataset

**Dataset:** Titanic (Seaborn)

The dataset contains passenger information such as:

* Passenger Class
* Sex
* Age
* Fare
* Embarked
* Family Information
* Survival Status (Target Variable)

**Target Variable**

* `survived`

  * **0** → Did Not Survive
  * **1** → Survived

---

# 🔄 Project Workflow

1. Load Dataset
2. Basic Data Cleaning
3. Encode Categorical Features
4. Train-Test Split
5. Feature Scaling
6. Implement Logistic Regression from Scratch
7. Train the Model using Gradient Descent
8. Make Predictions
9. Evaluate Model Performance

---

# 🧠 Scratch Implementation

The custom implementation includes:

* Logistic Regression Class
* Sigmoid Function
* Gradient Descent
* Weight & Bias Initialization
* Parameter Updates
* Prediction Function
* Binary Classification

---

# 📊 Model Evaluation

The custom model achieved the following performance on the Titanic dataset:

| Metric        | Score      |
| ------------- | ---------- |
| **Accuracy**  | **81.01%** |
| **Precision** | **72.97%** |
| **Recall**    | **79.41%** |
| **F1 Score**  | **76.06%** |

---

# 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn (Dataset, Preprocessing & Evaluation)
* Jupyter Notebook

---

# 📁 Project Structure

```text
Logistic-Regression/
│
├── Logistic_Regression.ipynb
├── Logistic_Regression.py
├── README.md
├── requirements.txt
```

---

# 🎓 Learning Outcomes

Through this project, I learned:

* Binary Classification
* Logistic Regression Mathematics
* Sigmoid Function
* Log Loss
* Gradient Descent
* Parameter Optimization
* NumPy-based ML Implementation
* Model Evaluation Metrics

---

## Model Comparison

To validate the correctness of the custom implementation, the model was also compared with **scikit-learn's `LogisticRegression`** implementation using the same preprocessed dataset and train-test split.

| Metric        | Custom Model | Scikit-learn Model |
| ------------- | -----------: | -----------------: |
| **Accuracy**  |   **81.01%** |         **81.56%** |
| **Precision** |   **72.97%** |         **74.32%** |
| **Recall**    |   **79.41%** |         **79.71%** |
| **F1 Score**  |   **76.06%** |         **76.92%** |

The custom Logistic Regression model achieved performance very close to the scikit-learn implementation across all evaluation metrics. While the **scikit-learn** model achieved slightly higher Accuracy, Precision, Recall, and F1 Score, the results confirm that the scratch implementation correctly learns the decision boundary using **Gradient Descent** and the **Sigmoid Function**.

This comparison validates the correctness of the custom implementation while highlighting the optimization and numerical efficiency provided by production-ready machine learning libraries such as **scikit-learn**.


---

# 👨‍💻 Author

**Ayyan Ahmed**

Machine Learning & AI Enthusiast
