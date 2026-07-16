# Linear Regression From Scratch – Insurance Charges Prediction

## Overview

This project demonstrates the implementation of **Linear Regression from Scratch** using **NumPy**. Instead of using scikit-learn's built-in `LinearRegression` model, the algorithm is implemented manually using **Gradient Descent** to learn the optimal weights and bias.

The project also includes complete data preprocessing, feature engineering, model evaluation, and visualization.

---

## Dataset

The project uses the **Medical Insurance Charges** dataset.

Target Variable:

* `charges`

Features include:

* Age
* Gender
* BMI
* Number of Children
* Smoking Status
* Region

---

## Project Workflow

* Data Loading
* Exploratory Data Analysis (EDA)
* Data Cleaning
* Feature Engineering
* One-Hot Encoding
* Feature Scaling
* Train-Test Split
* Linear Regression Implementation from Scratch
* Model Training using Gradient Descent
* Model Evaluation
* Data Visualization

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Structure

```
Linear_Regression/
├── images
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
├── insurance.csv
├── insurance.ipynb
├── Linear_Regression.py
├── __pycache__
│   └── Linear_Regression.cpython-314.pyc
└── README.md
```

---

## Model Evaluation

* R² Score: **75.33%**
* Adjusted R² Score: **74.07%**
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

---

## Features of this Project

* Linear Regression implemented from scratch
* Gradient Descent optimization
* Manual calculation of weights and bias
* StandardScaler for numerical features
* Feature Selection
* Model Evaluation using multiple regression metrics
* Data Visualization

---

## Learning Outcomes

Through this project, I learned:

* Linear Regression Mathematics
* Cost Function (MSE)
* Gradient Descent
* Derivation of Weight and Bias Updates
* Feature Scaling
* Model Evaluation
* Building a Machine Learning algorithm from scratch

---

## Future Improvements

* Mini-Batch Gradient Descent
* Stochastic Gradient Descent
* Ridge Regression
* Lasso Regression
* Model Serialization
* Hyperparameter Tuning

---

## Author

Ayyan Ahmed
