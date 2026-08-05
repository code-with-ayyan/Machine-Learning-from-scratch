# Decision Tree Regressor From Scratch

## Overview

This project implements a **Decision Tree Regressor** completely from scratch using **Python** and **NumPy**. The objective was to understand the internal working of the Decision Tree Regression algorithm by building it without using machine learning libraries such as Scikit-learn.

The model was trained and evaluated on the **Insurance Dataset**, and its performance was compared with Scikit-learn's `DecisionTreeRegressor`.

---

## Features

* Decision Tree Regressor implemented from scratch
* Recursive tree construction
* Variance calculation
* Variance Reduction based splitting
* Best feature and threshold selection
* Prediction for single and multiple samples
* Comparison with Scikit-learn implementation

---

## Dataset

* **Dataset:** Insurance Dataset
* **Task:** Regression
* **Target Variable:** Insurance Charges

---

## Algorithm Workflow

1. Load the dataset.
2. Split the data into training and testing sets.
3. Build the decision tree recursively.
4. Calculate variance for the parent node.
5. Compute variance reduction for every possible split.
6. Select the feature and threshold with the highest variance reduction.
7. Continue splitting until stopping conditions are met.
8. Predict values for the test dataset.
9. Compare results with Scikit-learn's implementation.

---

## Evaluation Metrics

The following regression metrics were used:

* R² Score
* Adjusted R²
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

---

## Results

| Model                                | R² Score | Adjusted R² |    RMSE |     MAE |
| ------------------------------------ | -------: | ----------: | ------: | ------: |
| Custom Decision Tree Regressor       |   0.8941 |      0.8917 | 4411.54 | 2656.84 |
| Scikit-learn Decision Tree Regressor |   0.8941 |      0.8917 | 4411.54 | 2656.84 |

The custom implementation achieved the same performance as Scikit-learn's model, confirming that the algorithm was implemented correctly.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn (only for dataset splitting and model comparison)

---

## Learning Outcomes

Through this project, the following concepts were implemented and understood:

* Decision Tree Regression
* Recursive Tree Building
* Variance
* Variance Reduction
* Best Split Selection
* Binary Tree Traversal
* Regression Evaluation Metrics
* Scratch Machine Learning Implementation

---

## Conclusion

This project demonstrates that a Decision Tree Regressor can be successfully implemented from scratch while achieving results identical to Scikit-learn's implementation. Building the algorithm manually provides a deeper understanding of recursive tree construction, variance reduction, and the complete prediction process behind Decision Tree Regression.
