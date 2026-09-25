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