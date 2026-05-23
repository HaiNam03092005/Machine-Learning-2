# 🏀 Machine Learning Implementation Blueprint - NBA Prediction (2011 - 2026)

This document serves as a step-by-step blueprint for building a classification system to predict NBA achievements based on the cleaned and aggregated historical dataset.

---

## 🛠️ PART 1: Data Preparation

### 1. Train/Test Split Strategy
Since time-series/season data has a sequential nature, the data will be split using a realistic temporal strategy:
* **Training Set:** Seasons from **2011 to 2025** (Data with complete target labels: `Is_Playoff`, `Is_Semifinal`, `Is_Champion`).
* **Test/Prediction Set:** Current season **2026** (The ongoing season where the `Is_Champion` label is currently `None` and needs to be predicted).

### 2. Feature Selection
* **X (Features):** Drop all identifier columns (`Rk`, `Team`, `Season`, `Arena`) and target labels. Keep only the performance metrics: `Age`, `W`, `L`, `MOV`, `SOS`, `SRS`, `ORtg`, `DRtg`, `NRtg`, `Pace`, `FTr`, `3PAr`, `TS%`, `eFG%`, `TOV%`, `ORB%`, `FT/FGA`, `DRB%`.
* **y (Target Labels):** Three independent binary classification tasks:
    1.  `Is_Playoff` (Predicting whether a team makes the Playoffs)
    2.  `Is_Semifinal` (Predicting whether a team reaches the Conference Finals)
    3.  `Is_Champion` (Predicting whether a team wins the NBA Championship)

### 3. Feature Scaling
Distance-based algorithms (k-NN) and geometric classifiers (SVM) are highly sensitive to the scale of features (e.g., win counts versus decimal shooting percentages). Therefore, applying **StandardScaler** is mandatory:
$$z = \frac{x - \mu}{\sigma}$$

---

## 🤖 PART 2: Implementing Classification Models

Train and compare the performance of the following 3 core machine learning algorithms:

### 1. K-Nearest Neighbors (k-NN)
* Experiment with the neighborhood parameter $k$ ranging from 1 to 15 to find the optimal balance (preventing overfitting when $k$ is too small).
* Utilize the default Euclidean distance metric.

### 2. Support Vector Machines (SVM)
Train the SVM model and perform hyperparameter optimization by sweeping through the three key kernel functions:
* **Linear Kernel:** $$K(x, y) = xy^T$$
* **Polynomial Kernel:** (Experiment with degrees $d = 2, 3$)
    $$K(x, y) = (xy^T + 1)^d$$
* **Gaussian (RBF) Kernel:** $$K(x, y) = e^{-\frac{||x-y||^2}{2\sigma^2}}$$

### 3. Ensemble Model - Random Forest
Apply the Bagging mechanism (voting across multiple independent decision trees) to reduce the ensemble error variance ($e_{ensemble}$):
* Number of trees (`n_estimators`): Experiment with values of 50, 100, and 200.
* Use Gini impurity or Entropy for optimal node splitting criteria.

---

## 📊 PART 3: Model Evaluation Metrics

Use a validation split from the historical data (2011–2025) to generate a **Confusion Matrix** and calculate the following evaluation metrics:

| Actual / Predicted | Positive (1) | Negative (0) |
| :--- | :--- | :--- |
| **Positive (1)** | True Positive (TP) | False Negative (FN) |
| **Negative (0)** | False Positive (FP) | True Negative (TN) |

* **Accuracy (Overall Correctness):** $\frac{TP + TN}{P + N}$
* **Precision (Exactness of Positive Predictions):** $\frac{TP}{TP + FP}$
* **Recall (Sensitivity/True Positive Rate):** $\frac{TP}{TP + FN}$
* **ROC Curve & AUC Score:** Plot True Positive Rate (TPR) against False Positive Rate (FPR) at various thresholds to evaluate the model's capability to rank team strength.

---

## 🔮 PART 4: Inference and Predictions for the 2026 Season

1. Feed the 2026 season data (Test Set) into the best-performing trained models.
2. Instead of generating a hard binary label (`0` or `1`), use the `predict_proba()` method to extract the **probability percentage** of winning for each team.
3. Sort the teams in descending order of probability to identify the **Top 4 Contenders** predicted by the AI.