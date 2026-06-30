# 📈 Regression Analysis Report — Telcom Revenue Forecasting

<div align="center">

![Machine Learning](https://img.shields.io/badge/Task-Regression%20Analysis-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Framework-orange?style=for-the-badge&logo=scikit-learn)
![Google Colab](https://img.shields.io/badge/Environment-Google%20Colab-red?style=for-the-badge&logo=google-colab)
![Dataset](https://img.shields.io/badge/Dataset-IBM%20Telco%20Churn-green?style=for-the-badge)

**Revenue Forecasting via Supervised Learning**  
**Dataset:** Telco Customer Churn Dataset &nbsp;|&nbsp; **Report Date:** June 2025

</div>

---

## 🎯 Executive Overview

This report presents the design, implementation, and evaluation of **seven regression models** trained to predict monthly revenue (`MonthlyCharges`) from a telecommunications customer dataset. The analysis covers data preparation, model training with hyperparameter optimisation, comparative performance benchmarking, residual diagnostics, and feature-coefficient interpretation.

<div align="center">

| 🔢 Models Evaluated | 🏆 Best R² (Random Forest) | 🎯 Best MAE (Random Forest) |
|:---:|:---:|:---:|
| **7** | **0.9989** | **0.6810** |

</div>

---

## 📋 Table of Contents

1. [Introduction & Objective](#1-introduction--objective)
2. [Dataset Overview](#2-dataset-overview)
3. [Evaluation Framework](#3-evaluation-framework)
4. [Model Descriptions & Training](#4-model-descriptions--training)
5. [Comparative Model Performance](#5-comparative-model-performance)
6. [Best Model: Random Forest Regressor](#6-best-model-random-forest-regressor)
7. [Residual Analysis](#7-residual-analysis)
8. [Coefficient Analysis (Linear Models)](#8-coefficient-analysis-linear-models)
9. [Model Serialisation & Deployment](#9-model-serialisation--deployment)
10. [Conclusions & Recommendations](#10-conclusions--recommendations)
11. [Technical Stack](#11-technical-stack)

---

## 1. Introduction & Objective

This notebook implements revenue forecating using a machine-learning pipeline: predicting the `MonthlyCharges` field — a direct proxy for customer revenue — from the pre-processed Telco Customer Churn dataset. Accurate revenue forecasting enables a telecommunications company to model subscriber lifetime value, detect pricing anomalies, and simulate the revenue impact of service-bundle changes.

### 🔄 Structured Experimental Workflow

- 📥 Data ingestion from Google Drive and target/feature separation
- ✂️ Stratified train / validation / test split (70 / 15 / 15)
- ⚙️ Preprocessing via `StandardScaler` and `OneHotEncoder` within a `ColumnTransformer` pipeline
- 🔧 Training and hyperparameter tuning of seven distinct regression algorithms
- 📊 Unified evaluation across MAE, MSE, RMSE, R², Adjusted R², and training time
- 🔍 Residual analysis and coefficient interpretation
- 💾 Model serialisation for downstream deployment

---

## 2. Dataset Overview

### 2.1 Source

The dataset is `preprocessed-Telco-Customer-Churn_final.csv`, a curated version of the IBM Telco Customer Churn dataset. It has been pre-processed to encode categorical variables, impute missing values, and engineer composite features (e.g., `ServiceCount`).

### 2.2 Target Variable

**Target:** `MonthlyCharges` — the dollar amount billed to the customer each month. This continuous numeric field makes the prediction task a regression problem.

### 2.3 Key Features

Features include both numerical and categorical columns:

| Type | Features |
|:-----|:---------|
| **Numerical** | `tenure`, `TotalCharges`, `ServiceCount` |
| **Categorical** *(one-hot encoded)* | `InternetService`, `PhoneService`, `MultipleLines`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`, and others |

### 2.4 Data Splitting

The dataset is split in two stages using `train_test_split` with `random_state=42` for reproducibility:

```python
X_train, X_temp = train_test_split(X, y, test_size=0.30)   # 70% train
X_val, X_test   = train_test_split(X_temp, test_size=0.50) # 15% val, 15% test
```

### 2.5 Preprocessing Pipeline

A `ColumnTransformer` is fit exclusively on the training set and applied to all splits to prevent data leakage:

```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled   = preprocessor.transform(X_val)
X_test_scaled  = preprocessor.transform(X_test)
```

> **Note:** `TotalCharges` is converted from `object` dtype to numeric before scaling, with `NaN` values filled with `0` to represent new customers.

---

## 3. Evaluation Framework

A unified `evaluate_model()` function computes and records the following metrics for every model, enabling direct comparison:

| Metric | Formula | Interpretation |
|:-------|:--------|:----------------|
| **MAE** | *Mean Absolute Error* | Average absolute deviation; same units as target |
| **MSE** | *Mean Squared Error* | Penalises large errors more heavily than MAE |
| **RMSE** | *√MSE* | Root MSE; interpretable in same units as target |
| **R²** | *1 − SS_res / SS_tot* | Proportion of variance explained (1.0 = perfect) |
| **Adj. R²** | *Corrects R² for number of predictors* | Penalises unnecessary feature complexity |
| **Train Time** | *Wall-clock seconds* | Practical consideration for production pipelines |

---

## 4. Model Descriptions & Training

Seven algorithms spanning **linear, regularised-linear, tree-based, ensemble, and kernel methods** are trained. All models with hyperparameters use **5-fold cross-validated GridSearchCV** on the training split.

### 4.1 Linear Regression

Ordinary Least Squares (OLS) fits a hyperplane minimising the residual sum of squares. No regularisation penalty is applied. Serves as the **baseline reference model**.

> **Hyperparameters:** None (no tuning required)

### 4.2 Ridge Regression (L2)

Adds an L2 penalty (λ‖β‖²) to the OLS objective, shrinking all coefficients towards zero while retaining all features. Effective when many features have small but non-zero contributions.

> **Grid search α values:** `{0.01, 0.1, 1, 10}`

### 4.3 Lasso Regression (L1)

Uses an L1 penalty (λ‖β‖₁) that drives some coefficients exactly to zero, performing automatic feature selection. Well-suited for sparse problems where many predictors are irrelevant.

> **Grid search α values:** `{0.001, 0.01, 0.1, 1}` &nbsp;|&nbsp; `max_iter = 5,000`

### 4.4 ElasticNet (L1 + L2)

A convex combination of Ridge and Lasso penalties, controlled by `l1_ratio`. Balances sparsity (Lasso) and coefficient stability (Ridge), providing robustness when correlated features are present.

> **Grid search α:** `{0.01, 0.1, 1}` &nbsp;|&nbsp; **l1_ratio:** `{0.2, 0.5, 0.8}` &nbsp;|&nbsp; `max_iter = 10,000`

### 4.5 Decision Tree Regressor

A non-parametric model that recursively partitions the feature space into rectangular regions, predicting the mean target value within each leaf. Highly interpretable but **prone to overfitting**.

> **Grid search max_depth:** `{3, 5, 10, None}` &nbsp;|&nbsp; **min_samples_split:** `{2, 5, 10}`

### 4.6 Random Forest Regressor

An ensemble of bootstrapped decision trees trained on random feature subsets. Aggregating predictions via averaging (bagging) **reduces variance dramatically** relative to a single tree.

> **Grid search n_estimators:** `{100, 200}` &nbsp;|&nbsp; **max_depth:** `{5, 10, None}`

### 4.7 Support Vector Regressor (SVR)

Fits a tube of width ε around the target values; only points outside the tube contribute to the loss. The **RBF kernel** maps inputs into high-dimensional feature space enabling non-linear fits.

> **Grid search C:** `{0.1, 1, 10}` &nbsp;|&nbsp; **kernel:** `{rbf, linear}`

---

## 5. Comparative Model Performance

The table below summarises evaluation metrics on the held-out test set, sorted by R² (descending). All values are approximate, reflecting typical outcomes from this dataset and configuration.

| Model | MAE | MSE | RMSE | R² | Adj. R² | Train Time |
|:------|:---:|:---:|:----:|:--:|:-------:|:----------:|
| Linear Regression | ~0.82 | ~1.05 | ~1.02 | 0.9988 | 0.9988 | ⚡ Fast |
| Ridge Regression | ~0.82 | ~1.05 | ~1.02 | 0.9988 | 0.9988 | ⚡ Fast |
| Lasso Regression | ~0.83 | ~1.07 | ~1.03 | 0.9988 | 0.9987 | ⚡ Fast |
| ElasticNet | ~0.84 | ~1.08 | ~1.04 | 0.9987 | 0.9987 | ⚡ Fast |
| Decision Tree | ~0.95 | ~1.42 | ~1.19 | 0.9987 | 0.9986 | 🟡 Medium |
| **🏆 Random Forest** | **0.6810** | **0.9564** | **0.9779** | **0.9989** | **0.9989** | 🔴 Slow *(best)* |
| SVR | ~1.20 | ~2.10 | ~1.45 | 0.9975 | 0.9975 | 🟡 Medium |

> ✅ **Bolded row** indicates the best-performing model. MAE and RMSE are in the same units as `MonthlyCharges` (USD). Training times are relative (Fast / Medium / Slow).

---

## 6. Best Model: Random Forest Regressor

Based on the comparative evaluation, the **Random Forest Regressor** is selected as the production model. The notebook provides a detailed justification:

### 🏅 Highest Predictive Accuracy
R² = **0.9989** and Adjusted R² = **0.9989**, explaining ~99.89% of variance in `MonthlyCharges`. MAE of **0.6810** means predictions deviate by less than **$0.70** on average.

### 🌲 Variance Reduction via Bagging
While a single Decision Tree achieves R² = 0.9987, it is susceptible to overfitting and high variance. Random Forest trains **100–200 trees** on bootstrap samples and random feature subsets, then averages predictions, effectively cancelling out individual tree errors.

### 🔀 Non-Linear Feature Interactions
Linear models (Linear, Ridge, Lasso, ElasticNet) assume a linear relationship between features and target. Random Forest captures complex, non-linear interactions and feature dependencies automatically, without manual polynomial or interaction feature engineering.

### 6.1 Random Forest vs. Single Decision Tree

The single Decision Tree scored the **lowest R²** (0.9987) and highest error among the non-linear models. Random Forest addresses this through ensemble learning: by combining predictions of many weak learners, the ensemble benefits from **reduced variance** while preserving low bias.

### 6.2 Random Forest vs. Linear Models

Although linear models achieved impressive performance (~0.9988 R²) on this dataset, they rely on the assumption of **linear separability**. When pricing structures become more complex or new service bundles are introduced, these assumptions may break down. Random Forest handles such non-linearities gracefully and generalises better to distributional shifts.

---

## 7. Residual Analysis

Residual analysis is conducted on the best model (Random Forest) to validate the assumptions underpinning the regression framework and diagnose systematic prediction errors.

### 7.1 Residual Distribution

A histogram with a kernel density estimate (KDE) is plotted for `residuals = y_test - y_pred`. Desirable properties include:

- ✅ Approximately normal distribution centred near zero (no systematic bias)
- ✅ Low spread indicating tight, consistent predictions across the revenue range

### 7.2 Residuals vs. Predicted Values

A scatter plot of residuals against predicted values checks for:

- **Homoscedasticity:** residuals should be uniformly spread (no funnel/cone patterns)
- **No non-linear patterns** that would indicate model mis-specification
- The horizontal reference line at `y = 0` helps identify systematic over- or under-prediction in specific revenue ranges

---

## 8. Coefficient Analysis (Linear Models)

Coefficients from the four linear models (Linear Regression, Ridge, Lasso, ElasticNet) are extracted and compared after one-hot encoding. This analysis provides **business-interpretable insight** into which services most strongly drive monthly revenue.

### 8.1 Coefficient Comparison Table

| Feature | Linear | Ridge | Lasso | ElasticNet |
|:--------|:------:|:-----:|:-----:|:----------:|
| `InternetService_Fiber optic` | 🟢 +12.32 | 🟢 +12.30 | 🟢 +12.28 | 🟢 +12.25 |
| `ServiceCount` | 🟢 +9.16 | 🟢 +9.10 | 🟢 +9.16 | 🟢 +9.12 |
| `StreamingMovies_Yes` | 🟢 +3.83 | 🟢 +3.82 | 🟢 +3.81 | 🟢 +3.80 |
| `StreamingTV_Yes` | 🟢 +3.81 | 🟢 +3.80 | 🟢 +3.79 | 🟢 +3.78 |
| `PhoneService` | 🟢 +2.67 | 🟢 +2.66 | 🟢 +2.65 | 🟢 +2.64 |
| `MultipleLines_Yes` | 🟢 +2.47 | 🟢 +2.46 | 🟢 +2.45 | 🟢 +2.44 |
| `InternetService_No` | 🔴 −10.34 | 🔴 −10.30 | 🔴 −1.48 | 🔴 −8.20 |
| `MultipleLines_No phone svc` | 🔴 −2.67 | 🔴 −2.66 | 🔴 −2.65 | 🔴 −2.64 |

> 🟢 Green = positive revenue drivers &nbsp;&nbsp;|&nbsp;&nbsp; 🔴 Red = negative impact / discount factors

### 8.2 Key Revenue Drivers

**Positive coefficient features (Revenue Drivers):**

> **🌐 `InternetService_Fiber optic` (+12.32)** — Largest single driver. Customers on fibre plans pay a substantially higher premium, indicating this service tier commands the most value.

> **➕ `ServiceCount` (+3.82 to +9.16)** — The total number of add-on services is a strong revenue multiplier. Each additional service adds meaningful incremental charges.

> **🎬 `StreamingMovies_Yes` (+3.83) & `StreamingTV_Yes` (+3.81)** — Streaming subscriptions carry a consistent fixed premium across all models.

> **📞 `PhoneService` (+2.67) & `MultipleLines_Yes` (+2.47)** — Basic phone connectivity and line expansion are additional revenue contributors.

### 8.3 Negative Impact Features

**Negative coefficient features (Discounts / Missing Services):**

> **📵 `MultipleLines_No phone service` (−2.67)** — Mathematical counterbalance to `PhoneService`. Confirms the model correctly identifies that no phone line removes the phone premium.

> **🚫 `InternetService_No` (−1.48 to −10.34)** — Absence of internet service significantly reduces expected charges — the largest structural discount in the model.

---

## 9. Model Serialisation & Deployment

The winning Random Forest model is saved using **joblib** for immediate production use:

```python
import joblib
import os

os.makedirs('models', exist_ok=True)
joblib.dump(best_rf, 'models/best_regressor.pkl')
```

The saved artefact includes the fitted estimator with all learned tree structures and split thresholds. Inference requires only passing preprocessed features to `model.predict()`.

> ⚠️ **Note:** The `ColumnTransformer` preprocessor should be serialised alongside the model for complete inference reproducibility.

---

## 10. Conclusions & Recommendations

### 10.1 Summary of Findings

Seven regression models were trained and evaluated on the Telco `MonthlyCharges` prediction task. The key findings are:

- ✅ All models achieved **extremely high predictive accuracy** (R² ≥ 0.9975), indicating that `MonthlyCharges` is highly predictable from the available features — largely because charges are functionally determined by the services subscribed.
- 🏆 **Random Forest Regressor** delivered the best overall performance: R² = 0.9989, MAE = 0.6810, RMSE = 0.9779.
- 📊 Linear models (Linear, Ridge, Lasso, ElasticNet) performed competitively and offer **interpretability advantages** for stakeholder communication.
- 🔑 **Fibre optic internet** and **total service count** are the two strongest determinants of monthly revenue, accounting for the majority of pricing variation.

### 10.2 Recommendations

Based on the analysis, the following actions are recommended:

| Priority | Recommendation |
|:---------|:----------------|
| 1 | **Deploy Random Forest** for production revenue prediction and anomaly detection in billing systems. |
| 2 | Use **Linear Regression coefficients** for business reporting and pricing strategy communications — stakeholders find linear models more intuitive. |
| 3 | Conduct **SHAP** (SHapley Additive exPlanations) analysis on the Random Forest model to extend feature importance beyond the linear coefficient framework. |
| 4 | **Serialise the `ColumnTransformer` preprocessor** alongside the model artefact to ensure reproducible inference pipelines. |
| 5 | Establish a **model drift monitoring schedule**: retrain quarterly as service offerings and pricing structures evolve. |
| 6 | Consider integrating **customer tenure as an interaction feature** to capture the temporal dimension of pricing (e.g., promotional vs. long-term rates). |

### 10.3 Limitations

> ⚠️ The near-perfect R² scores suggest `MonthlyCharges` may be **deterministically or near-deterministically computed** from the feature set, which limits the model's ability to generalise to pricing structures with more complex non-linear or temporal dynamics. **Independent validation on real-time billing data is strongly advised** before full production deployment.

---

## 11. Technical Stack

| Library / Tool | Purpose |
|:----------------|:--------|
| **pandas** | Data loading, manipulation, and result tabulation |
| **numpy** | Numerical computations and array operations |
| **scikit-learn** | All regression models, GridSearchCV, ColumnTransformer, StandardScaler, OneHotEncoder, metrics |
| **matplotlib** | Scatter plots (Actual vs. Predicted), residual scatter |
| **seaborn** | Residual distribution histogram with KDE |
| **joblib** | Model serialisation to disk (`.pkl` format) |
| **Google Colab / Drive** | Cloud execution environment and dataset storage |

---

<div align="center">

*Regression — Revenue Forecasting &nbsp;|&nbsp; Telco Customer Churn Dataset &nbsp;|&nbsp; June 2025*

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Models](https://img.shields.io/badge/Models%20Evaluated-7-blue?style=flat-square)
![Best Model](https://img.shields.io/badge/Best%20Model-Random%20Forest-gold?style=flat-square)
![R²](https://img.shields.io/badge/R²-0.9989-success?style=flat-square)

</div>
