# 📡 Telecom Customer Churn Prediction & Revenue Forecasting

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**An end-to-end Machine Learning system combining churn classification and revenue regression to power proactive customer retention strategies.**

[🚀 Quick Start](#-installation--setup) · [📊 Results](#-results-summary) · [🏗️ Architecture](#️-project-architecture) · [🐳 Docker Deployment](#-deployment-with-docker)

</div>

---

## 📌 Project Overview

This project helps telecom companies tackle two critical business challenges simultaneously:

- **Churn Prediction** — Identify customers at risk of leaving using classification models
- **Revenue Forecasting** — Predict monthly charges using regression to prioritize high-value accounts

By combining both signals, the system enables data-driven retention campaigns, smarter resource allocation, and improved Customer Lifetime Value (CLV).

The solution covers the full ML lifecycle: data preprocessing → EDA → model training → interpretation → REST API → interactive dashboard.

---

## 📂 Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) via Kaggle

| Property | Details |
|---|---|
| Records | ~7,043 customers |
| Features | 21+ columns |
| Classification Target | `Churn` (Yes / No) |
| Regression Target | `MonthlyCharges` |

### Feature Categories

| Category | Examples |
|---|---|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, StreamingTV |
| Account Info | Contract, PaymentMethod, PaperlessBilling |
| Financial | MonthlyCharges, TotalCharges, Tenure |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/telecom-ml-assignment.git
cd telecom-ml-assignment
```

### 2. Create a Virtual Environment

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python --version
pip list
```

---

## 🚀 How to Run

### Option A — Use Pre-trained Models *(Recommended)*

Start the Flask backend API:

```bash
cd deployment/backend
python app_api.py
```

In a separate terminal, launch the Streamlit frontend:

```bash
cd deployment/frontend
streamlit run app_ui.py
```

Open your browser at `http://localhost:8501`

---

### Option B — Reproduce the Full Training Pipeline

Run the Jupyter notebooks **in order**:

```
notebooks/
├── 01_EDA.ipynb              # Exploratory Data Analysis
├── 02_preprocessing.ipynb    # Feature engineering & encoding
├── 03_classification.ipynb   # Churn prediction models
├── 04_regression.ipynb       # Revenue forecasting models
└── 05_interpretation.ipynb   # SHAP values & model insights
```

Launch any notebook with:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

### Running Unit Tests

```bash
pytest tests/
```

---

## 🐳 Deployment with Docker

Run the entire application in an isolated container:

```bash
# 1. Ensure Docker Desktop is running

# 2. Build the image
docker build -t telecom-churn-app .

# 3. Start the container
docker run -p 8501:8501 telecom-churn-app
```

Access the app at `http://localhost:8501`

---

## 📊 Results Summary

### 🏆 Churn Classification — Best Model: Logistic Regression

| Metric | Score |
|---|---|
| Accuracy | 78.39% |
| Precision | 61.30% |
| Recall | 50.36% |
| F1-Score | 55.29% |
| ROC-AUC | **82.74%** |
| Training Time | 1.94 seconds |

### 🏆 Revenue Forecasting — Best Model: Random Forest Regressor

| Metric | Score |
|---|---|
| MAE | 0.93 |
| MSE | 1.66 |
| RMSE | 1.29 |
| R² Score | **0.9982** |
| Adjusted R² | 0.9981 |

> Combining churn prediction with revenue forecasting allows telecom companies to **identify high-value customers at risk**, prioritize retention campaigns, and improve revenue planning.

---

## 🏗️ Project Architecture

```
telecom-ml-assignment/
│
├── data/
│   ├── Telecom-Customer-Churn.csv
│   ├── Cleaned-Telco-Customer.csv
│   └── preprocessed-Telco-Customer.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_regression.ipynb
│   └── 05_interpretation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── classifiers.py
│   ├── regressors.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
│   ├── best_classifier.pkl
│   ├── best_regressor.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── onehot_encoder.pkl
│   └── feature_columns.pkl
│
├── deployment/
│   ├── backend/
│   │   └── app_api.py          # Flask REST API
│   └── frontend/
│       └── app_ui.py           # Streamlit dashboard
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_evaluation.py
│
├── reports/
│   ├── Churn_Classification_Report.pdf
│   └── Regression_Analysis_Report.pdf
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn |
| **Model Persistence** | Joblib |
| **Backend API** | Flask |
| **Frontend Dashboard** | Streamlit |
| **Testing** | Pytest |
| **Containerization** | Docker |

### ML Algorithms

| Task | Algorithms Evaluated |
|---|---|
| Classification | Logistic Regression, Decision Tree, Random Forest, SVM, KNN |
| Regression | Linear Regression, Random Forest Regressor, SVR |

---

## 💡 Business Impact

| Outcome | Description |
|---|---|
| 🎯 Proactive Retention | Flag at-risk customers before they churn |
| 💰 Revenue Protection | Prioritize high-value accounts for targeted outreach |
| 📈 CLV Optimization | Maximize Customer Lifetime Value through data-driven decisions |
| 📋 Smarter Planning | Use revenue forecasts to guide resource allocation |

---

## 📄 Reports

Detailed analysis reports are available in the `reports/` directory:

- [`Churn_Classification_Report.md`](reports/Churn_Classification_Report.pdf)
- [`Regression_Analysis_Report.md`](reports/Regression_Analysis_Report.pdf)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. Ensure all tests pass before submitting a PR.

```bash
pytest tests/
```
