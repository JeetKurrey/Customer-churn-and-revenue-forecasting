**Telecom Customer Churn Prediction \& Revenue Forecasting**

**Project Overview:**



Telecom Customer Churn Prediction \& Revenue Forecasting is an end-to-end Machine Learning project designed to help telecom companies identify customers likely to churn and forecast their future revenue. The project combines classification and regression techniques to support customer retention strategies, improve business decision-making, and maximize revenue generation.



The solution includes data preprocessing, exploratory data analysis (EDA), churn classification, revenue prediction, model interpretation, testing, and deployment through a Flask API and Streamlit frontend.



**Dataset:**

Dataset Source



IBM Telco Customer Churn Dataset



🔗 Dataset Link:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn



Dataset Description



The dataset contains information about telecom customers, including:



Customer demographics

Subscription details

Contract information

Billing and payment methods

Service usage

Monthly charges

Customer tenure

Churn status

Key Features

Feature Category	Examples

Demographics	Gender, SeniorCitizen, Partner, Dependents

Services	PhoneService, InternetService, StreamingTV

Account Information	Contract, PaymentMethod, PaperlessBilling

Financial	MonthlyCharges, TotalCharges

Target Variables	Churn, MonthlyCharges



Dataset Size

Records: \~7,043 customers

Features: 21+

Classification Target: Churn

Regression Target: MonthlyCharges



**Installation \& Setup**

Clone Repository

git clone https://github.com/your-username/telecom-ml-assignment.git



cd telecom-ml-assignment



Create Virtual Environment



Linux / Mac

python -m venv venv

source venv/bin/activate



Windows

python -m venv venv

venv\\Scripts\\activate



**Install Dependencies**

pip install -r requirements.txt

Verify Installation

python --version

pip list

**How to Run**



**Option 1: Use Pre-trained Models**



pip install -r requirements.txt



cd deployment/backend

python app\_api.py



In another terminal:



cd deployment/frontend

streamlit run app\_ui.py



**Option 2: Reproduce Training Pipeline**



Run notebooks in order:



01\_EDA.ipynb

02\_preprocessing.ipynb

03\_classification.ipynb

04\_regression.ipynb

05\_interpretation.ipynb



ex: jupyter notebook notebooks/01\_EDA.ipynb



Run Unit Tests

pytest tests/

**Deployment with Docker**
To run this project in an isolated containerized environment:

Ensure Docker Desktop is installed.

Build the image: docker build -t telecom-churn-app .

Run the container: docker run -p 8501:8501 telecom-churn-app

Access the app at http://localhost:8501


**Results Summary**



Customer Churn Prediction



Best Classification Model: Logistic Regression

Metric	Score

Accuracy	78.39%

Precision	61.30%

Recall	50.36%

F1-Score	55.29%

ROC-AUC	82.74%

Training Time	1.94 seconds



Revenue Forecasting Results

🏆 Best Regression Model: Random Forest Regressor

Metric	Score

MAE	0.93

MSE	1.66

RMSE	1.29

R² Score	0.9982

Adjusted R²	0.9981



Combining churn prediction with revenue forecasting allows telecom companies to:



Identify high-value customers at risk

Prioritize retention campaigns

Reduce customer attrition

Improve revenue planning

Increase customer lifetime value (CLV)



**Project Architecture**

telecom-ml-assignment/

│

├── data/

│   ├── Telecom-Customer-Churn.csv

│   ├── Cleaned-Telco-Customer.csv

│   └── preprocessed-Telco-Customer.csv

│

├── notebooks/

│   ├── 01\_EDA.ipynb

│   ├── 02\_preprocessing.ipynb

│   ├── 03\_classification.ipynb

│   ├── 04\_regression.ipynb

│   └── 05\_interpretation.ipynb

│

├── src/

│   ├── data\_loader.py

│   ├── eda.py

│   ├── preprocessing.py

│   ├── classifiers.py

│   ├── regressors.py

│   ├── evaluation.py

│   └── utils.py

│

├── models/

│   ├── best\_classifier.pkl

│   ├── best\_regressor.pkl

│   ├── scaler.pkl

│   ├── label\_encoder.pkl

│   ├── onehot\_encoder.pkl

│   └── feature\_columns.pkl

│

├── deployment/

│   ├── backend/

│   │   └── app\_api.py

│   │

│   └── frontend/

│       └── app\_ui.py

│

├── tests/

│   ├── test\_preprocessing.py

│   └── test\_evaluation.py

│

├── reports/

│   ├── Churn\_Classification\_Report.pdf

│   └── Regression\_Analysis\_Report.pdf

│

├── requirements.txt

├── README.md

└── .gitignore



**Tech Stack**

Programming Language

Python



Used for:



Data Analysis

Machine Learning

API Development

Model Deployment

Data Processing

NumPy



Used for:



Numerical computations

Matrix operations



Pandas



Used for:



Data loading

Data cleaning

Feature engineering

Data Visualization



Matplotlib



Used for:



Statistical plots

Model performance visualization

Seaborn



Used for:



Correlation heatmaps

Distribution analysis

Machine Learning

Scikit-Learn



Used for:



Classification models

Regression models

Feature encoding

Scaling

Model evaluation



Algorithms Used:



Logistic Regression

Decision Tree

Random Forest

Support Vector Machine

KNN

Linear Regression

Random Forest Regressor

SVR

Model Persistence

Joblib



Used for:



Saving trained models

Saving encoders and scalers

Backend Development

Flask



Used for:



REST API creation

Real-time prediction serving

Frontend Development

Streamlit



Used for:



Interactive user interface

Churn prediction dashboard

Revenue forecasting dashboard

Testing

Pytest



Used for:



Unit testing

Pipeline validation

Reproducibility checks



**Project Impact**



Developed an end-to-end Telecom Customer Churn Prediction and Revenue Forecasting system using Machine Learning. Achieved 78.39% churn prediction accuracy and 82.74% ROC-AUC using Logistic Regression, while attaining 99.82% R² in revenue forecasting with Random Forest Regressor. The solution enables proactive customer retention, revenue protection, customer lifetime value optimization, and data-driven business decision-making through predictive analytics.

