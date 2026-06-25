import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, mean_absolute_error, mean_squared_error, r2_score
)

# ==========================================
# 1. CORE SYSTEM & DIRECTORY UTILITIES
# ==========================================

def ensure_directories(dirs=None):
    """Ensures necessary pipeline directories exist."""
    if dirs is None:
        dirs = ["models", "plots", "data_outputs"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")


def save_artifact(obj, filename, folder="models"):
    """Safely dumps joblib serializations."""
    ensure_directories([folder])
    path = os.path.join(folder, filename)
    joblib.dump(obj, path)
    print(f"Successfully saved artifact to: {path}")


def load_artifact(filename, folder="models"):
    """Loads a saved joblib serialization artifact."""
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"Artifact not found at path: {path}")


# ==========================================
# 2. FEATURE ENGINEERING & PREPROCESSING UTILITIES
# ==========================================

def compute_engineered_features(df):
    """
    Applies the custom mathematical engineering definitions across the dataset:
    1. AvgMonthlySpend
    2. ServiceCount
    3. ContractValue
    """
    df_copy = df.copy()
    
    # 1. Average Monthly Spend
    df_copy['AvgMonthlySpend'] = df_copy['MonthlyCharges'] / (df_copy['tenure'] + 1)
    
    # 2. Service Count Calculation
    service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup", 
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df_copy["ServiceCount"] = 0
    for col in service_cols:
        if col in df_copy.columns:
            # Safely check for 'Yes' strings
            df_copy["ServiceCount"] += (df_copy[col].astype(str).str.contains("Yes")).astype(int)
            
    # 3. Contract Value Estimation
    df_copy["ContractValue"] = (df_copy["MonthlyCharges"] * df_copy["tenure"])
    
    return df_copy
