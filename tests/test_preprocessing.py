import pytest
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from src.data_loader import get_data_path, load_and_validate_data
from src.eda import clean_and_prepare_data
from src.preprocessing import (engineer_features, split_data, encode_categorical_features,scale_features
     , handle_imbalance,feature_selection)

# Tests for get_data_path()

def test_get_data_path_default():
    path = get_data_path()
    assert path.endswith(os.path.join("data", "Telco-Customer-Churn.csv"))

# test for load_and_validate_data()

def test_load_and_validate_data():
    df = load_and_validate_data()

    assert not df.empty
    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert "MonthlyCharges" in df.columns
    assert "TotalCharges" in df.columns
    assert "tenure" in df.columns

#test for clean_and_prepare_data()

def test_clean_and_prepare_data():
    df = load_and_validate_data()

    result = clean_and_prepare_data(df)

    assert not result.empty
    assert "tenure_group" in result.columns
    assert "customerID" not in result.columns

# test for engineer_features
def test_engineer_features():
    # Load and preprocess data
    df = load_and_validate_data()
    df = clean_and_prepare_data(df)

    # Apply feature engineering
    result = engineer_features(df)

    # Verify new features were created
    assert "AvgMonthlySpend" in result.columns
    assert "ServiceCount" in result.columns
    assert "ContractValue" in result.columns

    # Verify target variable was encoded
    assert result["Churn"].isin([0, 1]).all()

# test for split_data
def test_split_data():
    # Load and prepare data
    df = load_and_validate_data()
    df = clean_and_prepare_data(df)
    df = engineer_features(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Verify data was split
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # Verify features and targets have matching lengths
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

#test for encoding_categorical_features
def test_encode_categorical_features():
    # Prepare data
    df = load_and_validate_data()
    df = clean_and_prepare_data(df)
    df = engineer_features(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Encode features
    X_train, X_val, X_test, le_dict = encode_categorical_features(
        X_train, X_val, X_test
    )

    # Verify outputs are not empty
    assert not X_train.empty
    assert not X_val.empty
    assert not X_test.empty

    # Verify binary encoders were created
    assert "gender" in le_dict
    assert "Partner" in le_dict

    # Verify train, validation and test have same columns
    assert list(X_train.columns) == list(X_val.columns)
    assert list(X_train.columns) == list(X_test.columns)

#test for scale features
def test_scale_features():
    # Prepare data
    df = load_and_validate_data()
    df = clean_and_prepare_data(df)
    df = engineer_features(df)

    # Split and encode
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, le_dict = encode_categorical_features(
        X_train, X_val, X_test
    )

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    # Verify outputs are not empty
    assert not X_train_scaled.empty
    assert not X_val_scaled.empty
    assert not X_test_scaled.empty

    # Verify scaler object was created
    assert scaler is not None

    # Verify important columns still exist
    assert "tenure" in X_train_scaled.columns
    assert "MonthlyCharges" in X_train_scaled.columns
    assert "TotalCharges" in X_train_scaled.columns

#test for handle class imbalance
def test_handle_imbalance():
    # Prepare data
    df = load_and_validate_data()
    df = clean_and_prepare_data(df)
    df = engineer_features(df)

    # Split, encode, and scale
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, le_dict = encode_categorical_features(
        X_train, X_val, X_test
    )
    X_train, X_val, X_test, scaler = scale_features(
        X_train, X_val, X_test
    )

    # Apply SMOTE
    X_res, y_res = handle_imbalance(X_train, y_train)

    # Verify output is not empty
    assert len(X_res) > 0
    assert len(y_res) > 0

    # Verify feature-target lengths match
    assert len(X_res) == len(y_res)

    # Verify classes are balanced
    assert y_res.value_counts()[0] == y_res.value_counts()[1]

#test for feature selection
def test_run_feature_selection():
    # Prepare data
    df = load_and_validate_data()
    df = clean_and_prepare_data(df)
    df = engineer_features(df)

    # Split, encode, scale, and balance
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, le_dict = encode_categorical_features(
        X_train, X_val, X_test
    )
    X_train, X_val, X_test, scaler = scale_features(
        X_train, X_val, X_test
    )
    X_train, y_train = handle_imbalance(X_train, y_train)

    # Run feature selection
    selected_features = feature_selection(X_train, y_train)

    # Verify output
    assert selected_features is not None
    assert len(selected_features) == 15
    assert isinstance(selected_features, list)
"""
# --- Mock Functions representing your notebook logic ---
def calculate_features(df):
    """ """Calculates AvgMonthlySpend and ServiceCount based on notebook steps""""""
    df = df.copy()
    # 1.1 Average Monthly Spend
    df['AvgMonthlySpend'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # 1.2 Service Count
    service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["ServiceCount"] = 0
    for col in service_cols:
        df["ServiceCount"] += df[col].str.contains("Yes").astype(int)
    return df

# --- Unit Tests ---

def test_calculate_features():
    # Arrange: Create a small mock DataFrame mimicking Telco data
    mock_data = pd.DataFrame({
        'MonthlyCharges': [30.0, 60.0],
        'tenure': [2, 0],
        'PhoneService': ['No', 'Yes'],
        'OnlineSecurity': ['Yes', 'No'],
        'OnlineBackup': ['No', 'No'],
        'DeviceProtection': ['No', 'No'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'No'],
        'StreamingMovies': ['No', 'No']
    })
    
    # Act
    processed_df = calculate_features(mock_data)
    
    # Assert 1: Verify AvgMonthlySpend calculation
    # Row 0: 30.0 / (2 + 1) = 10.0
    # Row 1: 60.0 / (0 + 1) = 60.0
    assert processed_df['AvgMonthlySpend'].iloc[0] == 10.0
    assert processed_df['AvgMonthlySpend'].iloc[1] == 60.0
    
    # Assert 2: Verify ServiceCount tracking (counts "Yes" strings)
    # Row 0 has 1 "Yes" (OnlineSecurity)
    # Row 1 has 1 "Yes" (PhoneService)
    assert processed_df['ServiceCount'].iloc[0] == 1
    assert processed_df['ServiceCount'].iloc[1] == 1


def test_label_encoding_consistency():
    # Arrange: Simulating binary valued columns encoding
    train_categories = ['Male', 'Female', 'Male']
    val_categories = ['Female', 'Male']
    
    le = LabelEncoder()
    
    # Act
    train_encoded = le.fit_transform(train_categories)
    val_encoded = le.transform(val_categories)
    
    # Assert
    assert np.array_equal(train_encoded, np.array([1, 0, 1]))
    assert np.array_equal(val_encoded, np.array([0, 1]))  """