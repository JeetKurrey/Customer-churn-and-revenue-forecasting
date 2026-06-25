import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Import local data finder from your custom loader module
from .data_loader import get_data_path


def engineer_features(df):
    """Applies domain-specific transformations and metrics equations."""
    print("Engineering new features...")
    df = df.copy()

    # Drop non-numeric groupings left over from EDA visualization
    if "tenure_group" in df.columns:
        df.drop(columns=["tenure_group"], inplace=True)

    # 1. Map target variable explicitly
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 2. Metric: Average Monthly Spend
    df["AvgMonthlySpend"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # 3. Metric: Service Interaction Count Summation
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["ServiceCount"] = 0
    for col in service_cols:
        df["ServiceCount"] += df[col].str.contains("Yes").astype(int)

    # 4. Metric: Cumulative Contract Valuation Value
    df["ContractValue"] = df["MonthlyCharges"] * df["tenure"]

    return df


def split_data(df, target_col="Churn", test_size=0.30, val_size=0.50):
    """Splits features and target into stratified Train (70%), Val (15%), and Test (15%)."""
    print("Splitting data into stratified Train, Val, and Test collections...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: Train vs Temporary (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Second split: Even split of temporary data into Validation and Test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42
    )

    print(f"   Train Set:      {X_train.shape[0]} rows ({X_train.shape[0]/len(df):.1%})")
    print(f"   Validation Set: {X_val.shape[0]} rows ({X_val.shape[0]/len(df):.1%})")
    print(f"   Test Set:       {X_test.shape[0]} rows ({X_test.shape[0]/len(df):.1%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_categorical_features(X_train, X_val, X_test):
    """Encodes categorical strings utilizing structured Label and One-Hot encoders."""
    print("Encoding categorical variables...")
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    # 1. Label Encode Binary Properties
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    le_dict = {}
    
    for col in binary_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])
        le_dict[col] = le  # Retain structures if multi-column pipelines expand later

    # 2. One-Hot Encode remaining structural configurations
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_val = pd.get_dummies(X_val, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Strict structural alignment to guarantee matrix layout schema matching
    X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Convert all boolean tracking columns to integer representations
    X_train = X_train.astype(int)
    X_val = X_val.astype(int)
    X_test = X_test.astype(int)

    return X_train, X_val, X_test, le_dict


def scale_features(X_train, X_val, X_test, method="standard"):
    """Scales numeric feature variants using either Standard or MinMaxScaler."""
    print(f"Scaling numeric variables via method='{method}'...")
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train_scaled[num_cols])
    X_val_scaled[num_cols] = scaler.transform(X_val_scaled[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test_scaled[num_cols])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def handle_imbalance(X_train, y_train):
    """Balances class distributions using synthetic over-sampling (SMOTE)."""
    print("Balancing training records utilizing SMOTE over-sampling...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"   Original metrics: {np.bincount(y_train)} -> Resampled: {np.bincount(y_res)}")
    return X_res, y_res


def feature_selection(X_train, y_train):
    """Executes engineering lookups to extract core contributing predictors."""
    print("Computing feature contribution selectors...")
    
    # Recursive Feature Elimination (RFE)
    lr = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator=lr, n_features_to_select=15)
    rfe.fit(X_train, y_train)
    selected_rfe = X_train.columns[rfe.support_].tolist()
    print(f"   Top RFE Chosen Elements: {selected_rfe[:5]}... (Total: {len(selected_rfe)})")
    return selected_rfe


def main():
    try:
        # Load the clean intermediate file from your data folder
        input_csv = get_data_path("Cleaned-Telco-Customer-Churn.csv")
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Missing intermediate dataset target: {input_csv}")
            
        raw_df = pd.read_csv(input_csv)
        
        # 1. Feature Engineering
        engineered_df = engineer_features(raw_df)

        # 2. Data Stratification Split
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(engineered_df)

        # 3. Transform Strings (Encoding)
        X_train_enc, X_val_enc, X_test_enc, le_dict = encode_categorical_features(X_train, X_val, X_test)

        # 4. Feature Scaling (Using your default choice: StandardScaler)
        X_train_scale, X_val_scale, X_test_scale, scaler = scale_features(X_train_enc, X_val_enc, X_test_enc, method="standard")

        # 5. Class Imbalance Remediation
        X_train_balanced, y_train_balanced = handle_imbalance(X_train_scale, y_train)

        # 6. Feature Selection Review
        _ = feature_selection(X_train_scale, y_train)

        # 7. Persist Transformation Assets to Models Directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.normpath(os.path.join(current_dir, "..", "models"))
        os.makedirs(models_dir, exist_ok=True)

        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
        joblib.dump(le_dict, os.path.join(models_dir, "label_encoder.pkl"))
        joblib.dump(X_train_enc.columns.tolist(), os.path.join(models_dir, "onehot_columns.pkl"))
        print(f"Trained scaling models and artifact weights saved inside: '{models_dir}/'")

        # 8. Export Compiled Complete Frame Product
        final_data_out = get_data_path("preprocessed-Telco-Customer-Churn_final.csv")
        reconstructed_df = pd.concat([engineered_df.drop(columns=["Churn"]), engineered_df["Churn"]], axis=1)
        reconstructed_df.to_csv(final_data_out, index=False)
        print(f"Master dataset product output saved cleanly to: {final_data_out}")
        print("\nPreprocessing Pipeline Complete!")

    except Exception as e:
        print(f"Execution stopped: {e}")


if __name__ == "__main__":
    main()