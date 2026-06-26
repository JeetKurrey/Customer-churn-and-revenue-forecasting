import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import path helper from local modules
from data_loader import get_data_path

# Force non-interactive plot rendering configuration
plt.ion()


def prepare_and_transform_data(df):
    """Cleans columns, isolates boundaries, and applies pipeline transformations."""
    print("Preparing data matrix boundaries and pipeline transformations...")
    df = df.copy()

    # Isolate targets and features
    y = df["MonthlyCharges"]
    X = df.drop(columns=["MonthlyCharges"])

    # Establish stratified training train/validation/test splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Coerce data types safely across partitions
    for subset in [X_train, X_val, X_test]:
        if "TotalCharges" in subset.columns and subset["TotalCharges"].dtype == "object":
            subset["TotalCharges"] = pd.to_numeric(subset["TotalCharges"], errors="coerce")
            subset["TotalCharges"].fillna(0, inplace=True)

    # Separate numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    # Create composite production pipeline preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    # Process sparse transforms across dataset partitions
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor


def execute_regression_search(X_train, y_train, X_test, y_test):
    """Runs a hyperparameter grid search across 7 candidate regression estimators."""
    print("Beginning cross-validated grid search optimizations across estimators...")

    model_definitions = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {}
        },
        "Ridge Regression": {
            "model": Ridge(),
            "params": {"alpha": [0.01, 0.1, 1, 10]}
        },
        "Lasso Regression": {
            "model": Lasso(max_iter=5000),
            "params": {"alpha": [0.001, 0.01, 0.1, 1]}
        },
        "ElasticNet": {
            "model": ElasticNet(max_iter=10000),
            "params": {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]}
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
        },
        "SVR": {
            "model": SVR(),
            "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}
        }
    }

    results_registry = []
    trained_estimators = {}

    n = len(y_test)
    p = X_test.shape[1]

    for name, config in model_definitions.items():
        print(f"Tuning parameters for model: {name}...")
        grid = GridSearchCV(config["model"], config["params"], cv=5, scoring="r2", n_jobs=-1)
        
        start_time = time.time()
        grid.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        best_model = grid.best_estimator_
        trained_estimators[name] = best_model
        
        # Calculate metric vectors
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        
        results_registry.append([name, mae, mse, rmse, r2, adj_r2, elapsed_time])

    comparison_df = pd.DataFrame(
        results_registry,
        columns=["Model", "MAE", "MSE", "RMSE", "R2", "Adjusted_R2", "Training_Time"]
    ).sort_values(by="R2", ascending=False)

    return comparison_df, trained_estimators


def generate_residual_analysis(model, X_test, y_test, output_dir):
    """Saves Actual vs Predicted plots and residual distributions for error tracking."""
    print("Constructing pipeline error metrics and distribution diagnostic plots...")
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # 1. Scatter distribution check
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#4682B4")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual Monthly Charges")
    plt.ylabel("Predicted Monthly Charges")
    plt.title("Actual vs. Predicted Revenue Values")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_actual_vs_predicted.png"))
    plt.close()

    # 2. Residual error tracking checks
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="#2E8B57")
    plt.title("Residual Error Terms Distribution")
    plt.xlabel("Residual Error Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_residual_distribution.png"))
    plt.close()


def main():
    try:
        # Load upstream preprocessed files
        input_csv = get_data_path("Processed_002-Telco-Customer-Churn_final.csv")
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Missing master preprocessed tracking dataset: {input_csv}")

        df = pd.read_csv(input_csv)

        # 1. Build and run preprocessing transforms
        X_train, X_test, y_train, y_test, preprocessor = prepare_and_transform_data(df)

        # 2. Run hyperparameter grid optimizations
        comparison_matrix, estimators = execute_regression_search(X_train, y_train, X_test, y_test)

        print("\n Regression Leaderboard Metrics (Sorted by R²):")
        print(comparison_matrix.to_string(index=False))

        # 3. Extract the top-performing model configuration
        champion_name = comparison_matrix.iloc[0]["Model"]
        champion_model = estimators[champion_name]
        print(f"\n Selected Pipeline Champion: {champion_name}")

        # 4. Generate visual plot diagnostic images
        current_dir = os.path.dirname(os.path.abspath(__file__))
        generate_residual_analysis(champion_model, X_test, y_test, current_dir)
        print(f"  Diagnostic plots saved successfully to: '{current_dir}/'")

        # 5. Persist champion binaries to disk
        models_directory = os.path.normpath(os.path.join(current_dir, "..", "models"))
        os.makedirs(models_directory, exist_ok=True)

        joblib.dump(champion_model, os.path.join(models_directory, "best_regressor.pkl"))
        joblib.dump(preprocessor, os.path.join(models_directory, "regression_preprocessor.pkl"))
        print(f" Trained model binaries and transforms saved to: '{models_directory}/'")

        print("\n Revenue Forecasting Regression Pipeline Completed Successfully!")

    except Exception as e:
        print(f" Execution pipeline stopped: {e}")


if __name__ == "__main__":
    main()