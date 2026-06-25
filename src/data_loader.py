import os
import pandas as pd


def get_data_path(filename="Telco-Customer-Churn.csv"):
    """Calculates the absolute path to the data directory from this script's location."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(current_dir, "..", "data", filename))


def load_and_validate_data(filename="Telco-Customer-Churn.csv"):
    """Loads the dataset and validates that essential columns are present."""
    data_path = get_data_path(filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at: {os.path.abspath(data_path)}"
        )

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("The loaded dataset is completely empty.")

    # Core columns needed for your analysis pipeline
    required_columns = ["customerID", "Churn", "MonthlyCharges", "TotalCharges", "tenure"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing critical columns: {missing_cols}")

    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Test execution block to verify the loader independently
    try:
        test_df = load_and_validate_data()
    except Exception as e:
        print(e)