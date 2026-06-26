import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import load_and_validate_data
from src.eda import clean_and_prepare_data
from src.preprocessing import engineer_features
from src.preprocessing import split_data
from src.preprocessing import encode_categorical_features
from src.preprocessing import scale_features
from src.evaluation import evaluate_classification_models,evaluate_regression_models
from sklearn.model_selection import train_test_split


def test_evaluate_classification_models():

    # Prepare data
    df = load_and_validate_data()
    df = clean_and_prepare_data(df)
    df = engineer_features(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Encode and scale
    X_train, X_val, X_test, _ = encode_categorical_features(
        X_train, X_val, X_test
    )

    X_train, X_val, X_test, _ = scale_features(
        X_train, X_val, X_test
    )

    # Train one simple model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Create models_dict in the format expected by evaluation function
    models_dict = {
        "Logistic Regression": (model, 1.0)
    }

    # Evaluate
    results = evaluate_classification_models(
        models_dict,
        X_test,
        y_test,
        save_dir="test_plots"
    )

    # Assertions
    assert isinstance(results, pd.DataFrame)
    assert not results.empty

    assert "Model" in results.columns
    assert "Accuracy" in results.columns
    assert "Precision" in results.columns
    assert "Recall" in results.columns
    assert "F1" in results.columns
    assert "ROC_AUC" in results.columns


def test_evaluate_regression_models():
    
    df = load_and_validate_data()
    df = clean_and_prepare_data(df)
    df = engineer_features(df)

    X = df.drop(columns=["MonthlyCharges"])
    y = df["MonthlyCharges"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train, X_test = pd.get_dummies(X_train), pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    models_dict = {
        "Random Forest": (model, 1.0)
    }

    results = evaluate_regression_models(
        models_dict,
        X_test,
        y_test
    )

    assert not results.empty