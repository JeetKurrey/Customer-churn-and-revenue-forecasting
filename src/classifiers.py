import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Import local path utility from data loader
from .data_loader import get_data_path

# Force non-interactive plot rendering configuration
plt.ion()


def prepare_and_scale_splits(df):
    """Splits features/targets and replicates data transformations identically to training pipelines."""
    print("Processing data boundaries and executing StandardScaler scales...")
    
    # 1. Isolate feature matrix and ground truth vector targets
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    
    # 2. Re-establish identical stratified train/validation/test cuts matching upstream blocks
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # 3. Apply explicit structural categorical dummy mapping alignments
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_val_enc = pd.get_dummies(X_val, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, drop_first=True)
    
    X_train_enc, X_val_enc = X_train_enc.align(X_val_enc, join="left", axis=1, fill_value=0)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    
    # 4. Fit scaling operators on training subsets and morph verification arrays
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_enc.columns.tolist()


def execute_model_search(X_train, y_train, X_test, y_test):
    """Runs GridSearchCV over 5 standard classifiers and collects baseline score tables."""
    print("Beginning cross-validated grid search optimizations across all candidate spaces...")
    
    # Define hyperparameter grid matrix maps
    model_definitions = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
        }
    }
    
    results_registry = []
    trained_estimators = {}
    
    for name, config in model_definitions.items():
        print(f"Tuning hyper-parameters for estimator architecture: {name}...")
        grid = GridSearchCV(config["model"], config["params"], cv=5, scoring="f1", n_jobs=-1)
        
        start_time = time.time()
        grid.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        best_model = grid.best_estimator_
        trained_estimators[name] = best_model
        
        # Performance Evaluation Calculations
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results_registry.append([name, acc, prec, rec, f1, roc_auc, elapsed_time])
        
    # Convert execution tracking metrics to sorting frame logs
    comparison_df = pd.DataFrame(
        results_registry, 
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Training Time"]
    ).sort_values(by="ROC_AUC", ascending=False)
    
    return comparison_df, trained_estimators


def generate_pipeline_plots(estimators, X_test, y_test, output_dir):
    """Generates unified Multi-Model ROC curves and a comprehensive Confusion Matrix grid layout."""
    print("Constructing combined evaluation plots and multi-model matrix frames...")
    
    # 1. Multi-Model ROC Curve Layout Build
    plt.figure(figsize=(10, 8))
    for name, model in estimators.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", linewidth=2)
        
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guessing (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve - All Models", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_model_roc_curve.png"))
    plt.close()

    # 2. Clean 2x3 Confusion Matrix Subplots Layout Grid
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 11))
    axes = axes.ravel()
    
    for i, (name, model) in enumerate(estimators.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=axes[i]
        )
        axes[i].set_title(f"{name} Confusion Matrix", fontsize=12, fontweight="bold", pad=8)
        axes[i].set_xlabel("Predicted Label", fontsize=10)
        axes[i].set_ylabel("True Label", fontsize=10)
        
    axes[5].axis("off")  # Disable unused trailing 6th panel placeholder slot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_model_confusion_matrices.png"))
    plt.close()


def export_champion_assets(model, scaler, columns, models_dir):
    """Persists model binaries and transformation schemas to the artifact folder."""
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(models_dir, "best_classifier.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(columns, os.path.join(models_dir, "feature_columns.pkl"))
    
    print(f"Champion configuration files successfully saved inside: '{models_dir}/'")


def main():
    try:
        # Resolve path boundaries pointing to preprocessed target files
        input_csv = get_data_path("Processed_002-Telco-Customer-Churn_final.csv")
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Missing master preprocessed tracking dataset: {input_csv}")
            
        df = pd.read_csv(input_csv)
        
        # 1. Transform arrays and structure dataset boundaries
        X_train, X_test, y_train, y_test, scaler, columns = prepare_and_scale_splits(df)
        
        # 2. Run training optimization searches across spaces
        comparison_matrix, estimators = execute_model_search(X_train, y_train, X_test, y_test)
        
        print("\nModel Performance Comparison Summary Leaderboard:")
        print(comparison_matrix.to_string(index=False))
        
        # 3. Isolate top architecture based on target ROC_AUC scores
        champion_name = comparison_matrix.iloc[0]["Model"]
        champion_estimator = estimators[champion_name]
        print(f"\nSelected Pipeline Champion: {champion_name}")
        
        # Print classification validation analytics report for the selected champion
        champion_preds = champion_estimator.predict(X_test)
        print(f"\nProduction Test Set Classification Report ({champion_name}):")
        print(classification_report(y_test, champion_preds))
        
        # 4. Generate visual plot analysis assets
        current_dir = os.path.dirname(os.path.abspath(__file__))
        generate_pipeline_plots(estimators, X_test, y_test, current_dir)
        print(f"Evaluation graph images saved directly within: '{current_dir}/'")
        
        # 5. Serialize best artifacts to persistent disks
        models_directory = os.path.normpath(os.path.join(current_dir, "..", "models"))
        export_champion_assets(champion_estimator, scaler, columns, models_directory)
        
        print("\n Model Tuning and Evaluation Classification Pipeline Completed Successfully!")

    except Exception as e:
        print(f"Execution pipeline stopped: {e}")


if __name__ == "__main__":
    main()