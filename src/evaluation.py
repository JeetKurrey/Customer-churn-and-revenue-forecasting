import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, mean_absolute_error, mean_squared_error, r2_score
)

def evaluate_classification_models(models_dict, X_test, y_test, save_dir="plots"):
    """
    Evaluates classification models, exports a comparison table, 
    and generates combined ROC curves and a confusion matrix grid.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    results = []
    roc_data = {}
    cm_data = {}
    
    for name, (model, t_time) in models_dict.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append([name, acc, prec, rec, f1, auc, t_time])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        cm_data[name] = confusion_matrix(y_test, y_pred)

    # Comparison Table
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Training_Time"]
    df_class = pd.DataFrame(results, columns=cols).sort_values(by="ROC_AUC", ascending=False)
    df_class.to_csv(os.path.join(save_dir, "classification_comparison.csv"), index=False)
    
    # ROC Curves Plot
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curves - All Classification Models', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_roc_curve.png"))
    plt.close()

    # Confusion Matrices Plot
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    axes = axes.ravel()
    for i, (name, cm) in enumerate(cm_data.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=axes[i])
        axes[i].set_title(f"{name} Confusion Matrix", fontweight='bold')
    axes[5].axis('off') # Clear unused slot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices_grid.png"))
    plt.close()
    
    return df_class


def evaluate_regression_models(models_dict, X_test, y_test, save_dir="plots"):
    """
    Evaluates regression models, generates a sorted comparison metric dataframe,
    and plots Actual vs Predicted scatter charts alongside a global residual plot.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    results = []
    p = X_test.shape[1]
    n = len(y_test)
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
    axes = axes.ravel()
    
    for i, (name, (model, t_time)) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        
        results.append([name, mae, mse, rmse, r2, adj_r2, t_time])
        
        # Plot Scatter
        axes[i].scatter(y_test, y_pred, alpha=0.5, edgecolors='k', color='skyblue')
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_title(f"{name}: Actual vs Predicted", fontweight='bold')
        axes[i].set_xlabel("Actual Values")
        axes[i].set_ylabel("Predicted Values")
        
    # Clean unused subplot grids
    for j in range(len(models_dict), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regression_actual_vs_predicted.png"))
    plt.close()
    
    cols = ["Model", "MAE", "MSE", "RMSE", "R2", "Adjusted_R2", "Training_Time"]
    df_reg = pd.DataFrame(results, columns=cols).sort_values(by="R2", ascending=False)
    df_reg.to_csv(os.path.join(save_dir, "regression_comparison.csv"), index=False)
    
    return df_reg
