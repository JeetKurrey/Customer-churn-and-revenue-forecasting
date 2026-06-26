import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import our custom loader function
from .data_loader import load_and_validate_data

plt.ion()

def clean_and_prepare_data(df):
    """Handles data formatting, drops missing entries, and generates tenure groups."""
    print("\nStarting data cleaning process...")
    tele_data = df.copy()

    # Convert TotalCharges to numeric, coercing errors to NaN
    tele_data["TotalCharges"] = pd.to_numeric(tele_data["TotalCharges"], errors="coerce")

    # Handle missing values (0.15% missing data is safe to drop)
    initial_rows = len(tele_data)
    tele_data.dropna(inplace=True)
    print(f"   Dropped {initial_rows - len(tele_data)} rows containing empty values.")

    # Generate demographic tenure bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    tele_data["tenure_group"] = pd.cut(
        tele_data["tenure"], range(1, 80, 12), right=False, labels=labels
    )

    # Drop non-predictive identifiers
    tele_data.drop(columns=["customerID"], inplace=True, errors="ignore")
    return tele_data


def generate_visualizations(df, output_dir):
    """Generates all the plot distributions and bivariate analyses from the notebook."""
    print("\nGenerating and saving pipeline visualizations...")

    # 1. Churn Class Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Churn", data=df)
    plt.title("Distribution in Churn Column")
    plt.savefig(os.path.join(output_dir, "churn_distribution.png"))
    plt.close()

    # 2. Numerical Features Histograms
    numerical_cols = ["MonthlyCharges", "TotalCharges"]
    df[numerical_cols].hist(figsize=(12, 5), bins=50)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "numerical_histograms.png"))
    plt.close()

    # 3. Outlier Analysis Boxplots & IQR Metrics
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Outlier Check: {col}")
        plt.savefig(os.path.join(output_dir, f"{col}_boxplot.png"))
        plt.close()

    # 4. Correlation Matrix Heatmap
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numerical Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    # 5. Full Categorical Subplots Grid
    categorical_features = df.drop(columns=["Churn", "TotalCharges", "MonthlyEntries"], errors="ignore").select_dtypes(include=["object", "category"]).columns
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(22, 20))
    axes = axes.flatten()

    for i, predictor in enumerate(categorical_features):
        sns.countplot(data=df, x=predictor, hue="Churn", palette="Set2", ax=axes[i])
        axes[i].set_title(f"Churn by {predictor}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "categorical_features_grid.png"))
    plt.close()

    # 6. Optimized Reusable Stacked Percentage Analysis Plotting Loop
    bivariate_targets = [
        ("Contract", "churn_vs_contract.png", "Percentage of Churn Across Contract Types"),
        ("OnlineSecurity", "churn_vs_onlinesecurity.png", "Percentage of Churn Across Online Security"),
        ("TechSupport", "churn_vs_techsupport.png", "Percentage of Churn Across Tech Support"),
        ("gender", "churn_vs_gender.png", "Percentage of Churn Across Genders"),
        ("tenure_group", "churn_vs_tenure.png", "Percentage of Churn Across Tenure Groups")
    ]

    for feature, filename, title in bivariate_targets:
        cross_tab = pd.crosstab(df[feature], df["Churn"], normalize="index") * 100
        cross_tab.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#66c2a5", "#fc8d62"])
        plt.title(title)
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=0 if feature != "tenure_group" else 30)
        plt.legend(title="Churn", loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def main():
    try:
        # Load Raw Data using module boundary loading
        raw_df = load_and_validate_data()
        
        # Track our operational directory paths
        current_src_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Clean data structures
        processed_data = clean_and_prepare_data(raw_df)
        
        # Run Analytics & Export Visualizations
        generate_visualizations(processed_data, current_src_dir)
        
        # Save down processed files to the root data workspace folder
        output_file_path = os.path.normpath(os.path.join(current_src_dir, "..", "data", "Cleaned-Telco-Customer-Churn.csv"))
        processed_data.to_csv(output_file_path, index=False)
        
        print(f"\nEDA Pipeline execution complete!")
        print(f" preprocessed data file saved to: {output_file_path}")
        print(f"All distribution plots have been saved inside the 'src/' folder.")

    except Exception as e:
        print(f"Pipeline stopped: {e}")


if __name__ == "__main__":
    main()