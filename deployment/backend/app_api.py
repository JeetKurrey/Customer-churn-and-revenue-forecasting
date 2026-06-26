from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Loading model artifacts using relative paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def _load(filename):
    """Load one artifact — raises a clear error if the file is missing."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n  ✗ Missing model file: {path}"
            f"\n  Make sure '{filename}' is inside the 'models/' folder "
            f"next to app_api.py.\n"
        )
    obj = joblib.load(path)
    print(f"  ✓ Loaded {filename}")
    return obj

print(f"\nLoading artifacts from: {MODELS_DIR}")
try:
    classifier    = _load("best_classifier.pkl")   # best classifier
    regressor     = _load("best_regressor.pkl")    # best regressor 
    std_scaler    = _load("scaler.pkl")            # StandardScaler fitted
    X_train_ref = _load("feature_columns.pkl")
    EXPECTED_FEATURES = list(X_train_ref.columns)
    print("\nExpected Features:")
    print(EXPECTED_FEATURES)
    print("Count:", len(EXPECTED_FEATURES))  
    print(f"\nAll artifacts loaded ✓  —  expecting {len(EXPECTED_FEATURES)} features.")
    print("Feature columns:", EXPECTED_FEATURES, "\n")
    print(type(std_scaler))
    print("Scaler features:", std_scaler.n_features_in_)
except Exception as e:
    # Print the full error and re-raise so Flask refuses to start
    print(f"\n{'='*60}")
    print("FATAL — could not load model artifacts. Fix the error below")
    print("before starting Flask, or predictions will always fail.")
    print(f"{'='*60}")
    raise

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("\nReceived Keys:")
        print(sorted(data.keys()))

        print("\nExpected Features:")
        print(sorted(EXPECTED_FEATURES))
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload received."}), 400

        final_input_df = pd.DataFrame([data])
        final_input_df = final_input_df.reindex(columns=EXPECTED_FEATURES,fill_value=0)

        print("\nDataFrame Columns:")
        print(sorted(final_input_df.columns.tolist()))

        scaled_input = std_scaler.transform(final_input_df)

        print("\nRaw Input:")
        print(final_input_df.iloc[0])

        print("\nScaled Input:")
        print(scaled_input[0])

        
        prediction = int(classifier.predict(scaled_input)[0])
        probability = float(classifier.predict_proba(scaled_input)[0][1])

        revenue = float(regressor.predict(scaled_input)[0])

        # ── Risk tier ───────────────────────────────────────────────────────
        if probability < 0.2:
            risk_level = "Low"
        elif probability < 0.5:
            risk_level = "Medium"
        elif probability < 0.8:
            risk_level = "High"
        else:
            risk_level = "Critical"

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "expected_revenue": round(revenue, 2)
        }), 200

    except Exception as e:
        print("Backend Error:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)