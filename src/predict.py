import joblib
import pandas as pd
from src.feature_engineering import add_features

BUNDLE_PATH = "artifacts/svm_bundle.joblib"

def predict_one(payload):
    bundle = joblib.load(BUNDLE_PATH)
    row = pd.DataFrame([payload])

    # Apply the same feature engineering and column ordering used in training.
    row = add_features(row)
    row = row.reindex(columns=bundle["input_columns"])
    transformed = bundle["preprocessor"].transform(row)
    names = bundle["preprocessor"].get_feature_names_out()
    frame = pd.DataFrame(transformed, columns=names)
    frame = frame.reindex(columns=bundle["selected_features"], fill_value=0)

    prediction = int(bundle["model"].predict(frame)[0])
    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    return {"churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4)}
