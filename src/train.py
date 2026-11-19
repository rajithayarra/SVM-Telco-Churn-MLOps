from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import add_features
from src.feature_selection import select_features

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

def run_training():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    df = add_features(clean_data(load_data()))
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Split before fitting transformations to avoid data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"],
        random_state=params["random_state"], stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ])

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)

    selected = select_features(X_train_df, y_train, params["lasso_alpha"])
    X_train_df = X_train_df[selected]
    X_test_df = X_test_df[selected]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_SVM_Churn")
    with mlflow.start_run(run_name=f'{params["version"]}_SVM') as run:
        search = GridSearchCV(
            SVC(probability=True, random_state=params["random_state"]),
            param_grid={
                "kernel": ["linear", "rbf", "poly"],
                "C": [0.1, 1, 10],
                "gamma": ["scale"],
            },
            scoring="f1",
            cv=3,
            n_jobs=-1
        )
        search.fit(X_train_df, y_train)
        model = search.best_estimator_

        pred = model.predict(X_test_df)
        prob = model.predict_proba(X_test_df)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob)
        }

        mlflow.log_param("version", params["version"])
        mlflow.log_param("selected_kernel", search.best_params_["kernel"])
        mlflow.log_param("selected_C", search.best_params_["C"])
        mlflow.log_param("selected_gamma", search.best_params_["gamma"])
        mlflow.log_param("feature_count", len(selected))
        mlflow.log_metrics(metrics)
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.sklearn.log_model(         model, artifact_path="svm_model", registered_model_name="TelcoSVMChurn"      )

        bundle = {
            "preprocessor": preprocessor,
            "selected_features": selected,
            "model": model,
            "metrics": metrics,
            "best_params": search.best_params_,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "input_columns": X.columns.tolist()
        }
        joblib.dump(bundle, ARTIFACT_DIR / "svm_bundle.joblib")
        pd.DataFrame([metrics]).to_csv(ARTIFACT_DIR / "metrics.csv", index=False)
        print("Best parameters:", search.best_params_)
        print("Metrics:", metrics)
        print("MLflow run ID:", run.info.run_id)
        return metrics

if __name__ == "__main__":
    run_training()
