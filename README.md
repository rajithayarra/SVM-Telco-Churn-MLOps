# SVM Telco Customer Churn – MLOps Lab Project

## 1. Dataset
Download the IBM Telco Customer Churn CSV (commonly named
`WA_Fn-UseC_-Telco-Customer-Churn.csv`) from Kaggle and place it in:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The CSV is intentionally not included in this ZIP. Use the dataset version permitted by your course.

## 2. Environment setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

## 3. Run the complete pipeline
```powershell
python main.py
```
This cleans data, creates `Avg_Monthly_Charge_Ratio`, encodes/scales features,
performs correlation-based feature filtering followed by LASSO feature selection, tunes SVM kernels and C, evaluates the model,
logs the run to MLflow, and saves the model bundle and metrics.

## 4. View MLflow
Open a second terminal in the project folder:
```powershell
mlflow ui
```
Open `http://127.0.0.1:5000` in your browser. Select `Telco_SVM_Churn`
and compare runs and metrics.

## 5. Create V1 and V2
Run V1:
```powershell
python main.py
```
Edit `params.yaml` (for example, change `version: V1` to `version: V2`,
or change `lasso_alpha` / random_state), then run:
```powershell
python main.py
```
For a meaningful V1-vs-V2 comparison, keep the same train/test split and
change a model setting or feature-selection setting. Compare Accuracy,
Precision, Recall, F1 and ROC-AUC in MLflow. Do not claim V2 is better unless
the measured metrics support that conclusion.

## 6. Git version control
```powershell
git init
git add .
git commit -m "Initial SVM churn pipeline"
```
After modifying params or code:
```powershell
git add .
git commit -m "Create V2 SVM experiment"
```

## 7. DVC dataset tracking
Initialize DVC after Git:
```powershell
dvc init
dvc add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
git add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc .gitignore
git commit -m "Track Telco dataset with DVC"
```
Run and reproduce the pipeline:
```powershell
dvc repro
```
To use a remote DVC store, configure one supported by your lab (local, shared or cloud); do not assume a remote is configured.

## 8. FastAPI deployment
Train first, then start the API from the project root:
```powershell
uvicorn api.app:app --reload
```
Open Swagger UI: `http://127.0.0.1:8000/docs`
- `GET /` checks service status.
- `POST /predict` predicts churn.
- `GET /metrics` returns saved evaluation metrics.

## 9. Example valid request
Use the following JSON in Swagger's `/predict` endpoint:
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 844.2
}
```

Invalid payload tests:
- Set `SeniorCitizen` to `5` (fails range validation).
- Set `MonthlyCharges` to `-10` (fails range validation).
- Remove `Contract` (fails required-field validation).
- Add an unexpected field (rejected because extra fields are forbidden).

## 10. Important lab notes
- SVM requires scaled features; scaling is fitted only on training data.
- The test set is held out from preprocessing fitting and model tuning.
- The positive class is Churn = Yes.
- ROC-AUC uses predicted probabilities (`probability=True`).
- The model bundle contains the fitted preprocessor, selected feature names and SVM model.
- This starter project logs each run to MLflow's local tracking store. Model Registry stage
  promotion is not automatic here; use the MLflow UI / registry workflow if your MLflow
  installation and backend support it.
- Current MLflow versions may use model aliases instead of the older Production stage.

\n## 11. Register and promote the best version\n\nEach training run registers a new version of `TelcoSVMChurn`. Start `mlflow ui`, open\n**Models → TelcoSVMChurn**, inspect the linked runs and compare F1 / ROC-AUC (and the other\nmetrics). Select the best version and assign the model alias `production`. For a current\nMLflow Python client, the equivalent operation is:\n\n```python\nfrom mlflow import MlflowClient\nclient = MlflowClient(tracking_uri="sqlite:///mlflow.db")\nclient.set_registered_model_alias("TelcoSVMChurn", "production", "2")  # replace 2 with winning version\n```\n\nOnly assign the alias after checking the actual results. A version number is an example,\nnot a claim that version 2 is best.\n