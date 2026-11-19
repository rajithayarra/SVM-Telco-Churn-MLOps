from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from src.predict import predict_one

app = FastAPI(title="Telco Customer Churn SVM API", version="1.0")

class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

@app.get("/")
def home():
    return {"message": "Telco Churn SVM API is running"}

@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        return predict_one(customer.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def metrics():
    path = Path("artifacts/metrics.csv")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    df = pd.read_csv(path)
    return df.iloc[0].to_dict()
