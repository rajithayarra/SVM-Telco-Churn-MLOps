from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def load_data(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nDownload the Telco Customer Churn CSV "
            "and place it in data/raw/ with the filename "
            "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    return pd.read_csv(path)
