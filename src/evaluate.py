import pandas as pd
from pathlib import Path

def show_metrics():
    path = Path("artifacts/metrics.csv")
    if not path.exists():
        raise FileNotFoundError("Train the model first: python main.py")
    print(pd.read_csv(path).to_string(index=False))

if __name__ == "__main__":
    show_metrics()
