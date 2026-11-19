
import pandas as pd
import os

TARGET = "Churn"

def clean_data(df):
    df = df.copy()

    # Remove duplicate records
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"])
    else:
        df = df.drop_duplicates()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Handle missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Remove rows without target
    df = df.dropna(subset=[TARGET])

    # Save processed dataset
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(
        "data/processed/cleaned_telco_churn.csv",
        index=False
    )

    print("Processed dataset saved successfully!")
    print("Shape:", df.shape)

    return df
