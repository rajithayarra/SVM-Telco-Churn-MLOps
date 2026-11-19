import pandas as pd

def add_features(df):
    df = df.copy()
    # Avoid division by zero for customers with zero tenure.
    tenure = df["tenure"].replace(0, 1)
    df["Avg_Monthly_Charge_Ratio"] = df["TotalCharges"] / tenure
    return df
