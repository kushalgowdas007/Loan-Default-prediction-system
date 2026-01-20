import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found")
    return pd.read_csv(path)

def clean_data(df):
    # Rename columns (Kaggle → Project format)
    df = df.rename(columns={
        "loan_amnt": "loan_amount",
        "int_rate": "interest_rate",
        "annual_inc": "income",
        "term": "loan_term",
        "loan_status": "loan_status"
    })

    # Keep only required columns
    df = df[
        ["loan_amount", "interest_rate", "income",
         "credit_score", "loan_term", "loan_status"]
    ]

    # Convert loan_status to binary
    df["loan_status"] = df["loan_status"].apply(
        lambda x: 1 if str(x).lower() in ["default", "charged off", "1"] else 0
    )

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df

if __name__ == "__main__":
    df = load_data("data/raw/loan_data.csv")
    df = clean_data(df)
    df.to_csv("data/processed/processed_data.csv", index=False)
    print("✅ Kaggle data preprocessing completed")
