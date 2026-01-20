import pandas as pd

df = pd.read_csv("data/processed/processed_data.csv")

df["loan_to_income_ratio"] = df["loan_amount"] / df["income"]

df.to_csv("data/processed/processed_data.csv", index=False)

print("✅ Feature engineering done")
