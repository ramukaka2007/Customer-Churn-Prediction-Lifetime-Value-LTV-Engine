import pandas as pd

# Load Dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Create New Feature
df["AvgMonthlySpend"] = (
    df["TotalCharges"] /
    (df["tenure"] + 1)
)

print("New Feature Created Successfully")

print("\nFirst 5 Rows:")
print(
    df[
        [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "AvgMonthlySpend"
        ]
    ].head()
)
print("\nAverage Monthly Spend by Churn:")

print(
    df.groupby("Churn")["AvgMonthlySpend"].mean()
)