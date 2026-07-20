import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Create Lifetime Value
df["LTV"] = df["MonthlyCharges"] * df["tenure"]

# Display first few rows
print(df.head())

# Dataset information
print(df.info())

# Basic statistics
print(df.describe())

# Save processed dataset
df.to_csv("data/ltv_dataset.csv", index=False)

print("LTV dataset created successfully.")