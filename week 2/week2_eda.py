import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

# Convert Target Column
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nChurn Values:")
print(df["Churn"].head())

print("\nChurn Count:")
print(df["Churn"].value_counts())

# Remove customerID

df = df.drop("customerID", axis=1)

# Convert all text columns
# Convert TotalCharges to numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges Type:")
print(df["TotalCharges"].dtype)
df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

print("\nEncoded Dataset Shape:")
print(df_encoded.shape)

print("\nFirst 5 Rows:")
print(df_encoded.head())