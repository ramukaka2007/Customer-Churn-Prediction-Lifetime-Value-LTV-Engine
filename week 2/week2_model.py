import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert Churn
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove unnecessary column
df = df.drop("customerID", axis=1)

# One-Hot Encoding
df = pd.get_dummies(
    df,
    drop_first=True
)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Set Shape:")
print(X_train.shape)

print("\nTesting Set Shape:")
print(X_test.shape)