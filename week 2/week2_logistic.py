import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove ID column
df = df.drop("customerID", axis=1)

# Encode categorical columns
df = pd.get_dummies(
    df,
    drop_first=True
)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Model Accuracy:")
print(round(accuracy * 100, 2), "%")