# ==========================================
# Week 3 - Day 2
# LTV Regression Models
# ==========================================

# Import Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load Dataset
# ==========================================

print("Loading LTV dataset...\n")

df = pd.read_csv("data/ltv_dataset.csv")

print(df.head())

# ==========================================
# Features and Target
# ==========================================

# ==========================================
# Select Numerical Features
# ==========================================

features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X = df[features]

y = df["LTV"]

print("\nFeatures Used:")
print(features)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain-Test Split")

print("Training Data :", X_train.shape)

print("Testing Data  :", X_test.shape)

# ==========================================
# Linear Regression
# ==========================================

print("\nTraining Linear Regression...")

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

# ==========================================
# Random Forest Regression
# ==========================================

print("\nTraining Random Forest Regressor...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# ==========================================
# Evaluation Function
# ==========================================

def evaluate_model(model_name, actual, predicted):

    print("\n====================================")

    print(model_name)

    print("====================================")

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = mse ** 0.5

    r2 = r2_score(actual, predicted)

    print("Mean Absolute Error :", round(mae, 2))

    print("Mean Squared Error  :", round(mse, 2))

    print("Root Mean Squared Error :", round(rmse, 2))

    print("R2 Score :", round(r2, 4))

# ==========================================
# Evaluate Models
# ==========================================

evaluate_model(
    "Linear Regression",
    y_test,
    linear_predictions
)

evaluate_model(
    "Random Forest Regressor",
    y_test,
    rf_predictions
)

# ==========================================
# Save Models
# ==========================================

joblib.dump(
    linear_model,
    "models/linear_regression.pkl"
)

joblib.dump(
    rf_model,
    "models/random_forest_regressor.pkl"
)

print("\nModels saved successfully!")

print("\nFiles created:")

print("models/linear_regression.pkl")

print("models/random_forest_regressor.pkl")