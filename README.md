# Customer Churn Prediction & Lifetime Value (LTV) Engine

## 📌 Project Overview
This project predicts customer churn and estimates Customer Lifetime Value (LTV) using Machine Learning. It also provides business insights for customer retention and will be deployed using FastAPI for real-time predictions.

---

## 🚀 Project Progress

| Week | Status |
|------|--------|
| ✅ Week 1 | Completed |
| ✅ Week 2 | Completed |
| ✅ Week 3 – Day 1 | Completed |
| ✅ Week 3 – Day 2 | Completed |
| ✅ Week 3 – Day 3 | Completed |
| ⏳ Week 3 – Day 4 | In Progress (FastAPI Development) |

---

## ✅ Week 1: Data Preprocessing & Exploratory Data Analysis
- Loaded the Telco Customer Churn dataset
- Performed Exploratory Data Analysis (EDA)
- Handled missing values
- Converted data types
- Visualized customer churn patterns
- Saved cleaned dataset (`Telco_Cleaned.csv`)

---

## ✅ Week 2: Feature Engineering & Churn Prediction
- Performed feature engineering
- Encoded categorical variables
- Trained Logistic Regression model
- Trained Random Forest Classifier
- Trained XGBoost Classifier
- Compared classification models
- Selected the best-performing classifier
- Implemented SHAP explainability
- Identified key factors influencing customer churn

---

## ✅ Week 3: Customer Lifetime Value (LTV)

### Day 1 – Feature Engineering
- Created `AvgChargePerMonth`
- Created `TotalServices`
- Created `TenureGroup`
- Created `ChargeCategory`
- Saved `Telco_LTV_FeatureEngineered.csv`

### Day 2 – Regression Modeling
- Created `EstimatedLTV` target variable
- Prepared regression dataset
- Split training and testing datasets
- Trained Linear Regression model
- Evaluated model performance using MAE, RMSE, and R² Score

### Day 3 – Model Comparison
- Trained Random Forest Regressor
- Trained XGBoost Regressor
- Compared regression models
- Selected XGBoost Regressor as the final LTV prediction model

#### 🏆 Best Model Performance (XGBoost)
- **MAE:** 32.21
- **RMSE:** 49.52
- **R² Score:** 0.999529

---

## 🔜 Upcoming Work
- Build FastAPI application
- Create Single Customer Prediction API
- Create Batch Prediction API
- Deploy the API
- Add business recommendation features
- Complete project documentation

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- FastAPI (Upcoming)
- GitHub

---
## 🚀 Project Progress
- ✅ Data Collection
- ✅ Data Cleaning & Preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering
- ✅ Estimated Customer Lifetime Value (LTV) Calculation
- ⏳ Machine Learning Model Training (In Progress)
- ⏳ Dashboard Development (Upcoming)
