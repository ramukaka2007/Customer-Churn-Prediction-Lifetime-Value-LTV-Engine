-- Total Customers
SELECT COUNT(*) AS total_customers
FROM churn_data;

-- Churn Distribution
SELECT churn, COUNT(*) AS customers
FROM churn_data
GROUP BY churn;

-- Contract Type Analysis
SELECT contract, COUNT(*) AS customers
FROM churn_data
GROUP BY contract;

-- Average Monthly Charges
SELECT ROUND(AVG(monthlycharges),2) AS average_monthly_charge
FROM churn_data;

-- Average Tenure
SELECT ROUND(AVG(tenure),2) AS average_tenure
FROM churn_data;

-- Senior Citizen Distribution
SELECT seniorcitizen, COUNT(*) AS customers
FROM churn_data
GROUP BY seniorcitizen;