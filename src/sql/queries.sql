-- 1. Average Revenue Per User (ARPU) by Contract type
SELECT 
    Contract,
    AVG(MonthlyCharges) AS ARPU,
    COUNT(*) AS CustomerCount
FROM cleaned_customer_data
GROUP BY Contract
ORDER BY ARPU DESC;

-- 2. Churn Rate by Internet Service and Payment Method
SELECT 
    InternetService,
    PaymentMethod,
    COUNT(*) as TotalCustomers,
    SUM(Churn) as ChurnedCustomers,
    CAST(SUM(Churn) AS FLOAT) / COUNT(*) AS ChurnRate
FROM cleaned_customer_data
GROUP BY InternetService, PaymentMethod
ORDER BY ChurnRate DESC;

-- 3. Customer Lifetime Value (CLV) proxy metrics (tenure * MonthlyCharges) 
-- Average CLV by Churn Status
SELECT 
    Churn,
    AVG(tenure * MonthlyCharges) AS Avg_CLV_Proxy,
    COUNT(*) AS CustomerCount
FROM cleaned_customer_data
GROUP BY Churn;

-- 4. High-Risk Revenue at Stake ($ loss per cohort)
-- Based on current MonthlyCharges for customers who churned
SELECT 
    Contract,
    InternetService,
    SUM(MonthlyCharges) AS MonthlyRevenueLost
FROM cleaned_customer_data
WHERE Churn = 1
GROUP BY Contract, InternetService
ORDER BY MonthlyRevenueLost DESC;
