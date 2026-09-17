import pandas as pd
import sqlite3
import argparse
import os

def load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM cleaned_customer_data", conn)
    conn.close()
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")
    
    # 1. RFM / Behavioral Proxy
    # Recency / Tenure Buckets
    df['TenureBucket'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 200], labels=['<1 yr', '1-2 yrs', '2-4 yrs', '4+ yrs'])
    
    # Multi-Product Adoption Index
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    # Count of active add-on services ('Yes' = 1, anything else like 'No internet service' or 'No' = 0)
    df['MultiProductAdoptionIndex'] = sum((df[service] == 'Yes').astype(int) for service in services if service in df.columns)

    # 2. Financial Features
    # Charge-to-Tenure Ratio
    df['ChargeToTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # Lifetime Discrepancy Ratio
    df['LifetimeDiscrepancyRatio'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1)
    
    # Drop customerID as it's not useful for modeling
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # 3. Automatic One-Hot Encoding for categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"One-hot encoding columns: {list(categorical_cols)}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    return df

def main(input_db: str, output_csv: str):
    df = load_data(input_db)
    df_features = feature_engineering(df)
    
    print(f"Saving engineered features to {output_csv}...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_features.to_csv(output_csv, index=False)
    print("Feature engineering complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_db', default='data/processed/churn_db.sqlite')
    parser.add_argument('--output_csv', default='data/processed/featured_data.csv')
    args = parser.parse_args()
    main(args.input_db, args.output_csv)
