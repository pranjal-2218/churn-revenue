import pandas as pd
import numpy as np
import sqlite3
import os
import argparse

def process_data(input_path: str, db_path: str):
    """
    Validates, cleans, and saves the customer churn data.
    """
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)

    print("Cleaning TotalCharges...")
    # Coerce to numeric, turning space strings ' ' into NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Impute missing TotalCharges. Use MonthlyCharges * tenure if tenure > 0, otherwise median or 0.
    # New customers (tenure=0) might have NaN TotalCharges. We'll set it to 0 or their MonthlyCharge.
    missing_mask = df['TotalCharges'].isna()
    df.loc[missing_mask, 'TotalCharges'] = np.where(
        df.loc[missing_mask, 'tenure'] > 0,
        df.loc[missing_mask, 'MonthlyCharges'] * df.loc[missing_mask, 'tenure'],
        0  # For 0 tenure, total charges are usually 0
    )

    print("Encoding Churn column...")
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print(f"Saving to SQLite database at {db_path}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create SQLite connection
    conn = sqlite3.connect(db_path)
    df.to_sql('cleaned_customer_data', conn, if_exists='replace', index=False)
    conn.close()
    
    print("Data cleaning and validation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--db', default='data/processed/churn_db.sqlite')
    args = parser.parse_args()
    process_data(args.input, args.db)
