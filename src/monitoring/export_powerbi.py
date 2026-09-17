import pandas as pd
import sqlite3
import argparse
import os

def export_for_powerbi(db_path: str, output_csv: str):
    print(f"Exporting Data Mart for Power BI from {db_path}...")
    
    if not os.path.exists(db_path):
        print("Database not found. Run previous pipeline steps first.")
        return
        
    conn = sqlite3.connect(db_path)
    # Extract the base clean data
    df = pd.read_sql_query("SELECT * FROM cleaned_customer_data", conn)
    conn.close()
    
    # Add a few derived columns to make Power BI dashboards easier
    df['CLV_Proxy'] = df['tenure'] * df['MonthlyCharges']
    
    # Export as CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Power BI Data Mart successfully saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_db', default='data/processed/churn_db.sqlite')
    parser.add_argument('--output_csv', default='data/processed/powerbi_churn_mart.csv')
    args = parser.parse_args()
    export_for_powerbi(args.input_db, args.output_csv)
