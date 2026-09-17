import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def generate_eda(db_path: str, output_dir: str):
    print("Generating EDA reports...")
    os.makedirs(output_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM cleaned_customer_data", conn)
    conn.close()

    # 1. Tenure distribution vs Churn status
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
    plt.title('Tenure Distribution by Churn Status')
    plt.savefig(os.path.join(output_dir, 'tenure_vs_churn.png'))
    plt.close()

    # 2. MonthlyCharges density plot by Churn
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False, fill=True)
    plt.title('Monthly Charges Density by Churn Status')
    plt.savefig(os.path.join(output_dir, 'monthly_charges_density.png'))
    plt.close()

    # 3. Correlation matrix (Numeric and Binary encoded variables)
    plt.figure(figsize=(12, 10))
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Numeric Feature Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # 4. Customer segmentation summary (Tenure buckets x Charge tiers)
    df['TenureBucket'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 100], labels=['<1 yr', '1-2 yrs', '2-4 yrs', '4+ yrs'])
    df['ChargeTier'] = pd.qcut(df['MonthlyCharges'], q=3, labels=['Low', 'Medium', 'High'])
    
    segmentation = df.groupby(['TenureBucket', 'ChargeTier'], observed=True)['Churn'].mean().unstack()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(segmentation, annot=True, cmap='Reds', fmt=".1%")
    plt.title('Churn Rate by Customer Segment (Tenure x Charge Tier)')
    plt.ylabel('Tenure Bucket')
    plt.xlabel('Monthly Charge Tier')
    plt.savefig(os.path.join(output_dir, 'customer_segmentation.png'))
    plt.close()

    print(f"EDA generation complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/processed/churn_db.sqlite')
    parser.add_argument('--output', default='artifacts/eda')
    args = parser.parse_args()
    generate_eda(args.db, args.output)
