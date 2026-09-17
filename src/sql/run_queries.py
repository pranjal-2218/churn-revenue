import sqlite3
import pandas as pd
import argparse
import os

def run_queries(db_path: str, queries_path: str):
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Please run clean_data.py first.")
        return

    with open(queries_path, 'r') as f:
        sql_script = f.read()
        
    queries = [q.strip() for q in sql_script.split(';') if q.strip()]
    
    conn = sqlite3.connect(db_path)
    
    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1} ---")
        # Extract a simplistic title from the preceding comment if exists
        lines = query.split('\n')
        for line in lines:
            if line.startswith('--'):
                print(line)
        
        try:
            df = pd.read_sql_query(query, conn)
            print(df.to_string(index=False))
        except Exception as e:
            print(f"Error executing query: {e}")
            
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/processed/churn_db.sqlite')
    parser.add_argument('--queries', default='src/sql/queries.sql')
    args = parser.parse_args()
    run_queries(args.db, args.queries)
