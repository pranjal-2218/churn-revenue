import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")
    
    # Use python executable from environment to ensure we use the venv
    cmd = [sys.executable] + command.split()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}.")
        sys.exit(1)

def main():
    print("Starting Customer Churn & Revenue Intelligence Platform Pipeline...\n")
    
    run_command("src/data/clean_data.py", "Data Validation & Cleaning")
    run_command("src/sql/run_queries.py", "SQL Customer Analytics")
    run_command("src/eda/generate_eda.py", "Exploratory Data Analysis")
    run_command("src/features/build_features.py", "Feature Engineering")
    run_command("src/models/train_model.py", "ML Pipeline & Evaluation")
    run_command("src/monitoring/export_powerbi.py", "Power BI Export Layer")
    
    print("\n🎉 Pipeline Execution Complete!")
    print("You can now start the FastAPI server with: uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    main()
