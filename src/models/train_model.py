import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, recall_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sqlite3
try:
    from xgboost import XGBClassifier
    USING_XGB = True
except Exception as e:
    print(f"Warning: XGBoost import failed ({e}). Falling back to sklearn GradientBoostingClassifier.")
    from sklearn.ensemble import GradientBoostingClassifier
    USING_XGB = False
    
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    print("Warning: mlflow not installed, skipping MLflow logging.")
    MLFLOW_AVAILABLE = False

import pickle
import os
import argparse
import warnings

# Add the src folder to sys.path to allow importing from src.features
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.transformers import FeatureEngineerTransformer, OutlierTransformer

warnings.filterwarnings('ignore')

def train_pipeline(db_path: str, models_dir: str):
    print(f"Loading data from {db_path}...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM cleaned_customer_data", conn)
    conn.close()
    
    print("Preparing data for modeling...")
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Building Pipeline...")
    # Define categorical columns to encode. 
    # Notice we include 'TenureBucket' which is created by FeatureEngineerTransformer
    # We must delay finding categorical columns until after feature engineering if we do it dynamically, 
    # but we know the exact columns:
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                            'MultipleLines', 'InternetService', 'OnlineSecurity', 
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                            'StreamingTV', 'StreamingMovies', 'Contract', 
                            'PaperlessBilling', 'PaymentMethod', 'TenureBucket']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    if USING_XGB:
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        classifier = XGBClassifier(
            eval_metric='logloss', 
            scale_pos_weight=pos_weight, 
            random_state=42
        )
    else:
        # Use standard sklearn class explicitly so IDE doesn't complain
        classifier = GradientBoostingClassifier(random_state=42)
        
    pipeline = Pipeline(steps=[
        ('feat_eng', FeatureEngineerTransformer()),
        ('outlier', OutlierTransformer(column='TotalCharges')),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    print("Training Pipeline...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    
    # Calculate optimal threshold for F1 score
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_prob_train)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    print(f"Optimal F1 Threshold (Training): {optimal_threshold:.4f}")
    
    # Test set evaluation
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_opt = (y_prob_test >= optimal_threshold).astype(int)
    
    print(f"\n--- Test Evaluation (Threshold={optimal_threshold:.4f}) ---")
    print(f"Recall: {recall_score(y_test, y_pred_opt):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_opt):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_test):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_opt))
    
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_churn_pipeline.pkl')
    print(f"\nSaving Pipeline to {model_path}...")
    
    # We save the pipeline, the optimal threshold, and feature names just in case
    with open(model_path, 'wb') as f:
        pickle.dump({
            'pipeline': pipeline, 
            'optimal_threshold': float(optimal_threshold)
        }, f)
    
    print("Pipeline training and saving complete!")
    
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("Churn_Prediction")
            with mlflow.start_run():
                mlflow.log_metric("Recall", recall_score(y_test, y_pred_opt))
                mlflow.log_metric("F1-Score", f1_score(y_test, y_pred_opt))
                mlflow.log_metric("ROC-AUC", roc_auc_score(y_test, y_prob_test))
                mlflow.log_metric("Optimal_Threshold", float(optimal_threshold))
                mlflow.sklearn.log_model(pipeline, "xgboost_churn_pipeline")
                print("Successfully logged to MLflow.")
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', default='data/processed/churn_db.sqlite')
    parser.add_argument('--models_dir', default='models')
    args = parser.parse_args()
    
    train_pipeline(args.db_path, args.models_dir)
