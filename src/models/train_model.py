import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    USING_XGB = True
except Exception as e:
    print(f"Warning: XGBoost import failed ({e}). Falling back to sklearn GradientBoostingClassifier.")
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    USING_XGB = False
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate(df: pd.DataFrame, models_dir: str):
    print("Preparing data for modeling...")
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}

    # 1. Logistic Regression
    print("Training Logistic Regression (Baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = {
        'Recall': recall_score(y_test, y_pred_lr),
        'F1-Score': f1_score(y_test, y_pred_lr),
        'ROC-AUC': roc_auc_score(y_test, y_prob_lr),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_lr)
    }

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'Recall': recall_score(y_test, y_pred_rf),
        'F1-Score': f1_score(y_test, y_pred_rf),
        'ROC-AUC': roc_auc_score(y_test, y_prob_rf),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_rf)
    }

    # 3. XGBoost with GridSearchCV
    print("Training Gradient Boosting Model...")
    
    if not USING_XGB:
        xgb_base = XGBClassifier(random_state=42)
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200]
        }
    else:
        xgb_base = XGBClassifier(eval_metric='logloss', random_state=42)
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'scale_pos_weight': [1, (len(y_train) - sum(y_train)) / sum(y_train)]
        }
    
    grid = GridSearchCV(estimator=xgb_base, param_grid=param_grid, scoring='recall', cv=skf, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_xgb = grid.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
    
    results['XGBoost'] = {
        'Recall': recall_score(y_test, y_pred_xgb),
        'F1-Score': f1_score(y_test, y_pred_xgb),
        'ROC-AUC': roc_auc_score(y_test, y_prob_xgb),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_xgb)
    }

    # Print results
    print("\n--- Model Evaluation Results ---")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            if metric_name != 'Confusion Matrix':
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}:\n{value}")

    # Save best model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_churn_model.pkl')
    print(f"\nSaving XGBoost model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': best_xgb, 'features': list(X.columns)}, f)

    # Feature Importance Chart
    print("Generating Feature Importance chart...")
    feature_importances = best_xgb.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    top_features = [X.columns[i] for i in sorted_idx][:10]
    top_importances = [feature_importances[i] for i in sorted_idx][:10]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title('Top 10 Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'feature_importance.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/featured_data.csv')
    parser.add_argument('--models_dir', default='models')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    train_and_evaluate(df, args.models_dir)
