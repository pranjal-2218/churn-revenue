import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Applies basic preprocessing (type coercion, imputation) and generates new features.
    No statistics are learned here, so fit() just returns self.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # We work on a copy to avoid modifying the original dataframe
        df = X.copy()
        
        # 1. Type Coercion and Imputation (for raw incoming data)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            missing_mask = df['TotalCharges'].isna()
            if missing_mask.any():
                df.loc[missing_mask, 'TotalCharges'] = np.where(
                    df.loc[missing_mask, 'tenure'] > 0,
                    df.loc[missing_mask, 'MonthlyCharges'] * df.loc[missing_mask, 'tenure'],
                    0
                )
                
        # 2. RFM / Behavioral Proxy Features
        # Recency / Tenure Buckets
        if 'tenure' in df.columns:
            df['TenureBucket'] = pd.cut(
                df['tenure'], 
                bins=[-1, 12, 24, 48, 200], 
                labels=['<1 yr', '1-2 yrs', '2-4 yrs', '4+ yrs']
            )
            # To ensure it gets treated as object/string for OneHotEncoder, convert categorical to string
            df['TenureBucket'] = df['TenureBucket'].astype(str)

        # Multi-Product Adoption Index
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        available_services = [s for s in services if s in df.columns]
        if available_services:
            df['MultiProductAdoptionIndex'] = sum((df[service] == 'Yes').astype(int) for service in available_services)
        
        # 3. Financial Features
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['ChargeToTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
            
        if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['LifetimeDiscrepancyRatio'] = df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges'] + 1)
            
        # Drop customerID as it's not useful for modeling
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
            
        return df

class OutlierTransformer(BaseEstimator, TransformerMixin):
    """
    Learns IQR bounds on a specified column during fit() and flags outliers during transform().
    """
    def __init__(self, column: str = 'TotalCharges'):
        self.column = column
        self.upper_bound_ = None
        self.lower_bound_ = None
        
    def fit(self, X, y=None):
        if self.column in X.columns:
            Q1 = X[self.column].quantile(0.25)
            Q3 = X[self.column].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound_ = Q1 - 1.5 * IQR
            self.upper_bound_ = Q3 + 1.5 * IQR
        else:
            self.upper_bound_ = float('inf')
            self.lower_bound_ = float('-inf')
        return self
        
    def transform(self, X):
        df = X.copy()
        if self.column in df.columns and self.upper_bound_ is not None:
            df['IsOutlierCharge'] = (df[self.column] > self.upper_bound_).astype(int)
        else:
            df['IsOutlierCharge'] = 0
        return df
