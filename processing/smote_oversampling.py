# processing/resampling/smote_oversampling.py
from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X, y, random_state=42):
    """
    Applies SMOTE to balance the dataset.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series or array): Target
        random_state (int): Seed for reproducibility
        
    Returns:
        X_resampled (pd.DataFrame), y_resampled (pd.Series)
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y.values.ravel())
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name="target")
