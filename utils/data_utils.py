import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def detect_task_type(df, target_col=None):
    """
    Auto-detect whether the task is classification or regression
    based on the target column characteristics.
    """
    if target_col is None:
        # Assume last column is target
        target_col = df.columns[-1]
    
    target = df[target_col]
    
    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(target):
        # Check number of unique values
        unique_values = target.nunique()
        total_values = len(target)
        
        # If unique values are less than 5% of total or less than 20, likely classification
        if unique_values < 20 or (unique_values / total_values) < 0.05:
            return 'classification'
        else:
            return 'regression'
    else:
        # Non-numeric target suggests classification
        return 'classification'

def preprocess_data(X, y, task_type='classification', handle_missing='Fill with mean/mode'):
    """
    Preprocess features and target variables.
    """
    X_processed = X.copy()
    y_processed = y.copy()
    
    # Handle missing values in features
    if X_processed.isnull().sum().sum() > 0:
        if handle_missing == 'Drop rows with missing values':
            # Drop rows with any missing values
            missing_mask = X_processed.isnull().any(axis=1)
            X_processed = X_processed[~missing_mask]
            y_processed = y_processed[~missing_mask]
            
        elif handle_missing == 'Fill with mean/mode':
            # Fill numeric columns with mean, categorical with mode
            for col in X_processed.columns:
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col].fillna(X_processed[col].mean(), inplace=True)
                else:
                    X_processed[col].fillna(X_processed[col].mode()[0], inplace=True)
                    
        elif handle_missing == 'Fill with median':
            # Fill numeric columns with median, categorical with mode
            for col in X_processed.columns:
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col].fillna(X_processed[col].median(), inplace=True)
                else:
                    X_processed[col].fillna(X_processed[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    categorical_columns = X_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    
    # Handle target variable for classification
    if task_type == 'classification':
        if not pd.api.types.is_numeric_dtype(y_processed):
            le_target = LabelEncoder()
            y_processed = le_target.fit_transform(y_processed.astype(str))
        else:
            # Ensure integer type for classification
            y_processed = y_processed.astype(int)
    
    # Scale features (important for some algorithms)
    scaler = StandardScaler()
    X_processed = pd.DataFrame(
        scaler.fit_transform(X_processed),
        columns=X_processed.columns,
        index=X_processed.index
    )
    
    return X_processed, y_processed

def get_dataset_info(df):
    """
    Get comprehensive information about the dataset.
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return info

def validate_dataset(df, min_rows=10, min_features=1):
    """
    Validate if the dataset meets minimum requirements for ML.
    """
    issues = []
    
    # Check minimum rows
    if len(df) < min_rows:
        issues.append(f"Dataset has only {len(df)} rows, minimum {min_rows} required")
    
    # Check minimum features
    if df.shape[1] < min_features:
        issues.append(f"Dataset has only {df.shape[1]} columns, minimum {min_features} required")
    
    # Check for empty dataset
    if df.empty:
        issues.append("Dataset is empty")
    
    # Check for all missing values in any column
    all_missing_cols = df.columns[df.isnull().all()].tolist()
    if all_missing_cols:
        issues.append(f"Columns with all missing values: {all_missing_cols}")
    
    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        issues.append(f"Constant columns (no variation): {constant_cols}")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues
    }
