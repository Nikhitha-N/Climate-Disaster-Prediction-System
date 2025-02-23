import pandas as pd
import numpy as np
from scipy.stats import zscore
def correcting_datatypes(df, date_cols=None, categorical_cols=None, float_cols=None):  
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
                    print(f"Converted '{col}' to datetime")
                except ValueError as e:
                    print(f"Warning: Could not convert column '{col}' to datetime. Error: {e}")
            else:
                print(f"Warning: Column '{col}' does not exist in the DataFrame.")

    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                try:
                    df[col] = df[col].astype('category')
                    print(f"Converted '{col}' to category")
                except ValueError as e:
                    print(f"Warning: Could not convert column '{col}' to category. Error: {e}")
            else:
                print(f"Warning: Column '{col}' does not exist in the DataFrame.")

    if float_cols:
        for col in float_cols:  
            if col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                    print(f"Converted '{col}' to float")
                except ValueError as e:
                    print(f"Warning: Could not convert column '{col}' to float. Error: {e}")
            else:
                print(f"Warning: Column '{col}' does not exist in the DataFrame.")
    
    return df

def remove_duplicates(df):
    duplicate_indices = df[df.duplicated()].index.tolist()
    if duplicate_indices:
        print("Removed duplicate rows at indices:", duplicate_indices)
    else:
        print("No duplicate rows found.")
    return df.drop_duplicates()

def remove_outliers_iqr(df1, col):
    """Remove outliers using the IQR method."""
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df1[(df1[col] >= lower_bound) & (df1[col] <= upper_bound)]