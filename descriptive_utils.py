import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

import pandas as pd

def describe_data(df):
    print("***Describing the data:***")
    num_rows, num_columns = df.shape  
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_columns}")
    
    print("\nColumn details:")
    for column in df.columns:
        col_data = df[column]
        col_dtype = col_data.dtype
        print(f"\nColumn: {column}, Type: {col_dtype}")

        if pd.api.types.is_numeric_dtype(col_data):
            min_val = col_data.min()
            max_val = col_data.max()
            mean_val = col_data.mean()
            median_val = col_data.median()
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Median: {median_val}")
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
            num_categories = col_data.nunique()
            print(f"  Number of categories: {num_categories}")
            if num_categories <= 10:  
                print("  Counts per category:")
                category_counts = col_data.value_counts()
                for index, value in category_counts.items():
                    print(f"    {index}: {value}")
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            min_date = col_data.min()
            max_date = col_data.max()
            print(f"  Date Range: {min_date} to {max_date}")
            print(f"  Number of unique dates: {col_data.nunique()}")
        else:
            unique_vals = col_data.unique()
            if len(unique_vals) <= 10:  
                print("  Unique values:")
                for val in unique_vals:
                    print(f"    {val}")

    return num_rows, num_columns

def count_nulls(df):
    print("Describing Nulls in the data:")
    
    null_counts_columns = df.isnull().sum()
    print("Null counts per variable:")
    print(null_counts_columns)
    
    null_counts_rows = df.isnull().sum(axis=1)
    max_nulls = null_counts_rows.max()
    rows_with_most_nulls = null_counts_rows[null_counts_rows == max_nulls].index.tolist()

    total_rows = len(df)
    rows_with_any_nulls = (null_counts_rows > 0).sum()
    percentage_with_nulls = (rows_with_any_nulls / total_rows) * 100

    print(f"\nRows with the highest number of nulls ({max_nulls} nulls):")
    print(rows_with_most_nulls)



import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def describe_numeric(df):
    print("*** Reporting on Numeric Variables ***")
    
    numeric_vars = df.select_dtypes(include=['int64', 'float64'])
    descriptions = numeric_vars.describe()
    print(descriptions)

    # Define the directory where images will be saved
    directory = "../Images"

    # Ensure the "Images" directory exists
    os.makedirs(directory, exist_ok=True)

    for column in numeric_vars:
        data = numeric_vars[column].dropna()
        if data.empty:
            print(f"No data available for histogram of {column} after removing NaNs.")
            continue
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1]})
        
        sns.histplot(data, ax=ax1, color='blue', alpha=0.7, kde=False, element='bars', stat='count')
        ax1.set_title(f'Histogram of {column}')
        ax1.set_xlabel(column)
        ax1.set_ylabel('Frequency')
        ax1.grid(True)

        sns.boxplot(y=data, ax=ax2, color='green')
        ax2.set_title(f'Box Plot of {column}')
        ax2.set_ylabel('Values')
        ax2.set_xlabel('Box plot')

        plt.tight_layout()  
        
        # Save the plot inside "Images" folder directly
        filename = os.path.join(directory, f"{column}.png")
        plt.savefig(filename, format='png', dpi=300)
        plt.close(fig)  
        print(f" {filename} has been saved successfully.")


def plot_correlation_matrix(df):
    # Select only numeric columns
    numeric_vars = df.select_dtypes(include=['int64', 'float64', 'float32', 'int32'])

    if numeric_vars.empty:
        print("No numeric variables found in the DataFrame.")
        return
    
    # Compute correlation matrix
    correlation_matrix = numeric_vars.corr()

    # Create a directory if it doesn't exist
    os.makedirs("Images", exist_ok=True)

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    
    plt.title("Correlation Matrix", fontsize=14)
    
    # Save the correlation matrix plot
    filename = "../Images/correlation_matrix.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Correlation matrix plot saved as: {filename}")
