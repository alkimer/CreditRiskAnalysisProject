import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

def get_feature_target(
    app_train: pd.DataFrame, app_test: pd.DataFrame, target_column: str = 'TARGET'
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    x_train = app_train.drop(columns=[target_column])
    y_train = app_train[target_column]

    x_test = app_test.drop(columns=[target_column])
    y_test = app_test[target_column]

    return x_train, y_train, x_test, y_test

def get_train_val_sets(
    x_train:pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,stratify=y_train,test_size=0.20,random_state=42, shuffle=True)
    return x_train, x_val, y_train, y_val

def df_to_csv(df, filename, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Full file path
    output_file = os.path.join(output_folder, filename)

    # Save DataFrame to CSV
    try:
        df.to_csv(output_file, index=False, sep='\t', encoding='utf-8')
        print(f"File saved successfully to: '{output_file}'")
    except Exception as e:
        print(f"Failed to save file: {e}")


def summarize_column_counts(df, column):
    """
    Group by a column, count occurrences including missing values,
    and append a summary row with the total count.
    
    Parameters:
    - df: pandas DataFrame
    - column: str, column name to summarize
    
    Returns:
    - pandas DataFrame with value counts and a total summary row
    """
    result = df.groupby(column, dropna=False).size().reset_index(name='count')
    
    # Append summary row
    summary = pd.DataFrame({
        column: ['TOTAL'],
        'count': [result['count'].sum()]
    })
    
    result_with_summary = pd.concat([result, summary], ignore_index=True)
    
    return result_with_summary



def load_txt_with_mapped_columns(txt_file_path, xls_file_path):
    """
    Loads a .txt file without headers and applies column names based on an Excel mapping.

    Parameters:
    - txt_file_path (str): Path to the .txt file without column names.
    - xls_file_path (str): Path to the Excel file containing the variable mappings.

    Returns:
    - pandas.DataFrame: DataFrame with renamed columns.
    """

    # Load Excel mapping
    map_df = pd.read_excel(xls_file_path)
    map_df.columns = map_df.columns.str.strip()
    map_df['Var_Id'] = pd.to_numeric(map_df['Var_Id'], errors='coerce')
    map_df = map_df.dropna(subset=['Var_Id'])
    map_df['Var_Id'] = map_df['Var_Id'].astype(int) - 1  # Assuming Var_Id is 1-based
    column_mapping = dict(zip(map_df['Var_Id'], map_df['Var_Title']))

    # Load TXT file
    df = pd.read_csv(txt_file_path, delimiter='\t', header=None, encoding='latin1', low_memory=False)
    df.columns = list(range(df.shape[1]))  # Assign temporary numeric headers

    # Rename columns using mapping
    rename_mapping = {col: column_mapping[col] for col in df.columns if col in column_mapping}
    df.rename(columns=rename_mapping, inplace=True)

    return df

