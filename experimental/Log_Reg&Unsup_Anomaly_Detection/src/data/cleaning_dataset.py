import pandas as pd
import numpy as np
import re
import os

sum_groups = {
    'TOTAL_INCOME': ['OTHER_INCOMES', 'PERSONAL_MONTHLY_INCOME'], #Income and Asset Combinations
    "TOTAL_BANK_ACCOUNTS": ["QUANT_BANKING_ACCOUNTS","QUANT_SPECIAL_BANKING_ACCOUNTS"],       #Banking Profile
    "STABILITY_INDEX":["MONTHS_IN_RESIDENCE","MONTHS_IN_THE_JOB"]
    }
count_groups_1 = {
    'TOTAL_CREDIT_CARDS': ['FLAG_VISA', 'FLAG_MASTERCARD',"FLAG_DINERS","FLAG_AMERICAN_EXPRESS","FLAG_OTHER_CARDS"],  #Income and Asset Combinations
    "DOC_CONFIRMATION": ["FLAG_INCOME_PROOF","FLAG_CPF","FLAG_RG","FLAG_HOME_ADDRESS_DOCUMENT","FLAG_EMAIL"],
    'CONTACTABILITY': ['FLAG_RESIDENCIAL_PHONE', 'FLAG_MOBILE_PHONE',"FLAG_PROFESSIONAL_PHONE","FLAG_EMAIL"],
    "IS_NOT_FOREIGNER":["NACIONALITY"],
    "HAS_PREMIUM_CARD": ["FLAG_DINERS","FLAG_AMERICAN_EXPRESS"],
    "HAS_ASSETS&HAS_PREMIUM_CARD": ["QUANT_CARS","PERSONAL_ASSETS_VALUE","FLAG_DINERS","FLAG_AMERICAN_EXPRESS"]  
    
    }
count_groups_0 = {
    'NO_CREDIT_HISTORY': ['FLAG_VISA', 'FLAG_MASTERCARD',"FLAG_DINERS","FLAG_AMERICAN_EXPRESS","FLAG_OTHER_CARDS",
                          "QUANT_BANKING_ACCOUNTS","QUANT_SPECIAL_BANKING_ACCOUNTS"]  #Income and Asset Combinations
    }

compare_columns = [
        ('STATE_OF_BIRTH', 'RESIDENCIAL_STATE'),
        ('CITY_OF_BIRTH', 'RESIDENCIAL_CITY'),
        ("PROFESSIONAL_CITY","RESIDENCIAL_CITY"),
        ("PROFESSIONAL_STATE","RESIDENCIAL_STATE"),
        ('RESIDENCIAL_BOROUGH', 'PROFESSIONAL_BOROUGH')
    ]
ratios_columns = [
    ('MONTHS_IN_THE_JOB', 'AGE'), 
    ('TOTAL_INCOME',"QUANT_DEPENDANTS"),
    ('PERSONAL_MONTHLY_INCOME','OTHER_INCOMES'),
    ("TOTAL_INCOME","AGE"),
    ("TOTAL_INCOME","TOTAL_BANK_ACCOUNTS"),
    ("TOTAL_INCOME","TOTAL_CREDIT_CARDS"),
    ("MONTHS_IN_THE_JOB","MONTHS_IN_RESIDENCE"),
    ("MONTHS_IN_RESIDENCE","AGE"),
    ("TOTAL_CREDIT_CARDS","TOTAL_INCOME"),
    ("PERSONAL_ASSETS_VALUE","TOTAL_CREDIT_CARDS")

]
cat_columns = ["PAYMENT_DAY", "POSTAL_ADDRESS_TYPE", "MARITAL_STATUS", "EDUCATION_LEVEL", "NACIONALITY", "FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS","FLAG_AMERICAN_EXPRESS","FLAG_OTHER_CARDS",
               "RESIDENCE_TYPE", "PROFESSION_CODE", "OCCUPATION_TYPE", "MATE_PROFESSION_CODE", "PRODUCT","RESIDENCIAL_ZIP_3","PROFESSIONAL_ZIP_3","FLAG_INCOME_PROOF","FLAG_CPF","FLAG_RG",
               "FLAG_HOME_ADDRESS_DOCUMENT","FLAG_EMAIL",'FLAG_RESIDENCIAL_PHONE', 'FLAG_MOBILE_PHONE',"EDUCATION_LEVEL.1"]


ethically_sensitive_columns = ["SEX"]



def clean_dataset(df: pd.DataFrame,
                  file: str = None, 
                  output_folder: str = None,
                  sum_columns: dict = sum_groups,
                  count_groups_1: dict = count_groups_1,
                  count_groups_0: dict = count_groups_0,
                  compare_columns: list[tuple] = compare_columns,
                  ratios_columns: list[tuple] = ratios_columns,
                  ethically_sensitive_columns: list = ethically_sensitive_columns,
                  enable_ethically_sensitive_columns:bool = True,
                  enable_count_groups_1: bool = True,
                  enable_count_groups_0: bool = True
                  #cat_columns:list = cat_columns
                  ) -> pd.DataFrame:
    """
    Cleans the dataset by:
    - Normalizing text and standardizing missing values
    - Dropping columns that are completely empty or contain only one unique value
    - Renaming any column that contains 'target' (case-insensitive) to 'TARGET'
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        file (str): Optional input filename (used for saving cleaned output).
        output_folder (str): Optional folder to save cleaned output.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("Starting dataset cleaning...")
    
    # Normalize string columns: strip and uppercase
    str_cols = df.select_dtypes(include=['object', 'category'] ).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
    print(f"Normalized string columns (stripped and uppercased): {list(str_cols)}")
    
    # Columns to drop ethical
    ethical_cols = []
    if enable_ethically_sensitive_columns:
        ethical_cols = [col for col in ethically_sensitive_columns if col in df.columns]
        if ethical_cols:
            df = df.drop(columns=ethical_cols)

    # Change numeric to categorical
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")
    print(f"Numeric columns changed to Category: {list(cat_columns)}")
   
    #Contar columnas con 1
    all_cols_to_count_1 = []
    if enable_count_groups_1:
        for new_col, cols_to_count in count_groups_1.items():
            # Convert to numeric: coerce non-convertible values to NaN (optional)
            temp = df[cols_to_count].replace({'Y': 1, 'N': 0})
            temp = temp.apply(pd.to_numeric, errors='coerce')

            # Count how many values equal to 1 (after conversion) per row
            df[new_col] = temp.eq(1).sum(axis=1, skipna=True)

            all_cols_to_count_1.extend(cols_to_count)
            print(f"Created column '{new_col}' by counting 1s in: {cols_to_count}")
        #df = df.drop(columns=list(set(all_cols_to_count_1)))
    
    #Contar columnas con 0
    all_cols_to_count_0 = []
    if enable_count_groups_0:
        for new_col, cols_to_count in count_groups_0.items():
            # Convert to numeric: coerce non-convertible values to NaN (optional)
            temp = df[cols_to_count].replace({'Y': 1, 'N': 0})
            temp = df[cols_to_count].apply(pd.to_numeric, errors='coerce')

            # Count how many values equal to 1 (after conversion) per row
            df[new_col] = temp.eq(0).sum(axis=1, skipna=True)

            all_cols_to_count_0.extend(cols_to_count)
            print(f"Created column '{new_col}' by counting 0s in: {cols_to_count}")
        #df = df.drop(columns=list(set(all_cols_to_count_0)))
   
    # Column Group Sums
    all_cols_to_drop_SUM = []
    if sum_columns:
        for new_col, cols_to_sum in sum_columns.items():
            df[new_col] = df[cols_to_sum].sum(axis=1, skipna=True)
            all_cols_to_drop_SUM.extend(cols_to_sum)
            print(f"Created column '{new_col}' by summing: {cols_to_sum}")
    
    #df = df.drop(columns=list(set(all_cols_to_drop_SUM)))

    # Calculate new columns as ratios
    if ratios_columns:
        for numerator, denominator in ratios_columns:
            if numerator in df.columns and denominator in df.columns:
                new_col = f"{numerator}_DIV_{denominator}"
                df[new_col] = df[numerator] / df[denominator].replace({0: 1})
                print(f"Created ratio column '{new_col}' as {numerator} / {denominator}")
            else:
                print(f"⚠️ Skipping ratio: '{numerator}' or '{denominator}' not found in DataFrame")

    #Column Equality Indicators
    if compare_columns:
        for col1, col2 in compare_columns:
            new_col_name = f"{col1}_EQ_{col2}"
            df[new_col_name] = df[col1] == df[col2]
            print(f"Created column '{new_col_name}' to indicate equality between '{col1}' and '{col2}'")

    

    # Standardize Missing Values
    df.replace([
        'NA', ' ', '',None, 'N/A', 'NULL', 'MISSING', 'NONE',"null","NAN",
        '#DIV/0!', '#N/A', '#VALUE!', '#REF!', '#NAME?', '#NUM!', '#NULL!'
    ], np.nan, inplace=True)
    print("Standardized missing values to NaN.")

    # Completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()

    # Constant columns
    one_unique_value_cols = df.columns[df.nunique(dropna=False) == 1].tolist()

    # Combine and drop
    cols_to_drop = list(set(empty_cols + one_unique_value_cols ))
    df = df.drop(columns=cols_to_drop)
    if cols_to_drop:
        print(f"Dropped columns: {cols_to_drop}")
    else:
        print("No columns were dropped.")

    # Rename any column that contains 'target' (case-insensitive) to 'TARGET'
    renamed_target = False
    for col in df.columns:
        if re.search(r'target', col, re.IGNORECASE):
            df = df.rename(columns={col: 'TARGET'})
            print(f"Renamed column '{col}' to 'TARGET'")
            renamed_target = True
            break               # Only rename the first match, then exit loop
    if not renamed_target:
        print("No column containing 'target' was found to rename.")
    
    # Remove exact duplicate rows early
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)
    print(f"Dropped {before_dup - after_dup} duplicate rows.")

    # Identify and delete ID-like columns 
    id_columns = [
        col for col in df.columns
        if 'ID' in col.upper() and df[col].nunique(dropna=False) == len(df)
    ]
    df = df.drop(columns=id_columns)
    print(f"Dropped likely ID columns: {id_columns}")

    # Optionally save cleaned file
    if file and output_folder:
        filename = os.path.splitext(os.path.basename(file))[0]
        output_file = os.path.join(output_folder, f"{filename}_cleaned.txt")
        df.to_csv(output_file, index=False, sep='\t', encoding='utf-8')
        print(f"Saved cleaned file to: {output_file}")

    print("Dataset cleaning completed.")
    return df