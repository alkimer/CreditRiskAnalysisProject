import glob
import os
import pandas as pd

def rename_columns_in_txt_files(
    txt_folder='data/raw',
    xls_file='data/external/PAKDD2010_VariablesList.xls',
    output_folder='data/interim'
):
    """
    Renames columns in all .txt files within a folder using a mapping from an Excel file.

    Parameters:
    - txt_folder (str): Path to the folder containing .txt files.
    - xls_file (str): Path to the Excel file containing the variable mappings.
    - output_folder (str): Folder where the processed files will be saved.
    """
    
    # Find all .txt files in the specified folder
    txt_files = glob.glob(os.path.join(txt_folder, '*.txt'))

    # Load the Excel file containing the variable mapping
    map_df = pd.read_excel(xls_file)
    map_df.columns = map_df.columns.str.strip()
    map_df['Var_Id'] = pd.to_numeric(map_df['Var_Id'], errors='coerce')
    map_df = map_df.dropna(subset=['Var_Id'])
    map_df['Var_Id'] = map_df['Var_Id'].astype(int) - 1
    column_mapping = dict(zip(map_df['Var_Id'], map_df['Var_Title']))

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each text file
    for file in txt_files:
        df = pd.read_csv(file, delimiter='\t', header=None, encoding='latin1', low_memory=False)
        df.columns = list(range(df.shape[1]))

        rename_mapping = {col: column_mapping[col] for col in df.columns if col in column_mapping}
        df.rename(columns=rename_mapping, inplace=True)

        filename = os.path.splitext(os.path.basename(file))[0]
        output_file = os.path.join(output_folder, filename + '_with_columns.txt')
        df.to_csv(output_file, index=False, sep='\t', encoding='utf-8')

    print(f"{len(txt_files)} files processed and saved to '{output_folder}'")
