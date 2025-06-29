from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import boto3
import os
import pandas as pd
from typing import Tuple
from collections import Counter
from sklearn.model_selection import train_test_split

from credit_risk_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, BUCKET_NAME, PREFIX
from credit_risk_analysis.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

app = typer.Typer()

def download_file_from_s3(bucket_name: str, prefix: str, local_path: Path):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    # Listar objetos en ese prefijo (asume que querés todo el folder)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response:
        print("Files not found.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        filename = os.path.basename(key)
        if not filename:  # en caso de que el key termine en "/"
            continue

        local_folder = os.path.join(local_path, filename)
        print(f"Downloading {key} to {local_folder}...")

        s3.download_file(bucket_name, key, local_folder)

        print("Download complete.")

def load_variables_list(file_path: Path) -> pd.DataFrame:
    """Load variables list from a given file path.
    Args:
        file_path (Path): Path to the variables list file.
    Returns:
        pd.DataFrame: DataFrame containing the variables list.
    """
    variables = pd.read_excel(os.path.join(file_path, 'PAKDD2010_VariablesList.xls'))
    return variables

def load_dataset_with_colnames(file_path: Path) -> pd.DataFrame:
    """Load dataset from a given file path and ensure column names are unique.
    Args:
        file_path (Path): Path to the dataset file.
    Returns:
        pd.DataFrame: Loaded DataFrame with unique column names.
    """

    # Cargar la lista de variables
    logger.info("Loading variables list...")
    variables = load_variables_list(file_path)
    logger.info("Variables list loaded successfully.")
    colnames = variables['Var_Title'].values.tolist()
    print("Number of variables:", len(set(colnames)))

    counts = {k:v for k,v in Counter(colnames).items() if v > 1}
    newlist = colnames[:]

    for i in reversed(range(len(colnames))):
        item = colnames[i]
        if item in counts and counts[item]:
            newlist[i] += str(counts[item])
            counts[item]-=1

    # Cargar el CSV
    df = pd.read_csv(
        #file_path
        os.path.join(file_path, 'PAKDD-2010 training data.zip'),
        compression='zip', sep='\t', encoding='latin1',
        header=None, names=newlist,
        low_memory=False
    )

    return df

def get_train_val_sets(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split training dataset into two new sets used for train and validation.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=2023, shuffle=True
    )
    return X_train, X_val, y_train, y_val

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_path: Path = RAW_DATA_DIR, #/ "dataset.csv",
    #output_path: Path = PROCESSED_DATA_DIR, #/ "dataset.csv",
    prefix: str = PREFIX,
    bucket_name: str = BUCKET_NAME
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    #logger.info("Processing dataset...")
    #for i in tqdm(range(10), total=10):
    #    if i == 5:
    #        logger.info("Something happened for iteration 5.")
    #logger.success("Processing dataset complete.")
    logger.info("Downloading dataset from S3...")
    download_file_from_s3(bucket_name=bucket_name, prefix=prefix, local_path=output_path)

    df = load_dataset_with_colnames(output_path)
    logger.info("Dataset loaded successfully.")

    # Asegurar que la variable target esté bien definida
    target_col = "TARGET_LABEL_BAD=1"  

    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Aplicar split
    X_train, X_val, y_train, y_val = get_train_val_sets(X, y)

    # Opcional: mostrar shapes
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    # Guardar resultados
    X_train.to_csv("data/interim/X_train.csv", index=False)
    X_val.to_csv("data/interim/X_val.csv", index=False)
    y_train.to_csv("data/interim/y_train.csv", index=False)
    y_val.to_csv("data/interim/y_val.csv", index=False)
    print("Datos guardados en data/interim/")
    logger.success("Dataset downloaded successfully.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
