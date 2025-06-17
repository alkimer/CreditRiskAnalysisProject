from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import boto3
import os

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
    logger.success("Dataset downloaded successfully.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
