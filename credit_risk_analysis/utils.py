import boto3
import os

from dotenv import load_dotenv
load_dotenv()

def descargar_credit_data():
    bucket_name = "anyoneai-datasets"
    prefix = "credit-data-2010/"
    local_folder = "./raw"

    #aws_access_key_id = "COMPLETAR CON LA INFO DEL ACADEMY"
    #aws_secret_access_key = "COMPLETAR CON LA INFO DEL ACADEMY"

    # Crear la carpeta local si no existe
    os.makedirs(local_folder, exist_ok=True)

    # Crear cliente de S3
    s3 = boto3.client(
        "s3",
        #aws_access_key_id=aws_access_key_id,
        #aws_secret_access_key=aws_secret_access_key
    )

    # Listar objetos en ese prefijo (asume que querés todo el folder)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response:
        print("No se encontraron archivos.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        filename = os.path.basename(key)
        if not filename:  # en caso de que el key termine en "/"
            continue

        local_path = os.path.join(local_folder, filename)
        print(f"Descargando {key} a {local_path}...")

        s3.download_file(bucket_name, key, local_path)

    print("Descarga completa.")


descargar_credit_data()