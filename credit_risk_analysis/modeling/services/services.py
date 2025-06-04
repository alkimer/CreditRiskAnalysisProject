import json
import time
from uuid import uuid4

import redis

from settings import Settings

db = redis.Redis(
    host=Settings.REDIS_IP,
    port=Settings.REDIS_PORT,
    db=Settings.REDIS_DB_ID,
    password=Settings.REDIS_PASSWORD,
    decode_responses=True
)



async def model_predict(id_client):
    print(f"Credit Risk Analysis predicting for id_client {id_client}...")
    """
    Receives and id_client an predicts its Credit Risk.
    It uses Redis in-between to async prediction.

    Parameters
    ----------
    id_client : str

    Returns
    -------
    prediction, : float
    Score for credit.
    """
    # Generar un ID único para el trabajo
    job_id = str(uuid4())

    # Crear diccionario con los datos del trabajo
    job_data = {
        "id": job_id,
        "id_client": id_client
    }

    # Enviar el trabajo a Redis (cola)
    db.lpush(Settings.REDIS_QUEUE, json.dumps(job_data))

    # Esperar respuesta del modelo
    while True:
        # Luego de que el modelo realice la predicción,
        # volverá a colocar en redis el resutlado con key=job_id
        output = db.get(job_id)

        if output is not None:
            output = json.loads(output)
            score = output["score"]

            db.delete(job_id)
            break

        time.sleep(Settings.API_SLEEP)

    return True, score