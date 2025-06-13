import json
import time
from uuid import uuid4

import redis.asyncio as redis

from settings import Settings
import logging
import sys

db = redis.Redis(
    host=Settings.REDIS_IP,
    port=Settings.REDIS_PORT,
    db=Settings.REDIS_DB_ID,
    password=Settings.REDIS_PASSWORD,
    decode_responses=True
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),    # envía logs a stdout (Docker los recoge)
    ]
)
logger = logging.getLogger(__name__)
logger.info("----INIT Model predict service----")


import asyncio

async def model_predict(id_client):
    print(f"Credit Risk Analysis predicting for id_client {id_client}...")

    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "id_client": id_client
    }

    await db.lpush(Settings.REDIS_PENDING_PREDICTION, json.dumps(job_data))
    logger.info(f"----Esperando que se complete el job_id  {job_id}")

    while True:
        output = await db.get(job_id)  # <- ahora es await

        if output is not None:
            logger.info(f"consumido job :  {job_id}")

            output = json.loads(output)
            score = output["score"]

            await db.delete(job_id)  # <- también await
            break

        await asyncio.sleep(Settings.API_SLEEP)  # <- no uses time.sleep en async

    return True, score
