import logging
import sys

import redis.asyncio as redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from credit_risk_analysis.db.prediction_orm import insert_prediction
from settings import Settings

db_redis = redis.Redis(
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


# Setup Postgres DB

db_url = f"postgresql://{Settings.POSTGRES_USER}:{Settings.POSTGRES_PASS}@{Settings.POSTGRES_CONTAINER}:{Settings.POSTGRES_PORT}/{Settings.POSTGRES_DB_NAME}"

# Crear el engine
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
session = Session()



import json
import asyncio
import logging
from uuid import uuid4
from datetime import datetime


logger = logging.getLogger(__name__)


async def model_predict(predict_request):
    job_id = str(uuid4())
    start_time = datetime.utcnow()

    logger.info(f"[{job_id}] ▶️ Starting prediction for request: {predict_request}")

    # Preparar datos del trabajo
    try:
        job_data = {
            "id": job_id,
            "input": predict_request.model_dump(),  # usar dict() si es Pydantic v1
            "timestamp": start_time.isoformat()
        }
    except Exception as e:
        logger.exception(f"[{job_id}] ❌ Error serializing predict_request: {e}")
        raise

    # Encolar solicitud
    try:
        await db_redis.lpush(Settings.REDIS_PENDING_PREDICTION, json.dumps(job_data))
        logger.debug(f"[{job_id}] 📩 Job enqueued")
    except Exception as e:
        logger.exception(f"[{job_id}] ❌ Error enqueuing job in Redis: {e}")
        raise

    # Esperar respuesta del worker
    score = None
    timeout = Settings.API_TIMEOUT  # en segundos
    sleep_interval = Settings.API_SLEEP

    logger.debug(f"[{job_id}] 💤 Waiting for response (timeout={timeout}s)...")

    try:
        total_wait = 0
        while total_wait < timeout:
            output_raw = await db_redis.get(job_id)
            if output_raw:
                logger.info(f"[{job_id}] ✅ Job result received")
                try:
                    output = json.loads(output_raw)
                    score = output.get("score", None)
                    if score is None:
                        raise ValueError("Missing 'score' field in result")
                except Exception as e:
                    logger.exception(f"[{job_id}] ❌ Error decoding result: {e}")
                    raise

                    # Guardar en PostgreSQL

                insert_prediction(
                    session=session,
                    request_data=predict_request.model_dump(),  # ✅ Esto es un dict serializable
                    score=score,
                    model_name="MockModel-v1"
                )

                await db_redis.delete(job_id)
                break

            await asyncio.sleep(sleep_interval)
            total_wait += sleep_interval

        if score is None:
            logger.warning(f"[{job_id}] ⏰ Timeout after {timeout} seconds waiting for result")
            raise TimeoutError("No prediction result returned in time")

    except Exception as e:
        logger.exception(f"[{job_id}] ❌ Error while waiting for job result: {e}")
        raise

    elapsed = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"[{job_id}] 🏁 Prediction finished in {elapsed:.2f}s — Score: {score}")

    return True, score
