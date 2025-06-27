import json
import random
import asyncio
import logging
import sys

import redis.asyncio as redis

from credit_risk_analysis.modeling.model.model import model_predict
from credit_risk_analysis.modeling.schema import PredictResponse
from settings import Settings

# ----------------------
# Configuración del logger
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("async-consumer")

# ----------------------
# Inicialización Redis async
# ----------------------
db = redis.Redis(
    host=Settings.REDIS_IP,
    port=Settings.REDIS_PORT,
    db=Settings.REDIS_DB_ID,
    password=Settings.REDIS_PASSWORD,
    decode_responses=True
)

# ----------------------
# Loop principal asincrónico
# ----------------------
async def consume_predictions():
    logger.info("🔁 Iniciando consumidor de predicciones (async)...")
    logger.debug(f"🔌 Conectado a Redis en {Settings.REDIS_IP}:{Settings.REDIS_PORT}, DB={Settings.REDIS_DB_ID}")

    while True:
        try:
            logger.debug("⏳ Esperando mensaje con BRPOP...")
            result = await db.brpop(Settings.REDIS_PENDING_PREDICTION, timeout=0)  # bloqueante async

            if result is None:
                logger.warning("⏰ Timeout esperando mensajes")
                continue

            _, job_json = result
            logger.info(f"📦 Mensaje recibido crudo: {job_json}")

            try:
                job_data = json.loads(job_json)
                job_id = job_data["id"]
                logger.info(f"📩 Procesando job_id={job_id}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"❌ Error parseando job: {e} | Payload: {job_json}")
                continue

            # Call Real model
            success, score = await model_predict(job_data)
            response = PredictResponse(success=success, score=score)

            try:
                await db.setex(job_id, 300, json.dumps(response.model_dump()))  # TTL de 5 minutos
                logger.info(f"✅ Predicción generada y almacenada: job_id={job_id}, score={score}")
            except Exception as e:
                logger.error(f"❌ Error guardando resultado en Redis: {e}")
                continue

            await asyncio.sleep(1)  # simula procesamiento

        except Exception as e:
            logger.exception(f"🔥 Error inesperado en el loop principal: {e}")
            await asyncio.sleep(5)  # backoff

# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    try:
        asyncio.run(consume_predictions())
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown solicitado por el usuario")
