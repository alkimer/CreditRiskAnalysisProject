import json
import logging
import sys
from typing import List

from fastapi import Depends
from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from sqlalchemy.orm import Session

from credit_risk_analysis.db.prediction_orm import get_all_predictions, get_db
from credit_risk_analysis.modeling.schema import PredictRequest, PredictResponse, PredictionRecord
from credit_risk_analysis.modeling.services.prediction_job_producer import model_predict

app = FastAPI(
    title="Credit Risk Analysis API",
    version="1.0.0"
)

# Configuración de logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
logger.info("----INIT CREDIT RISK ANALYSIS ROUTER API SERVICE----")

model_name = "stacking_model.pkl"

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Leemos el body como bytes
    body_bytes = await request.body()

    try:
        body_data = json.loads(body_bytes)
    except Exception:
        body_data = body_bytes.decode(errors='ignore')

    logger.info(f"[VALIDATION ERROR] Request body received: {body_data}")

    error_report = []
    for error in exc.errors():
        loc = ".".join(str(e) for e in error["loc"])
        msg = error["msg"]
        error_type = error["type"]
        error_report.append({"field": loc, "error": msg, "type": error_type})
        logger.warning(f"[VALIDATION ERROR] field={loc} → {msg} (type: {error_type})")

    return JSONResponse(
        status_code=412,
        content={
            "detail": "Validation failed",
            "errors": error_report,
            "body_received": body_data,  # opcional, podés incluirlo en el response
        },
    )


# Router para endpoints relacionados al modelo
model_router = APIRouter(tags=["Model"], prefix="/model")

@model_router.post("/predict", response_model=PredictResponse)
async def predict(predict_request: PredictRequest):
    logger.debug("→ /predict() invoked: predict_request=%s , model_name:%s", predict_request, model_name)
    logger.info("→ /predict() invoked: payload:%s", predict_request)

    # response = PredictResponse(
    #     risk_percentage=1
    # )
    #
    # response.risk_percentage = await model_predict(predict_request, model_name)

    return await model_predict(predict_request, model_name)


@model_router.get("/predictions", response_model=List[PredictionRecord])
def get_predictions(db: Session = Depends(get_db)):
    return get_all_predictions(db)

# Incluir el router en la app principal
app.include_router(model_router)
