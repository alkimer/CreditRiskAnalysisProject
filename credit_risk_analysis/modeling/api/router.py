import logging
import sys
from fastapi import APIRouter, FastAPI, Depends
from typing import List
from sqlalchemy.orm import Session

from credit_risk_analysis.modeling.schema import PredictRequest, PredictResponse, PredictionRecord
from credit_risk_analysis.db.prediction_orm import get_all_predictions, get_db
from credit_risk_analysis.modeling.services.prediction_job_producer import model_predict

# Inicializar la app FastAPI
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
model_name = "stacking_model.pkl"
logger = logging.getLogger(__name__)
logger.info("----INIT CREDIT RISK ANALYSIS ROUTER API SERVICE----")

# Router para endpoints relacionados al modelo
model_router = APIRouter(tags=["Model"], prefix="/model")

@model_router.post("/predict", response_model=PredictResponse)
async def predict(predict_request: PredictRequest):
    logger.debug("→ /predict() invoked: predict_request=%s , model_name:%s", predict_request, model_name)

    response = PredictResponse(
        success=False,
        score=1
    )

    response.success, response.score = await model_predict(predict_request, model_name)

    return response

@model_router.get("/predictions", response_model=List[PredictionRecord])
def get_predictions(db: Session = Depends(get_db)):
    return get_all_predictions(db)

# Incluir el router en la app principal
app.include_router(model_router)
