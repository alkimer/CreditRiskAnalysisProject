import logging
import sys

from fastapi import APIRouter, FastAPI

from credit_risk_analysis.modeling.schema import PredictRequest, PredictResponse
from credit_risk_analysis.modeling.services.services import model_predict

app = FastAPI(
    title="Credit Risk Analysis API",
    version="1.0.0"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),    # envía logs a stdout (Docker los recoge)
    ]
)


model_router = APIRouter(tags=["Model"], prefix="/model")
logger = logging.getLogger(__name__)
logger.info("----INIT CREDIT RISK ANALYSIS ROUTER API SERVICE----")

@model_router.post("/predict", response_model=PredictResponse)
async def predict(predict_request: PredictRequest):
    logger.debug("→ /predict() invoked: predict_request=%s", predict_request)

    response = PredictResponse(
        success=False,
        score=1
    )

    response.success, response.score = await model_predict(predict_request)

    return response


app.include_router(model_router)