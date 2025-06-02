import logging
import sys

from fastapi import APIRouter, FastAPI

from credit_risk_analysis.modeling.schema import PredictRequest, PredictResponse

app = FastAPI(
    title="Credit Risk Analysis API",
    version="1.0.0"
)


model_router = APIRouter(tags=["Model"], prefix="/model")




logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),    # envía logs a stdout (Docker los recoge)
    ]
)
logger = logging.getLogger(__name__)
logger.info("----INIT CREDIT RISK ANALYSIS ROUTER API SERVICE----")

@model_router.post("/predict", response_model=PredictResponse)
async def predict(predict_request: PredictRequest):
    logger.debug("→ /predict() invoked: predict_request=%s", predict_request.id_client)

    # response_model.success = False
    #
    #
    # # prediction = await model_predict(image_hash_name)
    #
    # # Paso 4: Armar respuesta
    # response["success"] = True
    # response["score"] = 1
    response = PredictResponse(
        success=True,
        prediction=1
    )

    #
    # # return PredictResponse(**response)
    return response



app.include_router(model_router)