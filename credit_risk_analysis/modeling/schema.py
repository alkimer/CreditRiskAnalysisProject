from pydantic import BaseModel

# Using pydantic for static type validation in FASTAPI

class PredictRequest(BaseModel):
    id_client: int


class PredictResponse(BaseModel):
    success: bool
    prediction: float
