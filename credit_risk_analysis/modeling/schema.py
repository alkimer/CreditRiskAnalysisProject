from pydantic import BaseModel

# Using pydantic for static type validation in FASTAPI

class PredictRequest(BaseModel):
    client_information: dict


class PredictResponse(BaseModel):
    success: bool
    score: float
