# Using pydantic for static type validation in FASTAPI
from datetime import datetime

from pydantic import BaseModel

class PredictRequest(BaseModel):
    MARITAL_STATUS: int
    MONTHS_IN_RESIDENCE: int
    AGE: int
    OCCUPATION_TYPE: int
    SEX: str
    FLAG_RESIDENCIAL_PHONE: str
    STATE_OF_BIRTH: str
    RESIDENCIAL_STATE: str
    RESIDENCE_TYPE: int
    PROFESSIONAL_STATE: str
    PRODUCT: str
    RESIDENCIAL_CITY: str
    RESIDENCIAL_BOROUGH: str
    RESIDENCIAL_PHONE_AREA_CODE: int
    RESIDENCIAL_ZIP_3: int
    PROFESSIONAL_ZIP_3: int


class PredictResponse(BaseModel):
    risk_percentage: float


class PredictionRecord(BaseModel):
    id: int
    prediction_date: datetime
    request_json: str
    score: float
    model: str

    class Config:
        orm_mode = True  # 🔑 Esto permite que funcione con SQLAlchemy
