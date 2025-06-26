# Using pydantic for static type validation in FASTAPI
from datetime import datetime

from pydantic import BaseModel


class PredictRequest(BaseModel):
    age: int
    income: float
    loan_debt: float
    education: str
    credit_type: str


class PredictResponse(BaseModel):
    success: bool
    score: float


class PredictionRecord(BaseModel):
    id: int
    prediction_date: datetime
    request_json: str
    score: float
    model: str

    class Config:
        orm_mode = True  # 🔑 Esto permite que funcione con SQLAlchemy
