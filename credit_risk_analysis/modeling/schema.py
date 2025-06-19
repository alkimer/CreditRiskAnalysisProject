from pydantic import BaseModel

# Using pydantic for static type validation in FASTAPI

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
