from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 📦 Cargar modelo
model = joblib.load("model.pkl")

# 🚀 Instancia de FastAPI
app = FastAPI()

# ✅ Estructura del input (ya codificado)
class Applicant(BaseModel):
    AGE: int
    SEX: str
    MARITAL_STATUS: int
    OCCUPATION_TYPE: int
    MONTHS_IN_RESIDENCE: int
    FLAG_RESIDENCIAL_PHONE: str
    STATE_OF_BIRTH: str
    RESIDENCIAL_STATE: str
    RESIDENCE_TYPE: int
    RESIDENCIAL_CITY: str
    RESIDENCIAL_BOROUGH: str
    RESIDENCIAL_PHONE_AREA_CODE: int
    RESIDENCIAL_ZIP_3: int
    PROFESSIONAL_STATE: str
    PROFESSIONAL_ZIP_3: int
    PRODUCT: str

# 🧠 Clasificación de riesgo
def classify(score: float) -> str:
    if score < 35:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

# 🔮 Endpoint de predicción
@app.post("/predict")
def predict(data: Applicant):
    input_df = pd.DataFrame([data.dict()])

    # Seleccionar solo variables numéricas si fue entrenado así
    numeric_cols = input_df.select_dtypes(include=["int64", "float64"]).columns
    input_numeric = input_df[numeric_cols]

    proba = model.predict_proba(input_numeric)[0][1] * 100
    label = classify(proba)

    return {
        "risk_percentage": round(proba, 1),
        "risk_class": label
    }
