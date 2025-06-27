from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 🚀 Inicializar FastAPI
app = FastAPI()

# 💾 Cargar el modelo entrenado + preprocesamiento (Pipeline)
model = joblib.load("model.pkl")

# 🧾 Esquema de entrada con todas las nuevas variables
class Cliente(BaseModel):
    AGE: int
    SEX: str
    MARITAL_STATUS: str
    OCCUPATION_TYPE: str
    MONTHS_IN_RESIDENCE: int
    FLAG_RESIDENCIAL_PHONE: int
    STATE_OF_BIRTH: str
    RESIDENCIAL_STATE: str
    RESIDENCE_TYPE: str
    RESIDENCIAL_CITY: str
    RESIDENCIAL_BOROUGH: str
    RESIDENCIAL_PHONE_AREA_CODE: int
    RESIDENCIAL_ZIP_3: int
    PROFESSIONAL_STATE: str
    PROFESSIONAL_ZIP_3: int
    PRODUCT: str

# 🧠 Endpoint de predicción
@app.post("/predict")
def predict_risk(data: Cliente):
    # Convertir input a DataFrame
    df_input = pd.DataFrame([data.dict()])

    # Realizar predicción
    pred_clase = model.predict(df_input)[0]
    pred_proba = model.predict_proba(df_input)[0][1] * 100  # Probabilidad de clase positiva

    # Retornar resultado
    return {
        "riesgo_clase": str(pred_clase),
        "riesgo_porcentaje": round(pred_proba, 2)
    }
