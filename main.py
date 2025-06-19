from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import numpy as np
import shap

# Cargar modelo y escalador juntos
model, scaler = joblib.load("model.pkl")  # 👈 Ambos elementos

# Inicializar SHAP
try:
    explainer = shap.TreeExplainer(model)
    shap_enabled = True
except Exception as e:
    explainer = None
    shap_enabled = False
    print(f"SHAP deshabilitado: {e}")

# Iniciar la API
app = FastAPI()

# Clase para recibir datos del usuario
class UserData(BaseModel):
    age: int
    income: float
    loan_debt: float
    education: Literal["Bachillerato", "Universitario", "Postgrado", "Otro"]
    credit_type: Literal["Auto", "Casa", "Educación", "Tarjeta de Crédito"]

# Funciones de codificación
def encode_education(level):
    return {"Bachillerato": 0, "Universitario": 1, "Postgrado": 2, "Otro": 3}[level]

def encode_credit_type(tipo):
    return {"Auto": 0, "Casa": 1, "Educación": 2, "Tarjeta de Crédito": 3}[tipo]

# Ruta principal
@app.post("/predict")
def predict_risk(data: UserData):
    try:
        raw_features = [
            data.age,
            data.income,
            data.loan_debt,
            encode_education(data.education),
            encode_credit_type(data.credit_type)
        ]

        scaled_features = scaler.transform([raw_features])  # 👈 Aplicar escalado
        prob = model.predict_proba(scaled_features)[0][1] * 100
        label = "Alto Riesgo" if prob > 60 else "Bajo Riesgo"

        response = {
            "risk_score": round(prob, 2),
            "risk_label": label
        }

        # Explicaciones SHAP
        if shap_enabled:
            shap_result = explainer.shap_values(scaled_features)
            shap_values = shap_result[0][0].tolist() if isinstance(shap_result, list) else shap_result[0].tolist()
            feature_names = ["Edad", "Ingresos", "Deuda", "Educación", "Tipo Crédito"]
            response["shap_values"] = shap_values
            response["feature_names"] = feature_names
        else:
            response["shap_values"] = []
            response["feature_names"] = []

        return response

    except Exception as e:
        return {"error": str(e)}