from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 🚀 Initialize FastAPI
app = FastAPI()

# 💾 Load the trained model (pipeline)
model = joblib.load("model.pkl")

# 🧾 Input schema with all variables
class Applicant(BaseModel):
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

# 🔁 Optional: English → Spanish mapping if needed
def map_english_to_spanish(df):
    replacements = {
        "SEX": {"Male": "M", "Female": "F"},
        "MARITAL_STATUS": {
            "Single": "Soltero",
            "Married": "Casado",
            "Common-Law Union": "Unión libre",
            "Divorced": "Divorciado"
        },
        "OCCUPATION_TYPE": {
            "Public Employee": "Empleado público",
            "Private Employee": "Empleado privado",
            "Self-Employed": "Independiente",
            "Merchant": "Comerciante",
            "Technician": "Técnico",
            "Technologist": "Tecnólogo"
        },
        "RESIDENCE_TYPE": {
            "Owned": "Propia",
            "Rented": "Arrendada",
            "Family": "Familiar",
            "Company-Provided": "Empresa"
        },
        "PRODUCT": {
            "Mortgage Loan": "Crédito Hipotecario",
            "Consumer Credit": "Crédito Consumo",
            "Vehicle Loan": "Crédito Vehicular",
            "Credit Card": "Tarjeta de Crédito"
        }
    }

    for col, mapping in replacements.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

# 🧠 Prediction endpoint
@app.post("/predict")
def predict_risk(data: Applicant):
    df_input = pd.DataFrame([data.dict()])
    df_input = map_english_to_spanish(df_input)

    pred_proba = model.predict_proba(df_input)[0][1] * 100  # Score: 0–100
    pred_proba = round(pred_proba, 2)

    # ⬇️ Definir clase de riesgo según escala personalizada
    if pred_proba <= 35:
        risk_class = "Low"
    elif 36 <= pred_proba <= 69:
        risk_class = "Medium"
    else:
        risk_class = "High"

    return {
        "risk_class": risk_class,
        "risk_percentage": pred_proba
    }
