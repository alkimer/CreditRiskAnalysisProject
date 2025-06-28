from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

try:
    model = joblib.load("model.pkl")
    print("✅ Modelo cargado")
except Exception as e:
    print("❌ Error cargando modelo:", e)

class ApplicantData(BaseModel):
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
    PRODUCT: int

@app.post("/predict")
def predict(data: ApplicantData):
    try:
        payload = data.dict()
        print("📥 Payload recibido:", payload)

        df = pd.DataFrame([payload])
        numeric_cols = [
            "AGE", "MARITAL_STATUS", "OCCUPATION_TYPE", "MONTHS_IN_RESIDENCE",
            "RESIDENCIAL_PHONE_AREA_CODE", "RESIDENCIAL_ZIP_3",
            "PROFESSIONAL_ZIP_3", "RESIDENCE_TYPE", "PRODUCT"
        ]
        input_numeric = df[numeric_cols]
        print("🔢 Input numérico:", input_numeric)

        score = model.predict_proba(input_numeric)[0][1] * 100

        def classify(s):
            return "Low" if s < 35 else "Medium" if s < 70 else "High"

        result = {
            "risk_percentage": round(score, 1),
            "risk_class": classify(score)
        }
        print("✅ Resultado:", result)
        return result

    except Exception as e:
        print("❌ Error durante la predicción:", e)
        return {"error": str(e)}
