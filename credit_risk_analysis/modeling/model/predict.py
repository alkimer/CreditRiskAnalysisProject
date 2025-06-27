
import os
import sys

import joblib
import numpy as np
import typer

sys.path.append(os.path.abspath('../'))
from credit_risk_analysis.processing import process_all_features_v2

app = typer.Typer()

MODELS_DIR = "../../models"

import logging
import sys
import pandas as pd
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model-logger")


FINAL_COLUMNS = [
    'ID_CLIENT', 'CLERK_TYPE', 'PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE',
    'QUANT_ADDITIONAL_CARDS', 'POSTAL_ADDRESS_TYPE', 'SEX', 'MARITAL_STATUS',
    'QUANT_DEPENDANTS', 'EDUCATION_LEVEL', 'STATE_OF_BIRTH', 'CITY_OF_BIRTH',
    'NACIONALITY', 'RESIDENCIAL_STATE', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH',
    'FLAG_RESIDENCIAL_PHONE', 'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCE_TYPE',
    'MONTHS_IN_RESIDENCE', 'FLAG_MOBILE_PHONE', 'FLAG_EMAIL',
    'PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'FLAG_VISA', 'FLAG_MASTERCARD',
    'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS',
    'QUANT_BANKING_ACCOUNTS', 'QUANT_SPECIAL_BANKING_ACCOUNTS',
    'PERSONAL_ASSETS_VALUE', 'QUANT_CARS', 'COMPANY', 'PROFESSIONAL_STATE',
    'PROFESSIONAL_CITY', 'PROFESSIONAL_BOROUGH', 'FLAG_PROFESSIONAL_PHONE',
    'PROFESSIONAL_PHONE_AREA_CODE', 'MONTHS_IN_THE_JOB', 'PROFESSION_CODE',
    'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE', 'EDUCATION_LEVEL.1',
    'FLAG_HOME_ADDRESS_DOCUMENT', 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF',
    'PRODUCT', 'FLAG_ACSP_RECORD', 'AGE', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3'
]


DEFAULTS = {
    'ID_CLIENT': 1000,
    'CLERK_TYPE': 'C',
    'PAYMENT_DAY': 1,
    'APPLICATION_SUBMISSION_TYPE': 'Web',
    'QUANT_ADDITIONAL_CARDS': 1,
    'POSTAL_ADDRESS_TYPE': 1,
    'QUANT_DEPENDANTS': 0,
    'EDUCATION_LEVEL': 1,
    'CITY_OF_BIRTH': 'unknown',
    'NACIONALITY': 'BR',
    'FLAG_MOBILE_PHONE': "N",
    'FLAG_EMAIL': 0,
    'PERSONAL_MONTHLY_INCOME': 0.0,
    'OTHER_INCOMES': 0.0,
    'FLAG_VISA': 0,
    'FLAG_MASTERCARD': 0,
    'FLAG_DINERS': 0,
    'FLAG_AMERICAN_EXPRESS': 0,
    'FLAG_OTHER_CARDS': 0,
    'QUANT_BANKING_ACCOUNTS': 1,
    'QUANT_SPECIAL_BANKING_ACCOUNTS': 1,
    'PERSONAL_ASSETS_VALUE': 0.0,
    'QUANT_CARS': 0,
    'COMPANY': 'Y',
    'PROFESSIONAL_CITY': 'unknown',
    'PROFESSIONAL_BOROUGH': 'unknown',
    'FLAG_PROFESSIONAL_PHONE': "N",
    'PROFESSIONAL_PHONE_AREA_CODE': '000',
    'MONTHS_IN_THE_JOB': 0,
    'PROFESSION_CODE': 1,
    'MATE_PROFESSION_CODE': 1,
    'EDUCATION_LEVEL.1': 1,
    'FLAG_HOME_ADDRESS_DOCUMENT': 0,
    'FLAG_RG': 0,
    'FLAG_CPF': 0,
    'FLAG_INCOME_PROOF': 0,
    'FLAG_ACSP_RECORD': "N",
}
#
# async def model_predict(data) -> Tuple[bool, float]:
#     logger.info(f"📦 prediciendo: {data}")
#     df = build_input_dataframe(data)
#
#     print("📄 Primera fila del DataFrame:")
#     for col, val in df.iloc[0].items():
#         print(f"{col}: {val}")
#
#     # Simulamos procesamiento asíncrono
#     await asyncio.sleep(0.1)
#
#     # Resultado random
#     success = True
#     score = round(random.uniform(0, 1), 4)  # un float entre 0 y 1 con 4 decimales
#
#     logger.info(f"✅ resultado: success={success}, score={score}")
#     return success, score


def build_input_dataframe(job_data: dict) -> pd.DataFrame:
    input_data = job_data.get("input", {})
    row = {}

    for col in FINAL_COLUMNS:
        if col in input_data:
            row[col] = input_data[col]
        else:
            row[col] = DEFAULTS.get(col, None)

    df = pd.DataFrame([row])
    logger.info(f"✅ dataframe construido: df={df}")

    return df


''' This function generates a predicion given the modelpath and a dataframe with the information
provided via de UI
'''
async def predict_credit_risk(ui_data, model_name="stacking_model.pkl", )-> Tuple[bool, float]:
    print(f"✅ Modelo {model_name} realizando predicción...")

    # model_path = os.path.join(MODELS_DIR, model_name)
    #Json to Dataframe and adding needed columns
    X_new = build_input_dataframe(ui_data)
    X_n = process_all_features_v2.process_features_json(X_new)
    model = joblib.load(os.path.join("credit_risk_analysis", "models", model_name))
    predictions = model.predict(X_n)
    probabilities = model.predict_proba(X_n)[:, 1]
    X_n["TARGET_LABEL_BAD=1"] = predictions
    X_n["PROBABILITY_BAD=1"] = probabilities
    print("✅ Predicciones de riesgo crediticio completadas.")

    return True,predictions