
import json
import logging
import os
import sys
from typing import Tuple

import joblib
import pandas as pd
import typer

sys.path.append(os.path.abspath('../'))

app = typer.Typer()

MODELS_DIR = "../../models"

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

    X_new = build_input_dataframe(ui_data)
    preprocessor = joblib.load('credit_risk_analysis/models/preprocessor.pkl')

    df = preprocessor.transform(X_new)

    with open('data/interim/final_columns.json', 'r') as file:
        final_columns = json.load(file)

    df = pd.DataFrame(df, columns=final_columns)

    pipeline = joblib.load('credit_risk_analysis/models/stacking_model.pkl')

    return pipeline.predict_proba(df)[:, 1][0]
