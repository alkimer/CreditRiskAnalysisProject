from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from credit_risk_analysis.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

from matplotlib.pyplot import grid
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE

app = typer.Typer()

# Frequency Encoder personalizado
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns
        for col in self.feature_names_in_:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps[col] = freq
        return self

    def transform(self, X):
        check_is_fitted(self, 'freq_maps')
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_encoded = X.copy()
        for col in self.feature_names_in_:
            X_encoded[col] = X_encoded[col].map(self.freq_maps[col]).fillna(0)
        return X_encoded.to_numpy()


def process_features(X_new_path=None,new_data=False, return_data=False):
    # Columnas
    vars_to_drop = [
        'ID_CLIENT', 'CLERK_TYPE', 'QUANT_ADDITIONAL_CARDS', 'POSTAL_ADDRESS_TYPE',
        'EDUCATION_LEVEL', 'FLAG_MOBILE_PHONE', 'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS',
        'QUANT_SPECIAL_BANKING_ACCOUNTS', 'COMPANY', 'PROFESSIONAL_CITY',
        'PROFESSIONAL_BOROUGH', 'MONTHS_IN_THE_JOB', 'MATE_PROFESSION_CODE',
        'FLAG_HOME_ADDRESS_DOCUMENT', 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF',
        'FLAG_ACSP_RECORD']

    num_cont = ['PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'PERSONAL_ASSETS_VALUE']
    num_disc = ['MARITAL_STATUS', 'QUANT_DEPENDANTS', 'MONTHS_IN_RESIDENCE',
                'QUANT_BANKING_ACCOUNTS', 'QUANT_CARS', 'AGE',
                'PROFESSION_CODE', 'OCCUPATION_TYPE']
    cat_bin = ['APPLICATION_SUBMISSION_TYPE', 'SEX', 'FLAG_RESIDENCIAL_PHONE', 'FLAG_EMAIL',
               'FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_OTHER_CARDS', 'FLAG_PROFESSIONAL_PHONE']
    cat_multi = ['STATE_OF_BIRTH', 'CITY_OF_BIRTH', 'RESIDENCIAL_STATE', 'RESIDENCIAL_CITY',
                 'RESIDENCIAL_BOROUGH', 'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCE_TYPE',
                 'PROFESSIONAL_STATE', 'PROFESSIONAL_PHONE_AREA_CODE', 'NACIONALITY',
                 'PRODUCT', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3']

    if new_data==False:
        # Cargar datos desde la carpeta correcta
        X_train = pd.read_csv("data/interim/X_train.csv")
        X_val = pd.read_csv("data/interim/X_val.csv")
        y_train = pd.read_csv("data/interim/y_train.csv").select_dtypes(include='number').iloc[:, 0].astype(int)
        y_val = pd.read_csv("data/interim/y_val.csv").select_dtypes(include='number').iloc[:, 0].astype(int)

        
        # Eliminar columnas
        X_train.drop(columns=vars_to_drop, inplace=True, errors='ignore')
        X_val.drop(columns=vars_to_drop, inplace=True, errors='ignore')

        # Separar multicategóricas
        multi_high_card = [col for col in cat_multi if X_train[col].nunique() > 30]
        multi_low_card = [col for col in cat_multi if col not in multi_high_card]

        # Pipelines
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_bin_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(drop='if_binary', sparse_output=False))
        ])

        cat_multi_low_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        cat_multi_high_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('freq', FrequencyEncoder())
        ])

        # ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, num_cont + num_disc),
            ('bin', cat_bin_pipeline, cat_bin),
            ('multi_low', cat_multi_low_pipeline, multi_low_card),
            ('multi_high', cat_multi_high_pipeline, multi_high_card)
        ])

        # Procesamiento
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        # Columnas finales
        bin_names = preprocessor.named_transformers_['bin']['encode'].get_feature_names_out(cat_bin)
        multi_low_names = preprocessor.named_transformers_['multi_low']['onehot'].get_feature_names_out(multi_low_card)
        multi_high_names = multi_high_card
        final_columns = num_cont + num_disc + list(bin_names) + list(multi_low_names) + multi_high_names

        with open('data/interim/final_columns.json', 'w') as file:
            json.dump(final_columns, file)

        X_train_final = pd.DataFrame(X_train_processed, columns=final_columns, index=X_train.index)
        X_val_final = pd.DataFrame(X_val_processed, columns=final_columns, index=X_val.index)

        #os.makedirs("data/processed/balanced", exist_ok=True)
        # Guardar preprocessor
        joblib.dump(preprocessor, 'models/preprocessor.pkl', compress=True)

        # Guardar data procesada
        X_train_final.to_csv("data/processed/X_train_processed.csv", index=False)
        X_val_final.to_csv("data/processed/X_val_processed.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        y_val.to_csv("data/processed/y_val.csv", index=False)

        # Aplicar SMOTE
        smote = SMOTE(random_state=42)
        X_train_final_bal, y_train_bal = smote.fit_resample(X_train_final, y_train)
        
        X_train_final_bal.to_csv("data/processed/X_train_processed_bal.csv", index=False)
        y_train_bal.to_csv("data/processed/y_train_bal.csv", index=False)

        print("✅ Archivos procesados y guardados correctamente.")
    else:
        X_new = pd.read_csv(X_new_path)
        X_new.drop(columns=vars_to_drop, inplace=True, errors='ignore')
        # Separar multicategóricas
        multi_high_card = [col for col in cat_multi if X_new[col].nunique() > 30]
        multi_low_card = [col for col in cat_multi if col not in multi_high_card]

        # cargar preprocessor previamente ajustado
        preprocessor = joblib.load('models/preprocessor.pkl')
        # Procesamiento
        X_new_processed = preprocessor.transform(X_new)
        # Columnas finales
        with open('data/interim/final_columns.json', 'r') as file:
            final_columns = json.load(file)

        X_new_final = pd.DataFrame(X_new_processed, columns=final_columns, index=X_new.index)
        #os.makedirs("data/processed/balanced", exist_ok=True)

        # Guardar data procesada
        X_new_final.to_csv("data/processed/X_new_processed.csv", index=False)
        print("✅ Nuevos datos procesados y guardados correctamente.")

@app.command()
def main(
        X_new_path=None,
        new_data=False
):

    print("Iniciando procesamiento de características...")
    process_features(X_new_path=X_new_path, new_data=new_data)
    print("Procesamiento completado.")

if __name__ == "__main__":
    main()