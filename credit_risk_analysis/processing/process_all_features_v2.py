import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

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


'''features seleccionadas para la predicción del modelo'''
feat_select = ['MARITAL_STATUS', 'MONTHS_IN_RESIDENCE', 'AGE', 'OCCUPATION_TYPE',
               'SEX', 'FLAG_RESIDENCIAL_PHONE', 'STATE_OF_BIRTH',
               'RESIDENCIAL_STATE', 'RESIDENCE_TYPE', 'PROFESSIONAL_STATE', 'PRODUCT',
               'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', 'RESIDENCIAL_PHONE_AREA_CODE',
               'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3']


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