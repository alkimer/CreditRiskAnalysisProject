import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, PowerTransformer

warnings.simplefilter(action='ignore', category=FutureWarning)

def process_numerical_continuous(X_train, X_val, use_power=False):
    """
    Preprocesa las variables numéricas continuas usando X_train y X_val:
      1) Imputa valores nulos con la mediana de X_train
      2) Trata outliers usando límites calculados en X_train (IQR o winsorización)
      3) (Opcional) Ajusta PowerTransformer (Yeo-Johnson) en X_train y transforma X_val
      4) Escala con StandardScaler ajustado en X_train y aplicado a X_val

    Devuelve:
      X_train_proc, X_val_proc
    """
    train = X_train.copy()
    val   = X_val.copy()

    train, val = process_personal_monthly_income(train, val, use_power)
    train, val = process_other_incomes       (train, val, use_power)
    train, val = process_personal_assets_value(train, val, use_power)

    return train, val

def process_personal_monthly_income(train, val, use_power):
    """
    - Imputa PERSONAL_MONTHLY_INCOME con mediana
    - Trata outliers con IQR
    - (Opcional) Aplica PowerTransformer (Yeo-Johnson)
    - Escala con StandardScaler
    """
    feature = "PERSONAL_MONTHLY_INCOME"
    
    # 1. Imputación
    med = train[feature].median()
    train[feature].fillna(med, inplace=True)
    val[feature].fillna(med, inplace=True)
    
    # 2. Outliers IQR
    Q1, Q3 = train[feature].quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    train[feature + "_CLIPPED"] = train[feature].clip(lower=low, upper=high)
    val  [feature + "_CLIPPED"] = val  [feature].clip(lower=low, upper=high)
    
    # 3. PowerTransformer
    if use_power:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        train[feature + "_PT"] = pt.fit_transform(train[[feature + "_CLIPPED"]])
        val  [feature + "_PT"] = pt.transform   (val  [[feature + "_CLIPPED"]])
        src = feature + "_PT"
    else:
        src = feature + "_CLIPPED"
    
    # 4. Escalado
    scaler = StandardScaler()
    train[feature + "_SCALED"] = scaler.fit_transform(train[[src]])
    val  [feature + "_SCALED"] = scaler.transform   (val  [[src]])

    return train, val

def process_other_incomes(train, val, use_power):
    """
    - Imputa OTHER_INCOMES con mediana
    - Trata outliers con winsorización (1%-99%)
    - (Opcional) PowerTransformer
    - Escala con StandardScaler
    """
    feature = "OTHER_INCOMES"
    
    # 1) Imputación
    med = train[feature].median()
    train[feature].fillna(med, inplace=True)
    val  [feature].fillna(med, inplace=True)

    # 2) Winsorización 1%-99%
    low, high = train[feature].quantile([0.01, 0.99])
    train[feature + "_CLIPPED"] = train[feature].clip(lower=low, upper=high)
    val  [feature + "_CLIPPED"] = val  [feature].clip(lower=low, upper=high)

    # 3) PowerTransformer (si aplica)
    if use_power:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        train[feature + "_PT"] = pt.fit_transform(train[[feature + "_CLIPPED"]])
        val  [feature + "_PT"] = pt.transform   (val  [[feature + "_CLIPPED"]])
        src = feature + "_PT"
    else:
        src = feature + "_CLIPPED"

    # 4) Escalado
    scaler = StandardScaler()
    train[feature + "_SCALED"] = scaler.fit_transform(train[[src]])
    val  [feature + "_SCALED"] = scaler.transform   (val  [[src]])

    return train, val

def process_personal_assets_value(train, val, use_power):
    """
    - Imputa PERSONAL_ASSETS_VALUE con mediana
    - Trata outliers con IQR
    - (Opcional) PowerTransformer
    - Escala con StandardScaler
    """
    feature = "PERSONAL_ASSETS_VALUE"
    
    # 1) Imputación
    med = train[feature].median()
    train[feature].fillna(med, inplace=True)
    val  [feature].fillna(med, inplace=True)

    # 2) Outliers IQR
    Q1, Q3 = train[feature].quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    train[feature + "_CLIPPED"] = train[feature].clip(lower=low, upper=high)
    val  [feature + "_CLIPPED"] = val  [feature].clip(lower=low, upper=high)

    # 3) PowerTransformer (si aplica)
    if use_power:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        train[feature + "_PT"] = pt.fit_transform(train[[feature + "_CLIPPED"]])
        val  [feature + "_PT"] = pt.transform   (val  [[feature + "_CLIPPED"]])
        src = feature + "_PT"
    else:
        src = feature + "_CLIPPED"

    # 4) Escalado
    scaler = StandardScaler()
    train[feature + "_SCALED"] = scaler.fit_transform(train[[src]])
    val  [feature + "_SCALED"] = scaler.transform   (val  [[src]])

    return train, val

