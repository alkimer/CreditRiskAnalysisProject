import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, PowerTransformer

warnings.simplefilter(action='ignore', category=FutureWarning)

def process_numerical_continuous_split(X_train, X_val, use_power=False):
    """
    Preprocesa las variables numéricas continuas usando X_train y X_val y
    devuelve solo las columnas procesadas (_SCALED).
    """
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

    train, val = _proc_personal_monthly_income(train, val, use_power)
    train, val = _proc_other_incomes       (train, val, use_power)
    train, val = _proc_personal_assets_value(train, val, use_power)

    # Al final, quedarse solo con las columnas *_SCALED
    cont_feats = [
        "PERSONAL_MONTHLY_INCOME_SCALED",
        "OTHER_INCOMES_SCALED",
        "PERSONAL_ASSETS_VALUE_SCALED"
    ]
    return train[cont_feats].copy(), val[cont_feats].copy()


def _proc_personal_monthly_income(train, val, use_power):
    feat = "PERSONAL_MONTHLY_INCOME"
    
    # 1. Imputación
    med = train[feat].median()
    train[feat].fillna(med, inplace=True)
    val  [feat].fillna(med, inplace=True)
    
    # 2. Outliers con IQR
    Q1, Q3 = train[feat].quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    train[feat] = train[feat].clip(lower=low, upper=high)
    val  [feat] = val  [feat].clip(lower=low, upper=high)
    
    # 3. PowerTransformer opcional
    source = feat
    if use_power:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        train[feat + "_PT"] = pt.fit_transform(train[[feat]])
        val  [feat + "_PT"] = pt.transform   (val  [[feat]])
        source = feat + "_PT"
    
    # 4. Escalado
    scaler = StandardScaler()
    train[feat + "_SCALED"] = scaler.fit_transform(train[[source]])
    val  [feat + "_SCALED"] = scaler.transform   (val  [[source]])
    return train, val


def _proc_other_incomes(train, val, use_power):
    feat = "OTHER_INCOMES"
    
    # 1. Imputación
    med = train[feat].median()
    train[feat].fillna(med, inplace=True)
    val  [feat].fillna(med, inplace=True)
    
    # 2. Winsorización 1%-99%
    low, high = train[feat].quantile([0.01, 0.99])
    train[feat] = train[feat].clip(lower=low, upper=high)
    val  [feat] = val  [feat].clip(lower=low, upper=high)
    
    # 3. PowerTransformer opcional
    source = feat
    if use_power:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        train[feat + "_PT"] = pt.fit_transform(train[[feat]])
        val  [feat + "_PT"] = pt.transform   (val  [[feat]])
        source = feat + "_PT"
    
    # 4. Escalado
    scaler = StandardScaler()
    train[feat + "_SCALED"] = scaler.fit_transform(train[[source]])
    val  [feat + "_SCALED"] = scaler.transform   (val  [[source]])
    return train, val


def _proc_personal_assets_value(train, val, use_power):
    feat = "PERSONAL_ASSETS_VALUE"
    
    # 1. Imputación
    med = train[feat].median()
    train[feat].fillna(med, inplace=True)
    val  [feat].fillna(med, inplace=True)
    
    # 2. Outliers con IQR
    Q1, Q3 = train[feat].quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    train[feat] = train[feat].clip(lower=low, upper=high)
    val  [feat] = val  [feat].clip(lower=low, upper=high)
    
    # 3. PowerTransformer opcional
    source = feat
    if use_power:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        train[feat + "_PT"] = pt.fit_transform(train[[feat]])
        val  [feat + "_PT"] = pt.transform   (val  [[feat]])
        source = feat + "_PT"
    
    # 4. Escalado
    scaler = StandardScaler()
    train[feat + "_SCALED"] = scaler.fit_transform(train[[source]])
    val  [feat + "_SCALED"] = scaler.transform   (val  [[source]])
    
    return train, val


#if __name__ == "__main__":
    
    # X_train_in = pd.read_csv("./data/data_splitted/X_train.csv", header=0, encoding='UTF-8')
    # X_val_in   = pd.read_csv("./data/data_splitted/X_val.csv",   header=0, encoding='UTF-8')
    
    # train_cont, val_cont = process_numerical_continuous_split(X_train_in, X_val_in, use_power=False)
    
    # train_cont.to_csv("./data/processed/X_train_continous.csv", index=False)
    # val_cont.to_csv("./data/processed/X_val_continous.csv", index=False)