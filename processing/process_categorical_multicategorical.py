import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def start(path_train, path_val):
    # -------------------------------------------------------------
    # Carga los archivos CSV de entrenamiento y validación.
    # Devuelve únicamente las columnas de multicategóricas especificadas.
    # -------------------------------------------------------------
    X_train_full = pd.read_csv(path_train, header=0, encoding='utf-8')
    X_val_full   = pd.read_csv(path_val,   header=0, encoding='utf-8')

    location_cols = [
        'STATE_OF_BIRTH', 'CITY_OF_BIRTH',
        'RESIDENCIAL_STATE', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', 'RESIDENCIAL_PHONE_AREA_CODE',
        'PROFESSIONAL_STATE', 'PROFESSIONAL_PHONE_AREA_CODE'
    ]
    return X_train_full[location_cols], X_val_full[location_cols]



def normalize_and_impute(df_train, df_val):
    # -------------------------------------------------------------
    # Normaliza los datos categóricos:
    # - Convierte a mayúsculas
    # - Elimina espacios en blanco
    # - Reemplaza valores vacíos o nulos por 'OTRO'
    # Aplica los cambios tanto en train como en val.
    # -------------------------------------------------------------
    for col in df_train.columns:
        df_train[col] = df_train[col].astype(str).str.strip().str.upper().replace({'NAN': 'OTRO', '': 'OTRO'})
        df_val[col]   = df_val[col].astype(str).str.strip().str.upper().replace({'NAN': 'OTRO', '': 'OTRO'})

    df_train.fillna('OTRO', inplace=True)
    df_val.fillna('OTRO', inplace=True)
    return df_train, df_val



def clip_rare_categories(df_train, df_val, threshold=0.01):
    # -------------------------------------------------------------
    # Detecta y reemplaza categorías poco frecuentes (menos del threshold)
    # con el valor 'OTRO' para reducir la cardinalidad.
    # Se aplica a ambas particiones: train y val.
    # -------------------------------------------------------------
    for col in df_train.columns:
        frecuencia = df_train[col].value_counts(normalize=True)
        categorias_raras = frecuencia[frecuencia < threshold].index
        df_train[col] = df_train[col].replace(categorias_raras, 'OTRO')
        df_val[col] = df_val[col].replace(categorias_raras, 'OTRO')
    return df_train, df_val



def encode_combined(df_train, df_val):
    # -------------------------------------------------------------
    # Aplica codificación combinada:
    # - Usa One-Hot Encoding para columnas con ≤ 50 categorías únicas
    # - Usa Frequency Encoding para columnas con más cardinalidad
    # Asegura consistencia entre train y val.
    # -------------------------------------------------------------
    df_train_ohe = pd.DataFrame(index=df_train.index)
    df_val_ohe = pd.DataFrame(index=df_val.index)
    df_train_freq = pd.DataFrame(index=df_train.index)
    df_val_freq = pd.DataFrame(index=df_val.index)

    for col in df_train.columns:
        cardinalidad = df_train[col].nunique()

        if cardinalidad <= 50:
            # One-Hot Encoding
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe.fit(df_train[[col]])
            columnas_ohe = ohe.get_feature_names_out([col])

            df_train_encoded = pd.DataFrame(ohe.transform(df_train[[col]]), columns=columnas_ohe, index=df_train.index)
            df_val_encoded = pd.DataFrame(ohe.transform(df_val[[col]]), columns=columnas_ohe, index=df_val.index)

            df_train_ohe = pd.concat([df_train_ohe, df_train_encoded], axis=1)
            df_val_ohe = pd.concat([df_val_ohe, df_val_encoded], axis=1)

        else:
            # Frequency Encoding
            frecuencia = df_train[col].value_counts(normalize=True)
            df_train_freq[col + "_freq"] = df_train[col].map(frecuencia)
            df_val_freq[col + "_freq"] = df_val[col].map(frecuencia).fillna(0)

    # Combinar ambas codificaciones
    df_train_final = pd.concat([df_train_ohe, df_train_freq], axis=1)
    df_val_final = pd.concat([df_val_ohe, df_val_freq], axis=1)

    return df_train_final, df_val_final



def clean_all_multi():
    # -------------------------------------------------------------
    # Ejecuta toda la pipeline de procesamiento:
    # 1. Carga los datasets
    # 2. Normaliza e imputa
    # 3. Reemplaza categorías raras
    # 4. Codifica con OHE o Frequency según corresponda
    # 5. Guarda resultados como CSV
    # Devuelve los DataFrames codificados.
    # -------------------------------------------------------------
    path_train = './data/data_splitted/X_train.csv'
    path_val   = './data/data_splitted/X_val.csv'

    X_train, X_val = start(path_train, path_val)
    X_train, X_val = normalize_and_impute(X_train, X_val)
    # X_train, X_val = clip_rare_categories(X_train, X_val, threshold=0.01)
    X_train_enc, X_val_enc = encode_combined(X_train, X_val)

    # Guardar resultados
    X_train_enc.to_csv('data/processed/location_X_train.csv', index=False)
    X_val_enc.to_csv('data/processed/location_X_val.csv', index=False)

    print("✅ Variables de ubicación multicategóricas procesadas correctamente.")
    return X_train_enc, X_val_enc


if __name__ == "__main__":
    clean_all_multi()
