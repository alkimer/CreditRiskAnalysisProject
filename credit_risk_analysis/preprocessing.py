from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.utils import Bunch

def preprocess_data(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    max_categories: int = 50,
    random_state: int = 42
) -> Bunch:
    """
    Preprocesa datos con:
    - Imputación (num: media, cat: moda)
    - OneHotEncoder en categóricas de baja cardinalidad
    - SMOTE sobre entrenamiento
    - Transformación consistente de validación y test
    """

    # Detectar columnas numéricas
    numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Detectar y convertir categóricas a string
    initial_categorical = [col for col in x_train.columns if col not in numeric_features]
    for col in initial_categorical:
        x_train[col] = x_train[col].astype(str)
        x_val[col] = x_val[col].astype(str)
        x_test[col] = x_test[col].astype(str)

    # Filtrar categóricas con baja cardinalidad
    categorical_features = [
        col for col in initial_categorical
        if x_train[col].nunique() <= max_categories
    ]

    # Aviso sobre columnas descartadas
    dropped_cols = set(initial_categorical) - set(categorical_features)
    if dropped_cols:
        print(f"ℹ️  Columnas categóricas descartadas por alta cardinalidad (> {max_categories}): {dropped_cols}")

    # Pipelines de transformación
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=random_state))
    ])

    # Fit + SMOTE en entrenamiento
    x_train_balanced, y_train_balanced = pipeline.fit_resample(x_train, y_train)

    # Transformar val/test solo con el preprocesador
    preprocessor_fitted = pipeline.named_steps['preprocessor']
    x_val_processed = preprocessor_fitted.transform(x_val)
    x_test_processed = preprocessor_fitted.transform(x_test)

    # Obtener nombres de columnas finales
    cat_encoded_cols = []
    if categorical_features:
        ohe = preprocessor_fitted.named_transformers_['cat'].named_steps['encoder']
        cat_encoded_cols = ohe.get_feature_names_out(categorical_features).tolist()

    final_columns = numeric_features + cat_encoded_cols

    # Convertir a DataFrames
    df_train = pd.DataFrame(x_train_balanced, columns=final_columns)
    df_val = pd.DataFrame(x_val_processed, columns=final_columns)
    df_test = pd.DataFrame(x_test_processed, columns=final_columns)

    # Agregar target
    df_train[y_train.name] = y_train_balanced
    df_val[y_val.name] = y_val.values
    df_test[y_test.name] = y_test.values

    return Bunch(
        train_data=(df_train.drop(columns=[y_train.name]), df_train[y_train.name]),
        val_data=(df_val.drop(columns=[y_val.name]), df_val[y_val.name]),
        test_data=(df_test.drop(columns=[y_test.name]), df_test[y_test.name]),
        preprocessed_dfs={
            'train': df_train,
            'val': df_val,
            'test': df_test
        },
        preprocessor=preprocessor_fitted
    )
