import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def get_train_val_sets(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split training dataset into two new sets used for train and validation.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=2023, shuffle=True
    )
    return X_train, X_val, y_train, y_val


# ---------- USO: cargar datos y aplicar split ----------
if __name__ == "__main__":
    # Cargar el CSV
    df = pd.read_csv("data/processed/data_with_columns.csv")

    # Asegurar que la variable target esté bien definida
    target_col = "TARGET_LABEL_BAD=1"  

    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Aplicar split
    X_train, X_val, y_train, y_val = get_train_val_sets(X, y)

    # Opcional: mostrar shapes
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    # Guardar resultados
    X_train.to_csv("data/data_splitted/X_train.csv", index=False)
    X_val.to_csv("data/data_splitted/X_val.csv", index=False)
    y_train.to_csv("data/data_splitted/y_train.csv", index=False)
    y_val.to_csv("data/data_splitted/y_val.csv", index=False)
    print("Datos guardados en data/data_splitted/")