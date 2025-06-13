import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_numerical_discrete(csv_path, encode=False, binning=True, normalize=False, trim_max=6):
    """
    Abre un archivo CSV y realiza el preprocesamiento completo de variables numéricas/discretas
    usando funciones previamente definidas. Controla encoding, binning, normalización y trimming.
    """

    df = pd.read_csv(csv_path)

    df = process_payment_day(df, encode=encode)
    df = process_marital_status(df, encode=encode)
    df = process_quant_dependants(df, encode=encode, trim_max=trim_max, binning=binning)
    df = process_months_in_residence(df, encode=encode, binning=binning)
    df = process_quant_banking_accounts(df, normalize=normalize)
    df = process_quant_cars(df)
    df = process_months_in_the_job(df)
    df = process_education_level(df)
    df = process_quant_additional_cards(df)
    df = process_quant_special_banking_accounts(df)
    df = process_age(df, normalize=normalize)
    df = process_mate_profession_code(df)

    print("\n✅ Preprocesamiento finalizado. Vista general del DataFrame:")
    print(df.head())

    return df


def process_payment_day(df, encode=False):
    """
    La proporción de deudores no varía según el payment day, salvo el día 25 que tiene una proporción levemente
    superior de deudores
    Procesa PAYMENT_DAY:
    - Si encode=False: la convierte en categoría (PAYMENT_DAY_CAT). XGboost, LightGBM, RandomForest
    - Si encode=True: además aplica target encoding (PAYMENT_DAY_TE). Para modelos lineales, logistic regression
    """
    df = df.copy()

    # Convertir a categórica
    df["PAYMENT_DAY_CAT"] = df["PAYMENT_DAY"].astype("category")

    df.rename(columns={"TARGET_LABEL_BAD=1": "TARGET_LABEL_BAD"}, inplace=True)

    if encode:
        df["PAYMENT_DAY_TE"] = df["PAYMENT_DAY_CAT"].map(
            df.groupby("PAYMENT_DAY_CAT")["TARGET_LABEL_BAD"].mean()
        )
        print("✅ Target encoding aplicado a PAYMENT_DAY_CAT.")
    else:
        print("ℹ️ PAYMENT_DAY convertido en categórica.")

    df.drop(columns=["PAYMENT_DAY"], inplace=True)

    return df


def process_marital_status(df, encode=False):
    """
    Preprocesa la columna MARITAL_STATUS:
    - Convierte a categoría
    - Si encode=True, aplica target encoding
    """
    df = df.copy()
    df["MARITAL_STATUS_CAT"] = df["MARITAL_STATUS"].astype("category")

    if encode:
        df["MARITAL_STATUS_TE"] = df["MARITAL_STATUS_CAT"].map(
            df.groupby("MARITAL_STATUS_CAT")["TARGET_LABEL_BAD"].mean()
        )
        print("✅ Target encoding aplicado a MARITAL_STATUS.")
    else:
        print("ℹ️ MARITAL_STATUS convertido en categórica.")
    df.drop(columns=["MARITAL_STATUS"], inplace=True)

    return df


def process_quant_dependants(df, encode=False, trim_max=6, binning=False):

    """
    Luego del trimming y binning se ve claramente que a mayor cantidad de dependientes
    major cantidad de deudores.

    Preprocesa QUANT_DEPENDANTS:
    - Si binning=False: trunca valores extremos a trim_max (por defecto 6)
    - Si binning=True: agrupa en categorías "0", "1–3", "4+"
    - Si encode=True, aplica target encoding
    """
    df = df.copy()

    if binning:
        def categorizar_dependants(x):
            if x == 0:
                return "0"
            elif 1 <= x <= 3:
                return "1-3"
            else:
                return "4+"

        df["QUANT_DEPENDANTS_BIN"] = df["QUANT_DEPENDANTS"].apply(categorizar_dependants).astype("category")

        if encode:
            df["QUANT_DEPENDANTS_TE"] = df["QUANT_DEPENDANTS_BIN"].map(
                df.groupby("QUANT_DEPENDANTS_BIN")["TARGET_LABEL_BAD"].mean()
            )
            print("✅ Target encoding aplicado a QUANT_DEPENDANTS (binned).")
        else:
            print("ℹ️ QUANT_DEPENDANTS agrupado en categorías '0', '1–3', '4+'.")

    else:
        df["QUANT_DEPENDANTS_TRIM"] = df["QUANT_DEPENDANTS"].clip(upper=trim_max)

        if encode:
            df["QUANT_DEPENDANTS_TE"] = df["QUANT_DEPENDANTS_TRIM"].map(
                df.groupby("QUANT_DEPENDANTS_TRIM")["TARGET_LABEL_BAD"].mean()
            )
            print(f"✅ Target encoding aplicado a QUANT_DEPENDANTS (truncado a {trim_max}).")
        else:
            print(f"ℹ️ QUANT_DEPENDANTS truncado a máximo {trim_max}.")

    df.drop(columns=["QUANT_DEPENDANTS"], inplace=True)

    return df


def process_months_in_residence(df, encode=False, binning=True):

    """
    Preprocesa MONTHS_IN_RESIDENCE:
    distribución muy sesgada a la izq, y varios nulos.
    Le dejamos el binning en True por default dado que refleja mejor la relación con la morosidad
    imputamos la mediana a los nulos
    CREA COLUMNAS :MONTHS_IN_RESIDENCE_BIN Y MONTHS_IN_RESIDENCE_IMPUTED
    - Imputa nulos con la mediana (6)
    - Si binning=True: agrupa en rangos
    - Si encode=True: aplica target encoding
    """
    df = df.copy()

    # Imputar nulos con la mediana
    mediana = df["MONTHS_IN_RESIDENCE"].median()
    df["MONTHS_IN_RESIDENCE_IMPUTED"] = df["MONTHS_IN_RESIDENCE"].fillna(mediana)

    if binning:
        def categorizar_mes(x):
            if x == 0:
                return "0"
            elif 1 <= x <= 5:
                return "1-5"
            elif 6 <= x <= 11:
                return "6-11"
            elif 12 <= x <= 23:
                return "12-23"
            else:
                return "24+"

        df["MONTHS_IN_RESIDENCE_BIN"] = df["MONTHS_IN_RESIDENCE_IMPUTED"].apply(categorizar_mes).astype("category")

        if encode:
            df["MONTHS_IN_RESIDENCE_TE"] = df["MONTHS_IN_RESIDENCE_BIN"].map(
                df.groupby("MONTHS_IN_RESIDENCE_BIN")["TARGET_LABEL_BAD"].mean()
            )
            print("✅ Target encoding aplicado a MONTHS_IN_RESIDENCE (binned).")
        else:
            print("ℹ️ MONTHS_IN_RESIDENCE imputado y binned en categorías.")
    else:
        if encode:
            df["MONTHS_IN_RESIDENCE_TE"] = df["MONTHS_IN_RESIDENCE_IMPUTED"].map(
                df.groupby("MONTHS_IN_RESIDENCE_IMPUTED")["TARGET_LABEL_BAD"].mean()
            )
            print("✅ Target encoding aplicado a MONTHS_IN_RESIDENCE (imputado, sin binning).")
        else:
            print("ℹ️ MONTHS_IN_RESIDENCE imputado sin binning.")

    df.drop(columns=["MONTHS_IN_RESIDENCE"], inplace=True)

    return df



def process_quant_banking_accounts(df, normalize=False):

    """
    Si bien no se especifica exactamente qué es , todo parece indicar que refleja cantidades, por lo tanto
    la tratamos como numérica discreta.

    Preprocesa QUANT_BANKING_ACCOUNTS como cantidad numérica discreta.
    - No requiere imputación
    - Si normalize=True: aplica MinMax normalization
    """
    df = df.copy()

    if normalize:
        min_val = df["QUANT_BANKING_ACCOUNTS"].min()
        max_val = df["QUANT_BANKING_ACCOUNTS"].max()
        df["QUANT_BANKING_ACCOUNTS_NORM"] = (df["QUANT_BANKING_ACCOUNTS"] - min_val) / (max_val - min_val)
        df.drop(columns=["QUANT_BANKING_ACCOUNTS"], inplace=True)

        print("✅ QUANT_BANKING_ACCOUNTS normalizada con MinMax.")
    else:
        print("ℹ️ QUANT_BANKING_ACCOUNTS mantenida como cantidad discreta sin normalizar.")

    return df


def process_quant_cars(df):
    """
    - No hace binning, imputación ni encoding.
    - solo valida que sea entero
    """
    df = df.copy()
    df["QUANT_CARS_CLEAN"] = df["QUANT_CARS"].astype(int)
    print("ℹ️ QUANT_CARS válida")
    df.drop(columns=["QUANT_CARS"], inplace=True)

    return df


def process_months_in_the_job(df):
    """
    Elimina la columna MONTHS_IN_THE_JOB si existe en el DataFrame.
    El 99% de los casos son cero , por lo tanto no es representativa.
    """
    df = df.copy()

    if "MONTHS_IN_THE_JOB" in df.columns:
        df.drop(columns=["MONTHS_IN_THE_JOB"], inplace=True)
        print("✅ Columna MONTHS_IN_THE_JOB eliminada del DataFrame.")
    else:
        print("ℹ️ La columna MONTHS_IN_THE_JOB no existe en el DataFrame.")

    return df



def process_education_level(df):
    """
    Elimina la columna EDUCATION_LEVEL si existe en el DataFrame.
    El 100% de los casos son cero , por lo tanto no es representativa.
    """
    df = df.copy()

    if "EDUCATION_LEVEL" in df.columns:
        df.drop(columns=["EDUCATION_LEVEL"], inplace=True)
        print("✅ Columna EDUCATION_LEVEL eliminada del DataFrame.")
    else:
        print("ℹ️ La columna EDUCATION_LEVEL no existe en el DataFrame.")

    return df

def process_quant_additional_cards(df):
    """
    Elimina la columna quant_additional_cards si existe en el DataFrame.
    El 100% de los casos son cero, por lo tanto no es representativa.
    """
    df = df.copy()

    if "QUANT_ADDITIONAL_CARDS" in df.columns:
        df.drop(columns=["QUANT_ADDITIONAL_CARDS"], inplace=True)
        print("✅ Columna QUANT_ADDITIONAL_CARDS eliminada del DataFrame.")
    else:
        print("ℹ️ La columna QUANT_ADDITIONAL_CARDS no existe en el DataFrame.")

    return df

def process_quant_special_banking_accounts(df):
    """
    Elimina la columna QUANT_SPECIAL_BANKING_ACCOUNTS si existe en el DataFrame.
    El 100% de los casos son cero, por lo tanto no es representativa.
    """
    df = df.copy()

    if "QUANT_SPECIAL_BANKING_ACCOUNTS" in df.columns:
        df.drop(columns=["QUANT_SPECIAL_BANKING_ACCOUNTS"], inplace=True)
        print("✅ Columna QUANT_SPECIAL_BANKING_ACCOUNTS eliminada del DataFrame.")
    else:
        print("ℹ️ La columna QUANT_SPECIAL_BANKING_ACCOUNTS no existe en el DataFrame.")

    return df


def process_age(df, normalize=False):
    """
    Preprocesa la variable AGE como numérica continua/discreta.
    - Si normalize=True: aplica MinMax normalization, para regresión lineal, logística, MLP
    - Normalize=False : random forest, xgboost , lightgbm
    """
    df = df.copy()

    if normalize:
        min_age = df["AGE"].min()
        max_age = df["AGE"].max()
        df["AGE_NORM"] = (df["AGE"] - min_age) / (max_age - min_age)
        print("✅ AGE normalizada con MinMax.")
    else:
        df["AGE_DISCRETE"] = df["AGE"].astype(int)
        print("ℹ️ AGE conservada como numérica discreta.")

    df.drop(columns=["AGE"], inplace=True)

    return df


def process_profession_code(df, encode=False, binning=False):
    """
    15% NULOS, IMPUTAMOS CON LA MODA (EL MÁS FRECUENTE)

    Preprocesa PROFESSION_CODE:
    - Imputa nulos con la moda
    - Si binning=True: agrupa valores poco frecuentes como 'OTROS'
    - Si encode=True: aplica target encoding
    """
    df = df.copy()

    # Imputación con la moda
    moda = df["PROFESSION_CODE"].mode()[0]
    df["PROFESSION_CODE_IMPUTED"] = df["PROFESSION_CODE"].fillna(moda)

    if binning:
        # Marcar como 'OTROS' los valores poco frecuentes (<100 ocurrencias)
        counts = df["PROFESSION_CODE_IMPUTED"].value_counts()
        valores_comunes = counts[counts >= 100].index
        df["PROFESSION_CODE_BINNED"] = df["PROFESSION_CODE_IMPUTED"].apply(
            lambda x: x if x in valores_comunes else "OTROS"
        ).astype("category")

        if encode:
            df["PROFESSION_CODE_TE"] = df["PROFESSION_CODE_BINNED"].map(
                df.groupby("PROFESSION_CODE_BINNED")["TARGET_LABEL_BAD"].mean()
            )
            print("✅ Target encoding aplicado a PROFESSION_CODE (binned).")
        else:
            print("ℹ️ PROFESSION_CODE binned aplicada.")

    else:
        df["PROFESSION_CODE_CAT"] = df["PROFESSION_CODE_IMPUTED"].astype("category")

        if encode:
            df["PROFESSION_CODE_TE"] = df["PROFESSION_CODE_CAT"].map(
                df.groupby("PROFESSION_CODE_CAT")["TARGET_LABEL_BAD"].mean()
            )
            print("✅ Target encoding aplicado a PROFESSION_CODE.")
        else:
            print("ℹ️ PROFESSION_CODE convertida en categórica.")

    df.drop(columns=["PROFESSION_CODE"], inplace=True)

    return df


def process_mate_profession_code(df):
    """
    60% de nulos, droppeando
    Elimina la columna MATE_PROFESSION_CODE si existe en el DataFrame.
    """
    df = df.copy()

    if "MATE_PROFESSION_CODE" in df.columns:
        df.drop(columns=["MATE_PROFESSION_CODE"], inplace=True)
        print("✅ Columna MATE_PROFESSION_CODE eliminada del DataFrame.")
    else:
        print("ℹ️ La columna MATE_PROFESSION_CODE no existe en el DataFrame.")

    return df


def process_education_level(df):
    """
    60% de nulos, droppeando
    Elimina la columna EDUCATION_LEVEL si existe en el DataFrame.
    """
    df = df.copy()

    if "EDUCATION_LEVEL" in df.columns:
        df.drop(columns=["EDUCATION_LEVEL"], inplace=True)
        print("✅ Columna EDUCATION_LEVEL eliminada del DataFrame.")
    else:
        print("ℹ️ La columna EDUCATION_LEVEL no existe en el DataFrame.")

    return df

def process_occupation_type(df, encode=False, binning=False, normalize=False):
    """
    14.6% de nulos, imputamos categoría más frecuente.
    Preprocesamiento para OCCUPATION_TYPE:
    - Imputa nulos con la categoría más frecuente.
    - Si encode=True, aplica OneHotEncoding.
    - binning y normalize están ignorados porque no aplican en este caso.
    """
    df = df.copy()
    most_frequent = df["OCCUPATION_TYPE"].mode()[0]
    df["OCCUPATION_TYPE_IMPUTED"] = df["OCCUPATION_TYPE"].fillna(most_frequent)

    if encode:
        dummies = pd.get_dummies(df["OCCUPATION_TYPE_IMPUTED"], prefix="OCCUPATION_TYPE")
        df = df.drop("OCCUPATION_TYPE", axis=1)
        df = pd.concat([df, dummies], axis=1)

    return df


def process_numerical_discrete(csv_path, encode=False, binning=True, normalize=False):
    """
    Abre un archivo CSV y realiza el preprocesamiento completo de variables numéricas/discretas
    usando funciones previamente definidas. Controla encoding, binning, normalización y trimming.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    df = process_payment_day(df, encode=encode)
    df = process_marital_status(df, encode=encode)
    df = process_quant_dependants(df, encode=encode, trim_max=6, binning=binning)
    df = process_months_in_residence(df, encode=encode, binning=binning)
    df = process_quant_banking_accounts(df, normalize=normalize)
    df = process_quant_cars(df)
    df = process_months_in_the_job(df)
    df = process_education_level(df)
    df = process_quant_additional_cards(df)
    df = process_quant_special_banking_accounts(df)
    df = process_age(df, normalize=normalize)
    df = process_mate_profession_code(df)
    df = process_profession_code(df, encode=encode)
    df = process_occupation_type(df)

    # Mostrar todas las columnas en consola
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    print("\n✅ Preprocesamiento finalizado. Vista general del DataFrame:")
    print(df.head())

    return df



if __name__ == "__main__":
    ###This is just for testing Purposes""""
    # Ruta al archivo CSV
    csv_path = "./data/data_splitted/X_train.csv"

    train_discrete = process_numerical_discrete(csv_path, encode=True, binning=True, normalize=True)
    train_discrete.to_csv("./data/processed/X_train_discrete.csv", index=False)


    # csv_path = Path("data-with-columns.csv")


    # Guardar resultado en archivo
    #df_procesado.to_csv("data-preprocessed.csv", index=False)
    print("\n✅ Archivo preprocesado guardado como data-preprocessed.csv")
