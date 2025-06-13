import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def load_data():
    global df, numeric_df
    import os
    import pandas as pd
    from IPython.display import display, HTML
    # Configuración para mostrar bien los DataFrames
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_colwidth', None)
    # === RUTAS ABSOLUTAS Y SEGURAS ===
    # Ruta absoluta al archivo Excel con nombres de columnas
    excel_path = os.path.abspath(os.path.join("data", "external", "PAKDD2010_VariablesList.XLS"))
    # Ruta absoluta al archivo de datos .txt
    txt_path = os.path.abspath(os.path.join("data", "external", "PAKDD2010_Modeling_Data.txt"))
    # Ruta absoluta al archivo destino .csv
    save_path = os.path.abspath(os.path.join("data", "processed", "data-with-columns.csv"))
    # === LECTURA DE ARCHIVOS ===
    # Leer archivo Excel (nombres de columnas)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo Excel en: {excel_path}")
    try:
        variables_df = pd.read_excel(excel_path)
        column_names = variables_df['Var_Title'].head(54).tolist()
    except Exception as e:
        raise RuntimeError(f"❌ Error leyendo el archivo Excel: {e}")
    # Leer archivo de datos .txt sin cabecera
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo de datos en: {txt_path}")
    try:
        data_df = pd.read_csv(txt_path, sep='\t', header=None, encoding='latin1')
    except Exception as e:
        raise RuntimeError(f"❌ Error leyendo el archivo de datos: {e}")
    # Verificar que coincidan las columnas#%%
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')  # %%
    df = pd.read_csv("./data/processed/data-with-columns.csv")
    df.rename(columns={"TARGET_LABEL_BAD=1": "TARGET_LABEL_BAD"}, inplace=True)
    numeric_df = df.select_dtypes(include=["int64", "float64"]).drop(columns=["ID_CLIENT"])
    numeric_df.head()
    return np, plt, sns


def analisis_de_nulos():
    global umbral
    import pandas as pd
    total_filas = len(numeric_df)
    nulos_por_columna = pd.DataFrame({
        'columna': numeric_df.columns,
        'nulos': numeric_df.isnull().sum()
    })
    nulos_por_columna['porcentaje_nulos'] = (nulos_por_columna['nulos'] / total_filas) * 100
    nulos_por_columna = nulos_por_columna[nulos_por_columna['nulos'] > 0].copy()
    umbral = 30  # % de nulos a partir del cual se sugiere eliminar
    nulos_por_columna['sugerencia'] = nulos_por_columna['porcentaje_nulos'].apply(
        lambda x: 'ELIMINAR' if x > umbral else 'CONSERVAR / IMPUTAR'
    )
    nulos_por_columna = nulos_por_columna.sort_values(by='porcentaje_nulos', ascending=False).reset_index(drop=True)
    # Imprimir los resultados en modo texto
    print("\nAnálisis de columnas con valores nulos:")
    print("----------------------------------------")
    for _, row in nulos_por_columna.iterrows():
        print(f"Columna: {row['columna']}")
        print(f"  → Nulos: {row['nulos']}")
        print(f"  → Porcentaje de nulos: {row['porcentaje_nulos']:.2f}%")
        print(f"  → Sugerencia: {row['sugerencia']}")
        print("")
    print(f"Total de filas del dataframe: {total_filas}")
    return pd

def analisis_para_quant_additional_cards():
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="POSTAL_ADDRESS_TYPE", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de POSTAL_ADDRESS_TYPE según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    conteo = df["POSTAL_ADDRESS_TYPE"].value_counts().sort_index()
    
    plt.figure(figsize=(10, 4))
    conteo.plot(kind="bar")
    plt.title("Frecuencia de cada valor en POSTAL_ADDRESS_TYPE")
    plt.xlabel("POSTAL_ADDRESS_TYPE")
    plt.ylabel("Frecuencia")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    

def analisis_para_postal_address_type():
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="MARITAL_STATUS", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de MARITAL_STATUS según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def analisis_para_marital_status():
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="QUANT_DEPENDANTS", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de QUANT_DEPENDANTS según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_quant_dependants():
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="EDUCATION_LEVEL", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de EDUCATION_LEVEL según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_education_level():
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="NACIONALITY", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de NACIONALITY según TARGET_LABEL_BAD")
    plt.xlabel("NACIONALITY")
    plt.ylabel("Cantidad")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()
    

def analisis_para_nacionality():
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="RESIDENCE_TYPE", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    residence_dist = df.groupby('RESIDENCE_TYPE')['TARGET_LABEL_BAD'].value_counts(normalize=True).unstack().rename(
        columns={0: "Proporción Pagadores", 1: "Proporción Deudores"}
    )
    
    residence_dist["Cantidad Total"] = df['RESIDENCE_TYPE'].value_counts()
    
    # Calcular el promedio general de pagadores y deudores
    general_distribution = df['TARGET_LABEL_BAD'].value_counts(normalize=True).rename(
        {0: "Promedio Pagadores", 1: "Promedio Deudores"}
    )
    
    
    print(residence_dist)

def analisis_para_residence_type():
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="MONTHS_IN_RESIDENCE", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de MONTHS_IN_RESIDENCE según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    df.rename(columns={"TARGET_LABEL_BAD=1": "TARGET_LABEL_BAD"}, inplace=True)
    
    # Definir los rangos para agrupar los meses de residencia
    bins = [0, 6, 12, 24, 60, 120, float("inf")]
    labels = ["0-6", "7-12", "13-24", "25-60", "61-120", "120+"]
    
    # Crear una nueva columna con los rangos
    df["RESIDENCE_BIN"] = pd.cut(df["MONTHS_IN_RESIDENCE"], bins=bins, labels=labels, include_lowest=True)
    
    residence_bin_dist = df.groupby("RESIDENCE_BIN")["TARGET_LABEL_BAD"].value_counts(normalize=True).unstack().rename(
        columns={0: "Proporción Pagadores", 1: "Proporción Deudores"}
    )
    
    residence_bin_dist["Cantidad Total"] = df["RESIDENCE_BIN"].value_counts()
    
    # Mostrar la tabla
    print(residence_bin_dist)
    

def analisis_para_months_in_residence():
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="FLAG_EMAIL", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de FLAG_EMAIL según TARGET_LABEL_BAD")
    plt.xlabel("¿Tiene email registrado? (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    

def analisis_para_flag_email():
    # Gráfico de distribución para PERSONAL_MONTHLY_INCOME
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="PERSONAL_MONTHLY_INCOME", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de PERSONAL_MONTHLY_INCOME según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="TARGET_LABEL_BAD", y="PERSONAL_MONTHLY_INCOME")
    plt.ylim(0, 5000)
    plt.title("Boxplot de Ingreso Mensual por Clase de TARGET_LABEL_BAD (rango acotado)")
    plt.xlabel("TARGET_LABEL_BAD (0 = Pagador, 1 = Deudor)")
    plt.ylabel("Ingreso Mensual Personal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def analisis_para_personal_monthly_income():
    
    
    # Crear columna logarítmica (agregando 1 para evitar log(0))
    df["LOG_OTHER_INCOMES"] = np.log1p(df["OTHER_INCOMES"])
    
    # Gráfico de distribución log-transformado
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="LOG_OTHER_INCOMES",
        hue="TARGET_LABEL_BAD",
        bins=50,
        element="step",
        stat="density",
        common_norm=False,
        palette="Set2"
    )
    plt.title("Distribución logarítmica de OTHER_INCOMES según TARGET_LABEL_BAD")
    plt.xlabel("Log(OTHER_INCOMES + 1)")
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def analisis_para_other_incomes():
    # Gráfico de barras para FLAG_VISA
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="FLAG_VISA", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de FLAG_VISA según TARGET_LABEL_BAD")
    plt.xlabel("¿Tiene tarjeta VISA? (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Tabla de proporciones y cantidad total
    visa_dist = (
        df.groupby("FLAG_VISA")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores", 1: "Proporción Deudores"})
    )
    
    visa_dist["Cantidad Total"] = df["FLAG_VISA"].value_counts()
    print(visa_dist)
    

def analisis_para_flag_visa():
    # Gráfico de barras para FLAG_MASTERCARD
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="FLAG_MASTERCARD", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de FLAG_MASTERCARD según TARGET_LABEL_BAD")
    plt.xlabel("¿Tiene tarjeta MASTERCARD? (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Tabla de proporciones y cantidad total
    mast_dist = (
        df.groupby("FLAG_MASTERCARD")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores", 1: "Proporción Deudores"})
    )
    
    mast_dist["Cantidad Total"] = df["FLAG_MASTERCARD"].value_counts()
    print(mast_dist)
    
    

def analisis_para_flag_mastercard():
    # Gráfico de barras para FLAG_DINERS
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="FLAG_DINERS", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de FLAG_DINERS según TARGET_LABEL_BAD")
    plt.xlabel("¿Tiene tarjeta DINERS? (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Tabla de proporciones y cantidad total
    diners_dist = (
        df.groupby("FLAG_DINERS")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores", 1: "Proporción Deudores"})
    )
    
    diners_dist["Cantidad Total"] = df["FLAG_DINERS"].value_counts()
    print(diners_dist)
    

def analisis_para_flag_diners():
    # Gráfico de barras para FLAG_AMERICAN_EXPRESS
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="FLAG_AMERICAN_EXPRESS", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de FLAG_AMERICAN_EXPRESS según TARGET_LABEL_BAD")
    plt.xlabel("¿Tiene tarjeta AMERICAN EXPRESS? (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Tabla de proporciones y cantidad total
    ae_dist = (
        df.groupby("FLAG_AMERICAN_EXPRESS")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores", 1: "Proporción Deudores"})
    )
    
    ae_dist["Cantidad Total"] = df["FLAG_AMERICAN_EXPRESS"].value_counts()
    
    print(ae_dist)  # Corregido: antes se imprimía diners_dist por error
    

def analisis_para_flag_american_express():
    # Gráfico de barras para FLAG_OTHER_CARDS
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="FLAG_OTHER_CARDS", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de FLAG_OTHER_CARDS según TARGET_LABEL_BAD")
    plt.xlabel("¿Tiene otras tarjetas? (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Tabla de proporciones y cantidad total
    OC_dist = (
        df.groupby("FLAG_OTHER_CARDS")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores", 1: "Proporción Deudores"})
    )
    
    OC_dist["Cantidad Total"] = df["FLAG_OTHER_CARDS"].value_counts()
    print(OC_dist)
    

def analisis_para_flag_other_cards():
    # Gráfico de distribución para QUANT_BANKING_ACCOUNTS
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="QUANT_BANKING_ACCOUNTS", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de QUANT_BANKING_ACCOUNTS según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Crear la feature combinada
    df["CANTIDAD_TARJETAS"] = df[[
        "FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS",
        "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS"
    ]].sum(axis=1)
    
    # Calcular proporción de pagadores y deudores por cantidad de tarjetas
    card_count_dist = df.groupby("CANTIDAD_TARJETAS")["TARGET_LABEL_BAD"].value_counts(normalize=True).unstack().rename(
        columns={0: "Proporción Pagadores", 1: "Proporción Deudores"}
    )
    
    # Agregar la cantidad total por grupo
    card_count_dist["Cantidad Total"] = df["CANTIDAD_TARJETAS"].value_counts()
    
    # Mostrar la tabla
    print(card_count_dist)
    

def analisis_para_quant_banking_accounts():
    # Gráfico de distribución para QUANT_SPECIAL_BANKING_ACCOUNTS
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="QUANT_SPECIAL_BANKING_ACCOUNTS", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de QUANT_SPECIAL_BANKING_ACCOUNTS según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_quant_special_banking_accounts():
    # Gráfico de distribución para PERSONAL_ASSETS_VALUE
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="PERSONAL_ASSETS_VALUE", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de PERSONAL_ASSETS_VALUE según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
    # Cargar dataset y renombrar columna target si es necesario
    df.rename(columns={"TARGET_LABEL_BAD=1": "TARGET_LABEL_BAD"}, inplace=True)
    
    # Crear columna logarítmica para activos personales
    df["LOG_ASSET_VALUE"] = np.log1p(df["PERSONAL_ASSETS_VALUE"])
    
    # Boxplot para comparar según TARGET_LABEL_BAD
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="TARGET_LABEL_BAD", y="LOG_ASSET_VALUE")
    plt.title("Boxplot de LOG(PERSONAL_ASSETS_VALUE + 1) por TARGET_LABEL_BAD")
    plt.xlabel("TARGET_LABEL_BAD (0 = Pagador, 1 = Deudor)")
    plt.ylabel("Logaritmo del valor de activos personales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def analisis_para_personal_assets_value():
    # Gráfico de barras para QUANT_CARS
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="QUANT_CARS", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de QUANT_CARS según TARGET_LABEL_BAD")
    plt.xlabel("¿Tiene al menos un auto? (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Tabla de proporciones por cantidad de autos
    cars_dist = (
        df.groupby("QUANT_CARS")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores", 1: "Proporción Deudores"})
    )
    
    cars_dist["Cantidad Total"] = df["QUANT_CARS"].value_counts()
    print(cars_dist)
    
    

def analisis_para_quant_cars():
    # Gráfico de distribución para MONTHS_IN_THE_JOB
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="MONTHS_IN_THE_JOB", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de MONTHS_IN_THE_JOB según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_months_in_the_job():
    # Gráfico de distribución para PROFESSION_CODE
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="PROFESSION_CODE", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de PROFESSION_CODE según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_profession_code():
    
    # Gráfico de distribución para OCCUPATION_TYPE
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="OCCUPATION_TYPE", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de OCCUPATION_TYPE según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
    # Seleccionar los tipos de ocupación más comunes (opcional, acá se usan todos)
    top_occupation_types = df["OCCUPATION_TYPE"].value_counts().index
    
    # Calcular proporciones por tipo de ocupación
    occupation_dist = (
        df[df["OCCUPATION_TYPE"].isin(top_occupation_types)]
        .groupby("OCCUPATION_TYPE")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores (%)", 1: "Proporción Deudores (%)"})
        * 100  # convertir a porcentaje
    )
    
    # Agregar cantidad total
    occupation_dist["Cantidad Total"] = df["OCCUPATION_TYPE"].value_counts().loc[top_occupation_types]
    
    # Redondear
    occupation_dist = occupation_dist.round(2)
    
    # Mostrar
    print(occupation_dist)
    

def analisis_para_occupation_type():
    # Gráfico de distribución para MATE_PROFESSION_CODE
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="MATE_PROFESSION_CODE", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de MATE_PROFESSION_CODE según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Calcular distribución en porcentaje para los más frecuentes
    top_mate_prof = df["MATE_PROFESSION_CODE"].value_counts().nlargest(15).index
    
    mate_prof_dist = (
        df[df["MATE_PROFESSION_CODE"].isin(top_mate_prof)]
        .groupby("MATE_PROFESSION_CODE")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores (%)", 1: "Proporción Deudores (%)"})
        * 100
    )
    
    # Agregar cantidad total
    mate_prof_dist["Cantidad Total"] = df["MATE_PROFESSION_CODE"].value_counts().loc[top_mate_prof]
    
    # Redondear
    mate_prof_dist = mate_prof_dist.round(2)
    
    print(mate_prof_dist)
    

def analisis_para_mate_profession_code():
    # Gráfico de distribución para EDUCATION_LEVEL.1
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="EDUCATION_LEVEL.1", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de EDUCATION_LEVEL.1 según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Calcular nulos
    nulos_edu = df["EDUCATION_LEVEL.1"].isna().sum()
    porcentaje_nulos_edu = (nulos_edu / len(df)) * 100
    
    # Rellenar nulos como categoría "NULO"
    df["EDUCATION_LEVEL.1_CAT"] = df["EDUCATION_LEVEL.1"].fillna("NULO")
    
    # Calcular proporciones
    top_edu_values = df["EDUCATION_LEVEL.1_CAT"].value_counts().nlargest(15).index
    edu_dist = (
        df[df["EDUCATION_LEVEL.1_CAT"].isin(top_edu_values)]
        .groupby("EDUCATION_LEVEL.1_CAT")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores (%)", 1: "Proporción Deudores (%)"})
        * 100
    )
    
    # Agregar cantidad total
    edu_dist["Cantidad Total"] = df["EDUCATION_LEVEL.1_CAT"].value_counts().loc[top_edu_values]
    edu_dist = edu_dist.round(2)
    
    print(edu_dist)
    

def analisis_para_education_level():
    # Gráfico de distribución para FLAG_HOME_ADDRESS_DOCUMENT
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="FLAG_HOME_ADDRESS_DOCUMENT", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de FLAG_HOME_ADDRESS_DOCUMENT según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_flag_home_address_document():
    # Gráfico de distribución para FLAG_RG
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="FLAG_RG", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de FLAG_RG según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_flag_rg():
    # Gráfico de distribución para FLAG_CPF
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="FLAG_CPF", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de FLAG_CPF según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_flag_cpf():
    # Gráfico de distribución para FLAG_INCOME_PROOF
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="FLAG_INCOME_PROOF", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de FLAG_INCOME_PROOF según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analisis_para_flag_income_proof():
    # Gráfico de barras para PRODUCT
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="PRODUCT", hue="TARGET_LABEL_BAD")
    plt.title("Distribución de PRODUCT según TARGET_LABEL_BAD")
    plt.xlabel("Tipo de producto (1, 2 o 7)")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Tratar valores nulos como categoría separada (aunque no haya en este caso)
    df["PRODUCT_CAT"] = df["PRODUCT"].fillna("NULO")
    
    # Calcular proporciones por valor de PRODUCT
    product_dist = (
        df.groupby("PRODUCT_CAT")["TARGET_LABEL_BAD"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: "Proporción Pagadores (%)", 1: "Proporción Deudores (%)"})
        * 100
    )
    
    # Agregar la cantidad total de registros por categoría
    product_dist["Cantidad Total"] = df["PRODUCT_CAT"].value_counts()
    
    # Redondear porcentajes
    product_dist = product_dist.round(2)
    
    # Mostrar la tabla
    print(product_dist)
    
    

def analisis_para_product():
    # Gráfico de distribución para AGE
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x="AGE", hue="TARGET_LABEL_BAD", bins=30, element="step", stat="density")
    plt.title("Distribución de AGE según TARGET_LABEL_BAD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Crear boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="TARGET_LABEL_BAD", y="AGE")
    plt.title("Boxplot de AGE por TARGET_LABEL_BAD")
    plt.xlabel("TARGET_LABEL_BAD (0 = Pagador, 1 = Deudor)")
    plt.ylabel("Edad")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def analisis_para_age():
    correlation_matrix = numeric_df.copy()
    correlation_matrix["TARGET_LABEL_BAD"] = df["TARGET_LABEL_BAD"]
    correlations = correlation_matrix.corr()["TARGET_LABEL_BAD"].sort_values(ascending=False)
    print(correlations)


    
