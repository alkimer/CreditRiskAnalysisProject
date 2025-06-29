import pandas as pd
from src.process_categorical_binary import clean_all_binary
from src.process_categorical_multicategorical import clean_all_multi
from src.process_numerical_continuous import process_numerical_continuous_split
from src.process_numerical_discrete import process_numerical_discrete
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from typing import Tuple
import boto3
import os



def start(access_id, secret_access):
    
    col_names = ['ID_CLIENT', 'CLERK_TYPE', 'PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'QUANT_ADDITIONAL_CARDS', 'POSTAL_ADDRESS_TYPE', 'SEX', 'MARITAL_STATUS', 'QUANT_DEPENDANTS', 'EDUCATION_LEVEL', 'STATE_OF_BIRTH', 'CITY_OF_BIRTH', 'NACIONALITY', 'RESIDENCIAL_STATE', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', 'FLAG_RESIDENCIAL_PHONE', 'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCE_TYPE', 'MONTHS_IN_RESIDENCE', 'FLAG_MOBILE_PHONE', 'FLAG_EMAIL', 'PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS', 'QUANT_BANKING_ACCOUNTS', 'QUANT_SPECIAL_BANKING_ACCOUNTS', 'PERSONAL_ASSETS_VALUE', 'QUANT_CARS', 'COMPANY', 'PROFESSIONAL_STATE', 'PROFESSIONAL_CITY', 'PROFESSIONAL_BOROUGH', 'FLAG_PROFESSIONAL_PHONE', 'PROFESSIONAL_PHONE_AREA_CODE', 'MONTHS_IN_THE_JOB', 'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE', 'EDUCATION_LEVEL', 'FLAG_HOME_ADDRESS_DOCUMENT', 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF', 'PRODUCT', 'FLAG_ACSP_RECORD', 'AGE', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3', 'TARGET_LABEL_BAD=1']
    modeling_data_path = './dataset/external/PAKDD2010_Modeling_Data.txt'
    
    if not os.path.exists(modeling_data_path):
        download_credit_data(access_id, secret_access)
        
    tagger('./dataset/external/PAKDD2010_Modeling_Data.txt',
           "./dataset/data_with_columns.csv", col_names)
    split_data("./dataset/data_with_columns.csv")
    final_processing("./dataset/splitted/X_train.csv", "./dataset/splitted/X_val.csv",
                    "./dataset/X_train_processed_unbalanced.csv",  "./dataset/y_train_processed_unbalanced.csv",
                    "./dataset/X_val_processed.csv", "./dataset/y_val.csv", smote=False)

    X_train_unb = pd.read_csv('./dataset/X_train_processed_unbalanced.csv')
    y_train_unb = pd.read_csv('./dataset/y_train_processed_unbalanced.csv')
    
    X_val = pd.read_csv('./dataset/X_val_processed.csv')
    y_val = pd.read_csv('./dataset/y_val.csv')
    
    print("\n-- Ready to Start --")
    
    return X_train_unb, y_train_unb, X_val, y_val


def start2():
    
    print("Reading Datasets...")
    X_train = pd.read_csv('./dataset/X_train_processed_unbalanced.csv')
    y_train = pd.read_csv('./dataset/y_train_processed_unbalanced.csv')

    X_val = pd.read_csv('./dataset/X_val_processed.csv')
    y_val = pd.read_csv('./dataset/y_val.csv')
    print("\n-- Ready to Start --")
    
    return X_train, y_train, X_val, y_val
        

def download_credit_data(aws_access_key_id, aws_secret_access_key):
    bucket_name = "anyoneai-datasets"
    prefix = "credit-data-2010/"
    local_folder = "./dataset/external"


    # Crear la carpeta local si no existe
    os.makedirs(local_folder, exist_ok=True)

    # Crear cliente de S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Listar objetos en ese prefijo (asume que querés todo el folder)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response:
        print("Files not found.")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        filename = os.path.basename(key)
        if not filename:  # en caso de que el key termine en "/"
            continue

        local_path = os.path.join(local_folder, filename)
        print(f"Downloading {key} to {local_path}")

        s3.download_file(bucket_name, key, local_path)

    print("Download complete.")


def tagger(path_txt, path_csv, col_names):
    """
    Convierte un archivo .txt a .csv agregando encabezado de columnas.
    
    Parámetros:
        path_txt (str): Ruta del archivo de texto de entrada.
        path_csv (str): Ruta del archivo CSV de salida.
        nombres_columnas (list): Lista con los nombres de las 54 columnas.
    """
    df = pd.read_csv(path_txt, header=None, sep='\t', encoding='latin1', low_memory=False)
    df[51] = pd.to_numeric(df[51], errors='coerce')
    df[52] = pd.to_numeric(df[52], errors='coerce')
    
    print("Reformatting Data...")
    
    # Dimensions checking
    if df.shape[1] != len(col_names):
        raise ValueError(f"Se esperaban {len(col_names)} columnas pero se encontraron {df.shape[1]}.")
    
    df.columns = col_names
    df.to_csv(path_csv, index=False)
    
    return


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


def split_data(path):

    print("Splitting Data...")
    df = pd.read_csv(path) #"./dataset/data_with_columns.csv")
    target_col = "TARGET_LABEL_BAD=1"  

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = get_train_val_sets(X, y)

    # Guardar resultados
    os.makedirs("./dataset/splitted", exist_ok=True)
    X_train.to_csv("./dataset/splitted/X_train.csv", index=False)
    X_val.to_csv("./dataset/splitted/X_val.csv", index=False)
    y_train.to_csv("./dataset/splitted/y_train.csv", index=False)
    y_val.to_csv("./dataset/splitted/y_val.csv", index=False)


def final_processing(path_train, path_val, X_train_output, y_train_output, X_val_output, y_val_output, smote=True):
    
    print("Preprocessing and Cleaning Data...")
    X_train_out, X_val_out, y_train, y_val = process_all(path_train, path_val)
    
    # Export X_train_out, X_val_out as csv files
    X_train_out.to_csv(X_train_output, index=False)
    X_val_out.to_csv(X_val_output, index=False)
    y_train.to_csv(y_train_output, index=False)
    y_val.to_csv(y_val_output, index=False)
   
    
    if smote:             
        
        xtrain_balanced, ytrain_balanced = randomOverSample(X_train_out, y_train)
        
        xtrain_balanced.to_csv(f"./dataset/X_train_balanced.csv", index=False)
        ytrain_balanced.to_csv(f"./dataset/y_train_balanced.csv", index=False)
        
        return xtrain_balanced, ytrain_balanced

    # X_train_out.to_csv(X_train_output, index=False)    
    # y_train.to_csv(y_train_output, index=False)
    
    
    return

def process_all(path_train, path_val):
    """ 
    This function takes the raw splitted dataframes and process the variables
    according to their nature (binary, multicategorical, continous, discrerte) using
    the 4 scripts dedicated to each group, and outputs the processed dataFrames,
    delivering the data ready to use in the pipeline.
    
    Args:
        path_train: a string with the path to the raw splitted X_train
        path_val:   a string with the path to the raw splitted X_val
        
    output: 
        (in ./dataset/processed/interim)
            X_tr.csv
            X_train_binary.csv
            X_train_continuous.csv
            X_train_discrete.csv
            X_train_multi.csv
            X_v.csv
            X_val_binary.csv
            X_val_continuous.csv
            X_val_discrete.csv
            X_val_multi.csv
        (in ./data/processed/)
            dropped_variables_list.txt    
            processed_variables_list.txt
            X_train_p.csv : output processed train dataset
            X_val_p.csv   : output processed val dataset
    
    return: None """
        
    path_y_train = "./dataset/splitted/y_train.csv"
    path_y_val   = "./dataset/splitted/y_val.csv"
    
    os.makedirs("./dataset/interim", exist_ok=True)
    path_Xtr = "./dataset/interim/X_tr.csv"
    path_Xv  = "./dataset/interim/X_v.csv"
        
    X_train_in = pd.read_csv(path_train, header=0, encoding='UTF-8')
    y_train_in = pd.read_csv(path_y_train,   header=0, encoding='UTF-8')
    X_val_in   = pd.read_csv(path_val,   header=0, encoding='UTF-8')
    y_val_in   = pd.read_csv(path_y_val,   header=0, encoding='UTF-8')
    
    Xtr = pd.concat([X_train_in, y_train_in], axis=1)
    Xv  = pd.concat([X_val_in, y_val_in], axis=1)
    
    Xtr.to_csv(path_Xtr, index=False)
    Xv.to_csv (path_Xv , index=False)
    
    
    train_binary,   val_binary = clean_all_binary(path_train, path_val)
    train_multi,     val_multi =  clean_all_multi(path_train, path_val)    
    train_continuous, val_continuous = process_numerical_continuous_split(X_train_in, X_val_in, use_power=False)
    train_discrete = process_numerical_discrete(path_Xtr, encode=True, binning=True, normalize=True)
    val_discrete   = process_numerical_discrete(path_Xv,  encode=True, binning=True, normalize=True)
    
    #os.makedirs("./dataset/processed_by_type", exist_ok=True)
    #train_continuous.to_csv("./dataset/processed_by_type/X_train_continuous.csv", index=False)
    #val_continuous.to_csv("./dataset/processed_by_type/X_val_continuous.csv", index=False)   
    #train_discrete.to_csv("./dataset/processed_by_type/X_train_discrete.csv", index=False)   
    #val_discrete.to_csv  ("./dataset/processed_by_type/X_val_discrete.csv", index=False)   
    
    X_train_out = pd.concat([train_binary, 
                             train_multi, 
                             train_continuous,
                             train_discrete], axis=1)

    X_val_out  =  pd.concat([val_binary, 
                             val_multi, 
                             val_continuous,
                             val_discrete], axis=1) 
    

    binary_cols = pd.DataFrame(train_binary.columns)
    multi_cols  = pd.DataFrame(trim(train_multi.columns))
    conti_cols  = pd.DataFrame(trim(train_continuous.columns))
    discr_cols  = pd.DataFrame(trim(train_discrete.columns))
    
    
    # print("\n\n--- END OF THE WHOLE VARIABLES PROCESSING ---")
    # print("\n\n\nBinary Variables Kept//\n")
    # print(binary_cols.to_string(index=False, header=False))
    # print("\n\nMulticategorical Variables Kept//\n")
    # print(multi_cols.to_string(index=False, header=False))
    # print("\n\nContinuous Variables Kept//\n")
    # print(conti_cols.to_string(index=False, header=False))
    # print("\n\nDiscrete Variables Kept//\n")
    # print(discr_cols.to_string(index=False, header=False))
    
    processed_col_names = pd.concat([binary_cols, multi_cols, conti_cols, discr_cols], axis=0)

    dropped = []    
    for i in X_train_in.columns:
        if i not in processed_col_names[0].tolist():
            dropped.append(i)           
            
    # print(f"\nOriginal # of columns: {len(X_train_in.columns)}")
    # print(f"Number of processed VARIABLES: {len(processed_col_names)}")
    # print(f"Number of dropped columns: {len(dropped)}")
    # print(f"Number of output columns: {len(X_train_out.columns)}")

    os.makedirs("./dataset/lists", exist_ok=True)
    pd.DataFrame(processed_col_names).to_csv(f"./dataset/lists/processed_variables_list.txt", index=False, header=False, sep='\t')
    pd.DataFrame(dropped).to_csv(f"./dataset/lists/dropped_variables_list.txt", index=False, header=False, sep='\t')
    
    # print(f"\n//  SHAPES OF THE OUTPUT DATASETS  //\n")
    # print("(X_train_p), (X_val_p)")
    # print(f"\n{X_train_out.shape}, {X_val_out.shape}\n")
    # print("Binary, Multicategorical, Continuous, Discrete\nTrain | Val")
    # print(train_binary.shape, val_binary.shape)
    # print(train_multi.shape,  val_multi.shape)
    # print(train_continuous.shape, val_continuous.shape)
    # print(train_discrete.shape, val_discrete.shape)
                
    
    return X_train_out, X_val_out, y_train_in, y_val_in


def trim(df):
    
    no_sufix = [col.rsplit('_', 1)[0] for col in df]
    
    unique_ordered = []
    seen = set()
    for col in no_sufix:
        if col not in seen:
            unique_ordered.append(col)
            seen.add(col)
    
    return unique_ordered


def randomOverSample(X, y, random_state=42):
    """
    Applies RandomOverSampler to balance the dataset with categorical features.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series or array): Target
        random_state (int): Seed for reproducibility

    Returns:
        X_resampled (pd.DataFrame), y_resampled (pd.Series)
    """
    
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    return X_resampled, y_resampled