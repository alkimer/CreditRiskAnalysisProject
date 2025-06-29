import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def start(path_train, path_val):
    X_train_full = pd.read_csv(path_train, header=0, encoding='UTF-8')
    X_val_full   = pd.read_csv(path_val,   header=0, encoding='UTF-8')
    var_names    = X_train_full.columns

    blurry = [21,24,25,28]
    binary_index = [3,6,16,33,37]
    binary_index.extend(blurry)
    binary_index.sort()

    # print("\n\nThe following are the binary categrorical variables\nin the dataset (first their original indexes):\n")

    # print(binary_index)
    # print(var_names[binary_index].tolist())

    # Create binary subdatasets
    X_train = X_train_full.iloc[:,binary_index]
    X_val   =   X_val_full.iloc[:,binary_index]
    
    return X_train, X_val


def process_APPLICATION_SUBMISSION_TYPE(var_train, var_val):    
    """There is a category called '0' but it remain as it is (for its high frequency). 
    There is no process for this variable """    
    null_or_empty_values(var_train, var_val, 'APPLICATION_SUBMISSION_TYPE')
        
    return var_train, var_val

def process_SEX(var_train, var_val):
    """
    This variable has 4 categories, but 2 of them have extremely low frequency.
    Here we take away those 2 categories: 'N' and ' '.
    
    Here we impute 'N' and ' ' in SEX variable with 'M' and 'F' in equal proportion,
    maintaining the original row count.
    """
    
    null_or_empty_values(var_train, var_val, 'SEX')
    
    # Detect positions with values to imputar
    mask_train = var_train.isin(['N', ' '])
    mask_val   = var_val.isin(['N', ' '])
    
    
    # Number of imputations
    n_train = mask_train.sum()
    n_val = mask_val.sum()
    
    imputations_train = create_balanced_imputations(n_train)
    imputations_val = create_balanced_imputations(n_val)

    # Copies to avoid modifying the original ones
    var_train_cleaned = var_train.copy()
    var_val_cleaned = var_val.copy()
    
    # Assign imputations
    var_train_cleaned.loc[mask_train] = imputations_train
    var_val_cleaned.loc[mask_val] = imputations_val

    # print("Original train length:", len(var_train))
    # print("Cleaned train length:", len(var_train_cleaned))
    # print("Original val length:", len(var_val))
    # print("Cleaned val length:", len(var_val_cleaned))
    
    null_or_empty_values(var_train_cleaned, var_val_cleaned, 'Cleaned variable SEX')

    return var_train_cleaned, var_val_cleaned


def process_FLAG_RESIDENCIAL_PHONE(var_train, var_val):
    """There are no comments from the EDA for this variable claeaning. 
    There is no process for this variable """   
    null_or_empty_values(var_train, var_val, 'FLAG_RESIDENCIAL_PHONE') 
 
    return var_train, var_val


def process_FLAG_EMAIL(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    null_or_empty_values(var_train, var_val, 'FLAG_EMAIL')
    
    return var_train, var_val


def process_FLAG_VISA(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    null_or_empty_values(var_train, var_val, 'FLAG_VISA')
    
    return var_train, var_val


def process_FLAG_MASTERCARD(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    null_or_empty_values(var_train, var_val, 'FLAG_MASTRECARD')
    
    return var_train, var_val

def process_FLAG_OTHER_CARDS(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    null_or_empty_values(var_train, var_val, 'FLAG_OTHER_CARDS')
    
    return var_train, var_val

def process_FLAG_PROFESSIONAL_PHONE(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """     
    null_or_empty_values(var_train, var_val, 'FLAG_PROFESSIONAL_PHONE')
    
    return var_train, var_val


def null_or_empty_values(df_train, df_val, varName):
    
    df_tr_str =  df_train.astype(str).map(lambda x: x.strip())
    null_values_tr = df_train.isnull()
    empty_values_tr = df_tr_str == ''
    
    ne_train = null_values_tr | empty_values_tr
    
    df_val_str = df_val.astype(str).map(lambda x: x.strip())
    null_values = df_val.isnull()
    empty_values = df_val_str == ''
    
    ne_val = null_values | empty_values
    
    # print(f"\n\nVariable: {varName} \n")
    
    #if ne_train.any().any():
        # print("Null or empty values in TRAIN. It's time to clean")
    #else:
        # print("Train OK. Continue!")
    #if ne_val.any().any():
        # print("Null or empty values in VAL. It's time to clean")
    #else:
        # print("Val OK. Continue!")
    
    return


def create_balanced_imputations(n):
    half = n // 2
    remainder = n % 2
    imputations = ['M'] * half + ['F'] * half + (['M'] if remainder else [])
    np.random.shuffle(imputations)
    return imputations


def clean_all_binary(in_X_train, in_X_val):
    """Makes the processing of the binary (categorical) variables.
    The dropped variables directly are not considered.
    Args: 
        in_X_train: PATH to X_train datast
        in_X_val:   PATH to X_val datast
    """
    
    X_train, X_val = start(in_X_train, in_X_val)
    
    # print('\n////////////////////')

    cleaned_train_APPLICATION_SUBMISSION_TYPE, cleaned_val_APPLICATION_SUBMISSION_TYPE = \
        process_APPLICATION_SUBMISSION_TYPE(X_train['APPLICATION_SUBMISSION_TYPE'], X_val['APPLICATION_SUBMISSION_TYPE'])
    
    cleaned_train_SEX, cleaned_val_SEX = \
        process_SEX(X_train['SEX'], X_val['SEX'])
    
    cleaned_train_FLAG_RESIDENCIAL_PHONE, cleaned_val_FLAG_RESIDENCIAL_PHONE = \
        process_FLAG_RESIDENCIAL_PHONE(X_train['FLAG_RESIDENCIAL_PHONE'], X_val['FLAG_RESIDENCIAL_PHONE'])
        
    cleaned_train_FLAG_EMAIL, cleaned_val_FLAG_EMAIL = \
        process_FLAG_EMAIL(X_train['FLAG_EMAIL'], X_val['FLAG_EMAIL'])
   
    cleaned_train_FLAG_VISA, cleaned_val_FLAG_VISA = \
        process_FLAG_VISA(X_train['FLAG_VISA'], X_val['FLAG_VISA'])
        
    cleaned_train_FLAG_MASTERCARD, cleaned_val_FLAG_MASTERCARD = \
        process_FLAG_MASTERCARD(X_train['FLAG_MASTERCARD'], X_val['FLAG_MASTERCARD'])
    
    cleaned_train_FLAG_OTHER_CARDS, cleaned_val_FLAG_OTHER_CARDS = \
        process_FLAG_OTHER_CARDS(X_train['FLAG_OTHER_CARDS'], X_val['FLAG_OTHER_CARDS'])
    
    cleaned_train_FLAG_PROFESSIONAL_PHONE, cleaned_val_FLAG_PROFESSIONAL_PHONE = \
        process_FLAG_PROFESSIONAL_PHONE(X_train['FLAG_PROFESSIONAL_PHONE'], X_val['FLAG_PROFESSIONAL_PHONE'])
        
    clean_X_train = pd.concat([cleaned_train_APPLICATION_SUBMISSION_TYPE, 
                               cleaned_train_SEX,
                               cleaned_train_FLAG_RESIDENCIAL_PHONE,
                               cleaned_train_FLAG_EMAIL,
                               cleaned_train_FLAG_VISA,
                               cleaned_train_FLAG_MASTERCARD,
                               cleaned_train_FLAG_OTHER_CARDS,
                               cleaned_train_FLAG_PROFESSIONAL_PHONE                               
                               ], axis=1)
    
    clean_X_val  =  pd.concat([cleaned_val_APPLICATION_SUBMISSION_TYPE, 
                               cleaned_val_SEX,
                               cleaned_val_FLAG_RESIDENCIAL_PHONE,
                               cleaned_val_FLAG_EMAIL,
                               cleaned_val_FLAG_VISA,
                               cleaned_val_FLAG_MASTERCARD,
                               cleaned_val_FLAG_OTHER_CARDS,
                               cleaned_val_FLAG_PROFESSIONAL_PHONE                               
                               ], axis=1)
    
    ordinalEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinalEncoder.fit(clean_X_train)
    ordinalEncoder.fit(clean_X_val)
    
    encoded_X_train = pd.DataFrame(
        ordinalEncoder.transform(clean_X_train),
        columns=clean_X_train.columns,
        index=clean_X_train.index)

    encoded_X_val = pd.DataFrame(
        ordinalEncoder.transform(clean_X_val),
        columns=clean_X_val.columns,
        index=clean_X_val.index)
    
    #encoded_X_train.to_csv('data/processed/interim/X_train_binary.csv', index=False)
    #encoded_X_val.to_csv('data/processed/interim/X_val_binary.csv', index=False)
      
    # print("\n\n--- End of the categorical binary variables processing ---\n\n")    
    return encoded_X_train, encoded_X_val


if __name__ == "__main__":
    encoded_X_train, encoded_X_val = clean_all_binary("./data/data_splitted/X_train.csv", "./data/data_splitted/X_val.csv")
    
    #encoded_X_train.to_csv("./data/processed/X_train_binary.csv", index=False)
    #encoded_X_train.to_csv("./data/processed/X_val_binary.csv", index=False)

