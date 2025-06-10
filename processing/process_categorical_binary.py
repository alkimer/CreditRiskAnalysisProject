import pandas as pd
import matplotlib.pyplot as plt


def start():
    X_train_full = pd.read_csv('./data/data_splitted/X_train.csv', header=0, encoding='UTF-8')
    X_val_full   = pd.read_csv('./data/data_splitted/X_val.csv',   header=0, encoding='UTF-8')
    var_names    = pd.read_csv('./external/var_names.txt', header=None, delimiter='\t', encoding='latin1')

    blurry = [5,21,24,25,28]
    binary_index = [3,6,16,33,37]
    binary_index.extend(blurry)
    binary_index.sort()

    print("The following are the binary categrorical variables\nin the dataset (first their indexes):\n")
    print(binary_index)
    display(var_names.iloc[binary_index,:])

    # Create binary subdatasets
    X_train = X_train_full.iloc[:,binary_index]
    X_val = X_val_full.iloc[:,binary_index]
    
    return X_train, X_val, var_names

def process_APPLICATION_SUBMISSION_TYPE(var_train, var_val):    
    """There are no comments from the EDA for this variable. 
    There is no process for this variable """
    
    null_or_empty_values(var_train, var_val)
        
    return var_train, var_val

def process_POSTAL_ADDRESS_TYPE(var_train, var_val):    
    """There are no comments from the EDA for this variable. 
    There is no process for this variable """
    
    null_or_empty_values(var_train, var_val)
        
    return var_train, var_val

def process_SEX(var_train, var_val):
    """This variable has 4 categories, but 2 of them have extremely low frequency.
    Here we take away those 2 categories: 'N' and ' '."""
    
    
    var_train_cleaned = var_train[~var_train.iloc[:, 0].isin(['N', ' '])]
    var_val_cleaned = var_val[~var_val.iloc[:, 0].isin(['N', ' '])]
    
    return var_train_cleaned, var_val_cleaned  


def process_FLAG_RESIDENCIAL_PHONE(var_train, var_val):
    """There are no comments from the EDA for this variable. 
    There is no process for this variable """    
 
    return var_train, var_val


def process_FLAG_EMAIL(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    return var_train, var_val


def process_FLAG_VISA(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    return var_train, var_val


def process_FLAG_MASTERCARD(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    return var_train, var_val

def process_FLAG_OTHER_CARDS(var_train, var_val):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    return var_train, var_val

def process_FLAG_PROFESSIONAL_PHONE(var_train, var_val):
    return


def null_or_empty_values(df_train, df_val):
    df_tr_str = df_train.astype(str)
    null_values_tr = df_train.isnull()
    empty_values_tr = df_tr_str.str.strip() == ''
    
    ne_train = null_values_tr or empty_values_tr
    
    df_val_str = df_val.astype(str)
    null_values = df_val.isnull()
    empty_values = df_val_str.str.strip() == ''
    
    ne_val = null_values or empty_values
    
    if ne_train:
        print("null or empty values in TRAIN. It's time to clean")
    else:
        print("Train OK. Continue!")
    if ne_val:
        print("null or empty values in VAL. It's time to clean")
    else:
        print("Val OK. Continue!")
    
    return


def bind(df, col): 
    """"df debe ser un Dpandas.DataFrame
        col debe ser una pd.Series
    """
    df = pd.concat([df, col], axis=1)
    return


#X_train, X_val, var_names = start()

# train0, val0 = cleaned_train_POSTAL_ADDRESS_TYPE, cleaned_val_POSTAL_ADDRESS_TYPE = process_POSTAL_ADDRESS_TYPE(X_train[0], X_val[0])
# train1, val1 = cleaned_train_SEX, cleaned_val_SEX = process_SEX(X_train[1], X_val[1])
# train2, val2 = cleaned_train_FLAG_RESIDENCIAL_PHONE, cleaned_val_FLAG_RESIDENCIAL_PHONE = process_FLAG_RESIDENCIAL_PHONE(X_train[2], X_val[2])
# train3, val3 = cleaned_train_FLAG_EMAIL, cleaned_val_FLAG_EMAIL = process_FLAG_EMAIL(X_train[3], X_val[3])
# train3, val4 = cleaned_train_FLAG_VISA, cleaned_val_FLAG_VISA = process_FLAG_VISA(X_train[4], X_val[4])
# train5, val5 = cleaned_train_FLAG_MASTERCARD, cleaned_val_FLAG_MASTERCARD = process_FLAG_MASTERCARD(X_train[5], X_val[5])

