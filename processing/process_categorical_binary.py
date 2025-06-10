import pandas as pd
import matplotlib.pyplot as plt


def start():
    X_train_full = pd.read_csv('./data/data_splitted/X_train.csv', header=None, encoding='UTF-8')
    X_val_full   = pd.read_csv('./data/data_splitted/X_val.csv',   header=None, encoding='UTF-8')
    var_names    = pd.read_csv('./external/var_names.txt', header=None, delimiter='\t', encoding='latin1')

    blurry = [5,21,24,25,26,27,28]
    binary_index = [6,16,33,37]
    binary_index.extend(blurry)
    binary_index.sort()

    print("The following are the binary categrorical variables\nin the dataset (first their indexes):\n")
    print(binary_index)
    display(var_names[binary_index])

    # Create binary subdatasets
    X_train = X_train_full.iloc[:,binary_index]
    X_val = X_val_full.iloc[:,binary_index]
    

def process_POSTAL_ADDRESS_TYPE():    
    return

def process_SEX():
    return

def process_FLAG_RESIDENCIAL_PHONE():
    return

def process_FLAG_EMAIL():
    return

def process_FLAG_VISA():
    return

def process_FLAG_MASTERCARD():
    return

def process_FLAG_DINERS():   
    return


def process_FLAG_FLAG_AMERICAN_EXPRESS():
    return

def process_FLAG_PROFESSIONAL_PHONE():
    return