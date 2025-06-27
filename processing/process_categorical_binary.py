import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def start(path_train, path_val, path_new=None, new_data=False):
    """Starts the processing of the binary categorical variables.
    Args:
        path_train: a string with the path to the raw splitted X_train
        path_val:   a string with the path to the raw splitted X_val
        path_new:   a string with the path to the new raw splitted X
        new_data:   a boolean indicating if the data is new or not
    Returns:
        X_train: a DataFrame with the processed X_train
        X_val:   a DataFrame with the processed X_val
        X:      a DataFrame with the processed X (if new_data is True)
    If new_data is False, it reads the data from the paths provided.
    If new_data is True, it reads the data from the path_new provided.
    It also prints the binary categorical variables in the dataset.
    """
    if new_data==False:
        X_train_full = pd.read_csv(path_train, header=0, encoding='UTF-8')
        X_val_full   = pd.read_csv(path_val,   header=0, encoding='UTF-8')
        var_names    = X_train_full.columns

        blurry = [21,24,25,28]
        binary_index = [3,6,16,33,37]
        binary_index.extend(blurry)
        binary_index.sort()

        print("\n\nThe following are the binary categrorical variables\nin the dataset (first their original indexes):\n")

        print(binary_index)
        print(var_names[binary_index].tolist())

        # Create binary subdatasets
        X_train = X_train_full.iloc[:,binary_index]
        X_val   =   X_val_full.iloc[:,binary_index]
        
        return X_train, X_val
    else:
        X_full = pd.read_csv(path_new, header=0, encoding='UTF-8')
        var_names = X_full.columns

        blurry = [21,24,25,28]
        binary_index = [3,6,16,33,37]
        binary_index.extend(blurry)
        binary_index.sort()

        print("\n\nThe following are the binary categrorical variables\nin the dataset (first their original indexes):\n")

        print(binary_index)
        print(var_names[binary_index].tolist())

        # Create binary subdatasets
        X = X_full.iloc[:,binary_index]
        
        return X


def process_APPLICATION_SUBMISSION_TYPE(var_train, var_val, var_new=None, new_data=False):    
    """There is a category called '0' but it remain as it is (for its high frequency). 
    There is no process for this variable """
    if new_data==False: 
        null_or_empty_values(var_train, var_val, 'APPLICATION_SUBMISSION_TYPE')
            
        return var_train, var_val
    else:
        null_or_empty_values(var_new, 'APPLICATION_SUBMISSION_TYPE')
        return var_new

def process_SEX(var_train, var_val, var_new=None, new_data=False):
    """
    This variable has 4 categories, but 2 of them have extremely low frequency.
    Here we take away those 2 categories: 'N' and ' '.
    
    Here we impute 'N' and ' ' in SEX variable with 'M' and 'F' in equal proportion,
    maintaining the original row count.
    """
    if new_data==False:
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

        print("Original train length:", len(var_train))
        print("Cleaned train length:", len(var_train_cleaned))
        print("Original val length:", len(var_val))
        print("Cleaned val length:", len(var_val_cleaned))
        
        null_or_empty_values(var_train_cleaned, var_val_cleaned, 'Cleaned variable SEX')

        return var_train_cleaned, var_val_cleaned
    else:
        null_or_empty_values(var_new, 'SEX')
        # Detect positions with values to imputar
        mask_new = var_new.isin(['N', ' '])

        # Number of imputations
        n_new = mask_new.sum()

        imputations_new = create_balanced_imputations(n_new)

        # Copies to avoid modifying the original ones
        var_new_cleaned = var_new.copy()

        # Assign imputations
        var_new_cleaned.loc[mask_new] = imputations_new

        print("Original new length:", len(var_new))
        print("Cleaned new length:", len(var_new_cleaned))

        null_or_empty_values(var_new_cleaned, 'Cleaned variable SEX')

        return var_new_cleaned


def process_FLAG_RESIDENCIAL_PHONE(var_train, var_val, var_new=None, new_data=False):
    """There are no comments from the EDA for this variable claeaning. 
    There is no process for this variable """   
    if new_data==False:
        null_or_empty_values(var_train, var_val, 'FLAG_RESIDENCIAL_PHONE')
        return var_train, var_val
    else:
        null_or_empty_values(var_new, 'FLAG_RESIDENCIAL_PHONE')
        return var_new


def process_FLAG_EMAIL(var_train, var_val, var_new=None, new_data=False):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    if new_data==False:
        null_or_empty_values(var_train, var_val, 'FLAG_EMAIL')
        return var_train, var_val
    else:
        null_or_empty_values(var_new, 'FLAG_EMAIL')
        return var_new


def process_FLAG_VISA(var_train, var_val, var_new=None, new_data=False):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    if new_data==False:
        null_or_empty_values(var_train, var_val, 'FLAG_VISA')
        return var_train, var_val
    else:
        null_or_empty_values(var_new, 'FLAG_VISA')
        return var_new


def process_FLAG_MASTERCARD(var_train, var_val, var_new=None, new_data=False):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    if new_data==False:
        null_or_empty_values(var_train, var_val, 'FLAG_MASTERCARD')
        return var_train, var_val
    else:
        null_or_empty_values(var_new, 'FLAG_MASTERCARD')
        return var_new

    null_or_empty_values(var_train, var_val, 'FLAG_VISA')
    
    return var_train, var_val


def process_FLAG_MASTERCARD(var_train, var_val, var_new=None, new_data=False):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    if new_data==False:
        null_or_empty_values(var_train, var_val, 'FLAG_MASTERCARD')
        return var_train, var_val
    else:
        null_or_empty_values(var_new, 'FLAG_MASTERCARD')
        return var_new

def process_FLAG_OTHER_CARDS(var_train, var_val, var_new=None, new_data=False):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """ 
    if new_data==False:
        null_or_empty_values(var_train, var_val, 'FLAG_OTHER_CARDS')
        return var_train, var_val
    else:
        null_or_empty_values(var_new, 'FLAG_OTHER_CARDS')
        return var_new
    

def process_FLAG_PROFESSIONAL_PHONE(var_train, var_val, var_new=None, new_data=False):
    """Following the EDA, this variable has nothing to process. 
    There is no process for this variable """     
    if new_data==False:
        null_or_empty_values(var_train, var_val, 'FLAG_PROFESSIONAL_PHONE')
        return var_train, var_val
    else:
        null_or_empty_values(var_new, 'FLAG_PROFESSIONAL_PHONE')
        return var_new


def null_or_empty_values(df_train, df_val, varName, df_new=None, new_data=False):
    if new_data==False:
        df_tr_str =  df_train.astype(str).map(lambda x: x.strip())
        null_values_tr = df_train.isnull()
        empty_values_tr = df_tr_str == ''
        
        ne_train = null_values_tr | empty_values_tr
        
        df_val_str = df_val.astype(str).map(lambda x: x.strip())
        null_values = df_val.isnull()
        empty_values = df_val_str == ''
        
        ne_val = null_values | empty_values
        
        print(f"\n\nVariable: {varName} \n")
        
        if ne_train.any().any():
            print("Null or empty values in TRAIN. It's time to clean")
        else:
            print("Train OK. Continue!")
        if ne_val.any().any():
            print("Null or empty values in VAL. It's time to clean")
        else:
            print("Val OK. Continue!")
    else:
        df_str = df_new.astype(str).map(lambda x: x.strip())
        null_values = df_new.isnull()
        empty_values = df_str == ''
        
        ne_new = null_values | empty_values
        
        print(f"\n\nVariable: {varName} \n")
        
        if ne_new.any().any():
            print("Null or empty values in NEW. It's time to clean")
        else:
            print("New OK. Continue!")
    
    return


def create_balanced_imputations(n):
    half = n // 2
    remainder = n % 2
    imputations = ['M'] * half + ['F'] * half + (['M'] if remainder else [])
    np.random.shuffle(imputations)
    return imputations


def clean_all_binary(in_X_train, in_X_val, in_X_new, new_data=False):
    """Makes the processing of the binary (categorical) variables.
    The dropped variables directly are not considered.
    Args: 
        in_X_train: PATH to X_train datast
        in_X_val:   PATH to X_val datast
    """
    if new_data==False:
        X_train, X_val = start(in_X_train, in_X_val)

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
        
        encoded_X_train.to_csv('data/interim/X_train_binary.csv', index=False)
        encoded_X_val.to_csv('data/interim/X_val_binary.csv', index=False)
        
        print("\n\n--- End of the categorical binary variables processing ---\n\n")    
        return encoded_X_train, encoded_X_val
    else:
        X = start(path_new=in_X_new, new_data=True)

        cleaned_X_APPLICATION_SUBMISSION_TYPE = \
            process_APPLICATION_SUBMISSION_TYPE(var_new=X['APPLICATION_SUBMISSION_TYPE'], new_data=True)
        cleaned_X_SEX = \
            process_SEX(var_new=X['SEX'], new_data=True)

        cleaned_X_FLAG_RESIDENCIAL_PHONE = \
            process_FLAG_RESIDENCIAL_PHONE(var_new=X['FLAG_RESIDENCIAL_PHONE'], new_data=True)

        cleaned_X_FLAG_EMAIL = \
            process_FLAG_EMAIL(var_new=X['FLAG_EMAIL'], new_data=True)

        cleaned_X_FLAG_VISA = \
            process_FLAG_VISA(var_new=X['FLAG_VISA'], new_data=True)

        cleaned_X_FLAG_MASTERCARD = \
            process_FLAG_MASTERCARD(var_new=X['FLAG_MASTERCARD'], new_data=True)

        cleaned_X_FLAG_OTHER_CARDS = \
            process_FLAG_OTHER_CARDS(var_new=X['FLAG_OTHER_CARDS'], new_data=True)

        cleaned_X_FLAG_PROFESSIONAL_PHONE = \
            process_FLAG_PROFESSIONAL_PHONE(var_new=X['FLAG_PROFESSIONAL_PHONE'], new_data=True)

        clean_X = pd.concat([cleaned_X_APPLICATION_SUBMISSION_TYPE,
                              cleaned_X_SEX,
                              cleaned_X_FLAG_RESIDENCIAL_PHONE,
                              cleaned_X_FLAG_EMAIL,
                              cleaned_X_FLAG_VISA,
                              cleaned_X_FLAG_MASTERCARD,
                              cleaned_X_FLAG_OTHER_CARDS,
                              cleaned_X_FLAG_PROFESSIONAL_PHONE
                                ], axis=1)
        

        
        ordinalEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ordinalEncoder.fit(clean_X)
        
        encoded_X = pd.DataFrame(
            ordinalEncoder.transform(clean_X),
            columns=clean_X.columns,
            index=clean_X.index)

        print("\n\n--- End of the categorical binary variables processing ---\n\n")
        return encoded_X

def main(
    path_train="./data/interim/X_train.csv",
    path_val="./data/data_splitted/X_val.csv",
    path_new=None,
    new_data=False
):
    """Main function to process the binary categorical variables.
    Args:
        path_train: PATH to X_train dataset
        path_val:   PATH to X_val dataset
        path_new:   PATH to new dataset (if new_data is True)
        new_data:   boolean indicating if the data is new or not
    Returns:
        encoded_X_train: DataFrame with processed X_train
        encoded_X_val:   DataFrame with processed X_val
    """
    clean_all_binary(path_train, path_val, path_new, new_data)

if __name__ == "__main__":
    main()



