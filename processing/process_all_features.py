import pandas as pd
from process_categorical_binary import clean_all_binary
from process_categorical_multicategorical import clean_all_multi
from process_numerical_continuous import process_numerical_continuous
from process_numerical_discrete import process_numerical_discrete


def process_all(in_train, in_val):
        
    path_train = in_train 
    path_val =   in_val
    X_train_in = pd.read_csv(path_train, header=0, encoding='UTF-8')
    X_val_in   = pd.read_csv(path_val,   header=0, encoding='UTF-8')
    
    clean_all_binary(in_train, in_val,
                 "data/processed/clean_X_train_binary.csv", "data/processed/clean_X_val_binary.csv")    
    clean_all_multi()    
    train_cont, val_cont = process_numerical_continuous(X_train_in, X_val_in, use_power=False)
    train_discr = process_numerical_discrete(path_train)
    val_discr   = process_numerical_discrete(path_val)
    
    



if __name__ == "__main__":
    process_all("./data/data_splitted/X_train.csv", "./data/data_splitted/X_val.csv")





-----------------



clean_X_train.to_csv(out_X_train, index=False)
clean_X_val.to_csv(out_X_val, index=False)   