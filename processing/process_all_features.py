import pandas as pd
from process_categorical_binary import clean_all_binary
from process_categorical_multicategorical import clean_all_multi
from process_numerical_continuous import process_numerical_continuous
from process_numerical_discrete import process_numerical_discrete


def process_all(in_train, in_val, X_train_output, X_val_output):
        
    path_train = in_train 
    path_val =   in_val
    X_train_in = pd.read_csv(path_train, header=0, encoding='UTF-8')
    X_val_in   = pd.read_csv(path_val,   header=0, encoding='UTF-8')
    
    train_binary,   val_binary = clean_all_binary(path_train, path_val)
    train_multi,     val_multi = clean_all_multi(path_train, path_val)    
    train_continous, val_continous = process_numerical_continuous(X_train_in, X_val_in, use_power=False)
    train_discrete = process_numerical_discrete(path_train)
    val_discrete   = process_numerical_discrete(path_val)
    
    X_train_out = pd.concat([train_binary, 
                             train_multi, 
                             train_continous,
                             train_discrete], axis=1)

    X_val_out  =  pd.concat([val_binary, 
                             val_multi, 
                             val_continous,
                             val_discrete], axis=1)
    
    X_train_out.to_csv(X_train_output, index=False)
    X_val_out.to_csv(X_val_output, index=False)   
    return


if __name__ == "__main__":
    process_all("./data/data_splitted/X_train.csv", "./data/data_splitted/X_val.csv",
                "./data/processed/X_train_p.csv",  "./data/processed/data_splitted/X_val_p.csv")