import pandas as pd
from process_categorical_binary import clean_all_binary
from process_categorical_multicategorical import clean_all_multi
from process_numerical_continuous import process_numerical_continuous_split
from process_numerical_discrete import process_numerical_discrete


def process_all(path_train, path_val, X_train_output, X_val_output):
        
    path_y_train = "./data/data_splitted/y_train.csv"
    path_y_val   = "./data/data_splitted/y_val.csv"
    
    path_Xtr = "./data/processed/interim/X_tr.csv"
    path_Xv  = "./data/processed/interim/X_v.csv"
        
    X_train_in = pd.read_csv(path_train, header=0, encoding='UTF-8')
    X_val_in   = pd.read_csv(path_val,   header=0, encoding='UTF-8')
    y_train_in = pd.read_csv(path_y_train,   header=0, encoding='UTF-8')
    y_val_in   = pd.read_csv(path_y_val,   header=0, encoding='UTF-8')
    
    Xtr = pd.concat([X_train_in, y_train_in], axis=1)
    Xv  = pd.concat([X_val_in, y_val_in], axis=1)
    
    Xtr.to_csv(path_Xtr, index=False)
    Xv.to_csv (path_Xv , index=False)
    
    
    train_binary,   val_binary = clean_all_binary(path_train, path_val)
    train_multi,     val_multi =  clean_all_multi(path_train, path_val)    
    train_continuous, val_continuous = process_numerical_continuous_split(X_train_in, X_val_in, use_power=False)
    train_discrete = process_numerical_discrete(path_Xtr, encode=True, binning=True, normalize=True)
    val_discrete   = process_numerical_discrete(path_Xv, encode=True, binning=True, normalize=True)
    
    train_continuous.to_csv("./data/processed/interim/X_train_continuous.csv", index=False)
    val_continuous.to_csv("./data/processed/interim/X_val_continuous.csv", index=False)   
    train_discrete.to_csv("./data/processed/interim/X_train_discrete.csv", index=False)   
    val_discrete.to_csv  ("./data/processed/interim/X_val_discrete.csv", index=False)   
    
    X_train_out = pd.concat([train_binary, 
                             train_multi, 
                             train_continuous,
                             train_discrete], axis=1)

    X_val_out  =  pd.concat([val_binary, 
                             val_multi, 
                             val_continuous,
                             val_discrete], axis=1)
    
    X_train_out.to_csv(X_train_output, index=False)
    X_val_out.to_csv(X_val_output, index=False)   
    
    show_dropped("train", X_train_in, X_train_out)
    return

def show_dropped(df_in, df_out):
    dropped = []
    for i in df_in.columns:
        if i not in df_out:
            dropped.append(i)
    
    pd.DataFrame(dropped).to_csv(f"./data/processed/dropped_variables.txt", index=False, columns=False, sep='\t')

    return


if __name__ == "__main__":
    process_all("./data/data_splitted/X_train.csv", "./data/data_splitted/X_val.csv",
                "./data/processed/X_train_p.csv",  "./data/processed/X_val_p.csv")