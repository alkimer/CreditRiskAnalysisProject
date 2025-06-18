import pandas as pd
from process_categorical_binary import clean_all_binary
from process_categorical_multicategorical import clean_all_multi
from process_numerical_continuous import process_numerical_continuous_split
from process_numerical_discrete import process_numerical_discrete
from imblearn.over_sampling import RandomOverSampler


def process_all(path_train, path_val):
    """ 
    This function takes the raw splitted dataframes and process the variables
    according to their nature (binary, multicategorical, continous, discrerte) using
    the 4 scripts dedicated to each group, and outputs the processed dataFrames,
    delivering the data ready to use in the pipeline.
    
    Args:
        path_train: a string with the path to the raw splitted X_train
        path_val:   a string with the path to the raw splitted X_val
        X_train_output: a string with the path to the processed output X_train_p
        X_val_output:   a string with the path to the processed output X_train_p
        
    output: 
        (in ./data/processed/interim)
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
    val_discrete   = process_numerical_discrete(path_Xv,  encode=True, binning=True, normalize=True)
    
    #train_continuous.to_csv("./data/processed/interim/X_train_continuous.csv", index=False)
    #val_continuous.to_csv("./data/processed/interim/X_val_continuous.csv", index=False)   
    #train_discrete.to_csv("./data/processed/interim/X_train_discrete.csv", index=False)   
    #val_discrete.to_csv  ("./data/processed/interim/X_val_discrete.csv", index=False)   
    
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
    
    
    print("\n\n--- END OF THE WHOLE VARIABLES PROCESSING ---")
    print("\n\n\nBinary Variables Kept//\n")
    print(binary_cols.to_string(index=False, header=False))
    print("\n\nMulticategorical Variables Kept//\n")
    print(multi_cols.to_string(index=False, header=False))
    print("\n\nContinuous Variables Kept//\n")
    print(conti_cols.to_string(index=False, header=False))
    print("\n\nDiscrete Variables Kept//\n")
    print(discr_cols.to_string(index=False, header=False))
    
    processed_col_names = pd.concat([binary_cols, multi_cols, conti_cols, discr_cols], axis=0)

    dropped = []    
    for i in X_train_in.columns:
        if i not in processed_col_names[0].tolist():
            dropped.append(i)           
            
    print(f"\nOriginal # of columns: {len(X_train_in.columns)}")
    print(f"Number of processed VARIABLES: {len(processed_col_names)}")
    print(f"Number of dropped columns: {len(dropped)}")
    print(f"Number of output columns: {len(X_train_out.columns)}")

    
    pd.DataFrame(processed_col_names).to_csv(f"./data/processed/processed_variables_list.txt", index=False, header=False, sep='\t')
    pd.DataFrame(dropped).to_csv(f"./data/processed/dropped_variables_list.txt", index=False, header=False, sep='\t')
    
    print(f"\n//  SHAPES OF THE OUTPUT DATASETS  //\n")
    print("(X_train_p), (X_val_p)")
    print(f"\n{X_train_out.shape}, {X_val_out.shape}\n")
    print("Binary, Multicategorical, Continuous, Discrete\nTrain | Val")
    print(train_binary.shape, val_binary.shape)
    print(train_multi.shape,  val_multi.shape)
    print(train_continuous.shape, val_continuous.shape)
    print(train_discrete.shape, val_discrete.shape)
    
            
    
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
    

def final_processing(path_train, path_val, X_train_output, y_train_output, X_val_output, y_val_output, smote=True):
    
    X_train_out, X_val_out, y_train, y_val = process_all(path_train, path_val)
    
    X_val_out.to_csv(X_val_output, index=False)
    y_val.to_csv(y_val_output, index=False)
    
        # Export X_train_out, X_val_out as csv files
    # X_train_out.to_csv("./data/processed/interim/X_train_X_train_unbalanced.csv", index=False)
    # X_val_out.to_csv("./data/processed/interim/X_val_X_val_unbalanced.csv", index=False)
    
    if smote:        
        
        #xtrain_balanced, ytrain_balanced = apply_smotenc(X_train_out, y_train)
        #xtrain_balanced, ytrain_balanced = smotenc(X_train_out, y_train, categorical_cols)
        xtrain_balanced, ytrain_balanced = randomOverSample(X_train_out, y_train)
        
        xtrain_balanced.to_csv(f"./data/processed/X_train_balanced.csv", index=False)
        ytrain_balanced.to_csv(f"./data/processed/y_train_balanced.csv", index=False)
        
        return xtrain_balanced, ytrain_balanced

    X_train_out.to_csv(X_train_output, index=False)    
    y_train.to_csv(y_train_output, index=False)
    
    
    return


if __name__ == "__main__":
    final_processing("./data/data_splitted/X_train.csv", "./data/data_splitted/X_val.csv",
                "./data/processed/X_train_p.csv",  "./data/processed/y_train_p.csv",
                "./data/processed/X_val_p.csv", "./data/processed/y_val.csv", smote=True)
