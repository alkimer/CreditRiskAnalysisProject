import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np

def start():
    X_train_full = pd.read_csv('./data/data_splitted/X_train.csv', header=None, encoding='UTF-8')
    X_val_full   = pd.read_csv('./data/data_splitted/X_val.csv',   header=None, encoding='UTF-8')
    y_train_full = pd.read_csv('./data/data_splitted/y_train.csv', header=None, encoding='UTF-8')
    y_val_full   = pd.read_csv('./data/data_splitted/y_val.csv',   header=None, encoding='UTF-8') #delimiter=','
    
    var_names = pd.read_csv('./external/var_names.txt', header=None, encoding='UTF-8') # delimiter=','
    binary_index = [6,16,33,37] # + Others, review twice or more!!!!
    
    print("The following are the binary categrorical variables \nin the dataset (first their indexes):")
    display(binary_index
            )
    # Create binary subdatasets
    X_train = X_train_full[binary_index]
    X_val = X_val_full[binary_index]
    y_train = y_train_full[binary_index]
    y_val = y_val_full[binary_index]