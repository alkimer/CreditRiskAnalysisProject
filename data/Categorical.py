import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def start(head_number):
    df_sup = pd.read_csv('../external/PAKDD2010_Modeling_Data.txt', header=None, delimiter='\t', encoding='latin1')
    print(f"The following is an Overview of the first {head_number} rows")
    display(df_sup.head(head_number))
    
    return df_sup
    
def get_categorical_features(df):
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()    
    print(f"Identified categorical variables: {categorical_vars}\n")
    var_names = pd.read_csv('../external/var_names.txt', header=None, delimiter='\t', encoding='latin1')
    [var_names[0][i] for i in categorical_vars]
    
    
    return categorical_features

    