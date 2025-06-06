import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np

def start(head_number):
    df_sup = pd.read_csv('./external/PAKDD2010_Modeling_Data.txt', header=None, delimiter='\t', encoding='latin1')
    var_names = pd.read_csv('./external/var_names.txt', header=None, delimiter='\t', encoding='latin1')
    print(f"The following is an Overview of the first {head_number} rows")
    display(df_sup.head(head_number))
    
    return df_sup, var_names
    
def get_categorical_features(df, var_names):
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()    
    print(f"Identified categorical variables indexes: {categorical_vars}")
    print(f"Number of categorical variables: {len(categorical_vars)}\n")    
    display([var_names[0][i] for i in categorical_vars])
    
    return categorical_vars

def categorical_distribution(categorical_vars, var_names, df_sup):

    for col in categorical_vars:
        print("\n////////////////////////////////////////////\n")
        print(f"--- Variable distribution: {var_names[0][col]} ---\n")
        print(f"Number of unique categories: {df_sup[col].nunique()}")
        print("Category frequency:")
        
        freq = df_sup[col].value_counts(dropna=False)
        percent = round((freq / len(df_sup)) * 100, 2)
        freq_percent = pd.DataFrame({'Frequency': freq, 'Percentage (%)': percent})
        freq_percent.index.name = None
        print(freq_percent)    
        print("\n")

        # Visualization
        plt.figure(figsize=(8, 4))
        value_counts = freq.head(10)  # Top 10
        ax = sns.barplot(x=value_counts.values, y=value_counts.index)

        # Add tags to the bars (count, percentage %)
        for i, (count, pct) in enumerate(zip(value_counts.values, percent.head(10))):
            label = f"{count} ({pct}%)"
            ax.text(count + max(value_counts.values)*0.01, i, label, va='center', fontsize=9)

        # Barplot
        #sns.barplot(x=value_counts.values, y=value_counts.index)
        plt.title(f"{var_names[0][col]} Distribution (Top 10)")
        plt.xlabel("Frequency")
        plt.ylabel("Category")
        plt.tight_layout()
        plt.show()

def correlation(categorical_vars, df_sup, var_names):
    results = []

    for col in categorical_vars:
        
        contingency_table = pd.crosstab(df_sup[col], df_sup[53])    
        
        # Apply Chi-square
        try:
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            results.append({'Variable': var_names.iloc[col][0], 'p-value': p, 'Chi2': chi2, 'dof': dof})
        except ValueError as e:
            print(f"Error with the variable {var_names.iloc[col]}: {e}")

    # Show results, ordered by p-value
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="p-value", inplace=True)

    print("\nChi-square results:")
    display(results_df)

    # Filter those with lowest p-value
    print("\nVariables with significant relationship (p < 0.05):")
    significant_vars = results_df[results_df["p-value"] < 0.05]
    number_significant = significant_vars.shape[0]
    display(significant_vars)

    print(f"Number of significant categorical variables: {number_significant} of {len(categorical_vars)}")


    non_significant = results_df[results_df["p-value"] >= 0.05]

    print(f"\nNumber of categorical variables non significantly associated with the target variable: {non_significant.shape[0]}")
    display(non_significant)
    
def ouliers_batch(batch, frecuency_threshold, categorical_vars, var_names, df_sup):  
    # frecuency_threshold in X %
    
    print(f"///---  Categorical Variables Batch # {batch}:  ---///")
    from_cat = (batch-1)*4
    to_cat = batch*4
    batch_indexes = categorical_vars[from_cat:to_cat]
    d = var_names.iloc[batch_indexes]  
    d.columns = ["Variable"]
    print(batch_indexes)
    display(d)

    #----
    print(f"\n--- Detecting Outliers in Catgorical Variables. Batch # {batch}---")

    for col in batch_indexes:        
        print(f"\nVariable: {var_names[0][col]}")
        print(f"Number of unique categories: {df_sup[col].nunique()}\n")
        value_counts = df_sup[col].value_counts(normalize=False) # Count
        value_percentages = df_sup[col].value_counts(normalize=True) # Percent
        category_summary = pd.DataFrame({'Count': value_counts,'Proportion': value_percentages})
        
        
        # Identify Categories Under the treshold
        low_frequency_categories = category_summary[category_summary['Proportion'] < frecuency_threshold]
        low_frequency_categories.index.name = None    
        low_frequency_categories.columns.name = None    

        if not low_frequency_categories.empty:
            print(f"  Posible outlier categories (frecuency < {frecuency_threshold*100:.2f} %):\n")
            print(low_frequency_categories)
            print("-" * 50)
        else:
            print(f" No categories under the frequency treshold: {frecuency_threshold*100:.2f} %).")
            print("-" * 50)

    print("\n--- --- --- --- ---")

def missing_values(batch, categorical_vars, var_names, df_sup):
    
    print(f"///---  Categorical Variables Batch # {batch}:  ---///")
    from_cat = (batch-1)*4
    to_cat = batch*4
    batch_indexes = categorical_vars[from_cat:to_cat]
    d = var_names.iloc[batch_indexes]  
    d.columns = ["Variable"]
    print(batch_indexes)
    display(d)

    #----
    print(f"\n--- Detecting missing values in catgorical variables. Batch # {batch}---")
    print("--- Identifying number of Missing Values ---\n")
    
    df_b = df_sup.iloc[:, batch_indexes]
    variables_actual_names = [var_names[0].to_list()[idx] for idx in batch_indexes]

    # Number of missing values per the feature
    missing_values_count = df_b.isnull().sum()
    
    # Missing values percentage per column
    missing_values_percentage = (missing_values_count / len(df_b))   #* 100
    # DataFrame with the coount and percentage of missing values
    missing_info = pd.DataFrame({
        'Variable Name': variables_actual_names,
        'Missing Count': missing_values_count,
        'Missing Proportion': missing_values_percentage
    })

    # Filter only the columns that have at least one missing value
    missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

    if missing_info.empty:
        print(" >> Good!, there are no missing value in any column of this batch.")
    else:
        print(f"Some missing values found un the following columns:")
        print(missing_info)
        print("\nSummary:")        
        
    print(f"   >> Columns with missing values in this batch: {missing_info.index.tolist()}")
    print(f"\n--- Batch # {batch} completed for missing values ---\n")
    print("--- " * 15)
    
def cardinality(categorical_vars, var_names, df_sup): 

    #----
    print(f"\n--- Analizing cardinality of all catgorical variables ---")
    print("----- Identifying Cardinality -----\n")
    
    cardinality_info = []

    for col in categorical_vars:
        unique_count = df_sup[col].nunique(dropna=False)
        idx = df_sup.columns.get_loc(col)
        variable_name = var_names.iloc[idx,0]

        if unique_count < 10:
            cardinality = '(.) Low'
        elif 10 <= unique_count <= 50:
            cardinality = '(..) Mid'
        else:
            cardinality = '(...) High'
        if unique_count == 1:
            cardinal_type = 'Single-category'
        elif unique_count == 2:
            cardinal_type = 'Binary'
        else:
            cardinal_type = 'Multicategorical'

        cardinality_info.append({
            'Variable Index': col,
            'Variable Name': variable_name,
            'Unique Categories': unique_count,
            'Cardinality Level': cardinality,
            'Category Type': cardinal_type
        })

    # Mostrar resultados ordenados por cantidad de categorías
    cardinality_df = pd.DataFrame(cardinality_info).sort_values(by='Unique Categories', ascending=False)
    cardinality_df_sorted_by_index = cardinality_df.sort_values(by="Variable Index", ascending=True)

    cardinality_df.index.name = None
    print("Cardinality analysis of categorical variables:")
    display(cardinality_df)
    print("--- "*25)
    print("\nCardinality analysis of categorical variables, sorted by index:")
    display(cardinality_df_sorted_by_index)
    
def cities(size, df_sup):
    unique_values = np.sort(df_sup.iloc[:, 11].dropna().unique())
    random_values = np.random.choice(unique_values, size=size, replace=False)
    
    
    for valor in random_values:
        print(valor)