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
    return categorical_vars

def categorical_distribution(categorical_vars):

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

def correlation(categorical_vars, df_sup):
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