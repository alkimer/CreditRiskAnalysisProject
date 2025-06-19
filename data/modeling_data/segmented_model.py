import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import mode
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


def segmentation(km_clusters, dt_leafs, path_X_train, path_y_train):

    try:
        X_train = pd.read_csv(path_X_train)
        y_train = pd.read_csv(path_y_train)
    except FileNotFoundError:
        print("Error: Make sure about the paths of all CSV files (X_train, y_train.csv,).")
        exit()
        
        
    df_kmeans_segmented = kmeans_segmentation(km_clusters, X_train, y_train)
    visualize('kmeans_segment', X_train, df_kmeans_segmented, km_clusters) 
    
    df_dt_segmented = decision_trees_segmentation(dt_leafs, X_train, y_train)
    visualize('dt_segment', X_train, df_dt_segmented, dt_leafs) 
        
    return df_kmeans_segmented, df_dt_segmented


def kmeans_segmentation(n_clusters_kmeans, X_train, y_train):
    
    print("\n--- K-Means Segmentation ---\n")


    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10) # n_init to suppress warning
    X_train['kmeans_segment'] = kmeans.fit_predict(X_train)
    
    kmeans_segmented_counts = X_train['kmeans_segment'].value_counts()

    print(f"K-Means segments distribution (N = {n_clusters_kmeans}):")
    print()
    
    # Merge X_train, y_train
    df_kmeans_segmented = X_train.copy()
    df_kmeans_segmented['target'] = y_train.iloc[:, 0] 
    
    
    print("\nTarget variable distribution within K-Means segments:")
    for segment_id in sorted(df_kmeans_segmented['kmeans_segment'].unique()):
        
        segment_data = df_kmeans_segmented[df_kmeans_segmented['kmeans_segment'] == segment_id]
        target_counts = segment_data['target'].value_counts(normalize=True)
        print(f"Segment {segment_id}:")
        print(target_counts)
        print("-" * 20)
    
    return df_kmeans_segmented
    
def decision_trees_segmentation(n_max_leaf_nodes, X_train, y_train):
    
    print("\n--- Decision Tree Segmentation ---")
    
    dt_segmenter = DecisionTreeClassifier(random_state=42, max_leaf_nodes=n_max_leaf_nodes)
    dt_segmenter.fit(X_train.drop('kmeans_segment', axis=1, errors='ignore'), y_train.iloc[:, 0]) # Ensure no kmeans_segment column
    
    X_train['dt_segment'] = dt_segmenter.apply(X_train.drop('kmeans_segment', axis=1, errors='ignore'))
        
    print(f"\nDecision Tree segments distribution (Max leaf nodes aimed: {n_max_leaf_nodes}):")
    print(X_train['dt_segment'].value_counts())    
    
    # Check homogeneity within Decision Tree segments using the target variable (y_train)
    df_dt_segmented = X_train.copy()
    df_dt_segmented['target'] = y_train.iloc[:, 0]
    
    print("\nTarget variable distribution within Decision Tree segments:")
    for segment_id in sorted(df_dt_segmented['dt_segment'].unique()):
        
        segment_data = df_dt_segmented[df_dt_segmented['dt_segment'] == segment_id]
        target_counts = segment_data['target'].value_counts(normalize=True)
        print(f"Segment {segment_id}:")
        print(target_counts)
        print("-" * 20)
        
    return df_dt_segmented
        
def visualize(model, X_train, dataframe, n_segments):
    
    # Visualize K-Means/ Decision Tree segment balance (by segment size)
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=model, data=X_train)    # 'kmeans_segment'  #'dt_segment'
    plt.title(f'{model} Sizes (N={n_segments})') #n_clusters_kmeans     #n_min_leaf_nodes
    plt.xlabel('Segment ID')
    plt.ylabel('Number of Samples')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 100, f'{int(height)}', ha='center')
    
    plt.show()

    # Visualize target balance within the segments
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x= model, hue='target', data=dataframe) # df_kmeans_segmented    #df_dt_segmented
    plt.title(f'Target Distribution within {model}s (N={n_segments})') #n_clusters_kmeans
    plt.xlabel('Segment ID')
    plt.ylabel('Number of Samples')
    plt.legend(title='Target')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 100, f'{int(height)}', ha='center')
    
    plt.show()
    
    return

def logistic_regression():
    
    warnings.filterwarnings('ignore')
    
    try:
        X_train_balanced = pd.read_csv('./data/processed/X_train_balanced.csv')
        X_val_p = pd.read_csv('./data/processed/X_val_p.csv')
        y_train_balanced = pd.read_csv('./data/processed/y_train_balanced.csv')
        y_val = pd.read_csv('./data/processed/y_val.csv')
    
    except FileNotFoundError:
        print("Error: Make sure all CSV files (X_train_p.csv, X_val_p.csv, y_train.csv, y_val.csv) are in the same directory.")
        exit()

    print("\n" + "="*50)
    print("Logistic Regression per K-Means Segment")
    print("="*50)
    
    kmeans_models = {}
    kmeans_performance = {}


    # -----


    unique_kmeans_segments_train = X_train_balanced['kmeans_segment'].unique()
    unique_kmeans_segments_val = X_val_p['kmeans_segment'].unique()

    all_predictions_kmeans = pd.Series(index=y_val.index, dtype=int)
    all_true_labels_kmeans = pd.Series(index=y_val.index, dtype=int)
  
    for segment_id in sorted(unique_kmeans_segments_train):
        print(f"\n--- Training and Evaluating for K-Means Segment {segment_id} ---")

        # Filter data for the current segment (training)
        segment_X_train = X_train_balanced[X_train_balanced['kmeans_segment'] == segment_id].drop(['kmeans_segment', 'dt_segment'], axis=1)
        segment_y_train = y_train_balanced[X_train_balanced['kmeans_segment'] == segment_id]

        # Filter data for the current segment (validation)
        segment_X_val = X_val_p[X_val_p['kmeans_segment'] == segment_id].drop(['kmeans_segment', 'dt_segment'], axis=1)
        segment_y_val = y_val[X_val_p['kmeans_segment'] == segment_id]

        if len(segment_X_train) < 2 or len(segment_X_val) < 2: # Need at least 2 samples for LR
            print(f"  Segment {segment_id} has too few samples for training/validation. Skipping.")
            print(f"  Train samples: {len(segment_X_train)}, Validation samples: {len(segment_X_val)}")
            continue

    return


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


def frecuency_by_segment(df_segmented):
    
    # Ensure the segment column exists
    if 'kmeans_segment' not in df_segmented.columns:
        raise ValueError("The DataFrame must contain a 'kmeans_segment' column.")
    
    segments = sorted(df_segmented['kmeans_segment'].unique())
    
    for segment_label in segments:
        print(f"\n=== Frequency Summary for Segment '{segment_label}' ===")
        
        # Filter the dataframe for the current segment
        segment_data = df_segmented[df_segmented['kmeans_segment'] == segment_label]
        
        # Loop through each column except the segment column
        for col in df_segmented.columns:
            if col == 'kmeans_segment':
                continue
            
            print(f"\n-- Column: {col} --")
            freq_table = segment_data[col].value_counts(dropna=False)
            print(freq_table)
    return

def correlation_by_segment(df_segmented):
    
    # Ensure the segment column exists
    if 'kmeans_segment' not in df_segmented.columns:
        raise ValueError("The DataFrame must contain a 'kmeans_segment' column.")
    
    # Identify numeric columns for statistics and plotting (excluding the segment column)
    # numeric_cols = df_segmented.select_dtypes(include='number').drop(columns=['kmeans_segment'], errors='ignore').columns

    segments = sorted(df_segmented['kmeans_segment'].unique())
    
    for segment_label in segments:
        
        segment_data = df_segmented[df_segmented['kmeans_segment'] == segment_label]
        # Correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(segment_data.corr(), annot=False, cmap='coolwarm', fmt=".2f")        
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8) 
        plt.title(f"Correlation Heatmap - Segment '{segment_label}'")
        plt.tight_layout()
        plt.show()
    return

def elbow_method():
    


    inertias = []
    K = range(1, 8)  # puedes ajustar este rango

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K, inertias, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(True)
    plt.show()
    
    return

def calculate_segment_homogeneity(df, segment_col='dt_segment'):
    """
    Calcula la homogeneidad de los segmentos en un DataFrame.
    La homogeneidad se mide por la varianza para columnas numéricas
    y por la proporción de la moda para columnas categóricas.

    Args:
        df (pd.DataFrame): El DataFrame con los datos y la columna de segmento.
        segment_col (str): El nombre de la columna que indica el segmento.

    Returns:
        pd.DataFrame: Un DataFrame con la homogeneidad promedio para cada segmento
                      y la homogeneidad global del dataset.
    """
    if segment_col not in df.columns:
        raise ValueError(f"La columna de segmento '{segment_col}' no se encuentra en el DataFrame.")

    homogeneity_results = {}
    numerical_cols = df.select_dtypes(include=np.number).columns.drop(segment_col, errors='ignore')
    categorical_cols = df.select_dtypes(exclude=np.number).columns.drop(segment_col, errors='ignore')

    # Calcula la homogeneidad para cada segmento
    for segment_id in df[segment_col].unique():
        segment_df = df[df[segment_col] == segment_id]
        segment_homogeneity = {}

        # Homogeneidad para columnas numéricas (usando varianza)
        if not numerical_cols.empty:
            # Reemplazar NaN con la media del segmento para evitar que la varianza sea NaN
            for col in numerical_cols:
                if segment_df[col].isnull().any():
                    segment_df.loc[:, col] = segment_df[col].fillna(segment_df[col].mean())
            segment_homogeneity['numerical_variance_avg'] = segment_df[numerical_cols].var().mean()

        # Homogeneidad para columnas categóricas (usando proporción de la moda)
        if not categorical_cols.empty:
            mode_proportions = []
            for col in categorical_cols:
                if not segment_df[col].empty:
                    # Calcula la moda y su frecuencia
                    current_mode, current_count = mode(segment_df[col].dropna())
                    if len(current_mode) > 0: # Si se encuentra una moda
                        mode_proportion = current_count[0] / len(segment_df[col].dropna()) if len(segment_df[col].dropna()) > 0 else 0
                        mode_proportions.append(mode_proportion)
            if mode_proportions:
                segment_homogeneity['categorical_mode_proportion_avg'] = np.mean(mode_proportions)
            else:
                segment_homogeneity['categorical_mode_proportion_avg'] = np.nan # No hay columnas categóricas o están vacías

        homogeneity_results[f'Segment_{segment_id}'] = segment_homogeneity

    # Calcula la homogeneidad global del dataset (promedio ponderado por el tamaño del segmento)
    global_homogeneity = {'numerical_variance_global_avg': 0, 'categorical_mode_proportion_global_avg': 0}
    total_rows = len(df)
    
    for segment_id in df[segment_col].unique():
        segment_df = df[df[segment_col] == segment_id]
        segment_size = len(segment_df)
        weight = segment_size / total_rows
        
        segment_result = homogeneity_results[f'Segment_{segment_id}']
        
        if 'numerical_variance_avg' in segment_result and not pd.isna(segment_result['numerical_variance_avg']):
            global_homogeneity['numerical_variance_global_avg'] += segment_result['numerical_variance_avg'] * weight
        
        if 'categorical_mode_proportion_avg' in segment_result and not pd.isna(segment_result['categorical_mode_proportion_avg']):
            global_homogeneity['categorical_mode_proportion_global_avg'] += segment_result['categorical_mode_proportion_avg'] * weight

    results_df = pd.DataFrame(homogeneity_results).T
    
    # Agrega la fila de homogeneidad global
    global_homogeneity_series = pd.Series(global_homogeneity, name='Global_Homogeneity')
    results_df = pd.concat([results_df, pd.DataFrame([global_homogeneity_series])])

    return results_df




if __name__ == '__main__':
    segmentation(3, 3, 
                 './data/processed/X_train_balanced.csv', 
                 './data/processed/y_train_balanced.csv')