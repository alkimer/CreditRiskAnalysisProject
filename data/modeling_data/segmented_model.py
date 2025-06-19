import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----
# First phase

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
    
    df_kmeans_segmented = X_train.copy()


    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10) # n_init to suppress warning
    df_kmeans_segmented['kmeans_segment'] = kmeans.fit_predict(X_train)
    
    #kmeans_segmented_counts = X_train['kmeans_segment'].value_counts()

    print(f"K-Means segments distribution (N = {n_clusters_kmeans}):")
    print()
    
    # Merge X_train, y_train
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
    
    df_dt_segmented = X_train.drop('kmeans_segment',axis=1, errors = 'ignore')
    
    dt_segmenter = DecisionTreeClassifier(random_state=42, max_leaf_nodes=n_max_leaf_nodes)
    dt_segmenter.fit(df_dt_segmented, y_train.iloc[:, 0]) #.drop('kmeans_segment', axis=1, errors='ignore'), y_train.iloc[:, 0])
    
    df_dt_segmented['dt_segment'] = dt_segmenter.apply(df_dt_segmented) #.drop('kmeans_segment', axis=1, errors='ignore'))
        
    print(f"\nDecision Tree segments distribution (Max leaf nodes aimed: {n_max_leaf_nodes}):")
    print(df_dt_segmented['dt_segment'].value_counts())    
    
    # Check homogeneity within Decision Tree segments using the target variable (y_train)
    df_dt_segmented['target'] = y_train.iloc[:, 0]
    
    
    print("\nTarget variable distribution within Decision Tree segments:")
    for segment_id in sorted(df_dt_segmented['dt_segment'].unique()):
        
        segment_data = df_dt_segmented[df_dt_segmented['dt_segment'] == segment_id]
        target_counts = segment_data['target'].value_counts(normalize=True)
        print(f"Segment {segment_id}:")
        print(target_counts)
        print("-" * 20)
        
    return df_dt_segmented
        
def visualize(model, dataframe, n_segments):
    
    # Visualize K-Means/ Decision Tree segment balance (by segment size)
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=model, data=dataframe)    # 'kmeans_segment'  #'dt_segment'
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

# ----
# Second phase

def train_logistic_regression(df):
    
    segment_col = df.columns[-2]  
    
    # Set characteristics apart
    X_train = df.iloc[:, :-2]  # All columns but the last two
    y_train = df.iloc[:, -1]   # Last column is 'target' 
        
    # X_val = pd.read_csv('../processed/interim/X_val_X_val_unbalanced.csv')
    # y_val = pd.read_csv('../processed/y_val.csv')   
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train) 

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
    }

    return metrics

def second_phase(df):
    
    
    X_val = pd.read_csv('../processed/X_val_p.csv')
    y_val = pd.read_csv('../processed/y_val.csv')
    
    # Logistic regression
    lr_model = train_logistic_regression(df)
    
    metrics = evaluate_model(lr_model, X_val, y_val)
    print(metrics)
    print("\n")
    
    return

# ----
# Utils

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


if __name__ == '__main__':
    segmentation(3, 3, 
                 './data/processed/X_train_balanced.csv', 
                 './data/processed/y_train_balanced.csv')