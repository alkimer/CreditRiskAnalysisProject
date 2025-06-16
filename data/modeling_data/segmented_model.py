import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def segmentation(km_clusters, dt_leafs, path_X_train_p, path_y_train):

    try:
        X_train_p = pd.read_csv(path_X_train_p)
        y_train = pd.read_csv(path_y_train)
    except FileNotFoundError:
        print("Error: Make sure about the paths of all CSV files (X_train_p.csv, X_val_p.csv, y_train.csv, y_val.csv).")
        exit()
        
        
    df_kmeans_segmented = kmeans_segmentation(km_clusters, X_train_p, y_train)
    visualize('kmeans_segment', X_train_p, df_kmeans_segmented, km_clusters) 
    
    df_dt_segmented = decision_trees_segmentation(dt_leafs, X_train_p, y_train)
    visualize('dt_segment', X_train_p, df_dt_segmented, dt_leafs) 
        
    return df_kmeans_segmented, df_dt_segmented


def kmeans_segmentation(n_clusters_kmeans, X_train_p, y_train):
    
    print("\n--- K-Means Segmentation ---\n")

    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10) # n_init to suppress warning
    X_train_p['kmeans_segment'] = kmeans.fit_predict(X_train_p)

    print(f"K-Means segments distribution (N = {n_clusters_kmeans}):")
    print(X_train_p['kmeans_segment'].value_counts())
    
    # Merge X_train, y_train
    df_kmeans_segmented = X_train_p.copy()
    df_kmeans_segmented['target'] = y_train.iloc[:, 0] 
    
    print("\nTarget variable distribution within K-Means segments:")
    for segment_id in sorted(df_kmeans_segmented['kmeans_segment'].unique()):
        
        segment_data = df_kmeans_segmented[df_kmeans_segmented['kmeans_segment'] == segment_id]
        target_counts = segment_data['target'].value_counts(normalize=True)
        print(f"Segment {segment_id}:")
        print(target_counts)
        print("-" * 20)
    
    return df_kmeans_segmented
    
def decision_trees_segmentation(n_min_leaf_nodes, X_train_p, y_train):
    
    print("\n--- Decision Tree Segmentation ---")
    
    dt_segmenter = DecisionTreeClassifier(random_state=42, max_leaf_nodes=n_min_leaf_nodes * 2)
    dt_segmenter.fit(X_train_p.drop('kmeans_segment', axis=1, errors='ignore'), y_train.iloc[:, 0]) # Ensure no kmeans_segment column
    
    X_train_p['dt_segment'] = dt_segmenter.apply(X_train_p.drop('kmeans_segment', axis=1, errors='ignore'))
        
    print(f"\nDecision Tree segments distribution (Min leaf nodes aimed: {n_min_leaf_nodes}):")
    print(X_train_p['dt_segment'].value_counts())    
    
    # Check homogeneity within Decision Tree segments using the target variable (y_train)
    df_dt_segmented = X_train_p.copy()
    df_dt_segmented['target'] = y_train.iloc[:, 0]
    
    print("\nTarget variable distribution within Decision Tree segments:")
    for segment_id in sorted(df_dt_segmented['dt_segment'].unique()):
        
        segment_data = df_dt_segmented[df_dt_segmented['dt_segment'] == segment_id]
        target_counts = segment_data['target'].value_counts(normalize=True)
        print(f"Segment {segment_id}:")
        print(target_counts)
        print("-" * 20)
        
    return df_dt_segmented
        
def visualize(model, X_train_p, dataframe, n_segments):
    
    # Visualize K-Means/ Decision Tree segment balance (by segment size)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=model, data=X_train_p)    # 'kmeans_segment'  #'dt_segment'
    plt.title(f'{model} Sizes (N={n_segments})') #n_clusters_kmeans     #n_min_leaf_nodes
    plt.xlabel('Segment ID')
    plt.ylabel('Number of Samples')
    plt.show()

    # Visualize target balance within the segments segments
    plt.figure(figsize=(10, 6))
    sns.countplot(x= model, hue='target', data=dataframe) # df_kmeans_segmented    #df_dt_segmented
    plt.title(f'Target Distribution within {model}s (N={n_segments})') #n_clusters_kmeans
    plt.xlabel('Segment ID')
    plt.ylabel('Number of Samples')
    plt.legend(title='Target')
    plt.show()
    
    return

def logistic_regression():
    
    y_train = pd.read_csv('/data/data_splitted/y_train.csv').iloc[:, 0]
    y_val = pd.read_csv('/data/data_splitted/y_val.csv').iloc[:, 0]    
    
    

    return


if __name__ == '__main__':
    segmentation(3, 3, 
                 './data/processed/X_train_p.csv', 
                 './data/data_splitted/y_train.csv')