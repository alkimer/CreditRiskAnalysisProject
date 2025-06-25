import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_auc_score)
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

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

def second_phase(df, X_val, y_val):
    
    segmented_dfs = segment_dataframes(df)
    
    metrics = []
    
    # Logistic regression + metrics
    for i in range(len(segmented_dfs)):
        print(f"// Segment # {i+1} //")
        lr_model = train_logistic_regression(segmented_dfs[i])
        metrics_current = evaluate_model(lr_model, X_val, y_val)
        metrics.append(metrics_current)
        print(metrics_current, "\n")
    
    return metrics


def train_logistic_regression(df):
    
    #segment_col = df.columns[-2]  
    
    # Set characteristics apart
    X_train = df.iloc[:, :-2]  # All columns but the last two
    y_train = df.iloc[:, -1]   # Last column is 'target' 
        

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train) 

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Predicciones de probabilidad (para ROC AUC)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva
    else:
        y_proba = None
    
    # Confusion matrix    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred),4),
        "precision":round(precision_score(y_test, y_pred),4),
        "recall":   round( recall_score(y_test, y_pred),4),
        "f1_score": round(f1_score(y_test, y_pred),4)
    }
    
    # ROC AUC if there are probabilities available
    if y_proba is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)

    return metrics


def segment_dataframes(dataframe):
    
    segment_column = dataframe.columns[-2]
    id_segments = dataframe[segment_column].unique()
    
    segmented_dfs = []
    for i in id_segments:
        segmented_dfs.append(dataframe[dataframe[segment_column] == i])

    return segmented_dfs


def metrics_summary(all_metrics, segmentation_names):
    
    summary = []
    for segmentation in all_metrics:
        avg_accuracy  = sum(d['accuracy'] for d in segmentation) / len(segmentation)
        avg_precision = sum(d['precision'] for d in segmentation) / len(segmentation)
        avg_recall    = sum(d['recall'] for d in segmentation) / len(segmentation)
        avg_f1_score  = sum(d['f1_score'] for d in segmentation) / len(segmentation)
        avg_roc_auc   = sum(d['roc_auc'] for d in segmentation) / len(segmentation)
        summary.append({'accuracy': avg_accuracy,
                        'precision': avg_precision, 
                        'recall':avg_recall,
                        'f1_score': avg_f1_score,
                        'roc_auc': avg_roc_auc})

    # Crear DataFrame
    df_summary = pd.DataFrame(summary)
    df_summary.index = segmentation_names
    
    mean_row = df_summary.mean()
    df_summary.loc['------------------- MEAN'] = mean_row

    return df_summary




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

def mlflow_phase1(df, column_name, clusters):
    # df_kmeans_segmented2, 'kmeans_segment'
    
    with mlflow.start_run(run_name=f"kmeans_{clusters}_logreg"):
        
        mlflow.log_param("Phase1_method", "kmeans")
        mlflow.log_param("kmeans_clusters", clusters)
    
        segment_report = []
        for seg_id in sorted(df[column_name].unique()):
            segment_data = df[df[column_name] == seg_id]
            target_counts = segment_data['target'].value_counts(normalize=True)
            segment_report.append(f"Segment # {seg_id}:\n{target_counts.to_string()}\n{'-'*20}")

        segment_text = "\n".join(segment_report)
        with open("segment_report.txt", "w") as f:
            f.write(segment_text)

        mlflow.log_artifact("segment_report.txt")


def mlflow_phase2(run_name, segment_name, metrics_list):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("fase2_modelo", "logistic_regression")
        mlflow.log_param("segmentacion", segment_name)

        # Registrar las métricas promedio
        avg_metrics = {
            "accuracy": np.mean([m["accuracy"] for m in metrics_list]),
            "recall": np.mean([m["recall"] for m in metrics_list]),
            "f1_score": np.mean([m["f1_score"] for m in metrics_list]),
            "roc_auc": np.mean([m.get("roc_auc", 0.0) for m in metrics_list])
        }

        for k, v in avg_metrics.items():
            mlflow.log_metric(k, v)

        print(f"Registered in MLflow: {run_name}\n\n")


if __name__ == '__main__':
    segmentation(3, 3, 
                 './data/processed/X_train_balanced.csv', 
                 './data/processed/y_train_balanced.csv')