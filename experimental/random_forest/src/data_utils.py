
import os
import boto3
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, accuracy_score, precision_score,
    recall_score, f1_score
)


def download_credit_data():
    bucket_name = "anyoneai-datasets"
    prefix = "credit-data-2010/"
    local_folder = "data/original"

    aws_access_key_id = "AKIA2JHUK4EGBVSQ5RUW"
    aws_secret_access_key = "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"

    # Create the local folder if it does not exist
    os.makedirs(local_folder, exist_ok=True)

    # Create S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # List objects
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response:
        print("No files found.")
        return
    
    target_files = [
        "credit-data-2010/PAKDD2010_VariablesList.XLS",
        "credit-data-2010/PAKDD2010_Modeling_Data.txt"
    ]

    for obj in response["Contents"]:
        key = obj["Key"]
        if key in target_files:
            filename = os.path.basename(key)
            local_path = os.path.join(local_folder, filename)
            print(f"Downloading {key} to {local_path}...")
            s3.download_file(bucket_name, key, local_path)

    print("Download completed.")


def convert_txt_to_csv(txt_path, xls_columns_path, output_csv):
    """
    Converts the PAKDD2010 TXT dataset to CSV using the provided XLS
    that contains column indices and corresponding names.

    Parameters:
    - txt_path: Path to the PAKDD2010_Modeling_Data.txt file
    - xls_columns_path: Path to the PAKDD2010_VariablesList.XLS file
    - output_csv: Output path to save the CSV
    """

    # Read XLS columns mapping
    columns_df = pd.read_excel(xls_columns_path, engine='xlrd')

    # Ensure correct columns are selected (first column = index, second column = name)
    column_names = columns_df.iloc[:, 1].tolist()

    print(f"Loaded {len(column_names)} column names from XLS.")

    # Read TXT file
    df = pd.read_csv(txt_path, sep='\t', header=None, encoding='cp1252')

    print(f"TXT file shape: {df.shape}")

    # Check consistency
    if len(column_names) != df.shape[1]:
        raise ValueError(f"Mismatch: TXT has {df.shape[1]} columns, XLS provided {len(column_names)} names.")

    # Set correct column names
    df.columns = column_names

    # Create output directory if needed
    output_folder = 'data/original'
    os.makedirs(output_folder, exist_ok=True)

    # Save CSV
    df.to_csv(f'{output_folder}/dataset.csv', index=False)
    print(f"CSV saved to {output_csv} with correct column names.")


def preprocess_credit_data(df):
    """
    Cleans and preprocesses the credit dataset:
    - Drops irrelevant columns.
    - Cleans and imputes missing values.
    - Applies frequency encoding to categorical variables.
    """
    df = df.copy()
    df = df.drop(columns=[
        'ID_CLIENT', 'CLERK_TYPE', 'QUANT_ADDITIONAL_CARDS', 'POSTAL_ADDRESS_TYPE',
        'EDUCATION_LEVEL', 'FLAG_MOBILE_PHONE', 'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS',
        'QUANT_SPECIAL_BANKING_ACCOUNTS', 'COMPANY', 'PROFESSIONAL_CITY',
        'PROFESSIONAL_BOROUGH', 'MONTHS_IN_THE_JOB', 'MATE_PROFESSION_CODE',
        'FLAG_HOME_ADDRESS_DOCUMENT', 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF',
        'FLAG_ACSP_RECORD'
    ], errors='ignore')

    # Clean and replace empty strings or 'nan' with 'unknown'
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip().str.lower().replace({'': 'unknown', 'nan': 'unknown'})

    # Fill numeric missing values with median
    for col in df.select_dtypes(include='number'):
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Frequency encoding for categorical variables
    for col in df.select_dtypes(include='object'):
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[col] = df[col].map(freq_map)
    
    # Create output directory if needed
    output_folder = 'data/preprocessed'
    os.makedirs(output_folder, exist_ok=True)

    return df


def split_and_save(df, target_column, output_folder):
    """
    Splits the dataset into training and testing sets and saves them to CSV files.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    os.makedirs(output_folder, exist_ok=True)
    X_train.to_csv(f"{output_folder}/X_train.csv", index=False)
    X_test.to_csv(f"{output_folder}/X_test.csv", index=False)
    y_train.to_csv(f"{output_folder}/y_train.csv", index=False)
    y_test.to_csv(f"{output_folder}/y_test.csv", index=False)
    print(f"Data split and saved to {output_folder}.")


def apply_smote(input_folder, output_folder):
    """
    Applies SMOTE to balance the classes in the training dataset and saves the new datasets.
    """
    X = pd.read_csv(f"{input_folder}/X_train.csv")
    y = pd.read_csv(f"{input_folder}/y_train.csv").iloc[:, 0]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    os.makedirs(output_folder, exist_ok=True)
    X_resampled.to_csv(f"{output_folder}/X_train_smote.csv", index=False)
    y_resampled.to_csv(f"{output_folder}/y_train_smote.csv", index=False)
    print(f"SMOTE applied. Balanced data saved to {output_folder}.")


def train_random_forest_model(X_train, y_train, X_test, y_test, params, model_output_path):
    """
    Trains a Random Forest model.
    Call the function that prints classification report, ROC AUC score and saves the model.
    """
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)

    print("✅ Best hyperparameters found:")
    print(clf.get_params())

    evaluate_and_plot_model(clf, X_train, y_train, X_test, y_test, model_output_path)


def grid_search_random_forest(X_train, y_train, X_test, y_test, scoring, class_weight, model_output_path):
    """
    Performs a Grid Search for hyperparameter tuning on Random Forest using cross-validation.

     Parameters:
    - X_train: Training features
    - y_train: Training labels
    - scoring: Scoring metric for GridSearchCV (default 'f1')
    - class_weight: Class weight to use in Random Forest (default 'balanced', can be None)
    - model_output_path: Path to save the model
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
        'max_features': ['sqrt'],
        'class_weight': [class_weight]
    }
    clf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(clf, param_grid, scoring=scoring, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    evaluate_and_plot_model(grid.best_estimator_, X_train, y_train, X_test, y_test, model_output_path)


def random_search_random_forest(X_train, y_train, X_test, y_test, scoring, class_weight, model_output_path):
    """
    Performs a Randomized Search for hyperparameter tuning on Random Forest using Randomized Search.
    """
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 8, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [class_weight]
    }
    clf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=25,
                                scoring=scoring, cv=5, n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)
    evaluate_and_plot_model(search.best_estimator_, X_train, y_train, X_test, y_test, model_output_path)


def select_top_features(X_train, y_train, top_n=17):
    """
    Selects the top N most important features using Random Forest feature importances.
    """
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    top_features = importances.nlargest(top_n).index.tolist()
    print(f"Top {top_n} features selected.")
    return top_features


def evaluate_and_plot_model(model, X_train, y_train, X_val, y_val, save_path=None):
    """
    Evaluates a trained model, printing metrics, displaying confusion matrix and ROC curves,
    on both training and validation sets.
    """
    # Training set evaluation
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    
    print("\n🧪 Evaluation on Training Set:")
    print(classification_report(y_train, y_pred_train))
    print(f"\n🔹 ROC AUC Score (Train): {roc_auc_score(y_train, y_proba_train):.4f}")
    print(f"🔹 Accuracy (Train): {accuracy_score(y_train, y_pred_train):.4f}")

    # Validation set evaluation
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print("\n📋 Classification Report (Validation):")
    print(classification_report(y_val, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_val, y_val)
    plt.title("ROC Curve (Validation)")
    plt.show()

    # Additional Metrics
    print(f"\n🔹 ROC AUC Score (Validation): {roc_auc_score(y_val, y_proba):.4f}")
    print(f"🔹 Accuracy:        {accuracy_score(y_val, y_pred):.4f}")
    print(f"🔹 Precision (1):   {precision_score(y_val, y_pred, pos_label=1):.4f}")
    print(f"🔹 Recall (1):      {recall_score(y_val, y_pred, pos_label=1):.4f}")
    print(f"🔹 F1-score (1):    {f1_score(y_val, y_pred, pos_label=1):.4f}")

    # Save the model if a path is provided
    if save_path: 

        # Ensure the parent directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Save the model
        print(f"✅ Saving model to: {save_path}")

        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        print(f"\n✅ Model saved to: {save_path}")
