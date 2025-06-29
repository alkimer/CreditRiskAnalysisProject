from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from credit_risk_analysis.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif

app = typer.Typer()

def normalize_categorical_columns(df):
    """
    Normalize categorical columns in the DataFrame.
    This function should be implemented based on the specific requirements of the dataset.
    """
    # Example implementation (to be replaced with actual logic)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower().str.strip()
    return df

def top_n_label_encoder(df, n=10):
    """
    Encode categorical columns using top-n label encoding.
    This function should be implemented based on the specific requirements of the dataset.
    """
    # Example implementation (to be replaced with actual logic)
    for col in df.select_dtypes(include=['object']).columns:
        top_n = df[col].value_counts().nlargest(n).index
        df[col] = df[col].where(df[col].isin(top_n), other='other')
        df[col] = df[col].astype('category').cat.codes
    return df

def select_features(X_train, y_train, selection_method=VarianceThreshold(threshold=0.01), k=15):
    """
    Select relevant features from the DataFrame.
    This function should be implemented based on the specific requirements of the dataset.
    """
    # Example implementation (to be replaced with actual logic)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Define preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean for numerical features
                #('scaler', StandardScaler())  # Scale numerical features
            ]), numerical_columns),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with mode for categorical features
                ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
            ]), categorical_columns)
        ]
    )

    pipeline = Pipeline([
        ('preprocessing', preprocessor),  # Preprocess the data
        ('feature_selection', selection_method), 
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train.values.ravel())

    # Get the selected feature names
    selected_features = pipeline.named_steps['feature_selection'].get_support(indices=True)
    preprocessed_feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    selected_feature_names = preprocessed_feature_names[selected_features]
    print("Selected Features:", selected_feature_names)

    return selected_feature_names, pipeline

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    stat_model_option: str = 'VarianceThreshold',  # Options: 'VarianceThreshold', 'SelectKBest_chi2', 'SelectKBest_f_classif', 'SelectKBest_mutual_info_classif'
    Threshold: float = 0.01,
    K: int = 15,
    # -----------------------------------------
):
    # -----------------------------------------
    logger.info("Loading training and validation data...")
    x_train_path = os.path.join(input_path, "X_train.csv")
    y_train_path = os.path.join(input_path, "y_train.csv")
    x_val_path = os.path.join(input_path, "X_val.csv")
    y_val_path = os.path.join(input_path, "y_val.csv")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    X_val = pd.read_csv(x_val_path)
    y_val = pd.read_csv(y_val_path)
    logger.success("Loading training and validation data completed.")
    X_train = X_train.drop(columns=['ID_CLIENT', 'CLERK_TYPE'], errors='ignore')
    X_train = normalize_categorical_columns(X_train)
    X_train = top_n_label_encoder(X_train, n=10)
    X_val = X_val.drop(columns=['ID_CLIENT', 'CLERK_TYPE'], errors='ignore')
    X_val = normalize_categorical_columns(X_val)
    X_val = top_n_label_encoder(X_val, n=10)
    logger.info("Setting up feature selection method...")
    if stat_model_option == 'VarianceThreshold':
        selection_method = VarianceThreshold(threshold=Threshold)
    elif stat_model_option == 'SelectKBest_chi2':
        selection_method = SelectKBest(chi2, k=K)
    elif stat_model_option == 'SelectKBest_f_classif':
        selection_method = SelectKBest(f_classif, k=K)
    elif stat_model_option == 'SelectKBest_mutual_info_classif':
        selection_method = SelectKBest(mutual_info_classif, k=K)
    else:
        logger.error("Invalid selection method. Defaulting to VarianceThreshold.")
        selection_method = VarianceThreshold(threshold=0.01)
    logger.success("Feature selection method set to: {}".format(selection_method))
    logger.info("Starting feature selection...")
    selected_features, pipeline = select_features(X_train, y_train, selection_method=selection_method)
    logger.success("Features selection complete.")
    # -----------------------------------------
    #print("Selected Features:", selected_features)
    # Save the selected features and pipeline
    selected_features_df_train = pd.DataFrame(pipeline.transform(X_train), columns=selected_features)
    selected_features_df_train.to_csv(os.path.join(output_path, "X_train_selected_features.csv"), index=False)
    selected_features_df_val = pd.DataFrame(pipeline.transform(X_val), columns=selected_features)
    selected_features_df_val.to_csv(os.path.join(output_path, "X_val_selected_features.csv"), index=False)
    logger.success("df train/val selected features saved to: {}".format(os.path.join(output_path, "X_train|val_selected_features.csv")))


if __name__ == "__main__":
    app()
