from pathlib import Path

from loguru import logger
import typer
import pandas as pd
import joblib
import sys, os
sys.path.append(os.path.abspath('../'))
from credit_risk_analysis.processing import process_all_features_v2

app = typer.Typer()

MODELS_DIR = "../../models"


''' This function generates a predicion given the modelpath and a dataframe with the information
provided via de UI
'''
def predict_credit_risk_df(model_name="stacking_model.pkl", ui_data):
    model_path = os.path.join(MODELS_DIR, model_name)
    # if use_fake_data:
    #     create_fake_new_data()
    #     features_path = 'data/external/X_fake_new_sample.csv'
    process_all_features_v2.main(X_new_path=features_path, new_data=True)
    X_n = pd.read_csv('data/processed/X_new_processed.csv')
    model = joblib.load(model_path)
    predictions = model.predict(X_n)
    probabilities = model.predict_proba(X_n)[:, 1]
    X_n["TARGET_LABEL_BAD=1"] = predictions
    X_n["PROBABILITY_BAD=1"] = probabilities
    X_n.to_csv(predictions_path, index=False)
    print("✅ Predicciones de riesgo crediticio completadas.")
    return X_n



def predict_credit_risk(model_path, features_path='data/external/X_fake_new_sample.csv', predictions_path='data/processed/X_new_predicted.csv', use_fake_data=False):
    if use_fake_data:
        create_fake_new_data()
        features_path = 'data/external/X_fake_new_sample.csv'
    process_all_features_v2.main(X_new_path=features_path, new_data=True)
    X_n = pd.read_csv('data/processed/X_new_processed.csv')
    model = joblib.load(model_path)
    predictions = model.predict(X_n)
    probabilities = model.predict_proba(X_n)[:, 1]
    X_n["TARGET_LABEL_BAD=1"] = predictions
    X_n["PROBABILITY_BAD=1"] = probabilities
    X_n.to_csv(predictions_path, index=False)
    print("✅ Predicciones de riesgo crediticio completadas.")
    return X_n

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "stacking_model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    use_fake_data: bool = False,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    #for i in tqdm(range(10), total=10):
    #    if i == 5:
    #        logger.info("Something happened for iteration 5.")
    predictions = predict_credit_risk(
        model_path=model_path,
        features_path=features_path,
        predictions_path=predictions_path,
        use_fake_data=use_fake_data
    )
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
