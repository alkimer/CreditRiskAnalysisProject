from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from credit_risk_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             average_precision_score, f1_score, fbeta_score, make_scorer)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.neural_network import MLPClassifier
import mlflow
import mlflow.sklearn

app = typer.Typer()

# --- MLflow Autologging ---
#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

# --- Data Loading ---
def load_data(x_path, y_path, x_val_path=None, y_val_path=None, target_column="TARGET_LABEL_BAD=1"):
    """ Load training and validation data from CSV files."""
    try:
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading training data: {e}. Make sure paths are correct.")

    if target_column not in y.columns:
        raise ValueError(f"Missing target column: {target_column}")
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
    print(f"Target distribution:\n{y[target_column].value_counts(normalize=True)}")

    X_val, y_val = None, None
    if x_val_path and y_val_path:
        try:
            X_val = pd.read_csv(x_val_path)
            y_val = pd.read_csv(y_val_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading validation data: {e}. Make sure paths are correct.")
        if target_column not in y_val.columns:
            raise ValueError(f"Missing target column in validation data: {target_column}")
        print(f"Loaded X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"Validation target distribution:\n{y_val[target_column].value_counts(normalize=True)}")
    return X, y[target_column], X_val, y_val[target_column]

# --- Pipeline Building ---
def build_pipeline(smote_sampling_strategy=0.7, xgb_params=None, n_components=0.95):
    if xgb_params is None:
        xgb_params = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "objective": "binary:logistic",
        }
    model = XGBClassifier(**xgb_params)
    pipeline = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        #("smote", SMOTE(random_state=42, sampling_strategy=smote_sampling_strategy)),
        ("smote_tomek", SMOTETomek(random_state=42)),
        ("pca", PCA(n_components=n_components)),   
        ("model", model)
    ])
    return pipeline

# --- Feature Importance Plotting ---
def plot_feature_importance(model, feature_names, output_path="reports/figures/feature_importance.png", top_n=15): # Changed default top_n to 15
    try:
        importances = model.named_steps["model"].feature_importances_
    except AttributeError:
        print("Model does not have feature_importances_ attribute.")
        return None

    indices = np.argsort(importances)[::-1]
    sorted_feature_names = np.array(feature_names)[indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances (Top {})".format(top_n))
    plt.bar(range(min(top_n, len(importances))), sorted_importances[:top_n])
    plt.xticks(range(min(top_n, len(importances))), sorted_feature_names[:top_n], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")
    return output_path

def save_model(model, output_path="models/xgb_model.pkl"):
    try:
        joblib.dump(model, output_path)
        print(f"Model saved to {output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def train_model(model_name="xgb_model",
                experiment_name="credit_risk_experiment",
                x_path=PROCESSED_DATA_DIR / "X_train_processed.csv",
                y_path=PROCESSED_DATA_DIR / "y_train.csv",
                x_val_path=PROCESSED_DATA_DIR / "X_val_processed.csv",
                y_val_path=PROCESSED_DATA_DIR / "y_val.csv",
                ):
    #mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()
    # Load data
    X, y, X_val, y_val = load_data(x_path, y_path, x_val_path, y_val_path)

    # Calculate scale_pos_weight
    neg_count = y.value_counts()[0]
    pos_count = y.value_counts()[1]
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1

    # Stratified train-test split (for final evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain set distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test set distribution:\n{y_test.value_counts(normalize=True)}")

    if model_name == "xgb_model":

        base_xgb_model = XGBClassifier(eval_metric="logloss",
                                   objective="binary:logistic",
                                   scale_pos_weight=scale_pos_weight_val)
        
        pipeline_for_tuning = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        #("smote", SMOTE(random_state=42)),
        #("smote_tomek", SMOTETomek(random_state=42)),
        ("pca", PCA()),
        ("model", base_xgb_model)
        ])

        param_grid = {
            #'smote__sampling_strategy': [0.5], #[0.5, 0.75, 1.0],
            'pca__n_components': [0.95], #[0.9, 0.95, 0.99],
            'model__n_estimators': [200], #[100, 200, 300],
            'model__learning_rate': [0.1], #[0.05, 0.1, 0.2],
            'model__max_depth': [3], #[3, 5, 7],
            'model__subsample': [1.0], #[0.7, 0.8, 1.0],
            'model__colsample_bytree': [0.7], #[0.7, 0.8, 1.0],
            'model__reg_alpha': [1], #[0, 0.1, 1],
            'model__reg_lambda': [1], #[1, 5, 10],
            'model__gamma': [0.5] #[0, 0.1, 0.5]
        }

        #scorer = make_scorer(f1_score, pos_label=1)
        scorer = make_scorer(fbeta_score, beta=0.5, pos_label=1)

        if X_val is not None and y_val is not None:
            print("\nUsing provided validation data for early stopping.")
            grid_search = GridSearchCV(
                pipeline_for_tuning,
                param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring=scorer,
                verbose=1,
                n_jobs=-1
        )
        else:
            print("\nNo validation data provided. Early stopping will not be used in GridSearchCV.")
            grid_search = GridSearchCV(
                pipeline_for_tuning,
                param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring=scorer,
                verbose=1,
                n_jobs=-1
        )

        print("\nStarting Hyperparameter Tuning...")
        with mlflow.start_run(run_name="Hyperparameter_Tuning"):
            grid_search.fit(X_train, y_train)

            best_pipeline = grid_search.best_estimator_
            print(f"\nBest parameters found: {grid_search.best_params_}")
            print(f"Best F1-score (CV): {grid_search.best_score_:.4f}")

            # Evaluate the best pipeline on the test set
            y_pred = best_pipeline.predict(X_test)
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
            y_pred_val = best_pipeline.predict(X_val)
            y_prob_val = best_pipeline.predict_proba(X_val)[:, 1]

            # Calculate and log final metrics
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_metric("test_pr_auc", pr_auc)

            # Calculate and log final metrics
            acc_val = accuracy_score(y_val, y_pred_val)
            report_val = classification_report(y_val, y_pred_val, output_dict=True)
            roc_auc_val = roc_auc_score(y_val, y_prob_val)
            pr_auc_val = average_precision_score(y_val, y_prob_val)

            mlflow.log_metric("val_accuracy", acc_val)
            mlflow.log_metric("val_roc_auc", roc_auc_val)
            mlflow.log_metric("val_pr_auc", pr_auc_val)


            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        mlflow.log_metric(f"test_{label}_{k}", v)
                else:
                    mlflow.log_metric(f"test_{label}", metrics)

            for label, metrics in report_val.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        mlflow.log_metric(f"val_{label}_{k}", v)
                else:
                    mlflow.log_metric(f"val_{label}", metrics)

            print(f"\nTest Accuracy: {acc:.4f}")
            print(f"Test ROC AUC: {roc_auc:.4f}")
            print(f"Test PR AUC: {pr_auc:.4f}")
            print("\nTest Classification Report:\n", classification_report(y_test, y_pred))

            print(f"\nVal Accuracy: {acc_val:.4f}")
            print(f"Val ROC AUC: {roc_auc_val:.4f}")
            print(f"Val PR AUC: {pr_auc_val:.4f}")
            print("\nVal Classification Report:\n", classification_report(y_val, y_pred_val))

            # Confusion Matrix plotting and logging
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix - Test Set")
            plt.tight_layout()
            conf_matrix_path_t = "reports/figures/confusion_matrix_xgb_test.png"
            plt.savefig(conf_matrix_path_t)
            plt.close()
            mlflow.log_artifact(conf_matrix_path_t)

            cm_val = confusion_matrix(y_val, y_pred_val)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_val)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix - Val Set")
            plt.tight_layout()
            conf_matrix_path_v = "reports/figures/confusion_matrix_xgb_val.png"
            plt.savefig(conf_matrix_path_v)
            plt.close()
            mlflow.log_artifact(conf_matrix_path_v)

            # Feature importance plot for the best model (now plots top 15 by default)
            if hasattr(best_pipeline.named_steps["model"], 'feature_importances_'):
                importance_path = plot_feature_importance(best_pipeline, X.columns, output_path="reports/figures/feature_importance_xgb.png", top_n=15)
                if importance_path:
                    mlflow.log_artifact(importance_path)
            else:
                print("Feature importance cannot be plotted.")
            # Save the best model
            save_model(best_pipeline, output_path="models/xgb_model.pkl")
            print(f"\nMLflow Run completed successfully: {mlflow.active_run().info.run_id}")
            return best_pipeline
        
    elif model_name == "stacking_model":
        # Implement stacking model training and evaluation
        #pass
        
        xgb = XGBClassifier(
            eval_metric="logloss",
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight_val,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            subsample=1.0,
            colsample_bytree=0.7,
            reg_alpha=1,
            reg_lambda=1,
            gamma=0.5,
            use_label_encoder=False,
            random_state=42
        )

        rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5)

        # Optional: SVM or Naive Bayes if you want to expand
        svc = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
        nb = GaussianNB()

        meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        
        stacking_model = StackingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('knn', knn),
            ('svc', svc),     # Optional
            ('nb', nb)        # Optional
        ],
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=True
        )

        stacking_pipeline = ImbPipeline(steps=[
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(k=20)),
            #('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
            ("pca", PCA(n_components=0.95)),
            ('stacking', stacking_model)
        ])


        print("\nTraining Stacked Model with Diverse Base Learners...")
        stacking_pipeline.fit(X_train, y_train)

        y_pred = stacking_pipeline.predict(X_test)
        y_prob = stacking_pipeline.predict_proba(X_test)[:, 1]

        print("\nStacked Model - Classification Report:")
        print(classification_report(y_test, y_pred))

        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print(f"PR AUC: {average_precision_score(y_test, y_prob):.4f}")

        # Confusion Matrix plotting and logging
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("reports/figures/confusion_matrix_stacking.png")
        plt.close()
        print("Confusion matrix plot saved.")

        y_pred_val = stacking_pipeline.predict(X_val)
        y_prob_val = stacking_pipeline.predict_proba(X_val)[:, 1]

        print("\nStacked Model - Classification Report Validation:")
        print(classification_report(y_val, y_pred_val))

        print(f"ROC AUC: {roc_auc_score(y_val, y_prob_val):.4f}")
        print(f"PR AUC: {average_precision_score(y_val, y_prob_val):.4f}")

        # Confusion Matrix plotting and logging
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("reports/figures/confusion_matrix_stacking.png")
        plt.close()
        print("Confusion matrix plot saved.")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("reports/figures/confusion_matrix_stacking.png")
        plt.close()
        print("Confusion matrix plot saved.")

        cm = confusion_matrix(y_val, y_pred_val)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("reports/figures/confusion_matrix_stacking_val.png")
        plt.close()
        print("Confusion matrix plot saved.")

        # Save the stacking model
        save_model(stacking_pipeline, output_path="models/stacking_model.pkl")
        print("Stacked model training complete and saved.")
        return stacking_pipeline

    elif model_name == "mlp_model":
        # Implement MLP model training and evaluation
        pass
        # --- MLP com early stopping ---
        base_mlp_model = MLPClassifier(
            random_state=42,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1
        )

        pipeline_for_tuning = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            #("smote_tomek", SMOTETomek(random_state=42)),
            ("pca", PCA()),
            ("model", base_mlp_model)
        ])

        # --- Hyperparameter tuning for MLP ---
        param_grid = {
            'pca__n_components': [0.95],
            'model__hidden_layer_sizes': [(100,), (64, 32), (128, 64)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [0.0001, 0.001],
            'model__learning_rate_init': [0.001, 0.01]
        }

        scorer = make_scorer(fbeta_score, beta=0.5, pos_label=1)

        grid_search = GridSearchCV(
            pipeline_for_tuning,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=scorer,
            verbose=1,
            n_jobs=-1
        )

        # --- Execution of GridSearch + evaluation ---
        print("\nStarting Hyperparameter Tuning... (MLP)...")
        with mlflow.start_run(run_name="MLP_Hyperparameter_Tuning"):
            grid_search.fit(X_train, y_train)

            best_pipeline = grid_search.best_estimator_
            print(f"\nBest parameters found: {grid_search.best_params_}")
            print(f"Best F0.5-score (CV): {grid_search.best_score_:.4f}")

            # Predictions
            y_pred = best_pipeline.predict(X_test)
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
            y_pred_val = best_pipeline.predict(X_val)
            y_prob_val = best_pipeline.predict_proba(X_val)[:, 1]

            # Metrics - Test
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_metric("test_pr_auc", pr_auc)

            # Metrics - Validation
            acc_val = accuracy_score(y_val, y_pred_val)
            report_val = classification_report(y_val, y_pred_val, output_dict=True)
            roc_auc_val = roc_auc_score(y_val, y_prob_val)
            pr_auc_val = average_precision_score(y_val, y_prob_val)

            mlflow.log_metric("val_accuracy", acc_val)
            mlflow.log_metric("val_roc_auc", roc_auc_val)
            mlflow.log_metric("val_pr_auc", pr_auc_val)

            # Logging detailed metrics
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        mlflow.log_metric(f"test_{label}_{k}", v)
                else:
                    mlflow.log_metric(f"test_{label}", metrics)

            for label, metrics in report_val.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        mlflow.log_metric(f"val_{label}_{k}", v)
                else:
                    mlflow.log_metric(f"val_{label}", metrics)

            print(f"\nTest Accuracy: {acc:.4f}")
            print(f"Test ROC AUC: {roc_auc:.4f}")
            print(f"Test PR AUC: {pr_auc:.4f}")
            print("\nTest Classification Report:\n", classification_report(y_test, y_pred))

            print(f"\nVal Accuracy: {acc_val:.4f}")
            print(f"Val ROC AUC: {roc_auc_val:.4f}")
            print(f"Val PR AUC: {pr_auc_val:.4f}")
            print("\nVal Classification Report:\n", classification_report(y_val, y_pred_val))

            # --- Confusion Matrix ---
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix - Test Set")
            conf_matrix_path_t = "reports/figures/confusion_matrix_mlp_test.png"
            plt.tight_layout()
            plt.savefig(conf_matrix_path_t)
            plt.close()
            mlflow.log_artifact(conf_matrix_path_t)

            cm_val = confusion_matrix(y_val, y_pred_val)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_val)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix - Val Set")
            conf_matrix_path_v = "reports/figures/confusion_matrix_mlp_val.png"
            plt.tight_layout()
            plt.savefig(conf_matrix_path_v)
            plt.close()
            mlflow.log_artifact(conf_matrix_path_v)

            # --- Curva de perda do MLP ---
            best_model = best_pipeline.named_steps["model"]
            if hasattr(best_model, "loss_curve_"):
                plt.plot(best_model.loss_curve_)
                plt.title("MLP Loss Curve")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                loss_path = "reports/figures/mlp_loss_curve.png"
                plt.tight_layout()
                plt.savefig(loss_path)
                plt.close()
                mlflow.log_artifact(loss_path)

            print(f"\nMLflow Run concluído com sucesso: {mlflow.active_run().info.run_id}")
            # Save the best model
            save_model(best_pipeline, output_path="models/mlp_model.pkl")
            return best_pipeline
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        return None
    

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "X_train_processed.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "y_train.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    model_name: str = "xgb_model",
    target_column: str = "TARGET_LABEL_BAD=1",
    experiment_name: str = "credit_risk_experiment",
    x_val_path: Path = PROCESSED_DATA_DIR / "X_val_processed.csv",
    y_val_path: Path = PROCESSED_DATA_DIR / "y_val.csv"
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    #for i in tqdm(range(10), total=10):
    #    if i == 5:
    #        logger.info("Something happened for iteration 5.")
    train_model(
        model_name=model_name,
        x_path=features_path,
        y_path=labels_path,
        x_val_path=x_val_path,
        y_val_path=y_val_path
    )
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
