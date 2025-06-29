from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from credit_risk_analysis.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

def plot_missing_values(df: pd.DataFrame, output_path: Path, save: bool = True):
    """
    Function to plot missing values.
    Args:
        df (DataFrame): The input DataFrame.
        output_path (Path): Path to save the missing values plot.
    """
    logger.info(f"Plotting missing values")
    # Here you would implement the actual plotting logic
    # For now, we just simulate a delay
    ax = df.isnull().sum().plot.bar(
    figsize=(20, 10),
    legend=False
    )
    ax.set_title("Missing Values", fontsize=24)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    plt.figure(figsize=(10, 6))
    plt.title("Missing Values")
    plt.xlabel("Features")
    plt.ylabel("Percentage of Missing Values")
    if save:
        plt.savefig(output_path)
        logger.info(f"Missing values plot saved to {output_path}")
    plt.close()
    logger.success("Missing values plot created successfully.")

def plot_categorical_feature_distribution(df: pd.DataFrame, output_path: Path = REPORTS_DIR, save: bool = True, plot_each_feature: bool = True):
    """
    Function to plot categorical feature distribution.
    Args:
        df (DataFrame): The input DataFrame.
        output_path (Path): Path to save the categorical feature distribution plot.
    """
    logger.info(f"Plotting categorical feature distribution and saving to {output_path}")
    # Here you would implement the actual plotting logic
    # For now, we just simulate a delay
    # Categorical Feature Distribution plotting and logging
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not categorical_cols:
        logger.warning("No categorical columns found in the DataFrame.")
        return
    logger.info(f"Categorical columns found: {categorical_cols}")


    desc = df[categorical_cols].describe()
    desc.loc['unique'].sort_values(ascending=False).plot.bar(figsize=(12, 6), title='Unique values per column')
    plt.figure(figsize=(10, 6))
    plt.xlabel("Columns")
    plt.ylabel("Number of Unique Values")
    plt.tight_layout()
    if save:
        plt.savefig(output_path / "unique_values_per_column.png")
        logger.info(f"Unique values per column plot saved to {output_path / 'unique_values_per_column.png'}")
    plt.close()

    if not plot_each_feature:
        logger.info("Plotting of individual categorical feature distributions is disabled.")
    else:
        logger.info("Plotting individual categorical feature distributions.")
        # Plotting individual categorical feature distributions
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            df[col].value_counts().plot(kind='bar')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            if save:
                plt.savefig(output_path / f"{col}_distribution.png")
                logger.info(f"Categorical feature distribution for {col} saved to {output_path / f'{col}_distribution.png'}")
            plt.close()
        
        logger.success("Categorical feature distributions plotted successfully.")

def plot_confusion_matrix(y_true, y_pred, output_path: Path = REPORTS_DIR, save: bool = True):
    """
    Function to plot confusion matrix.
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        output_path (Path): Path to save the confusion matrix plot.
    """
    logger.info(f"Plotting confusion matrix and saving to {output_path}")
    # Here you would implement the actual plotting logic
    # For now, we just simulate a delay
    # Confusion Matrix plotting and logging
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    if save:
        plt.savefig(output_path)
        logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()
    logger.success("Confusion matrix plotted successfully.")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
