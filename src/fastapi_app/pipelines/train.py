"""
Model Training Pipeline (Ensemble + LOOCV Selection)

This module:
- Loads dataset containing features
- Applies feature encoding (Label + OneHot)
- Trains multiple ML models (linear + tree-based)
- Evaluates models using Leave-One-Out Cross Validation (LOOCV)
- Selects top models based on MSE
- Trains final ensemble models on full dataset
- Saves trained models + metadata for inference

CLI usage (from repo root or src/):
    python -m pipelines.train
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone
from . import preprocess
from utils.cat_encoding import Cat_LabelEncoding, Cat_OneHotEncoding
from utils.logger import setup_logger
from config import (
    ENSEMBLE_NUM,
    ENSEMBLE_RANKING_METRIC,
    ENSEMBLE_STRATEGY
)
from ..models.models import LINEAR_MODELS, TREE_MODELS

ensemble_metrics_col = ["mae", "mse", "rmse", "top_models", "ensemble_type"]

# Logging setup
logger = setup_logger()


def loocv_metrics(model, X, y):
    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    n = len(X)
    logger.info(f"[LOOCV with {model}] Starting | samples={n}")

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        pred = model_clone.predict(X_test)

        y_true_all.extend(y_test.values)
        y_pred_all.extend(pred)

        if i % 50 == 0:
            logger.debug(f"\n[LOOCV with {model}] Progress {i}/{n}\n")

    mae = mean_absolute_error(y_true_all, y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse)

    logger.info(
        f"[LOOCV COMPLETE] "
        f"MAE={mae:.6f} | "
        f"MSE={mse:.6f} | "
        f"RMSE={rmse:.6f}"
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }


def train(X_df, y, category_cols):
    # Encoding
    X_label = Cat_LabelEncoding(X_df, category_cols)
    X_onehot = Cat_OneHotEncoding(X_df, category_cols)

    logger.info(f"Encoded datasets ready | Label: {X_label.shape}, OneHot: {X_onehot.shape}\n")

    results = {}

    # Linear models (one-hot encoding)
    for model_name, model in LINEAR_MODELS:
        metrics = loocv_metrics(model, X_onehot, y)
        results[model_name] = {
            **metrics,
            "model": model,
            "type": "linear"
        }

    # Tree models (label encoding)
    for model_name, model in TREE_MODELS:
        metrics = loocv_metrics(model, X_label, y)
        results[model_name] = {
            **metrics,
            "model": model,
            "type": "tree"
        }

    # Build metrics dataframe
    metrics_df = pd.DataFrame([
        {
            "model_name": name,
            "mse": result["mse"],
            "rmse": result["rmse"],
            "mae": result["mae"],
            "type": result["type"]
        }
        for name, result in results.items()
    ])

    # Rank using MSE only
    metrics_df = (
        metrics_df
        .sort_values(ENSEMBLE_RANKING_METRIC)
        .reset_index(drop=True)
    )

    metrics_df["rank"] = metrics_df.index + 1
    metrics_df["selected"] = False

    metrics_df.loc[
        :ENSEMBLE_NUM-1,
        "selected"
    ] = True

    # Choosing top models based on MSE
    top_models = metrics_df[metrics_df["selected"]]["model_name"].tolist()

    logger.info(f"\n\nSelected top models: {top_models}\n")

    # Retrain only selected models with ALL available data
    trained_models = {}

    logger.info("Retraining top models...\n")

    for model_name in top_models:

        model_result = results[model_name]

        try:
            # Create a fresh copy of the model
            model = clone(model_result["model"])

            # Choose correct feature representation
            # - Linear models → one-hot encoded features
            # - Tree models   → label encoded features
            if model_result["type"] == "linear":
                logger.info(f"Training {model_name} on OneHot features")
                model.fit(X_onehot, y)
            else:
                logger.info(f"Training {model_name} on LabelEncoded features")
                model.fit(X_label, y)

            # Store trained model + metadata
            trained_models[model_name] = {
                "model": model,
                "type": model_result["type"],
                "metrics": {
                    "mse": model_result["mse"],
                    "rmse": model_result["rmse"],
                    "mae": model_result["mae"]
                }
            }

            logger.info(f"Training {model_name} complete\n\n")

        except Exception as e:
            raise ValueError(f"❌ Failed training {model_name}: {repr(e)}")
            continue

    ensemble_metrics_df = (
        metrics_df[metrics_df["selected"]]
        .agg({
            "mae": ENSEMBLE_STRATEGY,
            "mse": ENSEMBLE_STRATEGY,
            "rmse": ENSEMBLE_STRATEGY
        })
        .to_frame()
        .T
    )

    ensemble_metrics_df["top_models"] = ", ".join(top_models)
    ensemble_metrics_df["ensemble_type"] = ENSEMBLE_STRATEGY

    return trained_models, X_onehot.columns, ensemble_metrics_df[ensemble_metrics_col]

if __name__ == "__main__":
    train()