"""
src/input/missing_data.py
Handles missing values using a threshold-based decision matrix.
"""

import logging
import pandas as pd
import numpy as np
import yaml
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)


def load_config(path="configs/imputation_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def handle_missing_data(df: pd.DataFrame, config_path="configs/imputation_config.yaml") -> pd.DataFrame:
    cfg         = load_config(config_path)["imputation"]
    low_thresh  = cfg["low_threshold"]
    high_thresh = cfg["high_threshold"]
    skip_cols   = cfg.get("skip_columns", [])
    knn_k       = cfg.get("knn_neighbors", 5)
    cat_fill    = cfg.get("categorical_fill", "Unknown")

    df          = df.copy()
    missingness = df.isnull().mean()
    knn_targets = []

    for col in df.columns:
        if col in skip_cols:
            continue

        ratio = missingness[col]
        if ratio == 0:
            continue

        is_categorical = (df[col].dtype == object
                          or str(df[col].dtype) in ("category", "string")
                          or df[col].dtype.name == "str"
                          or pd.api.types.is_string_dtype(df[col]))

        if is_categorical:
            fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else cat_fill
            df[col]  = df[col].fillna(fill_val)

        elif ratio < low_thresh:
            df = df.dropna(subset=[col])

        elif ratio <= high_thresh:
            df[col] = df[col].fillna(df[col].median())

        else:
            # Only queue numeric columns for KNN
            knn_targets.append(col)

    if knn_targets:
        imputer         = KNNImputer(n_neighbors=knn_k)
        df[knn_targets] = imputer.fit_transform(df[knn_targets])

    logger.debug(f"Missing remaining after imputation: {df.isnull().sum().sum()}")
    return df
