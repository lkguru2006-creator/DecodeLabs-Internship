"""
src/input/outlier_handler.py
Detects outliers via IQR and winsorizes using numpy.clip().
Never drops rows — preserves row count and sequential integrity.
"""

import logging
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


def load_config(path="configs/imputation_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_iqr_bounds(series: pd.Series, multiplier=1.5):
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - multiplier * IQR, Q3 + multiplier * IQR


def handle_outliers(df: pd.DataFrame, config_path="configs/imputation_config.yaml",
                    save_plots=True, figures_dir="reports/figures/") -> pd.DataFrame:
    cfg          = load_config(config_path)["outliers"]
    multiplier   = cfg.get("iqr_multiplier", 1.5)
    numeric_cols = cfg.get("numeric_columns", [])

    df = df.copy()

    for col in numeric_cols:
        if col not in df.columns:
            continue

        lower, upper = compute_iqr_bounds(df[col], multiplier)
        n_outliers   = ((df[col] < lower) | (df[col] > upper)).sum()

        if save_plots and n_outliers > 0:
            _save_boxplot(df[col], col, lower, upper, figures_dir)

        df[col] = np.clip(df[col].values, lower, upper)
        logger.debug(f"  '{col}': {n_outliers} outliers winsorized to [{lower:.2f}, {upper:.2f}]")

    return df


def _save_boxplot(series, col, lower, upper, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].boxplot(series.dropna())
    axes[0].axhline(lower, color="red",    linestyle="--", label=f"Lower: {lower:.0f}")
    axes[0].axhline(upper, color="orange", linestyle="--", label=f"Upper: {upper:.0f}")
    axes[0].set_title(f"{col} — Before")
    axes[0].legend(fontsize=8)

    axes[1].boxplot(np.clip(series.values, lower, upper))
    axes[1].set_title(f"{col} — After Winsorization")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"outlier_{col}.png"), dpi=120, bbox_inches="tight")
    plt.close()
