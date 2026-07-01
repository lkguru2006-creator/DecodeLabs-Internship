"""
src/process/collinearity.py
Detects and removes collinear features using Pearson correlation.
For each flagged pair, keeps the feature more correlated with the target.
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


def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).corr(method="pearson").abs()


def find_collinear_pairs(corr: pd.DataFrame, threshold=0.80) -> list:
    pairs = []
    cols  = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] >= threshold:
                pairs.append((cols[i], cols[j], round(corr.iloc[i, j], 4)))
    return pairs


def _weaker_feature(df: pd.DataFrame, col_a: str, col_b: str, target: str) -> str:
    """Return whichever of col_a / col_b correlates less with the target."""
    if target not in df.columns:
        return col_b
    r_a = abs(df[[col_a, target]].corr().iloc[0, 1])
    r_b = abs(df[[col_b, target]].corr().iloc[0, 1])
    return col_b if r_a >= r_b else col_a


def remove_collinearity(df: pd.DataFrame, config_path="configs/imputation_config.yaml",
                         save_heatmap=True, figures_dir="reports/figures/"):
    cfg       = load_config(config_path)["collinearity"]
    threshold = cfg.get("correlation_threshold", 0.80)
    target    = cfg.get("target_column", "TotalPrice")

    df   = df.copy()
    corr = build_correlation_matrix(df)

    if save_heatmap:
        _save_heatmap(corr, "corr_before", figures_dir)

    pairs = find_collinear_pairs(corr, threshold)

    dropped = set()
    for col_a, col_b, _ in pairs:
        if col_a in dropped or col_b in dropped:
            continue
        to_drop = _weaker_feature(df, col_a, col_b, target)
        dropped.add(to_drop)
        logger.debug(f"  Dropping '{to_drop}' (collinear with partner, r={_})")

    cols_to_drop = [c for c in dropped if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    if save_heatmap:
        _save_heatmap(build_correlation_matrix(df), "corr_after", figures_dir)

    return df, cols_to_drop


def _save_heatmap(corr: pd.DataFrame, name: str, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    size = max(8, len(corr))
    fig, ax = plt.subplots(figsize=(size, size - 2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(f"Correlation Heatmap — {name.replace('_', ' ').title()}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{name}.png"), dpi=120, bbox_inches="tight")
    plt.close()
