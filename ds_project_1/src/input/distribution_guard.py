"""
src/input/distribution_guard.py
Checks that cleaning didn't distort feature distributions.
Warns if std changes more than 15% — sign of over-imputation.
"""

import logging
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

NUMERIC_COLS = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]


def check_distribution_variance(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    report = {}

    for col in NUMERIC_COLS:
        if col not in df_before.columns or col not in df_after.columns:
            continue

        before_std      = df_before[col].std()
        after_std       = df_after[col].std()
        std_change_pct  = abs(after_std - before_std) / (before_std + 1e-9) * 100

        status = "WARNING" if std_change_pct > 15 else "OK"
        if status == "WARNING":
            logger.warning(f"  '{col}': std changed {std_change_pct:.1f}% — possible over-imputation")

        report[col] = {
            "before_std":     round(before_std, 4),
            "after_std":      round(after_std, 4),
            "std_change_pct": round(std_change_pct, 2),
            "status":         status,
        }

    return report


def run_normality_check(df: pd.DataFrame) -> dict:
    results = {}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        sample = df[col].dropna().sample(min(500, len(df)), random_state=42)
        stat, p = stats.shapiro(sample)
        results[col] = {"statistic": round(stat, 4), "p_value": round(p, 4), "normal": p > 0.05}
    return results
