"""
src/process/vectorizer.py
Vectorized feature extraction — no Python loops on DataFrame rows.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def vectorized_date_features(df: pd.DataFrame, date_col="Date") -> pd.DataFrame:
    """Extract date parts from the Date column using pandas .dt accessor."""
    df = df.copy()
    if date_col not in df.columns:
        return df

    dt = pd.to_datetime(df[date_col])
    df["Year"]      = dt.dt.year.values
    df["Month"]     = dt.dt.month.values
    df["DayOfWeek"] = dt.dt.dayofweek.values
    df["Quarter"]   = dt.dt.quarter.values
    df["IsWeekend"] = (dt.dt.dayofweek >= 5).astype(int).values

    logger.debug("Date features extracted: Year, Month, DayOfWeek, Quarter, IsWeekend")
    return df


def vectorized_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived revenue columns using vectorized NumPy ops."""
    df = df.copy()

    if "TotalPrice" in df.columns and "Quantity" in df.columns:
        df["EffectiveUnitPrice"] = np.where(
            df["Quantity"] > 0,
            df["TotalPrice"].values / df["Quantity"].values,
            0.0
        )

    if "ItemsInCart" in df.columns and "TotalPrice" in df.columns:
        df["RevenuePerCartItem"] = np.where(
            df["ItemsInCart"] > 0,
            df["TotalPrice"].values / df["ItemsInCart"].values,
            0.0
        )

    return df
