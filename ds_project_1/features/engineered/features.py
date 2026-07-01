"""
features/engineered/features.py
Three new predictive features derived from existing columns.
All use vectorized NumPy/pandas ops — no Python loops.
"""

import pandas as pd
import numpy as np


def add_revenue_intensity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    RevenueIntensityScore = TotalPrice / (Quantity × ItemsInCart)
    High score = premium low-volume order.
    Low score  = bulk/discount order.
    """
    df = df.copy()
    denom = df["Quantity"].values * df["ItemsInCart"].values
    df["RevenueIntensityScore"] = np.where(denom > 0, df["TotalPrice"].values / denom, 0.0)
    return df


def add_price_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    PriceDeviation = UnitPrice - mean(UnitPrice per Product)
    Positive = priced above category average (premium).
    Negative = priced below (discount/deal).
    Uses groupby.transform() — fully vectorized.
    """
    df = df.copy()
    category_mean       = df.groupby("Product")["UnitPrice"].transform("mean")
    df["PriceDeviation"] = df["UnitPrice"].values - category_mean.values
    return df


def add_spend_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    SpendTier = pd.cut(TotalPrice, 4 equal-width bins) → 0, 1, 2, 3
    0 = Low, 1 = Medium, 2 = High, 3 = Premium
    """
    df = df.copy()
    df["SpendTier"] = pd.cut(
        df["TotalPrice"], bins=4, labels=[0, 1, 2, 3], include_lowest=True
    ).astype(int)
    return df


def add_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_revenue_intensity_score(df)
    df = add_price_deviation(df)
    df = add_spend_tier(df)
    return df
