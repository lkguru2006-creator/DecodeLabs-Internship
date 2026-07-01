"""
src/output/point_in_time.py
Enforces temporal integrity for train/test splits.
Uses date ordering to prevent future data leaking into training labels.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def create_temporal_train_test_split(df: pd.DataFrame, date_col="Date",
                                      cutoff_date=None, train_ratio=0.80):
    """
    Sort by date and split at a cutoff — not a random shuffle.
    Random shuffling causes data leakage on time-ordered datasets.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if cutoff_date:
        cutoff   = pd.Timestamp(cutoff_date)
        df_train = df[df[date_col] <= cutoff].copy()
        df_test  = df[df[date_col] >  cutoff].copy()
    else:
        n        = int(len(df) * train_ratio)
        df_train = df.iloc[:n].copy()
        df_test  = df.iloc[n:].copy()

    logger.debug(f"Split: {len(df_train)} train / {len(df_test)} test")
    return df_train, df_test


def add_event_timestamp(df: pd.DataFrame, date_col="Date") -> pd.DataFrame:
    """Rename Date → event_timestamp for Feast compatibility."""
    df = df.copy()
    if date_col in df.columns:
        df = df.rename(columns={date_col: "event_timestamp"})
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    return df


def check_for_leakage(df_train: pd.DataFrame, df_test: pd.DataFrame,
                       date_col="event_timestamp") -> bool:
    """Assert train max date is strictly before test min date."""
    if date_col not in df_train.columns or date_col not in df_test.columns:
        return True
    no_leak = df_train[date_col].max() < df_test[date_col].min()
    if not no_leak:
        logger.warning("Data leakage detected — train and test date ranges overlap")
    return no_leak
