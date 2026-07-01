"""
src/output/feature_store.py
Simulates Feast offline/online feature store using CSV files.
In production, replace push_to_offline_store / push_to_online_store
with actual feast.FeatureStore API calls.
"""

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

OFFLINE_PATH = "data/processed/offline_feature_store.csv"
ONLINE_PATH  = "data/processed/online_feature_cache.csv"


def push_to_offline_store(df: pd.DataFrame, path=OFFLINE_PATH):
    """Save full training feature set to offline store (used for batch training)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.debug(f"Offline store: {len(df)} rows → {path}")


def push_to_online_store(df: pd.DataFrame, key_col="CustomerID", path=ONLINE_PATH):
    """Keep latest feature values per entity for real-time serving."""
    if key_col not in df.columns:
        return

    ts_col = "event_timestamp" if "event_timestamp" in df.columns else None
    latest = (df.sort_values(ts_col).groupby(key_col).last().reset_index()
              if ts_col else df.groupby(key_col).last().reset_index())

    os.makedirs(os.path.dirname(path), exist_ok=True)
    latest.to_csv(path, index=False)
    logger.debug(f"Online store: {len(latest)} entities → {path}")
