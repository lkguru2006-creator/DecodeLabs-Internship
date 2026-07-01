"""
main.py — DS Project 1 Pipeline
Advanced EDA & Feature Engineering
DecodeLabs Industrial Training Kit — Batch 2026
"""

import logging
import os
import sys
import time
import yaml
import pandas as pd
import numpy as np

# ── Logging: file only for modules, clean console for main ───────────────────
os.makedirs("logs", exist_ok=True)

file_handler    = logging.FileHandler("logs/pipeline.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))

# Root logger → file only (silences all module loggers from console)
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler])

# Console logger for main only
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("main")
logger.addHandler(console)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.input.missing_data       import handle_missing_data
from src.input.outlier_handler     import handle_outliers
from src.input.distribution_guard  import check_distribution_variance
from src.process.vectorizer        import vectorized_date_features, vectorized_revenue_features
from src.process.encoder           import encode_features
from src.process.collinearity      import remove_collinearity
from src.output.schema_contracts   import validate_raw_input, validate_processed_output
from src.output.point_in_time      import (create_temporal_train_test_split,
                                            add_event_timestamp, check_for_leakage)
from src.output.feature_store      import push_to_offline_store, push_to_online_store
from features.engineered.features  import add_all_engineered_features


def load_config(path="configs/pipeline_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline():
    cfg   = load_config()
    start = time.time()

    logger.info("DS Project 1 — EDA & Feature Engineering Pipeline")
    logger.info("=" * 50)

    # LOAD
    logger.info("Loading dataset...")
    df_raw = pd.read_excel(cfg["pipeline"]["dataset"])
    logger.info(f"  {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns loaded")

    # PHASE 1
    logger.info("\nPhase 1 — Input Fidelity")
    validate_raw_input(df_raw)
    df = handle_missing_data(df_raw.copy())
    df = handle_outliers(df, save_plots=True)
    check_distribution_variance(df_raw, df)
    os.makedirs("data/interim", exist_ok=True)
    df.to_csv(cfg["pipeline"]["output_interim"], index=False)
    logger.info("  Missing values imputed, outliers winsorized")

    # PHASE 2
    logger.info("\nPhase 2 — Vectorized Computation")
    df = vectorized_date_features(df)
    df = vectorized_revenue_features(df)
    df_encoded = encode_features(df)
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).copy()
    result = remove_collinearity(numeric_cols, save_heatmap=True)
    if isinstance(result, tuple):
        _, dropped = result
        df_encoded.drop(columns=[c for c in dropped if c in df_encoded.columns], inplace=True)
    logger.info(f"  Encoded & collinearity removed → {df_encoded.shape[1]} features")

    # PHASE 3
    logger.info("\nPhase 3 — Contracts & Serving")
    df_features = add_all_engineered_features(df)
    for col in ["RevenueIntensityScore", "PriceDeviation", "SpendTier"]:
        df_encoded[col] = df_features[col].values
    validate_processed_output(df_encoded)

    df_split = df.copy()
    for col in ["RevenueIntensityScore", "PriceDeviation", "SpendTier"]:
        df_split[col] = df_features[col].values
    df_train, df_test = create_temporal_train_test_split(df_split)
    df_train = add_event_timestamp(df_train)
    df_test  = add_event_timestamp(df_test)
    leakage_ok = check_for_leakage(df_train, df_test)

    push_to_offline_store(df_train)
    push_to_online_store(df_train)

    os.makedirs("data/processed", exist_ok=True)
    df_encoded.to_csv(cfg["pipeline"]["output_processed"], index=False)

    # SUMMARY
    elapsed = time.time() - start
    logger.info("\n" + "=" * 50)
    logger.info("Pipeline complete")
    logger.info(f"  Features  : {df_raw.shape[1]} raw → {df_encoded.shape[1]} final")
    logger.info(f"  Train/Test: {len(df_train)} / {len(df_test)}")
    logger.info(f"  Leakage   : {'PASSED' if leakage_ok else 'FAILED'}")
    logger.info(f"  Time      : {elapsed:.1f}s")
    logger.info(f"  Log       : logs/pipeline.log")
    logger.info("=" * 50)

    return df_encoded, df_train, df_test


if __name__ == "__main__":
    run_pipeline()
