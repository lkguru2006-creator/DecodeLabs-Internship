"""
src/output/schema_contracts.py
Runtime validation contracts — checks types, ranges, and allowed values.
Logs violations to contracts/failure_logs/failure_cases.log.
"""

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

FAILURE_LOG = "contracts/failure_logs/failure_cases.log"

VALID_PRODUCTS  = {"Monitor", "Phone", "Tablet", "Chair", "Printer", "Laptop", "Keyboard", "Desk"}
VALID_PAYMENT   = {"Debit Card", "Online", "Credit Card", "Gift Card", "Cash"}
VALID_STATUS    = {"Pending", "Shipped", "Delivered", "Returned", "Cancelled"}
VALID_REFERRAL  = {"Instagram", "Referral", "Email", "Facebook", "Google"}


def validate_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    errors = []

    if df["Quantity"].lt(1).any():
        errors.append(f"Quantity < 1 found in {df['Quantity'].lt(1).sum()} rows")
    if df["UnitPrice"].le(0).any():
        errors.append(f"UnitPrice <= 0 found in {df['UnitPrice'].le(0).sum()} rows")
    if df["TotalPrice"].le(0).any():
        errors.append(f"TotalPrice <= 0 found in {df['TotalPrice'].le(0).sum()} rows")

    bad_products = ~df["Product"].isin(VALID_PRODUCTS)
    if bad_products.any():
        errors.append(f"Unknown Product values: {list(df.loc[bad_products, 'Product'].unique())}")

    bad_status = ~df["OrderStatus"].isin(VALID_STATUS)
    if bad_status.any():
        errors.append(f"Unknown OrderStatus values: {list(df.loc[bad_status, 'OrderStatus'].unique())}")

    if errors:
        _write_failures(errors, "raw_input")
        for e in errors:
            logger.warning(f"Contract violation: {e}")
    else:
        logger.debug("Raw input schema passed")

    return df


def validate_processed_output(df: pd.DataFrame) -> pd.DataFrame:
    errors = []

    for col in ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]:
        if col not in df.columns:
            continue
        nulls = df[col].isnull().sum()
        if nulls > 0:
            errors.append(f"'{col}' has {nulls} null values in processed output")
        if df[col].lt(0).any():
            errors.append(f"'{col}' has negative values")

    if errors:
        _write_failures(errors, "processed_output")
        for e in errors:
            logger.warning(f"Contract violation: {e}")
    else:
        logger.debug("Processed output schema passed")

    return df


def _write_failures(errors: list, stage: str):
    os.makedirs(os.path.dirname(FAILURE_LOG), exist_ok=True)
    with open(FAILURE_LOG, "a") as f:
        f.write(f"\n{'='*50}\nSTAGE: {stage}\n{'='*50}\n")
        for e in errors:
            f.write(f"  {e}\n")
