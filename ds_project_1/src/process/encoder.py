"""
src/process/encoder.py
Encodes categorical features into coordinate space.
  - Ordinal (known order) → Label Encoding
  - Nominal (no order)    → One-Hot Encoding
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# OrderStatus has a natural progression — safe to label encode
ORDINAL_COLUMNS = {
    "OrderStatus": ["Pending", "Shipped", "Delivered", "Returned", "Cancelled"]
}

# No order implied — one-hot encode to avoid false ordinality
NOMINAL_COLUMNS = ["Product", "PaymentMethod", "ReferralSource", "CouponCode"]

# Drop before encoding — identifiers carry no signal
DROP_COLUMNS = ["OrderID", "CustomerID", "ShippingAddress", "TrackingNumber", "Date"]


def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, order in ORDINAL_COLUMNS.items():
        if col not in df.columns:
            continue
        cat       = pd.Categorical(df[col], categories=order, ordered=True)
        df[col + "_encoded"] = cat.codes
        logger.debug(f"Label encoded '{col}'")
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    cols = [c for c in NOMINAL_COLUMNS if c in df.columns]
    if not cols:
        return df
    df = pd.get_dummies(df, columns=cols, prefix=cols, drop_first=False, dtype=int)
    logger.debug(f"One-hot encoded: {cols}")
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    cols = [c for c in DROP_COLUMNS if c in df.columns]
    return df.drop(columns=cols)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_identifier_columns(df)
    df = label_encode(df)
    df = one_hot_encode(df)
    return df
