"""
tests/test_pipeline.py — Pipeline test suite (unittest-compatible)
Run: python3 -m unittest tests/test_pipeline.py -v
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.input.missing_data       import handle_missing_data
from src.input.outlier_handler     import handle_outliers, compute_iqr_bounds
from src.process.encoder           import encode_features, label_encode, one_hot_encode
from src.process.collinearity      import build_correlation_matrix, find_collinear_pairs
from src.output.point_in_time      import (create_temporal_train_test_split,
                                            check_for_leakage, add_event_timestamp)
from features.engineered.features  import add_all_engineered_features

RAW_PATH = "data/raw/dataset.xlsx"

def make_small_df():
    return pd.DataFrame({
        "OrderID":       ["ORD001","ORD002","ORD003","ORD004","ORD005"],
        "Date":          pd.to_datetime(["2023-01-01","2023-03-01","2023-06-01","2024-01-01","2024-06-01"]),
        "CustomerID":    ["C001","C002","C003","C004","C005"],
        "Product":       ["Phone","Monitor","Tablet","Chair","Printer"],
        "Quantity":      [1,2,3,4,5],
        "UnitPrice":     [100.0,200.0,300.0,400.0,500.0],
        "ShippingAddress":["A1","A2","A3","A4","A5"],
        "PaymentMethod": ["Cash","Online","Credit Card","Debit Card","Gift Card"],
        "OrderStatus":   ["Pending","Shipped","Delivered","Returned","Cancelled"],
        "TrackingNumber":["T1","T2","T3","T4","T5"],
        "ItemsInCart":   [1,2,3,4,5],
        "CouponCode":    ["SAVE10",None,"FREESHIP",None,"WINTER15"],
        "ReferralSource":["Email","Google","Facebook","Instagram","Referral"],
        "TotalPrice":    [100.0,400.0,900.0,1600.0,2500.0],
    })


class TestMissingData(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_excel(RAW_PATH)
        self.small = make_small_df()

    def test_no_nulls_after_imputation(self):
        cleaned = handle_missing_data(self.df)
        skip = ["OrderID","Date","CustomerID","ShippingAddress","TrackingNumber"]
        check_cols = [c for c in cleaned.columns if c not in skip]
        self.assertEqual(cleaned[check_cols].isnull().sum().sum(), 0,
                         "Nulls remain after imputation")

    def test_couponcode_filled(self):
        cleaned = handle_missing_data(self.df)
        self.assertEqual(cleaned["CouponCode"].isnull().sum(), 0)

    def test_row_count_not_dropped_for_categorical(self):
        cleaned = handle_missing_data(self.df)
        self.assertEqual(len(cleaned), len(self.df))


class TestOutlierHandler(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_excel(RAW_PATH)

    def test_iqr_bounds_catches_extreme(self):
        s = pd.Series([1,2,3,4,5,100])
        lower, upper = compute_iqr_bounds(s)
        self.assertLess(upper, 100, "IQR upper bound should be below 100")

    def test_no_values_outside_bounds(self):
        cleaned = handle_outliers(self.df, save_plots=False)
        for col in ["Quantity","UnitPrice","ItemsInCart","TotalPrice"]:
            lower, upper = compute_iqr_bounds(self.df[col])
            self.assertGreaterEqual(cleaned[col].min(), lower - 0.01)
            self.assertLessEqual(cleaned[col].max(), upper + 0.01)

    def test_row_count_unchanged(self):
        cleaned = handle_outliers(self.df, save_plots=False)
        self.assertEqual(len(cleaned), len(self.df), "Winsorization must not drop rows")


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.small = make_small_df()

    def test_orderstatus_label_encoded(self):
        encoded = label_encode(self.small)
        self.assertIn("OrderStatus_encoded", encoded.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(encoded["OrderStatus_encoded"]))

    def test_ohe_creates_binary_columns(self):
        encoded = one_hot_encode(self.small)
        ohe_cols = [c for c in encoded.columns if any(
            c.startswith(p+"_") for p in ["Product","PaymentMethod","ReferralSource","CouponCode"])]
        for col in ohe_cols:
            vals = set(encoded[col].unique())
            self.assertTrue(vals.issubset({0,1}), f"{col} has non-binary values: {vals}")

    def test_nominal_not_label_encoded(self):
        encoded = encode_features(self.small)
        self.assertNotIn("PaymentMethod_encoded", encoded.columns)


class TestCollinearity(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_excel(RAW_PATH)

    def test_no_collinear_pairs_after_removal(self):
        from src.process.collinearity import remove_collinearity
        numeric = self.df.select_dtypes(include=[np.number])
        result = remove_collinearity(numeric, save_heatmap=False)
        df_clean = result[0] if isinstance(result, tuple) else result
        corr = build_correlation_matrix(df_clean)
        pairs = find_collinear_pairs(corr, threshold=0.80)
        self.assertEqual(len(pairs), 0, f"Collinear pairs remain: {pairs}")

    def test_correlation_matrix_symmetric(self):
        numeric = self.df.select_dtypes(include=[np.number])
        corr = build_correlation_matrix(numeric)
        np.testing.assert_array_almost_equal(corr.values, corr.values.T)


class TestPointInTime(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_excel(RAW_PATH)

    def test_no_leakage(self):
        df_train, df_test = create_temporal_train_test_split(self.df)
        df_train = add_event_timestamp(df_train)
        df_test  = add_event_timestamp(df_test)
        self.assertTrue(check_for_leakage(df_train, df_test))

    def test_train_before_test(self):
        df_train, df_test = create_temporal_train_test_split(self.df)
        self.assertLessEqual(df_train["Date"].max(), df_test["Date"].min())

    def test_full_dataset_covered(self):
        df_train, df_test = create_temporal_train_test_split(self.df)
        self.assertEqual(len(df_train) + len(df_test), len(self.df))


class TestEngineeredFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_excel(RAW_PATH)

    def test_three_features_added(self):
        df_feat = add_all_engineered_features(self.df)
        for col in ["RevenueIntensityScore","PriceDeviation","SpendTier"]:
            self.assertIn(col, df_feat.columns)

    def test_revenue_intensity_non_negative(self):
        df_feat = add_all_engineered_features(self.df)
        self.assertTrue((df_feat["RevenueIntensityScore"] >= 0).all())

    def test_spend_tier_range(self):
        df_feat = add_all_engineered_features(self.df)
        self.assertTrue(df_feat["SpendTier"].isin([0,1,2,3]).all())

    def test_no_nulls_from_features(self):
        df_feat = add_all_engineered_features(self.df)
        for col in ["RevenueIntensityScore","PriceDeviation","SpendTier"]:
            self.assertEqual(df_feat[col].isnull().sum(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
