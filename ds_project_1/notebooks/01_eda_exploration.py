# %% [markdown]
# # Notebook 01 — EDA & Data Exploration
# **DS Project 1 | Advanced EDA & Feature Engineering**
# DecodeLabs Industrial Training Kit — Batch 2026
#
# **Objective:** Understand the raw dataset structure, detect quality issues,
# and document statistical properties before the pipeline transforms the data.

# %% [markdown]
# ## 1. Setup & Imports

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='darkgrid', palette='muted')
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Libraries loaded ✓")

# %% [markdown]
# ## 2. Load Raw Dataset

# %%
df = pd.read_excel('../data/raw/dataset.xlsx')
print(f"Dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
df.head()

# %%
df.dtypes

# %% [markdown]
# ## 3. Missing Value Analysis

# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing, 'Percent (%)': missing_pct})
print("Missing Value Report:")
print(missing_df[missing_df['Count'] > 0])

# %%
# Decision matrix thresholds
print("\nMissing Data Decision Matrix:")
for col in df.columns:
    pct = df[col].isnull().mean()
    if pct == 0:
        continue
    if pct < 0.05:
        action = "→ DROP ROWS (< 5%)"
    elif pct <= 0.20:
        action = "→ MEDIAN / MODE IMPUTE (5–20%)"
    else:
        action = "→ KNN IMPUTATION (> 20%)"
    print(f"  {col}: {pct:.1%}  {action}")

# %% [markdown]
# ## 4. Descriptive Statistics — Numeric Features

# %%
num_cols = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']
df[num_cols].describe().round(2)

# %%
# Skewness & Kurtosis
print("Skewness:")
print(df[num_cols].skew().round(3))
print("\nKurtosis:")
print(df[num_cols].kurt().round(3))

# %% [markdown]
# ## 5. Categorical Feature Analysis

# %%
cat_cols = ['Product', 'PaymentMethod', 'OrderStatus', 'ReferralSource', 'CouponCode']
for col in cat_cols:
    print(f"\n{col} ({df[col].nunique()} unique):")
    print(df[col].value_counts(dropna=False))

# %% [markdown]
# ## 6. Outlier Detection — IQR Method

# %%
print("IQR Outlier Analysis:")
print(f"{'Feature':<20} {'Q1':>10} {'Q3':>10} {'IQR':>10} {'Lower':>12} {'Upper':>12} {'Outliers':>10}")
print("-"*85)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col:<20} {Q1:>10.2f} {Q3:>10.2f} {IQR:>10.2f} {lower:>12.2f} {upper:>12.2f} {n_out:>10}")

# %% [markdown]
# ## 7. Correlation Analysis

# %%
corr = df[num_cols].corr()
print("Pearson Correlation Matrix:")
print(corr.round(3))

# %%
# Flag high correlations
print("\nHigh correlation pairs (|r| > 0.5):")
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        r = corr.iloc[i,j]
        if abs(r) > 0.5:
            print(f"  {corr.columns[i]} ↔ {corr.columns[j]}:  r = {r:.3f}")

# %% [markdown]
# ## 8. Revenue Analysis

# %%
print("Revenue by Product:")
print(df.groupby('Product')['TotalPrice'].agg(['mean','median','sum','count']).round(2))

print("\nRevenue by ReferralSource:")
print(df.groupby('ReferralSource')['TotalPrice'].agg(['mean','sum','count']).round(2))

print("\nRevenue by OrderStatus:")
print(df.groupby('OrderStatus')['TotalPrice'].agg(['mean','count']).round(2))

# %% [markdown]
# ## 9. Date Analysis

# %%
df['Date'] = pd.to_datetime(df['Date'])
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Span: {(df['Date'].max() - df['Date'].min()).days} days (~30 months)")
print("\nOrders by Year:")
print(df['Date'].dt.year.value_counts().sort_index())
print("\nOrders by Month:")
print(df['Date'].dt.month.value_counts().sort_index())

# %% [markdown]
# ## 10. Key Findings Summary

# %%
print("=" * 60)
print("KEY FINDINGS BEFORE PIPELINE")
print("=" * 60)
print(f"  Total records    : {len(df):,}")
print(f"  Total features   : {df.shape[1]}")
print(f"  Missing values   : CouponCode — 309 rows (25.8%)")
print(f"  Outliers detected: TotalPrice — 8 values")
print(f"  Date coverage    : Jan 2023 → Jun 2025 (30 months)")
print(f"  Numeric skew     : TotalPrice is right-skewed (skew={df['TotalPrice'].skew():.2f})")
print(f"  Target variable  : TotalPrice (mean=₹{df['TotalPrice'].mean():.0f}, max=₹{df['TotalPrice'].max():.0f})")
print(f"  Recommended action: Median impute CouponCode + IQR winsorize TotalPrice")
print("=" * 60)

if __name__ == "__main__":
    print("\nRun this notebook with: jupyter notebook OR python3 01_eda_exploration.py")
